"""Smart tool-output trimming.

A pure-heuristic, deterministic, dependency-free trimmer for verbose tool
output (build logs, test runs, command stdout/stderr). The cost of an agent
feeding 50 KB of raw build log into its own context is one full re-read of
that 50 KB on every subsequent message. Trimming it to ~4 KB while
preserving the errors saves the agent ~12 K tokens per turn going forward.

What we keep:
  - Lines that look like errors / failures / warnings (regex match)
  - First N and last N lines (configurable head/tail budget)
  - One representative line of any run of consecutive duplicates

What we drop or shrink:
  - Consecutive identical lines collapsed with a "[ x N ]" marker
  - Long opaque tokens (base64/hex/UUID-like runs > 60 chars) redacted
  - The vast middle of the log, replaced with "... [N lines omitted] ..."

Stays deterministic — no ML, no LLM call, no sampling. Same input → same output.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable


# Lines we always try to preserve (in priority order). Case-insensitive.
ERROR_PATTERNS = (
    r"\b(error|fatal|panic|assert(ion)?(\s+failed)?|exception|traceback|fail(ed|ure)?|stack\s*trace)\b",
    r"^\s*at\s+\S+",                       # Java/Node stack frames
    r"^\s*File\s+\".*?\",\s+line\s+\d+",   # Python tracebacks
    r"\b(unhandled|uncaught|critical|severe|panic|abort(ed)?)\b",
    r"\b(?:HTTP/)?[45]\d\d\b",            # HTTP 4xx/5xx status codes
    r"^\s*\[(ERROR|FATAL|WARN(ING)?)\]",   # Common log prefixes
    r"^\s*X\s",                             # Test-runner "X" fail markers
    r"\bFAIL(ED)?\b",
    r"\bSyntaxError\b|\bTypeError\b|\bValueError\b|\bKeyError\b|\bAttributeError\b",
)

_ERROR_RE = re.compile("|".join(f"(?:{p})" for p in ERROR_PATTERNS),
                        re.IGNORECASE)


# Long opaque tokens to redact (base64 / hex / UUIDs / very long IDs)
_OPAQUE_RE = re.compile(
    r"[A-Fa-f0-9]{60,}"                    # long hex
    r"|[A-Za-z0-9+/=]{80,}"                # long base64
    r"|sk-[A-Za-z0-9-_]{32,}"              # OpenAI-style secret keys
    r"|gh[pousr]_[A-Za-z0-9]{30,}"        # GitHub tokens
)


@dataclass
class CapResult:
    text: str
    original_chars: int
    trimmed_chars: int
    original_lines: int
    kept_lines: int
    dropped_lines: int
    deduped_runs: int
    redacted_tokens: int
    strategy: str


def _is_error_line(line: str) -> bool:
    return bool(_ERROR_RE.search(line))


def _collapse_dupes(lines: list[str]) -> tuple[list[str], int]:
    """Collapse runs of identical adjacent lines into 'LINE [ xN ]'.

    Returns (collapsed_lines, num_runs_collapsed)."""
    if not lines:
        return [], 0
    out: list[str] = []
    runs = 0
    i = 0
    n = len(lines)
    while i < n:
        j = i + 1
        while j < n and lines[j] == lines[i]:
            j += 1
        count = j - i
        if count > 1:
            runs += 1
            out.append(f"{lines[i].rstrip()}    [ x{count} ]")
        else:
            out.append(lines[i])
        i = j
    return out, runs


def _redact_opaque(line: str) -> tuple[str, int]:
    """Redact long opaque blobs in a line. Returns (line, num_redactions)."""
    count = [0]
    def _sub(m):
        count[0] += 1
        tok = m.group(0)
        return f"<{len(tok)}-char-blob-redacted>"
    new = _OPAQUE_RE.sub(_sub, line)
    return new, count[0]


def cap(
    text: str,
    max_chars: int = 4000,
    head_lines: int = 25,
    tail_lines: int = 50,
    preserve_errors: bool = True,
    redact_opaque: bool = True,
) -> CapResult:
    """Trim `text` to roughly `max_chars` characters while preserving the
    signal an LLM agent will want.

    Strategy:
      1. If text already fits, return unchanged (fast path).
      2. Otherwise: split to lines, collapse runs of duplicates, redact
         long opaque blobs (base64/hex/secrets) per `redact_opaque`.
      3. Identify error lines per `preserve_errors`.
      4. Greedy budget: head + tail + error lines, filling up to max_chars.
      5. Insert "... [N lines omitted] ..." where gaps fall.
    """
    original_chars = len(text)
    if original_chars <= max_chars:
        return CapResult(
            text=text, original_chars=original_chars,
            trimmed_chars=original_chars, original_lines=text.count("\n") + 1,
            kept_lines=text.count("\n") + 1, dropped_lines=0,
            deduped_runs=0, redacted_tokens=0, strategy="no_trim_needed",
        )

    raw_lines = text.splitlines()
    original_lines = len(raw_lines)

    # 1. Collapse dupes
    deduped, deduped_runs = _collapse_dupes(raw_lines)

    # 2. Redact opaque blobs in every kept line
    redacted_total = 0
    if redact_opaque:
        new_deduped = []
        for ln in deduped:
            ln, n = _redact_opaque(ln)
            redacted_total += n
            new_deduped.append(ln)
        deduped = new_deduped

    # 3. Pick which lines to keep
    n_after_dedupe = len(deduped)
    error_idx = (
        {i for i, ln in enumerate(deduped) if _is_error_line(ln)}
        if preserve_errors else set()
    )
    head_idx = set(range(min(head_lines, n_after_dedupe)))
    tail_idx = set(range(max(0, n_after_dedupe - tail_lines), n_after_dedupe))

    # 4. Greedy fill with priority: errors first (with 2 lines of context
    # above and below each error for traceback continuity), then head, then
    # tail. Errors must never be pushed out by routine head/tail budget.
    char_budget = max_chars - 80  # reserve for "[N lines omitted]" markers
    kept_set: set[int] = set()

    def _try_add(i: int) -> bool:
        if i < 0 or i >= n_after_dedupe or i in kept_set:
            return True
        cost = len(deduped[i]) + 1
        nonlocal char_budget
        if cost > char_budget:
            return False
        kept_set.add(i)
        char_budget -= cost
        return True

    # Pass 1: error lines + 2-line context window each side
    for i in sorted(error_idx):
        for off in (-2, -1, 0, 1, 2):
            if not _try_add(i + off):
                break

    # Pass 2: head — adds nothing if errors already used the budget
    for i in sorted(head_idx):
        if not _try_add(i):
            break

    # Pass 3: tail (from outside-in, so we always get the most-recent lines)
    for i in sorted(tail_idx, reverse=True):
        if not _try_add(i):
            break

    kept = sorted(kept_set)

    # 5. Build output with gap markers
    output_parts: list[str] = []
    prev = -1
    for i in kept:
        if prev >= 0 and i - prev > 1:
            gap = i - prev - 1
            output_parts.append(f"... [{gap} lines omitted] ...")
        output_parts.append(deduped[i])
        prev = i
    if kept and kept[-1] < n_after_dedupe - 1:
        gap = n_after_dedupe - 1 - kept[-1]
        output_parts.append(f"... [{gap} lines omitted] ...")

    out_text = "\n".join(output_parts)
    return CapResult(
        text=out_text,
        original_chars=original_chars,
        trimmed_chars=len(out_text),
        original_lines=original_lines,
        kept_lines=len(kept),
        dropped_lines=n_after_dedupe - len(kept),
        deduped_runs=deduped_runs,
        redacted_tokens=redacted_total,
        strategy=(f"head={len(head_idx)}, tail={len(tail_idx)}, "
                  f"errors={len(error_idx)}, dedupe_runs={deduped_runs}"),
    )


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # Tiny self-test
    sample = """
INFO: Starting build
DEBUG: Compiling foo.py
DEBUG: Compiling bar.py
DEBUG: Compiling bar.py
DEBUG: Compiling bar.py
DEBUG: Compiling bar.py
ERROR: Test test_streaming_proxy FAILED at line 87
Traceback (most recent call last):
  File "test.py", line 87
    assert prelude_present
AssertionError
INFO: Cleanup
""".strip()
    r = cap(sample, max_chars=400)
    print(f"original={r.original_chars} → trimmed={r.trimmed_chars}")
    print(f"lines: {r.original_lines} → {r.kept_lines} kept, {r.dropped_lines} dropped")
    print(f"deduped_runs={r.deduped_runs}, redacted={r.redacted_tokens}")
    print(f"strategy: {r.strategy}")
    print("---")
    print(r.text)
