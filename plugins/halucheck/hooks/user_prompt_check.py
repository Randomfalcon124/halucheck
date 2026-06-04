"""UserPromptSubmit hook: abs_check the user's prompt; surface a hint if flagged.

Strategy:
  1. Try the running proxy at HALUCHECK_PROXY_URL (default http://localhost:8000)
     via /v1/chat/completions/preflight. ~100 ms p50. Path-independent.
  2. Fall back to halucheck_cli.py preflight if the proxy is down AND we can
     find the CLI under HALUCHECK_CKPT_ROOT (set by plugin.json).
  3. Silent exit on any error — never block the user.

Non-blocking: injects `additionalContext`; doesn't prevent submission.
"""
import json
import os
import sys
import subprocess
from pathlib import Path

try:
    import urllib.request
    _HAS_URLLIB = True
except Exception:
    _HAS_URLLIB = False


PROXY_URL = os.getenv("HALUCHECK_PROXY_URL", "http://localhost:8000")


def _try_proxy(prompt: str) -> dict | None:
    if not _HAS_URLLIB:
        return None
    try:
        body = json.dumps({"messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(
            f"{PROXY_URL}/v1/chat/completions/preflight",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _try_cli(prompt: str) -> dict | None:
    ckpt_root = os.getenv("HALUCHECK_CKPT_ROOT")
    if not ckpt_root:
        return None
    cli = Path(ckpt_root) / "halucheck_cli.py"
    if not cli.exists():
        return None
    try:
        result = subprocess.run(
            [sys.executable, str(cli), "preflight", prompt],
            capture_output=True, text=True, timeout=8,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return

    prompt = payload.get("user_message") or payload.get("prompt") or ""
    if not prompt or len(prompt) < 10:
        return

    # Try proxy first; fall back to CLI; silently exit otherwise
    data = _try_proxy(prompt)
    if data is None:
        data = _try_cli(prompt)
    if data is None:
        return

    # Proxy preflight returns {flagged, abs_margin, rephrased_question, ...}
    # CLI preflight returns the same shape with `margin` rather than `abs_margin`
    flagged = data.get("flagged")
    if not flagged:
        return

    margin = data.get("abs_margin", data.get("margin"))
    rephrased = data.get("rephrased_question")
    recommend_sub = data.get("recommend_subagent")

    msg = (f"[HaluCheck preflight]: this prompt is flagged as potentially "
           f"hallucination-inducing (margin {margin}).")
    if rephrased:
        msg += (f" A safer rewrite is available: \"{rephrased[:200]}\". "
                f"Use it or ask the user to clarify rather than answering "
                f"the original directly.")
    else:
        msg += (" Consider asking the user for clarification, or use the "
                "halucheck_rephrase MCP tool to rewrite the question safely.")
    if recommend_sub:
        msg += (" The query is also exploration-shaped — delegate to a "
                "subagent rather than handling inline.")

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": msg,
        },
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
