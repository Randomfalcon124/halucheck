"""UserPromptSubmit hook: inject user-tunable style preferences + subagent
delegation reminder into every prompt.

Reads `~/.claude/user_preferences.json`. Edit that file to tune — changes
take effect on the next prompt, no restart needed (the harness re-reads
settings + spawns the hook fresh each time).

Multiple UserPromptSubmit hooks run in sequence and their additionalContext
gets concatenated — so this layers cleanly with halucheck/preflight.
"""
import json
import os
import sys
from pathlib import Path


CONFIG_PATH = Path.home() / ".claude" / "user_preferences.json"


def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _conciseness_text(level: int) -> str:
    if level >= 9:
        return ("Be as terse as possible. One-liner answers when possible. "
                "Never add preambles like 'Here's what...' or summaries at the end.")
    if level >= 7:
        return ("Be very concise. Skip preambles. Get to the point in the "
                "first sentence. No restating the question.")
    if level >= 4:
        return "Be concise but include enough context to be actionable."
    return "Provide full explanations including background and context."


def _complexity_text(level: int) -> str:
    if level >= 9:
        return ("Assume expert audience. Use domain terminology without "
                "defining it. Skip motivational/background framing.")
    if level >= 7:
        return ("Assume technical audience. Define only unusual terms. "
                "Reference common concepts without re-explaining.")
    if level >= 4:
        return ("Define moderately technical terms when introduced. "
                "Provide concrete examples.")
    return ("ELI5 mode: define every technical term, use analogies, "
            "avoid jargon.")


def _tone_text(tone: str) -> str:
    t = (tone or "").lower()
    if t == "direct":
        return ("Use direct tone. Skip qualifiers and hedges like 'I think' "
                "or 'it might be'. State conclusions plainly. Disagree when "
                "you disagree.")
    if t == "collegial":
        return "Use collegial tone. Suggest alternatives diplomatically."
    if t == "formal":
        return "Use formal tone. Avoid contractions and casual phrasing."
    return ""


def _subagent_reminder_text() -> str:
    return ("Reminder: per project CLAUDE.md, delegate exploration / "
            "open-ended search to a subagent rather than loading files inline. "
            "Subagents consume their own context; the parent keeps theirs.")


def main():
    # Drain stdin (harness sends a payload but we don't actually use it)
    try:
        sys.stdin.read()
    except Exception:
        pass

    cfg = _load_config()
    parts: list[str] = []

    conciseness = cfg.get("conciseness_level")
    if isinstance(conciseness, (int, float)):
        line = _conciseness_text(int(conciseness))
        if line:
            parts.append(line)

    complexity = cfg.get("complexity_level")
    if isinstance(complexity, (int, float)):
        line = _complexity_text(int(complexity))
        if line:
            parts.append(line)

    tone = cfg.get("tone")
    if tone:
        line = _tone_text(tone)
        if line:
            parts.append(line)

    if cfg.get("remind_subagent_delegation"):
        parts.append(_subagent_reminder_text())

    custom = cfg.get("custom_preamble")
    if custom:
        parts.append(str(custom))

    # First-run nudge: if setup hasn't been completed, prepend a one-time
    # reminder telling Claude to suggest /setup_prefs. Cleared once
    # _setup_complete: true is in the config.
    if not cfg.get("_setup_complete") and not cfg.get("_setup_dismissed"):
        parts.insert(0, ("First-run: the user has not configured preferences "
                          "yet. Suggest they run /setup_prefs (interactive "
                          "survey) once at the end of your response."))

    if not parts:
        return

    msg = "[User preferences]: " + " ".join(parts)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": msg,
        },
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
