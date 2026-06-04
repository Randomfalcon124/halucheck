"""Stop hook: run the judge on the final assistant message.

Triggered when Claude finishes a response. Judges the (last_user_message,
final_assistant_message) pair and emits a system message if hallucination
likely. Runs async (configured in plugin.json) so it doesn't block the user.

Strategy:
  1. If HALUCHECK_PROXY_URL is reachable, hit /v1/judge_route (uses both
     v1 and v2 judges so surface-form FPs like "Paris is the capital of
     France." get suppressed). Fast — ~58 ms p50 round-trip.
  2. Else fall back to the CLI judge (~3-5 s startup, no FP suppression).
"""
import json
import os
import sys
import subprocess
from pathlib import Path

try:
    import urllib.request
    import urllib.error
    _HAS_URLLIB = True
except Exception:
    _HAS_URLLIB = False


PROXY_URL = os.getenv("HALUCHECK_PROXY_URL", "http://localhost:8000")


def _try_proxy_judge(question: str, response: str) -> dict | None:
    """Try /v1/judge_route on the local proxy. Returns None on failure
    so the caller can fall back to the CLI."""
    if not _HAS_URLLIB:
        return None
    try:
        body = json.dumps({"question": question, "response": response}).encode()
        req = urllib.request.Request(
            f"{PROXY_URL}/v1/judge_route",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return

    # Stop hook payload includes session_id and recent message history
    # Find the last user message and last assistant message
    messages = payload.get("messages", [])
    if not messages:
        return

    last_user = None
    last_asst = None
    for m in reversed(messages):
        role = m.get("role")
        if role == "user" and last_user is None:
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
            last_user = content
        elif role == "assistant" and last_asst is None:
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
            last_asst = content
        if last_user and last_asst:
            break

    if not last_user or not last_asst:
        return

    # Try proxy /v1/judge_route first (fast + FP suppression)
    data = _try_proxy_judge(last_user, last_asst)
    judge_source = "proxy /v1/judge_route"

    # Fall back to CLI if proxy unreachable
    if data is None:
        plugin_root = Path(__file__).resolve().parent.parent.parent.parent
        cli = plugin_root / "halucheck_cli.py"
        if not cli.exists():
            return
        try:
            result = subprocess.run(
                [sys.executable, str(cli), "judge", last_user, last_asst],
                capture_output=True, text=True, timeout=8,
            )
            if result.returncode != 0:
                return
            data = json.loads(result.stdout)
            judge_source = "CLI fallback"
        except Exception:
            return

    if not data.get("hallucination"):
        return  # silent — clean response

    # Build message; surface routing rule if available so user can see whether
    # v2 override was considered.
    rule = data.get("rule", "")
    rule_str = f" [rule: {rule}]" if rule else ""
    output = {
        "systemMessage": (
            f"⚠️ HaluCheck: the final response may contain a hallucination "
            f"(judge margin {data.get('margin')}{rule_str}, via {judge_source}). "
            f"Suggested actions: ask the model to cite sources, "
            f"verify the answer against retrieved documents, or rephrase the question."
        ),
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
