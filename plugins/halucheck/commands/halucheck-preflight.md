---
description: Check whether a question is likely to cause an LLM to hallucinate
---

Use `halucheck_preflight` to check the question in $ARGUMENTS (or the user's last message if no argument).

Steps:
1. Take the question from `$ARGUMENTS`, or fall back to the user's most recent message.
2. Call `halucheck_preflight` with the question.
3. Report the result: flagged (true/false), margin, and verdict text.
4. If flagged, briefly explain WHY the question is likely problematic (underspecified, false premise, ambiguous, etc.) and offer to call `halucheck_rephrase`.
5. If not flagged, report that the question is structurally answerable and tell the user they can proceed.

$ARGUMENTS
