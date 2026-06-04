---
description: Rephrase a hallucination-inducing question to make it safer to answer
---

Use the HaluCheck rephraser to rewrite the question in $ARGUMENTS (or the user's last message if no argument given) so it can be answered safely.

Steps:
1. Take the question from `$ARGUMENTS`, or fall back to the user's most recent message in this conversation.
2. Call `halucheck_preflight` first to check whether the question is actually hallucination-inducing — if not, tell the user no rewrite is needed.
3. Call `halucheck_rephrase` to produce a rewritten version.
4. Check `safe_to_use` — if false, the rephrase drifted too far from the original intent; report that.
5. If safe, show the user the rewritten question and explain what was added (clarifying question, premise check, abstention instruction).
6. Offer to use the rewritten version going forward.

$ARGUMENTS
