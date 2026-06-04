---
description: HaluCheck-powered prompt audit — run a draft prompt through the proxy and report what to fix before sending
---

You have access to a running HaluCheck proxy at `http://localhost:8000`. Use it to audit the user's prompt and report what to fix.

## Steps

1. **The prompt is in `$ARGUMENTS`.** If empty, ask: "What prompt do you want me to audit?"

2. **Hit `/v1/chat/completions/preflight`** with one Bash call:
   ```bash
   curl -s http://localhost:8000/v1/chat/completions/preflight \
       -X POST -H 'Content-Type: application/json' \
       -d "$(python -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}]}))' "$ARGUMENTS")"
   ```
   (or use `python -c` inline with `httpx.post` if curl isn't handy)

3. **Parse the JSON and report concisely** to the user, in this exact structure:

   **Verdict**: one line — `clean`, `flagged (margin +X.X)`, or `flagged + rephraseable`.

   **Why** (only if flagged): one sentence describing the abs_check verdict in plain language (underspecified, false premise, opinion-as-fact, etc.). Use the margin sign and rough magnitude to gauge confidence.

   **Rephrased version** (only if `rephrased_question` is present): show the rephrased prompt verbatim in a code block. This is what the user should send instead.

   **Subagent recommendation** (only if `recommend_subagent` is true): one short sentence: "This query is exploration-shaped — delegate to a subagent rather than handling inline (reason: <subagent_reason>)."

4. **If the proxy is unreachable** (curl errors / connection refused), say so plainly and tell the user to run:
   ```
   cd C:/Users/Akshay/Projects/Neuron/gpt2_vg && \
       HALUCHECK_JUDGE_LORA=ckpt_judge_enriched \
       HALUCHECK_JUDGE_LORA_V2=ckpt_judge_enriched_v2 \
       python halucheck_proxy.py
   ```

## Rules

- Be terse. The user is already in a chat; don't ramble.
- Do NOT send the prompt to any LLM yourself. This command audits, it does not execute.
- Do NOT add caveats like "this is just a heuristic" unless the margin is very close to zero (between -1 and +1).
- If verdict is `clean`, report only "Verdict: clean — safe to send as-is" and stop.

$ARGUMENTS
