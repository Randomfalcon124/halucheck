---
description: HaluCheck-powered hallucination check on a (question, response) pair via the running proxy
---

Run the routed judge on a (question, response) pair from the conversation.

## Steps

1. **Parse `$ARGUMENTS`.** Two acceptable forms:
   - `question=...; response=...` (semicolon-separated)
   - Free text — then identify the most recent question and response in the conversation and use those.
   If neither is clear, ask: "Which question and which response should I judge? Paste both."

2. **POST to `http://localhost:8000/v1/judge_route`**:
   ```bash
   curl -s http://localhost:8000/v1/judge_route \
       -X POST -H 'Content-Type: application/json' \
       -d "$(python -c 'import json,sys; print(json.dumps({"question":sys.argv[1],"response":sys.argv[2]}))' "$Q" "$A")"
   ```

3. **Report concisely**, in this exact shape:

   **Verdict**: `clean` or `flagged (margin +X.X, rule=<rule>)`
   **v1 margin**: ±X.XX
   **v2 margin**: ±X.XX (or "n/a" if router inactive)
   **Repeat signal**: `repeat_count=N, fp_suspected=<bool>` — show only when repeat_count > 1.
   **Cache**: `cache_hit=<bool>` — show only when true.

4. **Interpret the result for the user in one line**:
   - If `rule == v2_override_likely_FP`: "Suppressed as likely surface-form FP via v2 LoRA."
   - If `fp_suspected == true`: "Same flag has fired 3+ times — likely a false positive; stop chasing it."
   - If `hallucination == true` otherwise: brief plain-language flag.
   - Else: "Looks accurate."

5. **If the proxy is unreachable**, tell the user to start it:
   ```
   cd C:/Users/Akshay/Projects/Neuron/gpt2_vg && \
       HALUCHECK_JUDGE_LORA=ckpt_judge_enriched \
       HALUCHECK_JUDGE_LORA_V2=ckpt_judge_enriched_v2 \
       python halucheck_proxy.py
   ```

Be terse. No caveats unless |margin| < 1.

$ARGUMENTS
