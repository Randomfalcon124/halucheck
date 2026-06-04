---
description: HaluCheck-powered claim verifier — is a factual claim supported by the evidence?
---

Verify a factual claim against supporting evidence using HaluCheck's HHEM backend.

## Steps

1. **Parse `$ARGUMENTS`.** Two acceptable forms:
   - `claim=...; evidence=...` (semicolon-separated)
   - Free text mixing both — then ask the user to clarify which is which.
   If empty, ask: "What claim should I verify and what evidence supports it?"

2. **POST to `http://localhost:8000/v1/check`**:
   ```bash
   curl -s http://localhost:8000/v1/check \
       -X POST -H 'Content-Type: application/json' \
       -d "$(python -c 'import json,sys; print(json.dumps({"claim":sys.argv[1],"evidence":sys.argv[2]}))' "$CLAIM" "$EVIDENCE")"
   ```

3. **Report**, exactly:

   **Supported**: yes / no
   **Score**: 0.XX (HHEM faithfulness; higher = more supported)
   **Backend**: hhem (or qwen-lora if HHEM unavailable)

4. **Interpretation**:
   - Score ≥ 0.7: "Evidence solidly supports the claim."
   - 0.3 ≤ score < 0.7: "Evidence is weak — claim is partially supported at best."
   - Score < 0.3: "Evidence does NOT support the claim. Do not assert it."

5. **If the proxy is unreachable**, give the same start command as `/smart_prompt`.

Be terse.

$ARGUMENTS
