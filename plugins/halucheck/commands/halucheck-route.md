---
description: Routed judge — combines primary + L4-specialised judges to suppress surface-form FPs
---

Run the routed judge (primary + secondary v2 LoRA) on the current conversation. This catches false positives the standard judge has on confident factual surface forms (the canonical case: "Paris is the capital of France." flagged as hallucination) without losing the primary judge's recall on real hallucinations.

Steps:
1. Identify the most recent (question, response) pair to judge.
2. Call `halucheck_judge_route` with those two strings.
3. Report:
   - `hallucination`: routed verdict
   - `v1_margin`, `v2_margin`: both judges' raw margins
   - `rule`: which rule fired (`v1_default`, `v2_override_likely_FP`, or `v1_only` if no v2 loaded)
4. When `router_active` is false (no v2 LoRA loaded), explain that the route fell back to v1 and the user should set `HALUCHECK_JUDGE_LORA_V2=ckpt_judge_enriched_v2` to enable routing.

Use `/halucheck-judge` if you only want the primary judge. Use `/halucheck-route` if you want both judges to vote.

$ARGUMENTS
