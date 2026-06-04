---
description: Show HaluCheck plugin status and configuration
---

Display the HaluCheck plugin's current state:

1. List the available MCP tools (halucheck_preflight, halucheck_rephrase, halucheck_judge, halucheck_judge_rag).
2. Read the environment variables HALUCHECK_BASE_MODEL, HALUCHECK_ABS_LORA, HALUCHECK_JUDGE_LORA and report them. If any are unset, note the defaults being used.
3. Verify the LoRA checkpoints exist at their resolved paths.
4. Report whether the MCP server has been started (check for the process or recent log entry).
5. Report the configured hooks (UserPromptSubmit and Stop) and whether they're enabled.
6. Summarise the system's published quality numbers:
   - abs_check: F1 0.755 on AbstentionBench, cross-benchmark to SQuAD 2.0 F1 0.623
   - Rephraser: 22% relative hallucination reduction on flagged items
   - Judge (mixed): AUC 0.85 in-dist, 0.68 cross-dist
7. End with a tip: "Use /halucheck-judge to evaluate a response, /halucheck-rephrase to rewrite a prompt, or /halucheck-preflight to check a question before sending."
