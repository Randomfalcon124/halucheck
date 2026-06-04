---
name: halucheck
description: Use this skill when the user wants to detect or prevent LLM hallucinations using the HaluCheck sidecar. Triggers on: questions about whether a response is hallucinating, requests to rephrase a hallucination-inducing prompt, RAG-faithfulness checks, output verification, or any time the user explicitly mentions "halucheck", "hallucination check", or asks to "verify this response."
---

# HaluCheck Skill

HaluCheck is a cheap (~3 GB VRAM) hallucination sidecar that wraps any LLM with three components:

1. **`abs_check`** — input-side detector for hallucination-inducing prompts (underspecified, ambiguous, false-premise, etc.). F1 0.755 on AbstentionBench, transfers cross-benchmark to SQuAD 2.0.
2. **Rephraser** — rewrites flagged prompts so they're safer to answer. 22% relative hallucination reduction on flagged items.
3. **Output judge** — detects hallucinations in LLM responses. AUC 0.85 in-distribution (TruthfulQA), AUC 0.68 cross-distribution (AbstentionBench-style).

## When to use HaluCheck tools

| User intent | Tool | Notes |
|---|---|---|
| "Is this question safe to ask an LLM?" | `halucheck_preflight` | Run on the question; flagged=true means it's likely to elicit a hallucination |
| "This question seems problematic, can you rewrite it?" | `halucheck_rephrase` | Returns a rewritten question. Check `safe_to_use` — if false, the rewrite drifted |
| "Did the LLM hallucinate in this response?" | `halucheck_judge` | Pass (question, response). Returns hallucination verdict and margin |
| "Is this RAG response faithful to the retrieved docs?" | `halucheck_judge_rag` | Pass (question, response, context). HHEM-style faithfulness check |

## How to invoke

The tools are MCP tools exposed by the `halucheck` MCP server. You can call them directly:

```
halucheck_preflight(question="When did Marie Curie discover penicillin?")
→ {flagged: true, margin: 13.4, verdict: "hallucination-inducing"}

halucheck_rephrase(question="When did Marie Curie discover penicillin?")
→ {rephrased: "Note: this question may rest on a false premise (Marie Curie did not discover penicillin). Can you confirm what you are asking about?", composition_overlap: 0.4, safe_to_use: true}

halucheck_judge(question="What is the capital of France?", response="Berlin is the capital.")
→ {hallucination: true, margin: 6.0, verdict: "likely hallucination"}

halucheck_judge_rag(question="What's the company policy on remote work?",
                    response="Remote work is allowed up to 5 days per week.",
                    context="Policy 4.2: Remote work is permitted up to 2 days per week.")
→ {unfaithful: true, margin: 4.2, note: "RAG faithfulness mode: 'unfaithful' = response not supported by context"}
```

## Known limitations to communicate to the user

- **Cross-distribution AUC is 0.68, not 0.85.** The judge generalises from TruthfulQA + AbstentionBench-style data to similar genres, but a completely new domain (medical, legal, code) benefits from `halucheck adapt` retraining.
- **Composition guard** drops rephrases that drift too far. If `safe_to_use=false`, fall back to the original question.
- **The judge is not a fact-checker.** It catches hallucinations whose telltale signs are detectable from the response surface; it cannot independently verify facts the 1.5 B model doesn't know.
- **English-only** training data; multilingual is best-effort via the multilingual base model.

## When NOT to use HaluCheck

- For *factual* hallucinations in your domain that require world knowledge the 1.5 B model doesn't have → use a frontier LLM-as-judge or retrieval-grounded verifier
- For RAG-faithfulness on enterprise documents → consider HHEM-2.1-Open which is purpose-built and smaller
- For prompt injection / jailbreak detection → use Lakera Guard or NVIDIA Prompt Guard 2
- For comprehensive observability dashboards → use Arize Phoenix, Langfuse, or Galileo
