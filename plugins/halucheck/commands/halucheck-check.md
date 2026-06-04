---
description: Verify a factual claim against supporting evidence before asserting it
---

Use HaluCheck's claim verifier — the "give Claude a check it can run" primitive — to ground a factual claim against evidence the user provides (a doc passage, a search result, prior conversation).

Steps:
1. Identify the claim and evidence:
   - If `$ARGUMENTS` is formatted as `claim: ... evidence: ...`, parse it.
   - Else, ask: "What claim should I verify, and what evidence (paste it)?"
2. Call the `halucheck_check` MCP tool with `(claim, evidence)`.
3. Report:
   - `supported`: true/false
   - `confidence`: 0-1
   - The judge's margin
4. If `supported` is false, say so plainly and recommend the user not state the claim without further evidence. Do not paper over.

Best used before asserting any claim that has nearby supporting text — RAG outputs, citations, paraphrases of search results, recalled doc snippets.

$ARGUMENTS
