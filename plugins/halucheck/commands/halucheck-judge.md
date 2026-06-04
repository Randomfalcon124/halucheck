---
description: Judge whether the most recent assistant response contains a hallucination
---

Run the HaluCheck output judge on the current conversation's last (question, response) pair.

Steps:
1. Identify the most recent user question in this conversation.
2. Identify the most recent assistant response.
3. Call the `halucheck_judge` MCP tool with those two strings.
4. Display the result (hallucination verdict, margin, and verdict text).
5. If hallucination is flagged, suggest concrete next steps: ask the model to cite sources, retrieve from a knowledge base, or rephrase the question.

If no recent (question, response) pair is found, ask the user to paste the question and response they want judged.

$ARGUMENTS
