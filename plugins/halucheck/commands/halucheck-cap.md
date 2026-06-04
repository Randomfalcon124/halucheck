---
description: Trim verbose tool output (logs, test runs, command stdout) while preserving errors
---

Trim a chunk of verbose output before feeding it back into context. Keeps the signal an agent needs — errors, tracebacks, head, tail — and collapses duplicates / redacts opaque tokens.

Steps:
1. Identify the output the user wants trimmed:
   - If `$ARGUMENTS` contains text, treat it as the output to trim.
   - Else, look at the most recent tool output / command result in conversation and offer to trim it.
   - Else, ask the user to paste the output.
2. Call the `halucheck_cap_output` MCP tool with the text. Default `max_chars=4000`.
3. Show the user:
   - The trim ratio (original → trimmed chars)
   - The strategy summary (head/tail/errors lines preserved, deduped runs, redacted tokens)
   - The trimmed text itself (in a code block).
4. Note any errors / tracebacks that were preserved so the user can see them immediately.

Use this any time a tool returns more than ~4 KB and the user wants to keep that output around for follow-up turns without burning the rest of their context.

$ARGUMENTS
