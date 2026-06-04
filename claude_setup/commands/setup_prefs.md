---
description: Interactive survey to set conciseness, complexity, tone, and hook reminders. Writes ~/.claude/user_preferences.json.
---

Walk the user through a survey to set their global Claude preferences. These get injected into every prompt via the `user_preferences.py` UserPromptSubmit hook.

## Steps

1. **Use the `AskUserQuestion` tool** to ask all preferences in one call (multi-question form). Questions:

   - **Conciseness** — header "Conciseness", question "How terse should responses be?", options:
     - "Maximally terse" (level 10) — one-line answers only
     - "Very concise" (level 8) — no preambles, get to point first sentence  [Recommended]
     - "Concise" (level 6) — to the point but with enough context
     - "Full" (level 3) — full explanations, background, context

   - **Complexity** — header "Audience", question "What's the expected reader expertise?", options:
     - "Expert" (level 9) — assume domain knowledge, no jargon defined  [Recommended]
     - "Technical" (level 7) — define unusual terms only
     - "General technical" (level 5) — define moderately technical terms
     - "ELI5" (level 2) — define everything, analogies

   - **Tone** — header "Tone", question "What conversational tone?", options:
     - "Direct" — no hedges, disagree when warranted  [Recommended]
     - "Collegial" — suggest alternatives diplomatically
     - "Formal" — no contractions, professional

   - **Subagent reminder** — header "Subagents", question "Add a reminder on every prompt to delegate exploration to subagents?", options:
     - "Yes" — useful for context discipline  [Recommended]
     - "No" — skip the reminder

2. **Map answers to integers** per the levels in parentheses. Then **write to `C:/Users/Akshay/.claude/user_preferences.json`** via a single Python Bash invocation:
   ```bash
   python -c "
   import json, pathlib
   p = pathlib.Path.home() / '.claude' / 'user_preferences.json'
   cfg = json.loads(p.read_text(encoding='utf-8'))
   cfg.update({
       'conciseness_level': <N>,
       'complexity_level': <N>,
       'tone': '<direct|collegial|formal>',
       'remind_subagent_delegation': <true|false>,
       '_setup_complete': True,
   })
   p.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
   print('saved:', cfg)
   "
   ```

3. **Confirm** in one line: "Saved. Takes effect on your next prompt."

4. **Also ask** whether they want a `custom_preamble` (free-form text appended every turn — like "always cite file:line" or project-specific reminders). If they provide one, add it to the JSON.

Be terse. Don't repeat the user's answers back. Don't add caveats.

$ARGUMENTS
