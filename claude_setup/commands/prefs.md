---
description: Show current Claude preferences and a one-liner to edit them
---

Read `C:/Users/Akshay/.claude/user_preferences.json` and report current values in a tight table. Show only the user-settable fields (skip keys starting with `_`).

Then show:
- "Edit: `notepad C:\Users\Akshay\.claude\user_preferences.json`" — for manual edit
- "Re-survey: `/setup_prefs`" — for guided re-do

If `$ARGUMENTS` is non-empty, parse as `key=value` and update that single field directly via Python, then re-show the table.

Be terse.

$ARGUMENTS
