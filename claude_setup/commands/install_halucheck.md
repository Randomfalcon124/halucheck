---
description: One-shot install of the halucheck plugin + hooks + preferences + slash commands
---

Run the halucheck installer end-to-end.

## Steps

1. **Locate `install.py`.** It lives in the halucheck source repo. Most likely paths in priority order:
   - `$ARGUMENTS` if non-empty (user passed the path explicitly)
   - `C:/Users/Akshay/Projects/Neuron/gpt2_vg/install.py`
   - `$HOME/halucheck/install.py`
   - `$HOME/Projects/halucheck/install.py`

   Use Bash: `ls <path>` to find which exists. If none, ask the user where they cloned the repo.

2. **Run it with Bash:**
   ```bash
   python <path>/install.py
   ```
   Capture the output and show only the `[OK] / [WARN]` lines and the final "Done." line — skip the noise.

3. **If the proxy check says `[WARN]`**, also paste the proxy start command into the output so the user can copy it.

4. **End with**: "Run /setup_prefs to take the survey."

Don't restart, don't explain — just run and report.

$ARGUMENTS
