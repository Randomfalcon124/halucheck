# HaluCheck — Claude Code best-practice toolkit

Cheap (1.5 B Qwen + LoRA) hallucination sidecar packaged as a Claude Code plugin. Ships an OpenAI-compatible proxy, 7 MCP tools, 7 slash commands, hooks, and tunable user preferences — install in one command.

## Install

```bash
git clone <your-fork>/halucheck.git
cd halucheck
python install.py
python start_proxy.py    # leave running; ~3 GB VRAM (or ~1.7 s p50 on CPU)
```

The installer:
1. Registers `halucheck-local` as a Claude Code plugin marketplace (`claude plugin marketplace add ./`)
2. Installs the `halucheck` plugin (`claude plugin install halucheck@halucheck-local`)
3. Patches `HALUCHECK_CKPT_ROOT` in the installed plugin.json to point at the cloned repo
4. Writes `UserPromptSubmit` + `Stop` hooks into `~/.claude/settings.json` (tagged for clean removal)
5. Drops `~/.claude/user_preferences.json` template + the `user_preferences.py` hook
6. Copies six slash commands (`install_halucheck`, `setup_prefs`, `prefs`, `smart_prompt`, `judge_response`, `check_claim`) into `~/.claude/commands/` AND `<cwd>/.claude/commands/`
7. Checks the proxy is reachable on `localhost:8000`

Hooks attach on the **next prompt** — no Claude Code restart needed (the harness re-reads settings.json every turn).

## After install, in any Claude Code session

- **`/setup_prefs`** — interactive survey to set conciseness, complexity, tone, subagent reminder
- **`/prefs`** — show / edit current preferences
- **`/smart_prompt <draft>`** — audit a prompt for hallucination triggers, return a safer rewrite
- **`/judge_response <q + r>`** — routed hallucination judge with FP suppression
- **`/check_claim <claim + evidence>`** — HHEM-backed faithfulness check
- **`/install_halucheck [path]`** — re-run installer (e.g. after `git pull`)

MCP tools (Claude calls these directly when needed): `halucheck_preflight`, `halucheck_rephrase`, `halucheck_judge`, `halucheck_judge_rag`, `halucheck_judge_route`, `halucheck_check`, `halucheck_cap_output`.

## What the hooks do automatically

- **UserPromptSubmit**: injects your style preferences and (if flagged) a HaluCheck preflight verdict + safer rewrite into every prompt Claude sees.
- **Stop**: silently runs the routed judge on every assistant response. If it flags hallucination, a ⚠ system message surfaces in the conversation.

## Uninstall

```bash
python install.py --uninstall
```

Removes the plugin, the sentinel-tagged hooks from settings.json, and the slash commands. Your `user_preferences.json` stays (in case you want to keep your tuning).

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` (torch, transformers, peft, sentence-transformers, fastapi, uvicorn, httpx, pydantic, mcp)
- A GPU is recommended (~3 GB VRAM). CPU works at ~1.7 s p50 per judge call.
- Claude Code Desktop or Claude Code CLI

## Repo layout

```
install.py                  # one-shot installer
start_proxy.py              # cwd-independent proxy launcher
halucheck_proxy.py          # FastAPI OpenAI-compatible proxy (9 endpoints)
halucheck_hhem_backend.py   # HHEM-2.1-Open RAG backend
halucheck_cap_output.py     # smart tool-output trimmer
halucheck_streaming.py      # SSE streaming judge
halucheck_cli.py            # standalone CLI
plugins/halucheck/          # Claude Code plugin (MCP server, hooks, slash commands)
.claude-plugin/             # marketplace manifest
claude_setup/               # user-scope files staged by install.py
ckpt_abs_check_sft/         # abs_check LoRA (17 MB)
ckpt_judge_enriched/        # primary judge LoRA (17 MB)
ckpt_judge_enriched_v2/     # L4-specialised secondary judge LoRA (17 MB)
requirements.txt
```

## Honest caveats

- Judge AUC: 0.85 in-distribution, 0.71 cross-distribution (AbstentionBench held-out). Not a turn-key hallucination cure.
- The "Paris is the capital" surface-form FP is partially mitigated by the v1+v2 router but not fully eliminated.
- HHEM is strict on numerical paraphrase (e.g. "12% revenue / 73% enterprise" scores 0.15 even when semantically faithful).
- See `HALUCHECK_GUARDRAILS_LAYERING.md` for composition with NeMo-class jailbreak/PII/content-moderation guards (not in scope here).

## License

MIT.
