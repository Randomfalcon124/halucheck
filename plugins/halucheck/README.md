# HaluCheck — Claude Code / Cowork Plugin

Plug-and-play hallucination sidecar for Claude Code and Cowork. Wraps any LLM call in your Claude Code session with:

- **Input-side abstention detection** (catches structurally bad prompts before they're sent)
- **Prompt rephrasing** (rewrites flagged prompts to be safer)
- **Output-side judging** (flags potential hallucinations in responses)
- **RAG faithfulness checking** (for grounded responses against retrieved docs)

All powered by a single 1.5 B-parameter backbone (~3 GB VRAM) with two ~17 MB LoRA adapters.

## Installation

### Option 1: Direct file copy (development)

```bash
# 1. Place this plugin directory under your Claude Code plugins root:
mkdir -p ~/.claude/plugins
cp -r plugins/halucheck ~/.claude/plugins/

# 2. Make sure the LoRA checkpoints are reachable from $HALUCHECK_CKPT_ROOT:
export HALUCHECK_CKPT_ROOT=/path/to/halucheck/ckpts

# 3. Restart Claude Code (or run /hooks to reload settings)
```

### Option 2: Add to your settings.json

```json
{
  "enabledPlugins": {
    "halucheck@local": true
  },
  "extraKnownMarketplaces": {
    "local": {
      "source": {
        "source": "directory",
        "path": "/absolute/path/to/halucheck/parent-dir"
      }
    }
  }
}
```

## What gets installed

| Surface | What it does |
|---|---|
| **MCP server** (`mcp/server.py`) | Exposes `halucheck_preflight`, `halucheck_rephrase`, `halucheck_judge`, `halucheck_judge_rag` as tools Claude can call directly |
| **`/halucheck-judge`** slash command | Manual: judge the latest assistant response |
| **`/halucheck-rephrase`** slash command | Manual: rewrite a hallucination-inducing prompt |
| **`/halucheck-preflight`** slash command | Manual: check a question without rewriting it |
| **`/halucheck-status`** slash command | Show plugin status, env vars, quality numbers |
| **`halucheck` skill** | Loaded when the user mentions hallucination checking |
| **UserPromptSubmit hook** | Auto-checks every user prompt; injects a warning hint if flagged |
| **Stop hook** | Async-judges the final assistant message; surfaces a warning if hallucination likely |

## Configuration

Environment variables (set in `settings.json` env block, or in your shell):

| Variable | Default | Purpose |
|---|---|---|
| `HALUCHECK_BASE_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Backbone model (HF id or local path) |
| `HALUCHECK_ABS_LORA` | `ckpt_abs_check_sft` | Path to input-detector LoRA |
| `HALUCHECK_JUDGE_LORA` | `ckpt_judge_mixed` | Path to output-judge LoRA (mixed-distribution recommended) |
| `HALUCHECK_DEVICE` | `cuda` if available, else `cpu` | Inference device |
| `HALUCHECK_CKPT_ROOT` | (auto) | Root directory under which to resolve LoRA names |

## Quality numbers (so users know what to expect)

| Component | Metric | Value |
|---|---|---|
| `abs_check` (input detector) | F1 on AbstentionBench | 0.755 |
| `abs_check` cross-benchmark | F1 on SQuAD 2.0 dev | 0.623 (above in-dist!) |
| Rephraser | Hallucination-proxy reduction | 22 % relative on flagged items |
| Judge (mixed) | AUC on TruthfulQA val | 0.85 |
| Judge (mixed) | AUC on TQA picked-answer (realistic) | 0.82 |
| **Judge (mixed) cross-distribution** | **AUC on held-out AB items** | **0.68** (vs 0.49 with TQA-only training) |

## Architecture diagram

```
  User prompt ────────────────────► UserPromptSubmit hook
                                          │
                                          ▼
                                    halucheck_preflight (MCP)
                                          │
                                          ├─ flagged=false ─► nothing; prompt proceeds
                                          │
                                          └─ flagged=true ───► additionalContext hint
                                                              (Claude sees "this prompt is risky")
                                          │
   Claude may invoke MCP tools here:      │
   - halucheck_rephrase                   │
   - halucheck_judge                      │
   - halucheck_judge_rag                  │
                                          ▼
                                    LLM responds
                                          │
                                          ▼
                                    Stop hook (async)
                                          │
                                          ▼
                                    halucheck_judge (last_user, last_response)
                                          │
                                          ├─ hallucination=false ► silent
                                          │
                                          └─ hallucination=true ─► systemMessage warning
```

## Latency on RTX 4060 Laptop (8.6 GB VRAM)

- abs_check preflight: ~60 ms / call (shared backbone amortised)
- Rephrase: ~800 ms / call (generation, ~120 tokens)
- Judge: ~60 ms / call (forward pass only)
- MCP server cold start: ~7 s (loads Qwen 1.5 B + 2 LoRAs once)

CPU fallback (no GPU): judge ~1.8 s / call, abs_check ~1.8 s / call. Workable for batch but slow for interactive UI.

## Cost overhead per Claude Code turn

The hooks fire **two** sidecar calls per turn (preflight on prompt, judge on response). Total overhead:
- ~120 ms on GPU (negligible vs Claude's typical 1-3 s response)
- ~3.6 s on CPU (noticeable; switch to async hooks or disable Stop hook for CPU deployments)

## Disabling components

If you only want the input-side detector:

```json
{
  "hooks": {
    "Stop": []
  }
}
```

If you only want the MCP tools (no automatic hooks):

```json
{
  "hooks": {
    "UserPromptSubmit": [],
    "Stop": []
  }
}
```

The MCP server stays available either way; Claude can invoke tools when relevant.

## When NOT to use this plugin

- For RAG-grounded faithfulness on enterprise documents: use Vectara HHEM-2.1-Open (~150 MB, purpose-built for RAG)
- For mission-critical factuality: use Patronus Lynx-70B or frontier-LLM-as-judge
- For comprehensive observability dashboards: use Arize Phoenix or Langfuse
- For prompt-injection / jailbreak defense: use Lakera Guard or NVIDIA Prompt Guard 2

See `HALUCHECK_FULL_VERDICT.md` for the full competitive analysis.

## Troubleshooting

- **MCP server fails to start**: check `HALUCHECK_ABS_LORA` and `HALUCHECK_JUDGE_LORA` paths. The server logs to stderr; check `~/.claude/logs/` or run `/halucheck-status`.
- **Hooks slow on CPU**: switch `Stop` hook to `async: true` (already default) so it doesn't block the user; disable `UserPromptSubmit` if its latency is noticeable.
- **False positives on simple factual answers**: the shipped judge is trained on TruthfulQA + AB; it has known calibration bias on short confident factual claims. Adjust `HALUCHECK_JUDGE_THRESHOLD` (default 0.0; try 2.0 or 3.0 for fewer false alarms) or retrain via `halucheck adapt`.
- **Rephrase drops by composition guard**: lower `HALUCHECK_COMPOSITION_MIN_OVERLAP` (default 0.15) to allow more creative rewrites.

## License

MIT for the plugin code. The Qwen-1.5B-Instruct backbone is Apache 2.0. LoRA adapters are MIT.
