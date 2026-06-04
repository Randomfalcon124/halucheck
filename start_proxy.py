"""Launcher for halucheck_proxy.py that works from any cwd.

Usage from another Claude Code session / shell:
    python C:/Users/Akshay/Projects/Neuron/gpt2_vg/start_proxy.py

Sets the right env vars (judge LoRAs, HHEM backend) and cwd before exec'ing
the proxy. Intended for sessions where you can't `cd` first.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

env_defaults = {
    "HALUCHECK_JUDGE_LORA": "ckpt_judge_enriched",
    "HALUCHECK_JUDGE_LORA_V2": "ckpt_judge_enriched_v2",
    "HALUCHECK_JUDGE_RAG_BACKEND": "hhem",
    "HALUCHECK_CHECK_BACKEND": "hhem",
    "HALUCHECK_CKPT_ROOT": str(REPO),
}
for k, v in env_defaults.items():
    os.environ.setdefault(k, v)

os.chdir(REPO)
# Run the proxy in-process so logs stream to stdout/stderr the same way
proxy = REPO / "halucheck_proxy.py"
print(f"[start_proxy] cwd={REPO}", flush=True)
print(f"[start_proxy] env: JUDGE_LORA={os.environ['HALUCHECK_JUDGE_LORA']} "
      f"V2={os.environ['HALUCHECK_JUDGE_LORA_V2']} "
      f"RAG={os.environ['HALUCHECK_JUDGE_RAG_BACKEND']}", flush=True)
exec(compile(proxy.read_text(encoding="utf-8"), str(proxy), "exec"),
     {"__name__": "__main__", "__file__": str(proxy)})
