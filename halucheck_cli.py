"""halucheck CLI — single entry point for the most common tasks.

Subcommands:
  halucheck preflight QUESTION
      One-shot: run abs_check on a question, print the decision.

  halucheck judge QUESTION RESPONSE
      One-shot: run the judge on a (Q, A) pair, print verdict.

  halucheck rephrase QUESTION
      One-shot: rewrite a question to make it safer to answer.

  halucheck adapt --kind {judge,abs_check} --data FILE.jsonl --out DIR
      Fine-tune a domain-specific LoRA adapter on the user's own data.
      Wraps halucheck_judge_sft.py / abs_check_sft.py with sensible defaults.

  halucheck serve [--port 8000] [--backend URL]
      Start the OpenAI-compatible proxy server.

  halucheck eval --data FILE.jsonl --judge-lora DIR
      Evaluate a judge on labeled (Q, A, label) data; print AUC + F1.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace") if sys.platform == "win32" else None


def cmd_preflight(args):
    """One-shot abs_check on a question."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    ABS_CHECK_PROMPT = (
        "Is the following question fully answerable from the information given, "
        "or is it ambiguous, underspecified, or missing required context? "
        "Answer with exactly YES (fully answerable) or NO (underspecified or unanswerable). "
        "Do not explain.\n\n"
        "Question: {q}\n\n"
        "Answer (YES or NO):"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, args.abs_lora).to(device).eval()
    yes_id = tok(" YES", add_special_tokens=False).input_ids[0]
    no_id = tok(" NO", add_special_tokens=False).input_ids[0]
    msgs = [{"role": "user", "content": ABS_CHECK_PROMPT.format(q=args.question)}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(p, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        lg = out.logits[0, -1]
        margin = float(lg[no_id].item() - lg[yes_id].item())
    flagged = margin > 0
    print(json.dumps({
        "flagged": flagged,
        "margin": round(margin, 3),
        "verdict": "hallucination-inducing (abstain or rephrase)" if flagged
                   else "answerable as-is",
    }, indent=2))


def cmd_judge(args):
    """One-shot judge on a (question, response) pair."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    JUDGE_PROMPT = """You are an expert fact-checker. Does the response contain a hallucination?

A hallucination is any of:
- A factual claim that is false or unverifiable
- Information that contradicts the prompt
- A fabricated entity, citation, statistic, date, or quote
- A confident answer to an unanswerable question (missing context, false premise, opinion-as-fact)

NOT a hallucination if the response:
- Is factually accurate
- Appropriately abstains or asks for clarification
- Stays within the prompt's information

Question: {q}

Response: {a}

Answer YES (hallucination) or NO (accurate or appropriately uncertain):"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, args.judge_lora).to(device).eval()
    yes_id = tok(" YES", add_special_tokens=False).input_ids[0]
    no_id = tok(" NO", add_special_tokens=False).input_ids[0]
    msgs = [{"role": "user", "content": JUDGE_PROMPT.format(q=args.question, a=args.response)}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(p, return_tensors="pt", truncation=True, max_length=3072).to(device)
    with torch.no_grad():
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        lg = out.logits[0, -1]
        margin = float(lg[yes_id].item() - lg[no_id].item())
    is_hall = margin > 0
    print(json.dumps({
        "hallucination": is_hall,
        "margin": round(margin, 3),
        "verdict": "likely hallucination" if is_hall else "likely accurate",
    }, indent=2))


def cmd_rephrase(args):
    """One-shot rephrase a question."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    REPHRASE_PROMPT = """The following user question is likely to cause a language model to give an unreliable or fabricated answer (it may be missing context, ambiguous, presuppose something false, ask about an obscure topic, or solicit an opinion as fact).

Rewrite the question so it can be answered safely by adding ONE of:
- An explicit clarifying question the user must answer
- A note that the premise needs to be verified
- An instruction to abstain if the answer is uncertain

Critical: preserve the user's original intent. Do not change the topic or drop the substance of what they are asking.

Original question: {q}

Rewritten question (one or two sentences, plain text, no preamble):"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                  torch_dtype=torch.bfloat16).to(device).eval()
    msgs = [{"role": "user", "content": REPHRASE_PROMPT.format(q=args.question)}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(p, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask,
            max_new_tokens=120, do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    gen = out[0, enc.input_ids.shape[1]:]
    rephrased = tok.decode(gen, skip_special_tokens=True).strip()
    # Composition guard
    a = set(args.question.lower().split())
    b = set(rephrased.lower().split())
    jaccard = len(a & b) / len(a | b) if a and b else 0
    print(json.dumps({
        "original": args.question,
        "rephrased": rephrased,
        "composition_overlap": round(jaccard, 3),
        "safe_to_use": jaccard >= 0.25,
    }, indent=2))


def cmd_adapt(args):
    """Train a domain-specific LoRA adapter."""
    if args.kind == "judge":
        sft = Path(__file__).parent / "halucheck_judge_sft.py"
    elif args.kind == "abs_check":
        sft = Path(__file__).parent / "abs_check_sft.py"
    else:
        sys.exit(f"unknown kind: {args.kind}")
    cmd = [
        sys.executable, str(sft),
        "--train_file", args.data,
        "--ckpt", args.out,
        "--epochs", str(args.epochs),
        "--base_name", args.base_model,
    ]
    print(f"Running: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    sys.exit(rc)


def cmd_serve(args):
    """Start the OpenAI proxy."""
    env = os.environ.copy()
    env["HALUCHECK_PORT"] = str(args.port)
    if args.backend:
        env["HALUCHECK_BACKEND_URL"] = args.backend
    env["HALUCHECK_BASE_MODEL"] = args.base_model
    env["HALUCHECK_ABS_LORA"] = args.abs_lora
    env["HALUCHECK_JUDGE_LORA"] = args.judge_lora
    proxy = Path(__file__).parent / "halucheck_proxy.py"
    subprocess.call([sys.executable, str(proxy)], env=env)


def cmd_eval(args):
    """Evaluate a judge on labeled data."""
    eval_script = Path(__file__).parent / "halucheck_judge_infer.py"
    cmd = [
        sys.executable, str(eval_script),
        "--lora_dir", args.judge_lora,
        "--pairs", args.data,
        "--out", args.out,
        "--base_name", args.base_model,
    ]
    print(f"Running: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    sys.exit(rc)


def main():
    p = argparse.ArgumentParser(prog="halucheck",
                                description="Cheap (1.5B) hallucination sidecar for LLMs.")
    p.add_argument("--base-model", dest="base_model",
                   default=os.getenv("HALUCHECK_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
                   help="HF model id or path of the shared backbone")
    p.add_argument("--abs-lora", dest="abs_lora",
                   default=os.getenv("HALUCHECK_ABS_LORA", "ckpt_abs_check_sft"))
    p.add_argument("--judge-lora", dest="judge_lora",
                   default=os.getenv("HALUCHECK_JUDGE_LORA", "ckpt_judge_sft"))
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("preflight", help="abs_check a question")
    pp.add_argument("question")
    pp.set_defaults(func=cmd_preflight)

    pj = sub.add_parser("judge", help="judge a (Q, A) pair for hallucination")
    pj.add_argument("question")
    pj.add_argument("response")
    pj.set_defaults(func=cmd_judge)

    pr = sub.add_parser("rephrase", help="rewrite a question to be safer to answer")
    pr.add_argument("question")
    pr.set_defaults(func=cmd_rephrase)

    pa = sub.add_parser("adapt", help="fine-tune a domain-specific LoRA")
    pa.add_argument("--kind", choices=["judge", "abs_check"], required=True)
    pa.add_argument("--data", required=True, help="training JSONL or JSON")
    pa.add_argument("--out", required=True, help="output dir for the LoRA adapter")
    pa.add_argument("--epochs", type=int, default=3)
    pa.set_defaults(func=cmd_adapt)

    ps = sub.add_parser("serve", help="start the OpenAI-compatible proxy")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--backend", default=None,
                    help="LLM backend URL (default OpenAI: https://api.openai.com/v1)")
    ps.set_defaults(func=cmd_serve)

    pe = sub.add_parser("eval", help="evaluate a judge on labeled (Q, A, label) data")
    pe.add_argument("--data", required=True, help="JSONL with {qid, question, response, label}")
    pe.add_argument("--out", default="eval_results.jsonl")
    pe.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
