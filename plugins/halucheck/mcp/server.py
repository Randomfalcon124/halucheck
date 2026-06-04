"""HaluCheck MCP server.

Exposes the three sidecar components as MCP tools that Claude can call inside
Claude Code / Cowork:

  halucheck_preflight(question)
      Check whether a user prompt is hallucination-inducing (structurally
      underspecified, false-premise, etc.). Returns {flagged: bool, margin: float}.

  halucheck_rephrase(question)
      Rewrite a prompt so the LLM can answer it safely (adds clarifying question
      or false-premise note). Returns {rephrased: str, safe_to_use: bool}.

  halucheck_judge(question, response)
      Judge whether a response is a hallucination. Returns {hallucination: bool,
      margin: float}.

  halucheck_judge_rag(question, response, context)
      RAG faithfulness judge: evaluate response against retrieved context.

The server loads Qwen-1.5B-Instruct + two LoRA adapters (~3 GB VRAM) on first
tool call. Subsequent calls reuse the loaded backbone.

Environment variables:
  HALUCHECK_BASE_MODEL    default Qwen/Qwen2.5-1.5B-Instruct
  HALUCHECK_ABS_LORA      default ckpt_abs_check_sft
  HALUCHECK_JUDGE_LORA    default ckpt_judge_mixed
  HALUCHECK_DEVICE        default cuda if available, else cpu
  HALUCHECK_LAZY_LOAD     default true (load on first tool call)
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import torch
from mcp.server.fastmcp import FastMCP
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Resolve plugin root (the plugin directory containing this script's parent)
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
# By default look for checkpoints in the project root (parent of plugin)
CKPT_ROOT = Path(os.getenv("HALUCHECK_CKPT_ROOT", str(PLUGIN_ROOT.parent.parent)))


ABS_CHECK_PROMPT = (
    "Is the following question fully answerable from the information given, "
    "or is it ambiguous, underspecified, or missing required context? "
    "Answer with exactly YES (fully answerable) or NO (underspecified or unanswerable). "
    "Do not explain.\n\n"
    "Question: {q}\n\n"
    "Answer (YES or NO):"
)

REPHRASE_PROMPT = """The following user question is likely to cause a language model to give an unreliable or fabricated answer (it may be missing context, ambiguous, presuppose something false, ask about an obscure topic, or solicit an opinion as fact).

Rewrite the question so it can be answered safely by adding ONE of:
- An explicit clarifying question the user must answer
- A note that the premise needs to be verified
- An instruction to abstain if the answer is uncertain

Critical: preserve the user's original intent. Do not change the topic or drop the substance of what they are asking.

Original question: {q}

Rewritten question (one or two sentences, plain text, no preamble):"""

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

RAG_JUDGE_PROMPT = """You are an expert RAG faithfulness checker. Does the response stay faithful to the retrieved document?

A response is UNFAITHFUL if it:
- Makes a factual claim NOT supported by the document
- Contradicts the document
- Adds information beyond what the document contains
- Cites a fact that isn't in the document

A response is FAITHFUL if it:
- Only states facts present in the document
- Appropriately says "the document doesn't say" when asked something outside scope
- Quotes / paraphrases accurately

Retrieved document: {context}

User question: {q}

Response: {a}

Answer YES (response is unfaithful to the document) or NO (response is faithful):"""


class Sidecar:
    """Lazily-loaded shared-backbone sidecar."""

    def __init__(self):
        self._loaded = False
        self.tok = None
        self.model = None
        self._yes = None
        self._no = None
        self.device = None

    def _ensure_loaded(self):
        if self._loaded:
            return
        base_name = os.getenv("HALUCHECK_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        abs_lora_name = os.getenv("HALUCHECK_ABS_LORA", "ckpt_abs_check_sft")
        judge_lora_name = os.getenv("HALUCHECK_JUDGE_LORA", "ckpt_judge_mixed")
        abs_lora = self._resolve_ckpt(abs_lora_name)
        judge_lora = self._resolve_ckpt(judge_lora_name)
        self.device = os.getenv(
            "HALUCHECK_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"[halucheck] Loading base model {base_name} on {self.device}",
              file=sys.stderr, flush=True)
        self.tok = AutoTokenizer.from_pretrained(base_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype).to(self.device)

        print(f"[halucheck] Loading abs_check LoRA: {abs_lora}",
              file=sys.stderr, flush=True)
        model = PeftModel.from_pretrained(base, str(abs_lora), adapter_name="abs_check")
        print(f"[halucheck] Loading judge LoRA: {judge_lora}",
              file=sys.stderr, flush=True)
        model.load_adapter(str(judge_lora), adapter_name="judge")
        # Optional secondary judge (for judge_route)
        self.has_judge_v2 = False
        judge_lora_v2_name = os.getenv("HALUCHECK_JUDGE_LORA_V2")
        if judge_lora_v2_name:
            try:
                judge_lora_v2 = self._resolve_ckpt(judge_lora_v2_name)
                print(f"[halucheck] Loading judge LoRA v2: {judge_lora_v2}",
                      file=sys.stderr, flush=True)
                model.load_adapter(str(judge_lora_v2), adapter_name="judge_v2")
                self.has_judge_v2 = True
            except Exception as e:
                print(f"[halucheck] judge v2 load failed ({e}); router degrades to v1-only",
                      file=sys.stderr, flush=True)
        self.model = model.to(self.device).eval()
        self._yes = self.tok(" YES", add_special_tokens=False).input_ids[0]
        self._no = self.tok(" NO", add_special_tokens=False).input_ids[0]
        self._loaded = True
        print(f"[halucheck] Sidecar ready (judge_v2={'present' if self.has_judge_v2 else 'absent'}).",
              file=sys.stderr, flush=True)

    def _resolve_ckpt(self, name: str) -> Path:
        """Resolve a checkpoint name to an absolute path. Try as-is, then under
        CKPT_ROOT."""
        p = Path(name)
        if p.is_absolute() and p.exists():
            return p
        candidate = CKPT_ROOT / name
        if candidate.exists():
            return candidate
        # Fall back: maybe it's relative to cwd
        if Path(name).exists():
            return Path(name).resolve()
        raise FileNotFoundError(
            f"Cannot find LoRA checkpoint '{name}'. Searched: {p}, {candidate}. "
            f"Set HALUCHECK_CKPT_ROOT or use absolute paths."
        )

    @torch.no_grad()
    def abs_check(self, question: str) -> dict:
        self._ensure_loaded()
        self.model.set_adapter("abs_check")
        msgs = [{"role": "user", "content": ABS_CHECK_PROMPT.format(q=question)}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(p, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        out = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        lg = out.logits[0, -1]
        margin = float(lg[self._no].item() - lg[self._yes].item())
        return {"flagged": bool(margin > 0), "margin": round(margin, 3),
                "verdict": "hallucination-inducing" if margin > 0 else "answerable as-is"}

    _embedder = None
    _embedder_load_attempted = False

    def _get_embedder(self):
        if not self._embedder_load_attempted:
            self._embedder_load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                print("[halucheck] composition guard: embedding mode active "
                      "(all-MiniLM-L6-v2, F1 0.73)",
                      file=sys.stderr, flush=True)
            except ImportError:
                print("[halucheck] composition guard: sentence-transformers NOT "
                      "installed — falling back to word-Jaccard (F1 0.25 vs "
                      "embedding 0.73). Install with: pip install sentence-transformers",
                      file=sys.stderr, flush=True)
                self._embedder = False
            except Exception as e:
                print(f"[halucheck] composition guard: failed to load embedder ({e}) "
                      "— falling back to word-Jaccard",
                      file=sys.stderr, flush=True)
                self._embedder = False
        return self._embedder

    @torch.no_grad()
    def rephrase(self, question: str, max_new_tokens: int = 120) -> dict:
        self._ensure_loaded()
        self.model.disable_adapter_layers()
        try:
            msgs = [{"role": "user", "content": REPHRASE_PROMPT.format(q=question)}]
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.tok(p, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            out = self.model.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
            gen = out[0, enc.input_ids.shape[1]:]
            rephrased = self.tok.decode(gen, skip_special_tokens=True).strip()
        finally:
            self.model.enable_adapter_layers()

        # Composition guard: embedding cosine (preferred) with word-Jaccard fallback
        emb = self._get_embedder()
        if emb:
            ea = emb.encode(question, convert_to_tensor=True, normalize_embeddings=True)
            eb = emb.encode(rephrased, convert_to_tensor=True, normalize_embeddings=True)
            cos = float(torch.dot(ea, eb).item())
            safe = cos >= 0.7
            return {"rephrased": rephrased,
                    "composition_method": "embedding_cosine",
                    "composition_score": round(cos, 3),
                    "safe_to_use": safe}
        # Fallback: word Jaccard
        a = set(question.lower().split())
        b = set(rephrased.lower().split())
        jaccard = len(a & b) / len(a | b) if (a and b) else 0
        return {"rephrased": rephrased,
                "composition_method": "word_jaccard_fallback",
                "composition_score": round(jaccard, 3),
                "safe_to_use": jaccard >= 0.15}

    @torch.no_grad()
    def judge(self, question: str, response: str) -> dict:
        self._ensure_loaded()
        self.model.set_adapter("judge")
        msgs = [{"role": "user", "content": JUDGE_PROMPT.format(q=question, a=response)}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(p, return_tensors="pt", truncation=True, max_length=3072).to(self.device)
        out = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        lg = out.logits[0, -1]
        margin = float(lg[self._yes].item() - lg[self._no].item())
        return {"hallucination": bool(margin > 0), "margin": round(margin, 3),
                "verdict": "likely hallucination" if margin > 0 else "likely accurate"}

    @torch.no_grad()
    def judge_rag(self, question: str, response: str, context: str) -> dict:
        self._ensure_loaded()
        self.model.set_adapter("judge")
        msgs = [{"role": "user", "content": RAG_JUDGE_PROMPT.format(
            context=context, q=question, a=response)}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(p, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        out = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        lg = out.logits[0, -1]
        margin = float(lg[self._yes].item() - lg[self._no].item())
        return {"unfaithful": bool(margin > 0), "margin": round(margin, 3),
                "note": "RAG faithfulness mode: 'unfaithful' = response not supported by context"}

    @torch.no_grad()
    def judge_v2(self, question: str, response: str) -> tuple[int, float] | None:
        if not self.has_judge_v2:
            return None
        self._ensure_loaded()
        self.model.set_adapter("judge_v2")
        msgs = [{"role": "user", "content": JUDGE_PROMPT.format(q=question, a=response)}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(p, return_tensors="pt", truncation=True, max_length=3072).to(self.device)
        out = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        lg = out.logits[0, -1]
        margin = float(lg[self._yes].item() - lg[self._no].item())
        return int(margin > 0), margin

    @torch.no_grad()
    def judge_route(self, question: str, response: str,
                     override_strong_no: float = -1.0) -> dict:
        """Combined v1+v2 judge with override rule. Mirror of the proxy's
        /v1/judge_route endpoint."""
        v1 = self.judge(question, response)
        v1_pred = int(v1["margin"] > 0)
        v1_margin = v1["margin"]
        if not self.has_judge_v2:
            return {**v1, "v1_margin": v1_margin, "v2_margin": None,
                    "rule": "v1_only", "router_active": False}
        v2_pred, v2_margin = self.judge_v2(question, response)  # type: ignore[misc]
        if v1_margin > 0 and v2_margin < override_strong_no:
            return {
                "hallucination": False,
                "margin": round(v2_margin, 3),
                "v1_margin": round(v1_margin, 3),
                "v2_margin": round(v2_margin, 3),
                "rule": "v2_override_likely_FP",
                "router_active": True,
                "verdict": "likely accurate (router overrode v1)",
            }
        return {
            "hallucination": bool(v1_pred),
            "margin": round(v1_margin, 3),
            "v1_margin": round(v1_margin, 3),
            "v2_margin": round(v2_margin, 3),
            "rule": "v1_default",
            "router_active": True,
            "verdict": "likely hallucination" if v1_pred else "likely accurate",
        }

    @torch.no_grad()
    def check(self, claim: str, evidence: str) -> dict:
        """Claim verifier — is `claim` supported by `evidence`?

        Built on the RAG judge with the evidence as context and the claim
        as the response. Returns claim-shaped output (supported/confidence).
        Plugin-side uses the Qwen judge_rag; the proxy /v1/check supports
        HHEM as a sharper alternative if HALUCHECK_CHECK_BACKEND=hhem is set.
        """
        rag = self.judge_rag(claim, claim, evidence)
        margin = rag["margin"]
        return {
            "supported": not rag["unfaithful"],
            "margin": margin,
            "confidence": round(1.0 / (1.0 + abs(margin)), 3),
            "note": "supported = (judge margin <= 0); use proxy /v1/check for HHEM backend",
        }


SIDECAR = Sidecar()
mcp = FastMCP("halucheck")


@mcp.tool()
def halucheck_preflight(question: str) -> dict:
    """Check whether a prompt is hallucination-inducing.

    Use this BEFORE submitting a question to an LLM to detect structurally
    problematic prompts (underspecified, ambiguous, false-premise, etc).

    Args:
      question: The user prompt to evaluate.

    Returns:
      {flagged: bool, margin: float, verdict: str}
      - flagged=true means the prompt is likely to elicit a hallucination
    """
    return SIDECAR.abs_check(question)


@mcp.tool()
def halucheck_rephrase(question: str) -> dict:
    """Rewrite a hallucination-inducing prompt to be safer.

    Adds a clarifying question, premise note, or abstention instruction while
    preserving the user's intent. Includes a composition-overlap safety check
    that returns safe_to_use=false if the rewrite drifted too far.

    Args:
      question: The user prompt to rewrite.

    Returns:
      {rephrased: str, composition_overlap: float, safe_to_use: bool}
    """
    return SIDECAR.rephrase(question)


@mcp.tool()
def halucheck_judge(question: str, response: str) -> dict:
    """Judge whether an LLM response contains a hallucination.

    Use this AFTER getting a response from an LLM to flag likely hallucinations
    (false facts, fabricated citations, confident answers to unanswerable
    questions).

    Args:
      question: The original user question.
      response: The LLM's response to evaluate.

    Returns:
      {hallucination: bool, margin: float, verdict: str}
    """
    return SIDECAR.judge(question, response)


@mcp.tool()
def halucheck_judge_rag(question: str, response: str, context: str) -> dict:
    """RAG faithfulness judge: evaluate whether a response is supported by retrieved context.

    Use this for RAG systems where you want to detect responses that go beyond
    the retrieved documents (HHEM-style faithfulness scoring).

    Args:
      question: The original user question.
      response: The LLM's response.
      context: The retrieved document(s) the response should be grounded in.

    Returns:
      {unfaithful: bool, margin: float, note: str}
    """
    return SIDECAR.judge_rag(question, response, context)


@mcp.tool()
def halucheck_judge_route(question: str, response: str,
                            override_strong_no: float = -1.0) -> dict:
    """Routed hallucination judge — combines primary + L4-specialised LoRAs.

    Default is the primary judge. If HALUCHECK_JUDGE_LORA_V2 is configured,
    a secondary judge runs in parallel; when v1 flags hallucination but v2
    strongly disagrees (margin < override_strong_no, default -1), v2's
    verdict wins. This catches surface-form false positives (e.g. judging
    "Paris is the capital of France." as hallucination) without losing
    primary recall.

    Args:
      question: The original user question.
      response: The LLM's response to evaluate.
      override_strong_no: Threshold below which v2 can override v1's HALL.

    Returns:
      {hallucination: bool, margin: float, v1_margin: float,
       v2_margin: float | null, rule: str, router_active: bool, verdict: str}
    """
    return SIDECAR.judge_route(question, response,
                                 override_strong_no=override_strong_no)


@mcp.tool()
def halucheck_check(claim: str, evidence: str) -> dict:
    """Claim verifier — is the claim supported by the evidence?

    The "give Claude a check it can run" primitive. Call this before
    asserting any factual claim that has supporting evidence available
    (a doc passage, a search result, etc.) — useful for grounding agent
    outputs.

    Args:
      claim: The factual claim to verify.
      evidence: A premise / document / passage the claim should follow from.

    Returns:
      {supported: bool, confidence: float, margin: float, note: str}
    """
    return SIDECAR.check(claim, evidence)


@mcp.tool()
def halucheck_cap_output(text: str, max_chars: int = 4000,
                           head_lines: int = 25, tail_lines: int = 50,
                           preserve_errors: bool = True,
                           redact_opaque: bool = True) -> dict:
    """Smart tool-output trimmer — keeps errors + head + tail, collapses
    duplicates, redacts long opaque blobs (base64/hex/secrets).

    Call this on any verbose tool output (build logs, test runs, command
    stdout) BEFORE feeding into the next prompt. The trimmed output
    preserves the signal an agent needs (errors, key context) while
    cutting token cost by ~10× on a typical verbose log. Pure heuristic
    — deterministic, no ML, no GPU.

    Args:
      text: The raw output to trim.
      max_chars: Target maximum character count (default 4000).
      head_lines: How many leading lines to preserve (default 25).
      tail_lines: How many trailing lines to preserve (default 50).
      preserve_errors: Always keep error/traceback lines (default True).
      redact_opaque: Redact long base64/hex/token blobs (default True).

    Returns:
      {text: str, original_chars: int, trimmed_chars: int,
       original_lines: int, kept_lines: int, dropped_lines: int,
       deduped_runs: int, redacted_tokens: int, strategy: str}
    """
    # Lazy import — the plugin can ship cap_output without loading torch
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "halucheck_cap_output",
        str(CKPT_ROOT / "halucheck_cap_output.py"))
    if spec is None or spec.loader is None:
        return {"error": "halucheck_cap_output module not found"}
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.cap(text, max_chars=max_chars, head_lines=head_lines,
                tail_lines=tail_lines, preserve_errors=preserve_errors,
                redact_opaque=redact_opaque)
    return {
        "text": r.text,
        "original_chars": r.original_chars,
        "trimmed_chars": r.trimmed_chars,
        "original_lines": r.original_lines,
        "kept_lines": r.kept_lines,
        "dropped_lines": r.dropped_lines,
        "deduped_runs": r.deduped_runs,
        "redacted_tokens": r.redacted_tokens,
        "strategy": r.strategy,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
