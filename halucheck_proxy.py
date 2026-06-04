"""HaluCheck Proxy: OpenAI-compatible HTTP server that wraps an LLM with sidecar
hallucination detection and prompt rephrasing.

For any existing OpenAI-API-using application:
  - Change base_url from "https://api.openai.com/v1" to "http://localhost:8000/v1"
  - Everything else stays the same.
  - The proxy:
    1. Runs abs_check on the last user message
    2. If flagged: rephrases the message (with composition error guard)
    3. Forwards the (possibly rephrased) request to the real OpenAI API
    4. Runs the judge on the response
    5. Returns the response, optionally with x-halucheck-* metadata headers

Usage:
  export OPENAI_API_KEY=sk-...
  python halucheck_proxy.py

  Then in your app:
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-...")
    response = client.chat.completions.create(model="gpt-4o", messages=[...])
    # Sidecar runs transparently; original response returned unchanged

Configuration via env:
  HALUCHECK_BACKEND_URL  (default https://api.openai.com/v1)
  HALUCHECK_PORT         (default 8000)
  HALUCHECK_BASE_MODEL   (default Qwen/Qwen2.5-1.5B-Instruct)
  HALUCHECK_ABS_LORA     (default ckpt_abs_check_sft)
  HALUCHECK_JUDGE_LORA   (default ckpt_judge_sft)
  HALUCHECK_DEVICE       (default cuda if available, else cpu)
  HALUCHECK_ENABLE_REPHRASE  (default true)
  HALUCHECK_ENABLE_JUDGE     (default true)
"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn

from halucheck_streaming import StreamingJudge, DEFAULT_CHECKPOINTS
from halucheck_cap_output import cap as _cap_output

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("halucheck")
sys.stdout.reconfigure(encoding="utf-8", errors="replace") if sys.platform == "win32" else None


# ----- Prompts (kept verbatim from the trained system) -----

ABS_CHECK_PROMPT = (
    "Is the following question fully answerable from the information given, "
    "or is it ambiguous, underspecified, or missing required context? "
    "Answer with exactly YES (fully answerable) or NO (underspecified or unanswerable). "
    "Do not explain.\n\n"
    "Question: {q}\n\n"
    "Answer (YES or NO):"
)

REPHRASE_PROMPT_V2 = """The following user question is likely to cause a language model to give an unreliable or fabricated answer (it may be missing context, ambiguous, presuppose something false, ask about an obscure topic, or solicit an opinion as fact).

Rewrite the question so it can be answered safely by adding ONE of:
- An explicit clarifying question the user must answer
- A note that the premise needs to be verified
- An instruction to abstain if the answer is uncertain

Critical: preserve the user's original intent. Do not change the topic or drop the substance of what they are asking.

Original question: {q}

Rewritten question (one or two sentences, plain text, no preamble):"""

# v3 (minimal-mod): from the L5 rephraser variant test — preserved 100 % of
# composition-guard passes (vs 93 % for v2) and 0.901 mean cosine (vs 0.762).
# Slightly higher hall_proxy (80 % vs 77 %), so default stays v2; v3 is
# selectable when faithfulness to the original surface form matters more.
REPHRASE_PROMPT_V3_MINIMAL = """The following user question is likely to elicit a hallucinated answer.

Output the question UNCHANGED, then append exactly this safeguard: "Answer only if you can verify the facts; if you are not certain or do not know, respond with 'I don't know.'"

Original question: {q}

Output (question + appended safeguard, on separate lines):"""

REPHRASE_PROMPTS = {
    "v2": REPHRASE_PROMPT_V2,
    "v3_minimal": REPHRASE_PROMPT_V3_MINIMAL,
}
DEFAULT_REPHRASE_STYLE = os.getenv("HALUCHECK_REPHRASE_STYLE", "v2").lower()

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


# RAG faithfulness prompt — used when the caller provides retrieved context.
# The judge evaluates whether the response is *supported by the retrieved document*,
# regardless of whether the response is true in the world. This is the HHEM-style
# evaluation for RAG systems.
MULTI_TURN_JUDGE_PROMPT = """You are an expert fact-checker. The conversation history is shown below. Does the LATEST response contain a hallucination, including any contradiction with what was established earlier in the conversation?

A hallucination is any of:
- A factual claim that is false or unverifiable
- A contradiction with a fact stated earlier in this conversation
- A change of identity, role, or persona that wasn't requested
- A fabricated entity, citation, statistic, date, or quote
- A confident answer to an unanswerable question (missing context, false premise, opinion-as-fact)

NOT a hallucination if the response:
- Is factually accurate
- Is consistent with the earlier conversation
- Appropriately abstains or asks for clarification
- Stays within the prompt's information

Conversation history (oldest first):
{history}

Latest question: {q}

Latest response: {a}

Does the latest response contain a hallucination or contradiction? Answer YES or NO:"""


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


# ----- Sidecar (shared backbone) -----

class Sidecar:
    def __init__(self, base_name: str, abs_lora: str, judge_lora: str,
                 device: str = "cuda", judge_lora_v2: str | None = None):
        """Load the shared backbone + abs_check + judge adapters.

        Optional `judge_lora_v2`: a second judge LoRA loaded as adapter
        "judge_v2". When configured, /v1/judge_route ensembles v1 and v2 for
        sharper L4-class FP suppression without losing v1's AB recall.
        """
        self.device = device
        log.info(f"Loading tokenizer {base_name}")
        self.tok = AutoTokenizer.from_pretrained(base_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        log.info(f"Loading base model on {device}")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype).to(device)
        log.info(f"Loading abs_check LoRA: {abs_lora}")
        model = PeftModel.from_pretrained(base, abs_lora, adapter_name="abs_check")
        log.info(f"Loading judge LoRA: {judge_lora}")
        model.load_adapter(judge_lora, adapter_name="judge")
        self.has_judge_v2 = False
        if judge_lora_v2:
            try:
                log.info(f"Loading judge LoRA v2 (router secondary): {judge_lora_v2}")
                model.load_adapter(judge_lora_v2, adapter_name="judge_v2")
                self.has_judge_v2 = True
            except Exception as e:
                log.warning(f"judge v2 load failed ({e}); router will degrade to v1-only")
        self.model = model.to(device).eval()
        self._yes = self.tok(" YES", add_special_tokens=False).input_ids[0]
        self._no = self.tok(" NO", add_special_tokens=False).input_ids[0]
        log.info(f"Sidecar ready (judge_v2={'present' if self.has_judge_v2 else 'absent'}).")

    @torch.no_grad()
    def abs_check(self, question: str) -> tuple[int, float]:
        self.model.set_adapter("abs_check")
        msgs = [{"role": "user", "content": ABS_CHECK_PROMPT.format(q=question)}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(p, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        out = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        lg = out.logits[0, -1]
        margin = float(lg[self._no].item() - lg[self._yes].item())
        return int(margin > 0), margin

    @torch.no_grad()
    def rephrase(self, question: str, max_new_tokens: int = 120,
                 style: str | None = None) -> str:
        """Rephrase a flagged question. `style` selects the prompt template:
          - "v2" (default): the trained-and-validated strategy (22 % hall reduction,
            93 % composition-guard pass)
          - "v3_minimal": preserves surface form, appends a single safeguard.
            Higher faithfulness (100 % guard pass, 0.901 cosine in L5 eval), at
            a slightly higher hall_proxy.
        Falls back to v2 if style is unknown.
        """
        # Rephraser uses base model (no LoRA)
        self.model.disable_adapter_layers()
        try:
            prompt_tpl = REPHRASE_PROMPTS.get(
                (style or DEFAULT_REPHRASE_STYLE).lower(),
                REPHRASE_PROMPTS["v2"],
            )
            msgs = [{"role": "user", "content": prompt_tpl.format(q=question)}]
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.tok(p, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            out = self.model.generate(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
            gen = out[0, enc.input_ids.shape[1]:]
            return self.tok.decode(gen, skip_special_tokens=True).strip()
        finally:
            self.model.enable_adapter_layers()

    @torch.no_grad()
    def judge(self, question: str, response: str) -> tuple[int, float]:
        self.model.set_adapter("judge")
        msgs = [{"role": "user", "content": JUDGE_PROMPT.format(q=question, a=response)}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(p, return_tensors="pt", truncation=True, max_length=3072).to(self.device)
        out = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        lg = out.logits[0, -1]
        margin = float(lg[self._yes].item() - lg[self._no].item())
        return int(margin > 0), margin

    @torch.no_grad()
    def judge_v2(self, question: str, response: str) -> tuple[int, float] | None:
        """Run the secondary (L4-specialised) judge LoRA. Returns None if not
        configured. Same prompt and margin convention as `judge()`."""
        if not self.has_judge_v2:
            return None
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
                     override_strong_no: float = -1.0,
                     override_min_v1: float = 2.0) -> dict:
        """Router: combine v1 and v2 judges to suppress L4-class FPs without
        losing v1's recall on subtle fabrications.

        Decision rule:
          - If v2 is not configured → behave identically to `judge()`.
          - Else run both. Use v1 by default.
          - Override v1 → v2 iff:
              v1_margin >= override_min_v1 (strongly positive, default 2.0)
              AND v2_margin < override_strong_no (default -1.0)
          - The `override_min_v1` floor was added after the Curie / penicillin
            FN: v1 +1.75 (mild HALL signal — likely real fabrication) was
            wrongly suppressed by v2 -3.50 (high tolerance for confident
            assertions). Real fabrications often sit in v1 ∈ (0, 2] while
            v2's tolerance is too generous to be the deciding voice there.
            Set `override_min_v1=0` to restore legacy behaviour.

        Returns a dict with both margins, the rule that fired, and the final
        verdict — so callers can see the routing decision.
        """
        v1_pred, v1_margin = self.judge(question, response)
        if not self.has_judge_v2:
            return {
                "hallucination": bool(v1_pred),
                "margin": v1_margin,
                "v1_margin": v1_margin,
                "v2_margin": None,
                "rule": "v1_only",
                "router_active": False,
            }
        v2_pred, v2_margin = self.judge_v2(question, response)  # type: ignore[misc]
        if (v1_margin >= override_min_v1
                and v2_margin < override_strong_no):
            return {
                "hallucination": False,
                "margin": v2_margin,
                "v1_margin": v1_margin,
                "v2_margin": v2_margin,
                "rule": "v2_override_likely_FP",
                "router_active": True,
            }
        # v1 wins if v1 is mildly positive (0, override_min_v1) — that's the
        # subtle-fabrication zone the previous rule wrongly suppressed.
        return {
            "hallucination": bool(v1_pred),
            "margin": v1_margin,
            "v1_margin": v1_margin,
            "v2_margin": v2_margin,
            "rule": ("v1_default_mild_zone"
                      if 0 < v1_margin < override_min_v1 else "v1_default"),
            "router_active": True,
        }

    @torch.no_grad()
    def judge_multi_turn(self, question: str, response: str,
                          history: list[dict]) -> tuple[int, float]:
        """Multi-turn judge: evaluate the latest response against prior conversation.

        Args:
          question: the latest user question
          response: the latest assistant response
          history: list of {"role": "user"|"assistant", "content": str} for prior turns
                   (excluding the latest question/response pair)

        Catches cross-turn contradictions, identity slips, etc.
        """
        self.model.set_adapter("judge")
        # Format history as a transcript
        if history:
            lines = []
            for m in history:
                role = m.get("role", "?").upper()
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(x.get("text", "")) for x in content
                                       if isinstance(x, dict))
                lines.append(f"{role}: {content}")
            history_text = "\n".join(lines)
        else:
            history_text = "(no prior turns)"
        prompt = MULTI_TURN_JUDGE_PROMPT.format(history=history_text, q=question, a=response)
        msgs = [{"role": "user", "content": prompt}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(p, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        out = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        lg = out.logits[0, -1]
        margin = float(lg[self._yes].item() - lg[self._no].item())
        return int(margin > 0), margin

    @torch.no_grad()
    def judge_rag(self, question: str, response: str, context: str) -> tuple[int, float]:
        """RAG faithfulness mode: evaluate response against retrieved context.
        Returns (unfaithful_flag, margin). Same threshold semantics as judge()."""
        self.model.set_adapter("judge")
        prompt = RAG_JUDGE_PROMPT.format(context=context, q=question, a=response)
        msgs = [{"role": "user", "content": prompt}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(p, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        out = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        lg = out.logits[0, -1]
        margin = float(lg[self._yes].item() - lg[self._no].item())
        return int(margin > 0), margin

    _embedder = None  # lazy-loaded sentence-transformer
    _embedder_load_attempted = False

    def _get_embedder(self):
        if not self._embedder_load_attempted:
            self._embedder_load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                log.info("composition guard: embedding mode active (all-MiniLM-L6-v2, "
                         "F1 0.73 on test set)")
            except ImportError:
                log.warning("composition guard: sentence-transformers NOT installed — "
                            "falling back to word-Jaccard (F1 0.25 vs embedding 0.73). "
                            "Install with: pip install sentence-transformers")
                self._embedder = False
            except Exception as e:
                log.warning(f"composition guard: failed to load all-MiniLM-L6-v2 ({e}) — "
                            "falling back to word-Jaccard (F1 0.25 vs embedding 0.73)")
                self._embedder = False
        return self._embedder

    @torch.no_grad()
    def composition_guard(self, original: str, rephrased: str,
                           min_overlap: float = 0.15,
                           min_cosine: float = 0.7,
                           use_embedding: bool = True) -> bool:
        """Composition error guard: semantic similarity between original and
        rephrased question. If too low, the rephraser has likely drifted from
        the user's intent — caller should fall back to the original.

        Two modes:
          - Embedding cosine (default, F1 0.73 on test set): uses all-MiniLM-L6-v2
            (~80 MB). Compare emb(orig) and emb(rephrased) by cosine similarity.
          - Word Jaccard (fallback if sentence-transformers unavailable, F1 0.25):
            simple token-overlap.

        Returns True if rephrasing is safe to use, False if drifted.
        """
        if use_embedding:
            emb = self._get_embedder()
            if emb:
                ea = emb.encode(original, convert_to_tensor=True, normalize_embeddings=True)
                eb = emb.encode(rephrased, convert_to_tensor=True, normalize_embeddings=True)
                cos = float(torch.dot(ea, eb).item())
                return cos >= min_cosine
        # Fallback: word Jaccard
        a = set(original.lower().split())
        b = set(rephrased.lower().split())
        if not a or not b:
            return False
        jaccard = len(a & b) / len(a | b)
        return jaccard >= min_overlap


# ----- FastAPI app -----

SIDECAR: Sidecar | None = None
BACKEND_URL = os.getenv("HALUCHECK_BACKEND_URL", "https://api.openai.com/v1")
ENABLE_REPHRASE = os.getenv("HALUCHECK_ENABLE_REPHRASE", "true").lower() == "true"
ENABLE_JUDGE = os.getenv("HALUCHECK_ENABLE_JUDGE", "true").lower() == "true"
JUDGE_THRESHOLD = float(os.getenv("HALUCHECK_JUDGE_THRESHOLD", "0.0"))
# Calibration: on the shipped mixed judge, threshold = 0 gives F1 0.78 at the
# in-distribution TruthfulQA val. Calibration analysis shows better operating
# points exist (recompute with halucheck_threshold_calibrate.py on your traffic):
#   thr = -1.0  →  F1 0.80, precision 0.72, recall 0.89  (balanced; recommended)
#   thr = -2.0  →  F1 0.79, precision 0.69, recall 0.93  (high-recall safety)
#   thr =  5.0  →  F1 0.62, precision 0.89, recall 0.48  (high-precision audit)
# The default stays at 0 for backward compatibility; production teams should
# re-calibrate on their own traffic.
COMPOSITION_MIN_OVERLAP = float(os.getenv("HALUCHECK_COMPOSITION_MIN_OVERLAP", "0.15"))
# v2 composition guard: embedding cosine similarity
USE_EMBEDDING_GUARD = os.getenv("HALUCHECK_USE_EMBEDDING_GUARD", "true").lower() == "true"
COMPOSITION_MIN_COSINE = float(os.getenv("HALUCHECK_COMPOSITION_MIN_COSINE", "0.7"))
# Streaming-mode mid-response judge (L6): emits early warnings before the full
# response completes. Default checkpoints come from halucheck_streaming. Set
# HALUCHECK_STREAM_CHECKPOINTS="32,128,512" to override (cumulative tok counts).
_stream_ckpt_env = os.getenv("HALUCHECK_STREAM_CHECKPOINTS", "")
if _stream_ckpt_env:
    try:
        STREAM_CHECKPOINTS = [int(x.strip()) for x in _stream_ckpt_env.split(",") if x.strip()]
    except ValueError:
        STREAM_CHECKPOINTS = list(DEFAULT_CHECKPOINTS)
else:
    STREAM_CHECKPOINTS = list(DEFAULT_CHECKPOINTS)
# Disable mid-stream judging entirely (only final judge_final at end of stream)
STREAM_JUDGE_ENABLED = os.getenv("HALUCHECK_STREAM_JUDGE", "true").lower() == "true"

# Judge result cache — addresses the guide's "bail-on-stuck" rule.
# Same (question, response) pair returns the cached margin without re-running
# the LoRA. On 3rd+ identical HALL hit, the response carries
# `repeat_count` + `fp_suspected: True` — the agent's signal to stop
# re-judging the same flag.
JUDGE_CACHE_ENABLED = os.getenv("HALUCHECK_JUDGE_CACHE", "1") != "0"
JUDGE_CACHE_MAX = int(os.getenv("HALUCHECK_JUDGE_CACHE_MAX", "256"))
JUDGE_CACHE_REPEAT_FP_THRESHOLD = int(
    os.getenv("HALUCHECK_JUDGE_CACHE_FP_AT", "3"))

# /v1/judge_rag backend selector. Attack-2 adoption of Vectara HHEM-2.1-Open.
#   "qwen-lora"  → use the Qwen LoRA judge (default; backwards-compat).
#   "hhem"       → use HHEM-2.1-Open (150 M encoder, RAG-specific, ~0.9 AUC).
# Lazy-loaded only when the endpoint is hit AND the backend is set to "hhem".
JUDGE_RAG_BACKEND = os.getenv("HALUCHECK_JUDGE_RAG_BACKEND", "qwen-lora").lower()
HHEM_BACKEND = None  # set on first /v1/judge_rag call when backend == "hhem"


def _get_hhem_backend():
    """Lazy-init HHEM. Returns the backend instance or None if unavailable."""
    global HHEM_BACKEND
    if HHEM_BACKEND is None:
        try:
            from halucheck_hhem_backend import HHEMBackend
            HHEM_BACKEND = HHEMBackend()
        except Exception as e:
            log.warning(f"HHEM backend import failed: {e}")
            return None
    return HHEM_BACKEND


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SIDECAR
    base_name = os.getenv("HALUCHECK_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    abs_lora = os.getenv("HALUCHECK_ABS_LORA", "ckpt_abs_check_sft")
    judge_lora = os.getenv("HALUCHECK_JUDGE_LORA", "ckpt_judge_mixed")
    judge_lora_v2 = os.getenv("HALUCHECK_JUDGE_LORA_V2")  # optional router secondary
    device = os.getenv("HALUCHECK_DEVICE",
                       "cuda" if torch.cuda.is_available() else "cpu")
    SIDECAR = Sidecar(base_name, abs_lora, judge_lora, device,
                       judge_lora_v2=judge_lora_v2)
    log.info(f"Backend: {BACKEND_URL}  rephrase={ENABLE_REPHRASE}  judge={ENABLE_JUDGE}")
    yield
    log.info("Shutdown.")


app = FastAPI(lifespan=lifespan, title="HaluCheck Proxy", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok", "backend": BACKEND_URL,
            "rephrase": ENABLE_REPHRASE, "judge": ENABLE_JUDGE}


class _JudgeCache:
    """Tiny LRU + repeat-counter for judge results.

    Keyed by SHA-1 of (question + '\\0' + response). Value stores the most
    recent verdict dict plus how many times that exact pair has been judged
    this session. Lifecycle is the proxy process — restart clears.
    """

    def __init__(self, max_size: int = 256):
        self._max_size = max_size
        # OrderedDict insertion order = LRU
        from collections import OrderedDict
        self._d: "OrderedDict[str, dict]" = OrderedDict()

    @staticmethod
    def _key(question: str, response: str) -> str:
        import hashlib
        h = hashlib.sha1()
        h.update(question.encode("utf-8", "replace"))
        h.update(b"\x00")
        h.update(response.encode("utf-8", "replace"))
        return h.hexdigest()

    def observe(self, question: str, response: str) -> dict | None:
        """Record an observation of this (q, r) pair and return the cached
        verdict (if any) with an updated repeat_count + fp_suspected. Used
        on the hit path BEFORE running the LoRA — caller short-circuits
        when this returns non-None.
        """
        if not JUDGE_CACHE_ENABLED:
            return None
        k = self._key(question, response)
        if k not in self._d:
            return None
        entry = self._d.pop(k)
        entry = dict(entry)
        entry["repeat_count"] = entry.get("repeat_count", 1) + 1
        entry["fp_suspected"] = bool(
            entry.get("hallucination")
            and entry["repeat_count"] >= JUDGE_CACHE_REPEAT_FP_THRESHOLD
        )
        self._d[k] = entry  # re-insert at MRU
        return entry

    def put(self, question: str, response: str, verdict: dict) -> dict:
        """Record a verdict for a freshly judged pair. First call has
        repeat_count = 1; subsequent calls go through `observe`."""
        if not JUDGE_CACHE_ENABLED:
            return verdict
        k = self._key(question, response)
        verdict = dict(verdict)
        verdict["repeat_count"] = 1
        verdict["fp_suspected"] = False
        self._d[k] = verdict
        while len(self._d) > self._max_size:
            self._d.popitem(last=False)
        return verdict

    def stats(self) -> dict:
        return {
            "enabled": JUDGE_CACHE_ENABLED,
            "size": len(self._d),
            "max_size": self._max_size,
            "fp_threshold": JUDGE_CACHE_REPEAT_FP_THRESHOLD,
        }

    def clear(self) -> int:
        n = len(self._d)
        self._d.clear()
        return n


JUDGE_CACHE = _JudgeCache(max_size=JUDGE_CACHE_MAX)


def _last_user_message(messages: list[dict]) -> tuple[int, str] | tuple[None, None]:
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):  # multi-modal content
                c = " ".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
            return i, c
    return None, None


def _make_sse_chunk(model_name: str, hc_payload: dict) -> str:
    """Build an OpenAI-shaped chat-completion chunk carrying only an x_halucheck
    extension. Default OpenAI clients will see empty deltas (no content); clients
    that look at x_halucheck see the sidecar event."""
    chunk = {
        "id": "halucheck-event",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "x_halucheck": hc_payload,
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if SIDECAR is None:
        raise HTTPException(503, "Sidecar not loaded")
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(400, "messages required")

    metadata = {}
    idx, user_q = _last_user_message(messages)
    if idx is None:
        raise HTTPException(400, "no user message found")

    # Stage 1: abs_check
    t0 = time.time()
    flagged, abs_margin = SIDECAR.abs_check(user_q)
    metadata["abs_check_ms"] = int((time.time() - t0) * 1000)
    metadata["abs_check_flagged"] = bool(flagged)
    metadata["abs_check_margin"] = round(abs_margin, 3)

    # Stage 2: rephrase if flagged + ENABLE_REPHRASE
    used_question = user_q
    if flagged and ENABLE_REPHRASE:
        t0 = time.time()
        rephrased = SIDECAR.rephrase(user_q)
        metadata["rephrase_ms"] = int((time.time() - t0) * 1000)
        # Composition guard (embedding-based by default; Jaccard fallback)
        if rephrased and SIDECAR.composition_guard(user_q, rephrased,
                                                    min_overlap=COMPOSITION_MIN_OVERLAP,
                                                    min_cosine=COMPOSITION_MIN_COSINE,
                                                    use_embedding=USE_EMBEDDING_GUARD):
            used_question = rephrased
            metadata["rephrased"] = True
            metadata["rephrased_question"] = rephrased
        else:
            metadata["rephrased"] = False
            metadata["rephrase_dropped_reason"] = "composition_guard"
            if rephrased:
                metadata["rephrase_attempt"] = rephrased  # surface what was tried

    # Update the messages array with possibly-rephrased user prompt
    forward_messages = list(messages)
    if used_question != user_q:
        m = dict(forward_messages[idx])
        m["content"] = used_question
        forward_messages[idx] = m

    # Stage 3: forward to backend
    auth = request.headers.get("authorization", "")
    headers = {"Authorization": auth, "Content-Type": "application/json"}
    forward_body = dict(body)
    forward_body["messages"] = forward_messages
    backend_url = f"{BACKEND_URL.rstrip('/')}/chat/completions"

    # Streaming branch (L6): emit OpenAI-shaped SSE with halucheck events embedded
    if body.get("stream"):
        return StreamingResponse(
            _stream_chat_completions(backend_url, forward_body, headers,
                                      used_question, metadata,
                                      body.get("model", "")),
            media_type="text/event-stream",
        )

    t0 = time.time()
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(backend_url, json=forward_body, headers=headers)
        except Exception as e:
            raise HTTPException(502, f"Backend error: {e}")
    metadata["backend_ms"] = int((time.time() - t0) * 1000)

    if r.status_code != 200:
        # Pass backend errors through
        return JSONResponse(r.json() if "application/json" in r.headers.get("content-type", "")
                            else {"error": r.text},
                            status_code=r.status_code)

    resp_body = r.json()

    # Stage 4: judge the response
    if ENABLE_JUDGE:
        try:
            choice = resp_body.get("choices", [{}])[0]
            msg = choice.get("message", {})
            response_text = msg.get("content", "") or ""
            if response_text:
                t0 = time.time()
                _, judge_margin = SIDECAR.judge(used_question, response_text)
                metadata["judge_ms"] = int((time.time() - t0) * 1000)
                # Apply configurable threshold (default 0; production should re-calibrate)
                judged_hall = judge_margin > JUDGE_THRESHOLD
                metadata["judge_hallucination"] = bool(judged_hall)
                metadata["judge_margin"] = round(judge_margin, 3)
                metadata["judge_threshold_used"] = JUDGE_THRESHOLD
        except Exception as e:
            log.warning(f"Judge failed: {e}")

    # Attach metadata to response (non-breaking — extra fields ignored by OpenAI clients)
    resp_body["x_halucheck"] = metadata

    return JSONResponse(resp_body)


async def _stream_chat_completions(backend_url: str, forward_body: dict,
                                    headers: dict, used_question: str,
                                    prelude_metadata: dict, model_name: str):
    """SSE generator. Forwards backend chunks verbatim, intercepts content deltas
    to feed a StreamingJudge, interleaves halucheck judge events as x_halucheck
    chunks (with empty deltas so default OpenAI clients pass them through).

    Termination contract:
      1. Yield prelude halucheck chunk carrying preflight metadata.
      2. Stream backend chunks one-for-one, augmenting with judge events at
         exponentially-spaced token checkpoints.
      3. On backend [DONE], emit a final judge chunk then forward [DONE].
    """
    t_start = time.time()
    # 1. Prelude: surface preflight decisions before the first content token
    prelude_metadata["stage"] = "preflight_done"
    yield _make_sse_chunk(model_name, dict(prelude_metadata))

    # 2. Set up the streaming judge (mid-stream warnings)
    sj = None
    if ENABLE_JUDGE and STREAM_JUDGE_ENABLED:
        sj = StreamingJudge(
            sidecar=SIDECAR,
            question=used_question,
            threshold=JUDGE_THRESHOLD,
            checkpoints=list(STREAM_CHECKPOINTS),
        )

    # 3. Forward to backend with streaming
    final_metadata = dict(prelude_metadata)
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", backend_url, json=forward_body,
                                       headers=headers) as r:
                if r.status_code != 200:
                    err_text = (await r.aread()).decode("utf-8", errors="replace")
                    yield _make_sse_chunk(model_name, {
                        "stage": "backend_error",
                        "status": r.status_code,
                        "error": err_text[:500],
                    })
                    yield "data: [DONE]\n\n"
                    return
                async for raw_line in r.aiter_lines():
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data:"):
                        continue
                    data = raw_line[5:].lstrip()
                    if data == "[DONE]":
                        # Don't emit [DONE] yet — we still need to finalize the judge
                        break
                    # Forward the backend chunk to the client verbatim
                    yield f"data: {data}\n\n"
                    # Try to extract a delta content and feed StreamingJudge
                    if sj is None:
                        continue
                    try:
                        ev = json.loads(data)
                        choices = ev.get("choices") or []
                        delta = ""
                        if choices:
                            d = choices[0].get("delta") or {}
                            delta = d.get("content") or ""
                    except Exception:
                        delta = ""
                    if not delta:
                        continue
                    try:
                        async for hc_evt in sj.feed(delta):
                            # We already passed the delta to the client; only the
                            # judge events are new information here.
                            if hc_evt.get("type") == "delta":
                                continue
                            yield _make_sse_chunk(model_name, hc_evt)
                    except Exception as e:
                        log.warning(f"StreamingJudge.feed failed: {e}")
    except Exception as e:
        log.warning(f"Streaming backend forward failed: {e}")
        yield _make_sse_chunk(model_name, {"stage": "forward_error",
                                            "error": str(e)})
        yield "data: [DONE]\n\n"
        return

    # 4. Finalize judge and emit final summary
    if sj is not None:
        try:
            final = await sj.finalize()
            final_metadata["judge_margin"] = final.get("final_margin")
            final_metadata["judge_hallucination"] = final.get("final_flagged")
            final_metadata["judge_threshold_used"] = JUDGE_THRESHOLD
            final_metadata["judge_ms"] = final.get("total_judge_ms")
            final_metadata["judge_calls"] = final.get("total_judge_calls")
            final_metadata["judge_warned_during_stream"] = final.get("warned_during_stream")
            yield _make_sse_chunk(model_name, {**final, **{"stage": "judge_final"}})
        except Exception as e:
            log.warning(f"StreamingJudge.finalize failed: {e}")
    final_metadata["backend_ms"] = int((time.time() - t_start) * 1000)
    final_metadata["stage"] = "halucheck_done"
    yield _make_sse_chunk(model_name, final_metadata)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions/preflight")
async def preflight_only(request: Request):
    """Lightweight endpoint: run abs_check + (if flagged) rephrase, return the
    decision without forwarding to a backend. For clients that want to handle
    the LLM call themselves.

    Also returns a `recommend_subagent` signal (with reason) — the runtime
    operationalisation of the guide's "subagents for all exploration" rule:
    exploration-shaped queries cost less to delegate to a fresh subagent than
    to handle inline, because the intermediate file reads never enter the
    parent's context.
    """
    if SIDECAR is None:
        raise HTTPException(503, "Sidecar not loaded")
    body = await request.json()
    messages = body.get("messages", [])
    idx, user_q = _last_user_message(messages)
    if idx is None:
        raise HTTPException(400, "no user message found")
    flagged, abs_margin = SIDECAR.abs_check(user_q)
    out = {"flagged": bool(flagged), "abs_margin": round(abs_margin, 3),
           "original_question": user_q}
    # Optional rephrase-style override: body or query param. Falls back to env default.
    rephrase_style = (body.get("rephrase_style")
                      or request.query_params.get("rephrase_style")
                      or DEFAULT_REPHRASE_STYLE)
    if flagged and ENABLE_REPHRASE:
        rephrased = SIDECAR.rephrase(user_q, style=rephrase_style)
        if rephrased and SIDECAR.composition_guard(user_q, rephrased):
            out["rephrased_question"] = rephrased
            out["rephrase_style"] = rephrase_style
        else:
            out["rephrase_dropped"] = True
            out["rephrase_style"] = rephrase_style

    # Subagent-recommendation signal: heuristic over the query shape.
    rec_flag, rec_reason = _recommend_subagent(user_q)
    out["recommend_subagent"] = rec_flag
    if rec_flag:
        out["subagent_reason"] = rec_reason
    return out


# Patterns that mark a query as exploration-shaped. Conservative — fires only
# on queries that almost certainly burn parent context inline. Override with
# HALUCHECK_SUBAGENT_HINTS=word1,word2,... (comma-separated additional triggers).
_EXPLORATION_PHRASES = (
    "where is", "where are", "find all", "find every", "list all",
    "investigate", "trace through", "audit", "review the code",
    "how does", "how do", "walk through", "explain the architecture",
    "across the codebase", "in the codebase", "all the files",
    "every file", "every place", "all instances",
    "what files", "which files",
)
_EXPLORATION_PHRASES = _EXPLORATION_PHRASES + tuple(
    p.strip().lower()
    for p in os.getenv("HALUCHECK_SUBAGENT_HINTS", "").split(",")
    if p.strip()
)


def _recommend_subagent(query: str) -> tuple[bool, str]:
    """Heuristic classifier: should this query be delegated to a fresh-context
    subagent? Returns (recommended, reason)."""
    q = (query or "").lower()
    if not q:
        return False, ""

    # Length heuristic — very short queries almost always inline-fittable
    if len(q.split()) < 4:
        return False, ""

    # Trigger phrase match
    matched = [p for p in _EXPLORATION_PHRASES if p in q]
    if matched:
        return True, (f"exploration-shaped query matched phrase "
                      f"'{matched[0]}' — a subagent's read pass keeps "
                      "intermediate file contents out of the parent context")

    # Heuristic: multi-subject conjunctions ("X and Y and Z") tend to fan out
    and_count = q.count(" and ")
    if and_count >= 2:
        return True, ("multi-subject query (multiple 'and' clauses) — "
                       "a subagent can fan out the read passes in parallel")

    # Very long query → likely needs decomposition
    if len(q.split()) > 40:
        return True, ("long query (>40 words) — split into a subagent that "
                       "returns a structured plan rather than handling inline")

    return False, ""


@app.post("/v1/judge_cache_stats")
async def judge_cache_stats_endpoint():
    """Return cache stats — useful for debugging FP-loops or capacity tuning."""
    return JUDGE_CACHE.stats()


@app.post("/v1/judge_cache_clear")
async def judge_cache_clear_endpoint():
    """Clear the judge cache (e.g. after the user pivots topics)."""
    cleared = JUDGE_CACHE.clear()
    return {"cleared": cleared}


@app.post("/v1/judge")
async def judge_endpoint(request: Request):
    """Standalone judge endpoint. POST {"question": "...", "response": "..."}.

    Results are cached by (question, response) pair. Repeated identical
    queries return the cached margin without re-running the LoRA. On the
    3rd+ identical HALL hit, the response includes `fp_suspected: true` —
    the agent's signal that the same flag keeps firing and may be a false
    positive worth ignoring. Disable per-call with body `no_cache: true`,
    globally with HALUCHECK_JUDGE_CACHE=0.
    """
    if SIDECAR is None:
        raise HTTPException(503, "Sidecar not loaded")
    body = await request.json()
    q = body.get("question")
    a = body.get("response")
    if not q or not a:
        raise HTTPException(400, "question and response required")
    no_cache = bool(body.get("no_cache", False))
    if not no_cache:
        cached = JUDGE_CACHE.observe(q, a)
        if cached is not None:
            return {**cached, "cache_hit": True}
    pred, margin = SIDECAR.judge(q, a)
    verdict = {"hallucination": bool(pred), "margin": round(margin, 3)}
    if no_cache:
        return verdict
    out = JUDGE_CACHE.put(q, a, verdict)
    out["cache_hit"] = False
    return out


@app.post("/v1/judge_route")
async def judge_route_endpoint(request: Request):
    """Routed judge: combines the primary judge LoRA and the L4-specialised
    secondary (if configured) per-query to suppress Shakespeare-style FPs.

    POST {"question": "...", "response": "..."}. Returns:
      - hallucination (bool, the routed verdict)
      - margin (float, the margin of the judge that decided)
      - v1_margin, v2_margin (float | null)
      - rule ("v1_only" | "v1_default" | "v2_override_likely_FP")
      - router_active (bool, true when v2 is loaded)

    Configure with HALUCHECK_JUDGE_LORA (primary, default v1) and
    HALUCHECK_JUDGE_LORA_V2 (secondary, optional). When v2 isn't set,
    this endpoint behaves identically to /v1/judge.
    """
    if SIDECAR is None:
        raise HTTPException(503, "Sidecar not loaded")
    body = await request.json()
    q = body.get("question")
    a = body.get("response")
    if not q or not a:
        raise HTTPException(400, "question and response required")
    override_thr = float(body.get("override_strong_no",
                                    os.getenv("HALUCHECK_ROUTE_OVERRIDE_STRONG_NO", "-1.0")))
    override_min_v1 = float(body.get("override_min_v1",
                                       os.getenv("HALUCHECK_ROUTE_OVERRIDE_MIN_V1", "2.0")))
    no_cache = bool(body.get("no_cache", False))
    # Cache by (q, a, override_thr) — override changes the verdict
    cache_q = q
    cache_a = f"__route_{override_thr}__\x00{a}"
    if not no_cache:
        cached = JUDGE_CACHE.observe(cache_q, cache_a)
        if cached is not None:
            return {**cached, "cache_hit": True}
    result = SIDECAR.judge_route(q, a, override_strong_no=override_thr,
                                    override_min_v1=override_min_v1)
    # Round margins for response
    for k in ("margin", "v1_margin", "v2_margin"):
        if isinstance(result.get(k), float):
            result[k] = round(result[k], 3)
    if no_cache:
        return result
    out = JUDGE_CACHE.put(cache_q, cache_a, result)
    out["cache_hit"] = False
    return out


@app.post("/v1/judge_multi_turn")
async def judge_multi_turn_endpoint(request: Request):
    """Multi-turn hallucination judge. POST {"question", "response", "history": [...]}.

    Evaluates the latest (question, response) pair against prior conversation
    turns. Catches cross-turn contradictions, identity slips, and other
    hallucinations that only emerge when seen against the conversation history.
    """
    if SIDECAR is None:
        raise HTTPException(503, "Sidecar not loaded")
    body = await request.json()
    q = body.get("question")
    a = body.get("response")
    history = body.get("history", [])
    if not q or not a:
        raise HTTPException(400, "question and response required")
    pred, margin = SIDECAR.judge_multi_turn(q, a, history)
    return {"hallucination": bool(pred), "margin": round(margin, 3),
            "history_turns": len(history)}


@app.post("/v1/judge_rag")
async def judge_rag_endpoint(request: Request):
    """RAG faithfulness judge. POST {"question": "...", "response": "...", "context": "..."}.

    Evaluates whether the response is faithful to the retrieved context (HHEM-style),
    NOT whether the response is factually correct in the world. Use this when you have
    a RAG system and want to detect hallucinations relative to your retrieved documents.

    Backend selection via HALUCHECK_JUDGE_RAG_BACKEND env var:
      - "qwen-lora"  (default): use the Qwen LoRA judge with RAG prompt
      - "hhem":      use Vectara HHEM-2.1-Open (150 M encoder, ~0.9 published RAG AUC).
                     If HHEM fails to load, falls back to the Qwen judge and surfaces
                     the failure in the response metadata.
    """
    if SIDECAR is None:
        raise HTTPException(503, "Sidecar not loaded")
    body = await request.json()
    q = body.get("question")
    a = body.get("response")
    ctx = body.get("context")
    if not q or not a or not ctx:
        raise HTTPException(400, "question, response, and context required")

    # Backend dispatch
    backend_used = "qwen-lora"
    if JUDGE_RAG_BACKEND == "hhem":
        hhem = _get_hhem_backend()
        if hhem is not None:
            result = hhem.judge(q, a, ctx)
            if result is not None:
                # HHEM returned a verdict
                return {
                    "unfaithful": result["unfaithful"],
                    "margin": result["margin"],
                    "score": result["score"],
                    "threshold": result["threshold"],
                    "backend": "hhem",
                    "note": "HHEM-2.1-Open faithfulness score; higher score = more faithful",
                }
        # HHEM unavailable — fall through to Qwen with a warning flag
        backend_used = "qwen-lora-fallback"

    pred, margin = SIDECAR.judge_rag(q, a, ctx)
    return {"unfaithful": bool(pred), "margin": round(margin, 3),
            "backend": backend_used,
            "note": ("RAG faithfulness mode: 'unfaithful' = response not supported by context"
                     + (" (HHEM requested but unavailable; fell back to Qwen judge)"
                        if backend_used == "qwen-lora-fallback" else ""))}


@app.post("/v1/cap_output")
async def cap_output_endpoint(request: Request):
    """Smart tool-output trimmer for LLM agents.

    Verbose tool output (build logs, test runs, command stdout) feeds into
    every subsequent turn's context and gets re-read on each — token cost
    compounds. This endpoint trims aggressively while preserving the signal
    an agent needs (errors, head, tail), and is fully deterministic (no ML,
    same input → same output).

    POST {
      "text": "<raw output>",
      "max_chars": 4000,              # optional
      "head_lines": 25,               # optional
      "tail_lines": 50,               # optional
      "preserve_errors": true,        # optional
      "redact_opaque": true           # optional
    }
    -> {
      "text": "<trimmed>",
      "original_chars": int,
      "trimmed_chars": int,
      "original_lines": int,
      "kept_lines": int,
      "dropped_lines": int,
      "deduped_runs": int,
      "redacted_tokens": int,
      "strategy": str
    }
    """
    body = await request.json()
    text = body.get("text")
    if text is None:
        raise HTTPException(400, "text required")
    if not isinstance(text, str):
        raise HTTPException(400, "text must be a string")
    try:
        result = _cap_output(
            text,
            max_chars=int(body.get("max_chars", 4000)),
            head_lines=int(body.get("head_lines", 25)),
            tail_lines=int(body.get("tail_lines", 50)),
            preserve_errors=bool(body.get("preserve_errors", True)),
            redact_opaque=bool(body.get("redact_opaque", True)),
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"bad params: {e}")
    return {
        "text": result.text,
        "original_chars": result.original_chars,
        "trimmed_chars": result.trimmed_chars,
        "original_lines": result.original_lines,
        "kept_lines": result.kept_lines,
        "dropped_lines": result.dropped_lines,
        "deduped_runs": result.deduped_runs,
        "redacted_tokens": result.redacted_tokens,
        "strategy": result.strategy,
    }


@app.post("/v1/check")
async def check_endpoint(request: Request):
    """Generic claim-vs-evidence verifier — the "give Claude a check it can
    run" primitive. Useful as a tool any LLM agent can call before asserting
    a factual claim:

      POST {"claim": "Marie Curie discovered radium in 1898",
            "evidence": "Marie Curie's 1898 paper announced the discovery of radium and polonium..."}
      ->  {"supported": true, "confidence": 0.94, "score": 0.94, "backend": "hhem"}

    Differs from /v1/judge_rag only in semantics: this endpoint frames the
    decision as "is the claim supported?" rather than "is the response unfaithful
    to the context?" — the same primitive, agent-friendly framing.

    Backend: prefers HHEM (sharp faithfulness scores), falls back to the Qwen
    judge LoRA if HHEM isn't loaded or fails. Set the backend explicitly with
    `?backend=hhem|qwen-lora` query param, or via the
    HALUCHECK_CHECK_BACKEND env var.
    """
    if SIDECAR is None:
        raise HTTPException(503, "Sidecar not loaded")
    body = await request.json()
    claim = body.get("claim")
    evidence = body.get("evidence")
    if not claim or not evidence:
        raise HTTPException(400, "claim and evidence required")
    backend_pref = (request.query_params.get("backend")
                     or body.get("backend")
                     or os.getenv("HALUCHECK_CHECK_BACKEND", "hhem")).lower()

    # Try HHEM first (sharper on faithfulness)
    if backend_pref == "hhem":
        hhem = _get_hhem_backend()
        if hhem is not None:
            result = hhem.judge(claim, claim, evidence)  # premise=evidence, hypothesis=claim
            if result is not None:
                return {
                    "supported": not result["unfaithful"],
                    "confidence": result["score"],
                    "score": result["score"],
                    "threshold": result["threshold"],
                    "backend": "hhem",
                    "note": "supported = (HHEM score >= threshold); higher score = more supported",
                }
        # Fall through to Qwen on HHEM failure

    pred, margin = SIDECAR.judge_rag(claim, claim, evidence)
    # margin > 0 means unfaithful (claim not supported by evidence)
    return {
        "supported": not bool(pred),
        "confidence": round(1.0 / (1.0 + abs(margin)), 3),  # rough mapping
        "margin": round(margin, 3),
        "backend": "qwen-lora",
        "note": "supported = (judge margin <= 0); negative margin = strongly supported",
    }


if __name__ == "__main__":
    port = int(os.getenv("HALUCHECK_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
