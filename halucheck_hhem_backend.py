"""HHEM-2.1-Open backend for the /v1/judge_rag endpoint.

HHEM is a 150-M encoder trained by Vectara specifically for RAG faithfulness:
given a (premise, hypothesis) pair, it returns a probability in [0, 1] that
the hypothesis is consistent with the premise. Published RAG-faithfulness
AUC ~0.9 — substantially tighter than our Qwen-LoRA judge on this specific
task.

Usage from halucheck_proxy.py:
    from halucheck_hhem_backend import HHEMBackend
    hhem = HHEMBackend()         # lazy-loads on first call
    score = hhem.score(premise=context, hypothesis=response)
    is_unfaithful = score < THR  # higher score = more faithful

Compat note: requires a one-line shim against transformers>=5; see
`halucheck_hhem_probe.py` for the rationale.
"""
from __future__ import annotations
import logging
import os
import threading
from typing import Iterable

log = logging.getLogger("halucheck.hhem")


# Apply the compat shim eagerly. HHEM's remote modeling code targets older
# transformers; transformers >=5.x renamed `_tied_weights_keys` to
# `all_tied_weights_keys` (a dict) and calls `.keys()` on it during
# `_finalize_model_loading`. The HHEM custom class doesn't define it.
def _apply_compat_shim() -> None:
    try:
        from transformers.modeling_utils import PreTrainedModel
        # `all_tied_weights_keys` is consumed as a dict (`.keys()` called on it)
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}  # type: ignore[attr-defined]
        if not hasattr(PreTrainedModel, "get_expanded_tied_weights_keys"):
            def _get_expanded_tied_weights_keys(self, all_submodels=False):  # type: ignore[no-redef]
                legacy = getattr(self, "_tied_weights_keys", None)
                return legacy if isinstance(legacy, dict) else {}
            PreTrainedModel.get_expanded_tied_weights_keys = _get_expanded_tied_weights_keys  # type: ignore[attr-defined]
    except Exception as e:
        log.warning(f"HHEM compat shim could not be applied: {e}")


_apply_compat_shim()


class HHEMBackend:
    """Lazy-loaded HHEM-2.1-Open wrapper.

    Threading: HF .predict() is GIL-bound at the Python level but releases
    the GIL during the torch forward. A single instance is safe to call
    from multiple FastAPI workers; we use a lock to serialise model.predict
    only because batched calls compete for GPU memory.
    """

    MODEL_ID = "vectara/hallucination_evaluation_model"
    # Higher = more faithful. Vectara's card uses 0.5; we expose via env.
    DEFAULT_THRESHOLD = float(os.getenv("HALUCHECK_HHEM_THRESHOLD", "0.5"))

    def __init__(self, device: str | None = None):
        self.device = device or os.getenv("HALUCHECK_HHEM_DEVICE")
        if not self.device:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        self._model = None
        self._lock = threading.Lock()
        self._load_failed = False
        self._load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def available(self) -> bool:
        return self.loaded or not self._load_failed

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False
            try:
                import torch
                from transformers import AutoModelForSequenceClassification
                log.info(f"HHEM: loading {self.MODEL_ID} on {self.device}...")
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.MODEL_ID,
                    trust_remote_code=True,
                )
                # Manually tie encoder.embed_tokens to shared embedding —
                # transformers 5.x doesn't auto-tie due to our shim, leaving
                # encoder.embed_tokens with a random-init weight that breaks
                # scoring (faithful and unfaithful both score ~0.13).
                try:
                    shared = model.t5.transformer.shared
                    if hasattr(model.t5.transformer, "encoder"):
                        model.t5.transformer.encoder.embed_tokens = shared
                except Exception as tie_err:
                    log.warning(f"HHEM: manual weight-tying failed ({tie_err}); "
                                "scores may be invalid")
                model.eval()
                model.to(self.device)
                if hasattr(model, "t5"):
                    model.t5.to(self.device)
                self._model = model
                log.info("HHEM: loaded.")
                return True
            except Exception as e:
                self._load_failed = True
                self._load_error = f"{type(e).__name__}: {e}"
                log.warning(f"HHEM: load failed ({self._load_error}); "
                            "falling back to Qwen judge for /v1/judge_rag")
                return False

    def score(self, premise: str, hypothesis: str) -> float | None:
        """Return faithfulness probability in [0, 1], or None if unavailable."""
        if not self._ensure_loaded():
            return None
        import torch
        with self._lock, torch.no_grad():
            scores = self._model.predict([(premise, hypothesis)])
        return float(scores.detach().cpu().tolist()[0])

    def score_batch(
        self, pairs: Iterable[tuple[str, str]]
    ) -> list[float] | None:
        """Score multiple (premise, hypothesis) pairs in one forward."""
        if not self._ensure_loaded():
            return None
        pair_list = list(pairs)
        if not pair_list:
            return []
        import torch
        with self._lock, torch.no_grad():
            scores = self._model.predict(pair_list)
        return scores.detach().cpu().tolist()

    def judge(self, question: str, response: str, context: str,
              threshold: float | None = None) -> dict | None:
        """Same shape as Sidecar.judge_rag(): returns
            {"unfaithful": bool, "margin": float, "score": float, "backend": "hhem"}
        or None if HHEM is unavailable.

        We surface the score as `score` and also synthesise a Qwen-shape
        `margin` (= 2 * (threshold - score)) so downstream consumers that
        already use `margin > 0 → flagged` work without changes.
        """
        thr = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        s = self.score(context, response)
        if s is None:
            return None
        is_unfaithful = bool(s < thr)
        # Map [0, 1] score to a margin around 0: positive margin = flagged
        # (unfaithful). Scale so that a 0.5 threshold gives margin in roughly
        # [-1, +1] over the typical score range.
        margin = float(2.0 * (thr - s))
        return {
            "unfaithful": is_unfaithful,
            "margin": round(margin, 4),
            "score": round(s, 4),
            "threshold": thr,
            "backend": "hhem",
        }
