"""Streaming-time hallucination judge.

Wraps an LLM stream and runs the judge every N tokens, emitting an early-warning
signal as soon as the partial response crosses the hallucination threshold.

This is the Phase-2 niche expansion enabling streaming chat UIs. Without this,
the sidecar judge must wait for the full response (~1-3 sec extra latency,
breaking chat UX). With this, the judge runs in parallel with generation and
can flag mid-response.

Design:

```
LLM stream ────────────► accumulating buffer
       │                       │
       │                       ▼ (every N tokens)
       │                  judge(prompt, partial)
       │                       │
       └──► to user           │
            with sidecar      ▼
            metadata          early-warning event
                              if judge_margin > threshold
```

Architecture choices:
- Judge runs on PARTIAL responses (truncated at the buffer length)
- Trade-off: more frequent judging = more compute but earlier warnings
- We use exponentially-spaced check points: N tokens, 2N, 4N, 8N...
  This costs O(log T) judge calls per response instead of O(T/N).

Usage (from a FastAPI handler):

  async for delta_chunk in stream_from_llm(...):
      async for event in stream_judge.feed(delta_chunk):
          yield event   # passes through user-bound tokens + emits judge events

Compatible with OpenAI's server-sent-events streaming format.
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

# Default check points (in cumulative token count) — exponentially spaced
DEFAULT_CHECKPOINTS = [16, 32, 64, 128, 256, 512, 1024]


@dataclass
class StreamingJudge:
    """Wraps an LLM stream with mid-stream hallucination judging.

    Args:
      sidecar: a Sidecar instance with a .judge(question, partial_response) method
      question: the user prompt being responded to
      threshold: judge margin above which an early warning is emitted (default 0)
      checkpoints: list of cumulative token counts at which to run the judge
      tokens_fn: callable to estimate token count from accumulated text (default len.split())
    """
    sidecar: object  # has .judge(q, a) -> (pred, margin)
    question: str
    threshold: float = 0.0
    checkpoints: list = field(default_factory=lambda: list(DEFAULT_CHECKPOINTS))
    tokens_fn: Callable[[str], int] = field(
        default_factory=lambda: (lambda text: len(text.split())))

    def __post_init__(self):
        self._buffer = ""
        self._next_check_idx = 0
        self._warning_emitted = False
        self._judge_calls = 0
        self._judge_time_ms = 0.0
        self._last_margin = float("-inf")

    async def feed(self, delta: str) -> AsyncIterator[dict]:
        """Feed a delta chunk of text from the LLM stream. Yields events.

        Two kinds of events:
          {"type": "delta", "text": delta} - passes through the LLM output
          {"type": "judge_warning", "margin": x, "partial_tokens": n} - early warning
        """
        if delta:
            self._buffer += delta
            yield {"type": "delta", "text": delta}

        # Check if we crossed the next checkpoint
        n_tokens = self.tokens_fn(self._buffer)
        while (self._next_check_idx < len(self.checkpoints)
               and n_tokens >= self.checkpoints[self._next_check_idx]):
            checkpoint = self.checkpoints[self._next_check_idx]
            self._next_check_idx += 1
            # Run judge on partial response
            t0 = time.time()
            try:
                # Run synchronously in a thread to not block the event loop
                pred, margin = await asyncio.to_thread(
                    self.sidecar.judge, self.question, self._buffer)
            except Exception as e:
                yield {"type": "judge_error", "error": str(e)}
                continue
            self._judge_calls += 1
            self._judge_time_ms += (time.time() - t0) * 1000
            self._last_margin = margin

            if margin > self.threshold and not self._warning_emitted:
                self._warning_emitted = True
                yield {
                    "type": "judge_warning",
                    "margin": round(margin, 3),
                    "partial_tokens": n_tokens,
                    "checkpoint": checkpoint,
                    "note": "judge flagged hallucination on partial response",
                }
            else:
                yield {
                    "type": "judge_status",
                    "margin": round(margin, 3),
                    "partial_tokens": n_tokens,
                    "checkpoint": checkpoint,
                    "flagged": bool(margin > self.threshold),
                }

    async def finalize(self) -> dict:
        """Final judge call on the complete response. Returns summary stats."""
        if self._buffer:
            t0 = time.time()
            pred, margin = await asyncio.to_thread(
                self.sidecar.judge, self.question, self._buffer)
            self._judge_calls += 1
            self._judge_time_ms += (time.time() - t0) * 1000
            self._last_margin = margin
        return {
            "type": "judge_final",
            "final_margin": round(self._last_margin, 3),
            "final_flagged": bool(self._last_margin > self.threshold),
            "total_judge_calls": self._judge_calls,
            "total_judge_ms": round(self._judge_time_ms, 1),
            "tokens_judged": self.tokens_fn(self._buffer),
            "warned_during_stream": self._warning_emitted,
        }


# Example integration with the proxy (would go in halucheck_proxy.py to support
# OpenAI-style server-sent-events streaming):
"""
@app.post("/v1/chat/completions")
async def chat_completions_streaming(request: Request):
    body = await request.json()
    if not body.get("stream"):
        return await chat_completions(request)  # non-streaming path

    user_q = _last_user_message(body["messages"])[1]
    # (skip abs_check + rephrase for brevity; reuse from non-streaming handler)
    forward_body = dict(body)
    forward_body["stream"] = True

    async def stream_generator():
        sj = StreamingJudge(SIDECAR, user_q)
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", BACKEND_URL, json=forward_body) as r:
                async for chunk in r.aiter_lines():
                    if chunk.startswith("data: "):
                        data = chunk[6:]
                        if data.strip() == "[DONE]":
                            break
                        # Parse OpenAI delta, extract content
                        try:
                            ev = json.loads(data)
                            delta = ev["choices"][0].get("delta", {}).get("content", "")
                        except Exception:
                            delta = ""
                        async for evt in sj.feed(delta):
                            yield f"data: {json.dumps(evt)}\\n\\n"
        final = await sj.finalize()
        yield f"data: {json.dumps(final)}\\n\\n"
        yield "data: [DONE]\\n\\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
"""


# ----- Standalone demo -----

class MockSidecar:
    """For local testing without GPU: judge returns 'hallucination' for any
    response containing the word 'fabricated'."""
    def judge(self, question, response):
        is_hall = "fabricated" in response.lower()
        margin = 5.0 if is_hall else -2.0
        return int(is_hall), margin


async def demo():
    """Demo: simulate a streaming LLM response, route through StreamingJudge."""
    sc = MockSidecar()
    sj = StreamingJudge(sc, "What is the capital of France?",
                        checkpoints=[8, 16, 24, 32])

    # Simulated tokens from a fictitious response
    fake_tokens = [
        "The ", "capital ", "of ", "France ", "is ", "Paris. ",
        "Paris ", "was ", "fabricated ", "in ", "the ", "Roman ", "era. ",
        "It ", "sits ", "on ", "the ", "Seine ", "River.",
    ]

    print("STREAMING JUDGE DEMO")
    print("=" * 60)
    for i, tok in enumerate(fake_tokens):
        async for event in sj.feed(tok):
            if event["type"] == "delta":
                print(f"{event['text']}", end="", flush=True)
            elif event["type"] == "judge_warning":
                print(f"\n[!] JUDGE WARNING at token {event['partial_tokens']}: "
                      f"margin {event['margin']}", flush=True)
            elif event["type"] == "judge_status":
                print(f"\n  judge check at {event['partial_tokens']} tokens: "
                      f"margin {event['margin']} flagged={event['flagged']}", flush=True)
    final = await sj.finalize()
    print(f"\n\nFinal: {final}")


if __name__ == "__main__":
    asyncio.run(demo())
