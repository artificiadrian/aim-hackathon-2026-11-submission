"""OpenRouter calls: chat with tool calling (the fixed small model), vision, embeddings and Jev decisions.

All calls are plain httpx POSTs on one shared client whose transport comes from ``arena.recorder``, so every request
and response is recorded (the openai SDK runs on httpx2, which the recorder cannot see). Each response's
``usage.cost`` (USD, added by OpenRouter) feeds a CostTracker. Endpoints come from ``OPENROUTER_BASE_URL``.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from arena import client as _  # noqa: F401  (loads .env)
from arena import recorder

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAIN_MODEL = os.environ.get("ARENA_MAIN_MODEL", "google/gemma-4-31b-it")
JEV_MODEL = os.environ.get("ARENA_JEV_MODEL", "typesafe/jev-1.14")
EMBED_MODEL = os.environ.get("ARENA_EMBED_MODEL", "qwen/qwen3-embedding-8b")
# Where the fixed model runs. In the final the organisers' gateway sets exactly this on every request (whatever we
# send), so dev runs use it too and their costs match the final.
PROVIDER_ORDER = ["deepinfra/fp8", "crusoe", "novita"]

TIMEOUT = 180.0
RETRIES = 5


@dataclass
class CostTracker:
    """Thread-safe local tally of spend. The platform's score uses the real OpenRouter key usage."""

    total: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, usage: dict | None) -> float:
        usage = usage or {}
        cost = float(usage.get("cost") or 0.0)
        with self._lock:
            self.total += cost
            self.calls += 1
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            self.by_model[model] = self.by_model.get(model, 0.0) + cost
        return cost


class LLMError(RuntimeError):
    pass


class BudgetExhausted(LLMError):
    """403 from OpenRouter: the key's limit is reached. Every later call fails too, so stop calling models."""


def _key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        raise LLMError(
            "OPENROUTER_API_KEY is not set (use the team key from the Arena dashboard)."
        )
    return k


def base_url() -> str:
    return (os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def jev_url() -> str:
    """Jev is on the same host, with ``/v1`` replaced: ``.../api/v1`` -> ``.../api/alpha/decisions``."""
    return f"{base_url().removesuffix('/v1')}/alpha/decisions"


_http: dict[str, httpx.Client] = {}
_http_lock = threading.Lock()


def _client() -> httpx.Client:
    """One pooled client for all model calls, created on first use (after the recorder has started)."""
    with _http_lock:
        if "c" not in _http:
            _http["c"] = httpx.Client(
                transport=recorder.transport(),
                timeout=TIMEOUT,
                headers={
                    "Authorization": f"Bearer {_key()}",
                    "X-Title": "Harness Arena team-10",
                },
            )
        return _http["c"]


def _retry_after_secs(value: Any) -> float:
    try:
        return min(float(value), 120.0)
    except (TypeError, ValueError):
        return 0.0


def _post(
    url: str, body: dict, timeout: float = TIMEOUT, retries: int = RETRIES
) -> dict:
    """POST with retries on network errors, 408, 429, 5xx and 402 + Retry-After (in-flight budget full)."""
    last = ""
    for attempt in range(retries + 1):
        wait = 0.0
        try:
            r = _client().post(url, json=body, timeout=timeout)
        except httpx.TransportError as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                data = r.json()
                if "error" not in data or "choices" in data or "answers" in data:
                    return data
                last = f"upstream error: {data['error']}"  # OpenRouter sometimes returns 200 + error
            elif r.status_code == 403:
                raise BudgetExhausted(f"{url} -> 403: {r.text[:300]}")
            elif (
                r.status_code in (408, 429)
                or r.status_code >= 500
                or (r.status_code == 402 and "retry-after" in r.headers)
            ):
                last = f"{r.status_code}: {r.text[:300]}"
                wait = _retry_after_secs(r.headers.get("retry-after", 0))
            else:
                raise LLMError(f"{url} -> {r.status_code}: {r.text[:500]}")
        if attempt < retries:
            time.sleep(max(min(2 * 2**attempt, 30), wait))
    raise LLMError(f"{url} failed after retries: {last}")


def chat(
    messages: list[dict],
    *,
    model: str = MAIN_MODEL,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
    response_format: dict | None = None,
    max_tokens: int = 2048,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = None,
    tracker: CostTracker | None = None,
) -> dict:
    """One chat/completions call. Returns the assistant ``message`` dict plus ``_finish_reason``.

    Thinking tokens count against ``max_tokens`` and cost money; if they use it up, ``content`` is null.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if temperature is not None:
        body["temperature"] = temperature
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    if model == MAIN_MODEL:
        body["provider"] = {
            "order": PROVIDER_ORDER,
            "only": PROVIDER_ORDER,
            "allow_fallbacks": True,
        }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice or "auto"
    if response_format:
        body["response_format"] = response_format
    data = _post(f"{base_url()}/chat/completions", body)
    if tracker:
        tracker.add(model, data.get("usage"))
    if not data.get("choices"):
        raise LLMError(f"No choices in response: {str(data)[:300]}")
    choice = data["choices"][0]
    return {**choice["message"], "_finish_reason": choice.get("finish_reason")}


def describe_image(
    data_url: str,
    question: str,
    *,
    model: str = MAIN_MODEL,
    max_tokens: int = 700,
    tracker: CostTracker | None = None,
) -> str:
    """One vision call on an image passed as a base64 data URL."""
    msg = chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        model=model,
        max_tokens=max_tokens,
        tracker=tracker,
    )
    return msg.get("content") or ""


def embed(
    texts: list[str], *, model: str = EMBED_MODEL, tracker: CostTracker | None = None
) -> list[list[float]]:
    """Embedding vectors for ``texts`` (one request). Embed each distinct file once per run (files are sha256-keyed)."""
    data = _post(f"{base_url()}/embeddings", {"model": model, "input": texts})
    if tracker:
        tracker.add(model, data.get("usage"))
    rows = sorted(data["data"], key=lambda d: d["index"])
    if len(rows) != len(texts):
        raise LLMError(f"expected {len(texts)} embeddings, got {len(rows)}")
    return [r["embedding"] for r in rows]


# --- Jev: typed decisions, no text output -------------------------------------------------------------
def noul(instructions: str) -> dict:
    """Yes/no question; the answer carries the probability of 'yes'."""
    return {"type": "noul", "instructions": instructions}


def choice(instructions: str, criteria: dict[str, str]) -> dict:
    """Pick one option; the answer carries the chosen option and probabilities for all."""
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, low: str, high: str) -> dict:
    """Graded scale between two described end points."""
    return {"type": "score", "instructions": instructions, "criteria": [low, high]}


def decide(
    state: Any,
    questions: dict[str, dict],
    *,
    model: str = JEV_MODEL,
    tracker: CostTracker | None = None,
) -> dict[str, Any]:
    """Ask Jev typed questions about ``state``. Returns ``{name: answer}``, e.g. ``{"inj": {"noul": 0.97}}``.

    An answer can be ``{"type": "refusal"}`` with no value: treat it as unknown. Keep ``state`` under ~32 KB.
    """
    data = _post(
        jev_url(), {"model": model, "state": state, "questions": questions}, timeout=60
    )
    if tracker:
        tracker.add(model, data.get("usage"))
    return data.get("answers") or {}
