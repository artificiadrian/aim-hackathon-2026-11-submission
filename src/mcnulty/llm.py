"""OpenRouter client: chat with tool calling (the fixed small model), vision, embeddings and Jev decisions.

Only three models are allowed at runtime (see AGENTS.md). Every request is logged to a Recorder.
"""

import threading
import time
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr

from mcnulty.recorder import Frozen, LlmExchange, LlmFailed, Recorder
from mcnulty.settings import Settings

MAIN_MODEL = "google/gemma-4-31b-it"
JEV_MODEL = "typesafe/jev-1.14"
EMBED_MODEL = "qwen/qwen3-embedding-8b"
# Where the fixed model runs. In the final the organisers' gateway sets exactly this on every request, so dev runs
# use it too and their costs match the final.
PROVIDER = {
    "order": ["deepinfra/fp8", "crusoe", "novita"],
    "only": ["deepinfra/fp8", "crusoe", "novita"],
    "allow_fallbacks": True,
}
RETRIES = 5

type ReasoningEffort = Literal["none", "low", "medium", "high"]


class LLMError(RuntimeError):
    pass


class BudgetExhausted(LLMError):
    """403 from OpenRouter: the key's limit is reached. Every later call fails too, so stop calling models."""


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    id: str
    type: str = "function"
    function: dict[str, str]  # {"name": ..., "arguments": "<json text>"}


class ChatReply(Frozen):
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None
    usage: Usage


class CostTracker(BaseModel):
    """Thread-safe tally of model spend in this process."""

    total: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, float] = Field(default_factory=dict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def add(self, model: str, usage: Usage):
        with self._lock:
            self.total += usage.cost
            self.calls += 1
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.by_model[model] = self.by_model.get(model, 0.0) + usage.cost


# --- Jev questions and answers ------------------------------------------------------------------------
class YesNo(Frozen):
    """Yes/no question; the answer carries the probability of 'yes' in ``noul``."""

    type: Literal["noul"] = "noul"
    instructions: str


class PickOne(Frozen):
    """Pick one option; the answer carries the chosen key in ``choice`` and ``probabilities`` for all."""

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]  # option key -> description


class Scale(Frozen):
    """Graded scale between two described end points; the answer carries ``score``."""

    type: Literal["score"] = "score"
    instructions: str
    criteria: tuple[str, str]  # (low end, high end)


JevQuestion = Annotated[YesNo | PickOne | Scale, Field(discriminator="type")]


class JevAnswer(BaseModel):
    """One answer. ``type == "refusal"`` carries no value: treat it as unknown, not as no."""

    model_config = ConfigDict(frozen=True, extra="allow")

    type: str | None = None
    noul: float | None = None
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    score: float | None = None

    @property
    def refused(self):
        return self.type == "refusal"


class OpenRouterClient(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    http: httpx.Client
    base_url: str
    tracker: CostTracker

    @classmethod
    def connect(cls, settings: Settings, tracker: CostTracker):
        http = httpx.Client(
            timeout=180.0,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}",
                "X-Title": "Harness Arena team-10",
            },
        )
        return cls(
            http=http,
            base_url=settings.openrouter_base_url.rstrip("/"),
            tracker=tracker,
        )

    @property
    def jev_url(self):
        """Jev is on the same host with ``/v1`` replaced: ``.../api/v1`` -> ``.../api/alpha/decisions``."""
        return f"{self.base_url.removesuffix('/v1')}/alpha/decisions"

    def post(
        self,
        rec: Recorder,
        url: str,
        body: dict[str, JsonValue],
        timeout: float = 180.0,
    ):
        """POST with retries on network errors, 408, 429, 5xx and 402 + Retry-After (in-flight budget full)."""
        last = ""
        for attempt in range(RETRIES + 1):
            wait = 0.0
            t0 = time.time()
            try:
                r = self.http.post(url, json=body, timeout=timeout)
            except httpx.TransportError as e:
                last = f"{type(e).__name__}: {e}"
                rec.log(LlmFailed(url=url, request=body, error=last))
            else:
                ms = int((time.time() - t0) * 1000)
                response = rec.body(r.content, r.headers.get("content-type", ""))
                rec.log(
                    LlmExchange(
                        url=url,
                        status=r.status_code,
                        ms=ms,
                        request=body,
                        response=response,
                    )
                )
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
                    wait = min(float(r.headers.get("retry-after", 0) or 0), 120.0)
                else:
                    raise LLMError(f"{url} -> {r.status_code}: {r.text[:500]}")
            if attempt < RETRIES:
                time.sleep(max(min(2 * 2**attempt, 30), wait))
        raise LLMError(f"{url} failed after retries: {last}")

    def chat(
        self,
        rec: Recorder,
        messages: list[JsonValue],
        *,
        tools: list[JsonValue] | None = None,
        response_format: JsonValue = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        reasoning_effort: ReasoningEffort | None = None,
    ):
        """One chat/completions call on the fixed model. Thinking tokens count against ``max_tokens``."""
        body: dict[str, JsonValue] = {
            "model": MAIN_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "usage": {"include": True},
            "provider": PROVIDER,
        }
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if response_format is not None:
            body["response_format"] = response_format
        data = self.post(rec, f"{self.base_url}/chat/completions", body)
        if not data.get("choices"):
            raise LLMError(f"no choices in response: {str(data)[:300]}")
        usage = Usage.model_validate(data.get("usage") or {})
        self.tracker.add(MAIN_MODEL, usage)
        choice = data["choices"][0]
        message = choice["message"]
        return ChatReply(
            content=message.get("content"),
            tool_calls=tuple(
                ToolCall.model_validate(c) for c in message.get("tool_calls") or ()
            ),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
        )

    def describe_image(
        self, rec: Recorder, data_url: str, question: str, max_tokens: int = 700
    ):
        """One vision call on an image passed as a base64 data URL."""
        content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        return (
            self.chat(
                rec, [{"role": "user", "content": content}], max_tokens=max_tokens
            ).content
            or ""
        )

    def embed(self, rec: Recorder, texts: list[str]):
        """Embedding vectors for ``texts`` in one request."""
        data = self.post(
            rec,
            f"{self.base_url}/embeddings",
            {"model": EMBED_MODEL, "input": list(texts)},
        )
        self.tracker.add(EMBED_MODEL, Usage.model_validate(data.get("usage") or {}))
        rows = sorted(data["data"], key=lambda d: d["index"])
        if len(rows) != len(texts):
            raise LLMError(f"expected {len(texts)} embeddings, got {len(rows)}")
        return [row["embedding"] for row in rows]

    def decide(
        self, rec: Recorder, state: JsonValue, questions: dict[str, JevQuestion]
    ):
        """Ask Jev typed questions about ``state``; returns ``{name: JevAnswer}``. Keep ``state`` under ~32 KB."""
        body = {
            "model": JEV_MODEL,
            "state": state,
            "questions": {
                name: q.model_dump(mode="json") for name, q in questions.items()
            },
        }
        data = self.post(rec, self.jev_url, body, timeout=60.0)
        self.tracker.add(JEV_MODEL, Usage.model_validate(data.get("usage") or {}))
        return {
            name: JevAnswer.model_validate(a)
            for name, a in (data.get("answers") or {}).items()
        }
