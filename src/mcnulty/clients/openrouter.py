"""OpenRouter client: chat with tool calling (the fixed small model), vision and Jev decisions.

Only three models are allowed at runtime (see AGENTS.md). Requests and responses are typed models; every exchange is
logged to a Recorder.
"""

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Annotated, ClassVar, Literal

import httpx
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, ValidationError

from mcnulty.clients.http import header, retry_after
from mcnulty.frozen import Frozen
from mcnulty.record.events import ErrorInfo, LlmExchange, LlmFailed
from mcnulty.record.log import Recorder
from mcnulty.settings import Settings

MAIN_MODEL = "google/gemma-4-31b-it"
JEV_MODEL = "typesafe/jev-1.14"
RETRIES = 5
BUDGET = 600.0  # seconds for one call with all its retries, unless the caller has less time

type ReasoningEffort = Literal["none", "low", "medium", "high"]


class LLMError(RuntimeError):
    status: int | None

    def __init__(self, url: str, status: int | None, detail: object):
        super().__init__(f"{url} -> {status}: {detail}")
        self.status = status


class BudgetExhausted(LLMError):
    """403 from OpenRouter (the key's limit) or our own spend cap: later calls fail too, so stop calling models."""


class RetriesExhausted(LLMError):
    """Every retry failed; ``detail`` holds the last error."""


class MalformedReply(LLMError):
    """A 200 response that does not parse as the expected model."""


class Reply(BaseModel):
    """A platform response: tolerate fields OpenRouter adds."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="allow")


# --- chat messages ------------------------------------------------------------------------------------
class TextPart(Frozen):
    type: Literal["text"] = "text"
    text: str


class ImageUrl(Frozen):
    url: str  # data:image/...;base64,...


class ImagePart(Frozen):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class FunctionCall(Reply):
    name: str
    arguments: str  # JSON text, as the model wrote it: parse and validate before use


class ToolCall(Reply):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class SystemMessage(Frozen):
    role: Literal["system"] = "system"
    content: str


class UserMessage(Frozen):
    role: Literal["user"] = "user"
    content: str | tuple[ContentPart, ...]


class AssistantMessage(Reply):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None


class ToolMessage(Frozen):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


Message = Annotated[SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")]


class FunctionSpec(Frozen):
    name: str
    description: str
    parameters: dict[str, JsonValue]  # JSON schema of the arguments


class ToolDefinition(Frozen):
    type: Literal["function"] = "function"
    function: FunctionSpec


# --- requests -----------------------------------------------------------------------------------------
class Provider(Frozen):
    """Where the fixed model runs. In the final the organisers' gateway sets exactly this on every request, so dev
    runs use it too and their costs match the final."""

    order: tuple[str, ...] = ("deepinfra/fp8", "crusoe", "novita")
    only: tuple[str, ...] = ("deepinfra/fp8", "crusoe", "novita")
    allow_fallbacks: bool = True


class Reasoning(Frozen):
    effort: ReasoningEffort


class IncludeUsage(Frozen):
    include: bool = True  # OpenRouter adds usage.cost


class ChatRequest(Frozen):
    model: str = MAIN_MODEL
    messages: tuple[Message, ...]
    max_tokens: int
    temperature: float
    usage: IncludeUsage = IncludeUsage()
    provider: Provider = Provider()
    reasoning: Reasoning | None = None
    tools: tuple[ToolDefinition, ...] | None = None
    tool_choice: Literal["auto", "required"] | None = None


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


class JevRequest(Frozen):
    model: str = JEV_MODEL
    state: JsonValue
    questions: dict[str, JevQuestion]


class JevAnswer(Reply):
    """One answer. ``type == "refusal"`` carries no value: treat it as unknown, not as no."""

    type: str | None = None
    noul: float | None = None
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    score: float | None = None

    @property
    def refused(self):
        return self.type == "refusal"


# --- responses ----------------------------------------------------------------------------------------
class Usage(Reply):
    """Token counts and cost; chat replies say prompt/completion tokens, Jev says input/output tokens."""

    prompt_tokens: int = Field(default=0, validation_alias=AliasChoices("prompt_tokens", "input_tokens"))
    completion_tokens: int = Field(default=0, validation_alias=AliasChoices("completion_tokens", "output_tokens"))
    cost: float = 0.0


class Choice(Reply):
    message: AssistantMessage
    finish_reason: str | None = None


class ChatResponse(Reply):
    choices: tuple[Choice, ...] = Field(min_length=1)
    usage: Usage = Usage()
    provider: str | None = None


class JevResponse(Reply):
    answers: dict[str, JevAnswer]
    usage: Usage = Usage()


# --- cost ---------------------------------------------------------------------------------------------
@dataclass(slots=True)
class CostTracker:
    """Thread-safe tally of model spend, with an optional hard cap in USD. A child (one task) adds to its parent too,
    and the parent's cap stops it."""

    cap: float | None = None
    parent: "CostTracker | None" = None
    total: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, float] = field(default_factory=dict[str, float])
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, usage: Usage):
        with self.lock:
            self.total += usage.cost
            self.calls += 1
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.by_model[model] = self.by_model.get(model, 0.0) + usage.cost
        if self.parent is not None:
            self.parent.add(model, usage)

    @property
    def capped(self) -> bool:  # recursive
        return (self.cap is not None and self.total >= self.cap) or (self.parent is not None and self.parent.capped)

    def child(self):
        return CostTracker(parent=self)


# --- client -------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OpenRouterClient:
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
        return cls(http=http, base_url=settings.openrouter_base_url.rstrip("/"), tracker=tracker)

    def for_task(self):
        """The same client with its own tracker for one task (spend per task), under this client's cap."""
        return replace(self, tracker=self.tracker.child())

    @property
    def jev_url(self):
        """Jev is on the same host with ``/v1`` replaced: ``.../api/v1`` -> ``.../api/alpha/decisions``."""
        return f"{self.base_url.removesuffix('/v1')}/alpha/decisions"

    def post[R: Reply](
        self,
        rec: Recorder,
        url: str,
        request: Frozen,
        reply: type[R],
        timeout: float = 180.0,
        budget: float = BUDGET,
    ):
        """POST with retries on network errors, 408, 429, 5xx, 402 + Retry-After and 200 + error body. ``timeout``
        is per attempt; ``budget`` is the time in seconds for all attempts and waits together."""
        if self.tracker.capped:
            raise BudgetExhausted(url, None, f"spend cap ${self.tracker.cap} reached (${self.tracker.total:.4f})")
        body = request.model_dump(mode="json", exclude_none=True)
        stop = time.time() + budget
        last = "no time left for an attempt"
        for attempt in range(RETRIES + 1):
            if stop - time.time() < 1.0:
                break
            wait = 0.0
            t0 = time.time()
            try:
                r = self.http.post(url, json=body, timeout=min(timeout, stop - t0))
            except httpx.TransportError as e:
                last = f"{type(e).__name__}: {e}"
                rec.log(LlmFailed(url=url, request=body, error=ErrorInfo.from_exception(e)))
            else:
                ms = int((time.time() - t0) * 1000)
                response = rec.body(r.content, header(r, "content-type"))
                rec.log(LlmExchange(url=url, status=r.status_code, ms=ms, request=body, response=response))
                if r.status_code == 200:
                    try:
                        return reply.model_validate_json(r.content)
                    except ValidationError as e:
                        if b'"error"' not in r.content:
                            raise MalformedReply(url, 200, r.text[:300]) from e
                        last = f"upstream error: {r.text[:300]}"  # OpenRouter sometimes returns 200 + error
                elif r.status_code == 403:
                    raise BudgetExhausted(url, 403, r.text[:300])
                elif (
                    r.status_code in {408, 429}
                    or r.status_code >= 500
                    or (r.status_code == 402 and "retry-after" in r.headers)
                ):
                    last = f"{r.status_code}: {r.text[:300]}"
                    wait = min(retry_after(header(r, "retry-after")), 120.0)
                else:
                    raise LLMError(url, r.status_code, r.text[:500])
            if attempt < RETRIES:
                time.sleep(max(0.0, min(max(min(2 * (1 << attempt), 30), wait), stop - time.time())))
        raise RetriesExhausted(url, None, last)

    def chat(
        self,
        rec: Recorder,
        messages: tuple[Message, ...],
        *,
        tools: tuple[ToolDefinition, ...] = (),
        max_tokens: int = 2048,
        temperature: float = 0.0,
        reasoning_effort: ReasoningEffort | None = None,
        require_tool: bool = False,
        budget: float = BUDGET,
    ):
        """One chat/completions call on the fixed model. Thinking tokens count against ``max_tokens``.
        ``require_tool``: the reply must call one of ``tools``. ``budget``: seconds for all attempts."""
        request = ChatRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning=Reasoning(effort=reasoning_effort) if reasoning_effort else None,
            tools=tools or None,
            tool_choice=("required" if require_tool else "auto") if tools else None,
        )
        response = self.post(rec, f"{self.base_url}/chat/completions", request, ChatResponse, budget=budget)
        self.tracker.add(MAIN_MODEL, response.usage)
        return response

    def describe_image(self, rec: Recorder, data_url: str, question: str, max_tokens: int = 700):
        """One vision call on an image passed as a base64 data URL."""
        message = UserMessage(content=(TextPart(text=question), ImagePart(image_url=ImageUrl(url=data_url))))
        return self.chat(rec, (message,), max_tokens=max_tokens).choices[0].message.content or ""

    def decide(self, rec: Recorder, state: JsonValue, questions: dict[str, JevQuestion], budget: float = BUDGET):
        """Ask Jev typed questions about ``state``; returns ``{name: JevAnswer}``. Keep ``state`` under ~32 KB."""
        request = JevRequest(state=state, questions=questions)
        response = self.post(rec, self.jev_url, request, JevResponse, timeout=min(60.0, budget), budget=budget)
        self.tracker.add(JEV_MODEL, response.usage)
        return response.answers
