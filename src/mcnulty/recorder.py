"""Run recorder: every event of a run as one JSON line in ``<run_dir>/events.jsonl``.

Large or binary bodies go to ``<run_dir>/blobs/<sha256>.gz``. A ``Recorder`` carries its context (the task it logs
for); ``for_task`` gives a recorder for one task that writes to the same file.
"""

import gzip
import hashlib
import itertools
import json
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Annotated, Literal, override

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr

INLINE_MAX = 64_000  # bytes
SECRET_PATTERN = re.compile(r"(arena_[A-Za-z0-9_-]{8,}|sk-or-[A-Za-z0-9-]{16,}|Bearer\s+[A-Za-z0-9._-]{8,})")


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


# --- bodies -------------------------------------------------------------------------------------------
class JsonBody(Frozen):
    kind: Literal["json"] = "json"
    value: JsonValue


class TextBody(Frozen):
    kind: Literal["text"] = "text"
    value: str


class BlobRef(Frozen):
    kind: Literal["blob"] = "blob"
    sha256: str
    size: int


Body = Annotated[JsonBody | TextBody | BlobRef, Field(discriminator="kind")]


# --- errors -------------------------------------------------------------------------------------------
REPR_MAX = 2_000  # characters per local variable or attribute


def short_repr(value: object):
    try:
        text = repr(value)
    except Exception as e:
        text = f"<repr failed: {type(e).__name__}>"
    return text if len(text) <= REPR_MAX else text[:REPR_MAX] + f"... [{len(text)} chars]"


class Frame(Frozen):
    file: str
    line: int | None
    function: str
    code: str | None
    locals: dict[str, str]  # name -> repr, cut to REPR_MAX


class ErrorInfo(Frozen):
    """An exception with everything it carries: type, message, attributes, notes, traceback with locals, and the
    exception it came from (``raise ... from`` or raised while handling another)."""

    type: str  # qualified name, e.g. "httpx.ConnectError"
    message: str
    attributes: dict[str, str]  # public instance attributes (e.g. status), as repr
    notes: tuple[str, ...]
    frames: tuple[Frame, ...]  # outermost first
    cause: "ErrorInfo | None"

    @classmethod
    def from_exception(cls, e: BaseException, depth: int = 0) -> "ErrorInfo":
        frames = tuple(
            Frame(
                file=summary.filename,
                line=summary.lineno,
                function=summary.name,
                code=summary.line,
                locals={k: short_repr(v) for k, v in frame.f_locals.items() if not k.startswith("__")},
            )
            for frame, summary in zip(
                (f for f, _ in traceback.walk_tb(e.__traceback__)),
                traceback.extract_tb(e.__traceback__),
                strict=True,
            )
        )
        chained = e.__cause__ or (None if e.__suppress_context__ else e.__context__)
        return cls(
            type=f"{type(e).__module__}.{type(e).__qualname__}",
            message=str(e),
            attributes={k: short_repr(v) for k, v in vars(e).items() if not k.startswith("_")},
            notes=tuple(getattr(e, "__notes__", ())),
            frames=frames,
            cause=cls.from_exception(chained, depth + 1) if chained is not None and depth < 10 else None,
        )


# --- events -------------------------------------------------------------------------------------------
class CliStarted(Frozen):
    kind: Literal["cli_started"] = "cli_started"
    argv: tuple[str, ...]


class HttpExchange(Frozen):
    """One Task API request and its response (each retry attempt is its own exchange)."""

    kind: Literal["http"] = "http"
    method: str
    path: str
    status: int
    ms: int
    request: JsonValue
    response: Body


class HttpFailed(Frozen):
    kind: Literal["http_failed"] = "http_failed"
    method: str
    path: str
    request: JsonValue
    error: ErrorInfo


class LlmExchange(Frozen):
    """One OpenRouter request (chat, embeddings or Jev) and its response."""

    kind: Literal["llm"] = "llm"
    url: str
    status: int
    ms: int
    request: JsonValue
    response: Body


class LlmFailed(Frozen):
    kind: Literal["llm_failed"] = "llm_failed"
    url: str
    request: JsonValue
    error: ErrorInfo


class TaskCrashed(Frozen):
    kind: Literal["task_crashed"] = "task_crashed"
    error: ErrorInfo

    @classmethod
    def from_exception(cls, e: BaseException):
        return cls(error=ErrorInfo.from_exception(e))


class TaskFinished(Frozen):
    """The end of one task: what was submitted and how it went (the stdout line, as data)."""

    kind: Literal["task_finished"] = "task_finished"
    outcome: str
    seconds: float
    run_cost_usd: float  # model spend of the whole run so far
    score: float | None
    feedback: tuple[str, ...]


class RunFinished(Frozen):
    kind: Literal["run_finished"] = "run_finished"
    cost_usd: float
    calls: int
    prompt_tokens: int
    completion_tokens: int
    by_model: dict[str, float]


class SandboxRan(Frozen):
    """One run of model-written code in the sandbox (``src/mcnulty/sandbox.py``)."""

    kind: Literal["sandbox"] = "sandbox"
    code: str
    stdout: str  # cut to the sandbox's output limit
    stderr: str
    exit_code: int | None  # None when the run timed out
    seconds: float
    timed_out: bool


class CheckResult(Frozen):
    """One step of ``mcnulty check``."""

    kind: Literal["check"] = "check"
    name: str
    ok: bool
    detail: str


Event = Annotated[
    CliStarted
    | HttpExchange
    | HttpFailed
    | LlmExchange
    | LlmFailed
    | TaskCrashed
    | TaskFinished
    | RunFinished
    | SandboxRan
    | CheckResult,
    Field(discriminator="kind"),
]


class Envelope(Frozen):
    seq: int
    t: float
    task_id: str | None
    event: Event


# --- recorder -----------------------------------------------------------------------------------------
class EventLog(BaseModel):
    """The files of one run. Shared by every Recorder of the run; thread-safe."""

    run_dir: Path
    secrets: tuple[str, ...] = ()
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _seq: itertools.count = PrivateAttr(default_factory=lambda: itertools.count(1))
    _t0: float = PrivateAttr(default_factory=time.time)

    @override
    def model_post_init(self, context: object):
        (self.run_dir / "blobs").mkdir(parents=True, exist_ok=True)

    def redact(self, text: str):
        for secret in self.secrets:
            if len(secret) >= 8:
                text = text.replace(secret, "[REDACTED]")
        return SECRET_PATTERN.sub("[REDACTED]", text)

    def write(self, task_id: str | None, event: Event):
        with self._lock:
            envelope = Envelope(
                seq=next(self._seq),
                t=round(time.time() - self._t0, 3),
                task_id=task_id,
                event=event,
            )
            with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(self.redact(envelope.model_dump_json()) + "\n")

    def store(self, raw: bytes):
        sha = hashlib.sha256(raw).hexdigest()
        path = self.run_dir / "blobs" / f"{sha}.gz"
        if not path.exists():
            path.write_bytes(gzip.compress(raw))
        return BlobRef(sha256=sha, size=len(raw))


class Recorder(Frozen):
    events: EventLog
    task_id: str | None = None

    def for_task(self, task_id: str):
        return self.model_copy(update={"task_id": task_id})

    def log(self, event: Event):
        self.events.write(self.task_id, event)

    def body(self, raw: bytes, content_type: str):
        """A body as an event field: inline JSON or text when small, else a stored blob."""
        textual = "json" in content_type or "text" in content_type or not content_type
        if len(raw) <= INLINE_MAX and textual:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                return self.events.store(raw)
            try:
                return JsonBody(value=json.loads(text))
            except ValueError:
                return TextBody(value=text)
        return self.events.store(raw)
