"""Every event of a run as a typed model, and the error and body types they carry. Pure data; ``log`` writes them."""

import traceback
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from mcnulty.frozen import Frozen


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


def reprs(values: Mapping[str, object]):
    """Each public name's value as a short repr: locals of a frame, attributes of an error."""
    return {k: short_repr(v) for k, v in values.items() if not k.startswith("_")}


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
                locals=reprs(frame.f_locals),
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
            attributes=reprs(vars(e)),
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
    """One OpenRouter request (chat or Jev) and its response."""

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


class TaskSolved(Frozen):
    """What the solver returns: the submission, the checked actions with their findings, the steps used.
    (``mcnulty.solver.Solution`` as JSON: the recorder sits below the modules that define it.)"""

    kind: Literal["task_solved"] = "task_solved"
    solution: JsonValue


class TaskScored(Frozen):
    """Offline grade of one task against the revealed answers of a saved dev run (``src/mcnulty/eval.py``)."""

    kind: Literal["task_scored"] = "task_scored"
    expected_outcome: str | None
    outcome: str
    answer_score: float
    sources_hit: int
    sources_expected: int
    score: float
    steps: int = 0  # model steps of the loop
    calls: int = 0  # model calls, vision included
    prompt_tokens: int = 0
    cost_usd: float = 0.0


class RunFinished(Frozen):
    kind: Literal["run_finished"] = "run_finished"
    cost_usd: float
    calls: int
    prompt_tokens: int
    completion_tokens: int
    by_model: dict[str, float]


class FileSkipped(Frozen):
    """A workspace file not laid out in the task's sandbox: not downloaded, or a path outside the task root."""

    kind: Literal["file_skipped"] = "file_skipped"
    path: str
    reason: Literal["not_downloaded", "outside_root"]


class DeadlineReached(Frozen):
    """``mcnulty run`` reached its deadline and submitted the fallback for every task still open."""

    kind: Literal["deadline_reached"] = "deadline_reached"
    open_tasks: tuple[str, ...]


class ScopeDecided(Frozen):
    """The kinds of action a task asks for (``agent/loop.py`` ``asked_scope``); ``fell_back``: every kind allowed."""

    kind: Literal["scope_decided"] = "scope_decided"
    kinds: tuple[str, ...]
    fell_back: bool


class ToolFailed(Frozen):
    """A tool of the model failed (a vision call, a sandbox error); the model gets an error text and goes on."""

    kind: Literal["tool_failed"] = "tool_failed"
    tool: str
    error: ErrorInfo


class Stream(Frozen):
    """One output stream of a sandbox run: its head and tail as text, and its real size."""

    text: str  # whole, or head and tail with a note in the middle when longer than the record keeps
    bytes: int
    lines: int
    capped: bool  # the run reached the file size limit on this stream


class SandboxRan(Frozen):
    """One run of model-written code in the sandbox (``src/mcnulty/agent/sandbox.py``); also its result."""

    kind: Literal["sandbox"] = "sandbox"
    code: str
    stdout: Stream
    stderr: Stream
    submitted: str | None  # the JSON of the last ``submit(...)`` the code called
    exit_code: int | None  # None when the run timed out; negative: killed by that signal
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
    | TaskSolved
    | TaskScored
    | RunFinished
    | FileSkipped
    | SandboxRan
    | ToolFailed
    | ScopeDecided
    | DeadlineReached
    | CheckResult,
    Field(discriminator="kind"),
]


class Envelope(Frozen):
    seq: int
    t: float
    task_id: str | None
    event: Event
