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
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr

INLINE_MAX = 64_000  # bytes
SECRET_PATTERN = re.compile(
    r"(arena_[A-Za-z0-9_-]{8,}|sk-or-[A-Za-z0-9-]{16,}|Bearer\s+[A-Za-z0-9._-]{8,})"
)


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
    error: str


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
    error: str


class TaskCrashed(Frozen):
    kind: Literal["task_crashed"] = "task_crashed"
    error: str


Event = Annotated[
    CliStarted | HttpExchange | HttpFailed | LlmExchange | LlmFailed | TaskCrashed,
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

    def model_post_init(self, context: object) -> None:
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
