"""Task API client (``{ARENA_BASE_URL}/api/v1``) and its data models. Every request is logged to a Recorder."""

import io
import shutil
import subprocess
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from mcnulty.blobs import BlobCache, BlobMismatch
from mcnulty.clients.http import header, retry_after
from mcnulty.frozen import Frozen, ModelInput
from mcnulty.record.events import ErrorInfo, HttpExchange, HttpFailed
from mcnulty.record.log import Recorder
from mcnulty.settings import REPO_ROOT, Settings

type Outcome = Literal["OK", "DENIED_SECURITY", "NEEDS_CLARIFICATION", "UNSUPPORTED"]
type RunKind = Literal["dev", "final"]
MAX_NOTE = 500  # characters in a run note
RETRIES = 7
ENDED = frozenset({"finished", "closed", "ended", "expired"})  # run statuses after which nothing is accepted
ZIP_MEMBER_MAX = 512 * 1024 * 1024  # bytes: a larger member is fetched by itself
NOT_RUN = frozenset({502, 503, 504})  # gateway answers: the request did not reach the Task API


class Open(BaseModel):
    """A platform response: tolerate fields the platform adds later."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="allow")


# --- models -------------------------------------------------------------------------------------------
class Task(Open):
    task_id: str
    instruction: str
    level: int | None = None


class Run(Open):
    run_id: str
    kind: RunKind
    theme: str | None = None
    seed: int | None = None
    variation: int | None = None
    deadline_at: str | None = None
    tasks: tuple[Task, ...] = ()


class RunSummary(Open):
    run_id: str
    kind: RunKind | None = None
    status: str | None = None


class RunList(Open):
    runs: tuple[RunSummary, ...] = ()


class ToolSpec(Open):
    name: str
    description: str = ""
    args: dict[str, str] = Field(default_factory=dict)


class ToolList(Open):
    tools: tuple[ToolSpec, ...] = ()


class Source(ModelInput):
    """A citation: the path plus the most specific locator for the file type (see docs/ARENA_DOCS.md). Lenient:
    citations come from model output ("line": "24" means 24)."""

    path: str
    page: int | None = None
    line: int | None = None
    record: str | None = None
    sheet: str | None = None
    row: int | None = None
    paragraph: int | None = None

    @property
    def locators(self):
        return (self.page, self.line, self.record, self.sheet, self.row, self.paragraph)

    def covers(self, other: "Source"):
        """``other`` has this path and every locator this citation sets (a path-only citation covers the file)."""
        return self.path == other.path and all(
            mine is None or mine == theirs for mine, theirs in zip(self.locators, other.locators, strict=True)
        )


class ToolResult(Open):
    """A tool-level problem (file not found, bad argument) is ``ok: false`` with an error, not an exception."""

    ok: bool
    result: JsonValue = None
    error: str | None = None


class Submission(Frozen):
    outcome: Outcome
    answer: JsonValue
    sources: tuple[Source, ...] = ()


class SubmitResult(Open):
    received: bool = False
    task_id: str | None = None
    score: float | None = None
    feedback: tuple[str, ...] = ()


class TaskStatus(Open):
    """One task in a run summary. Dev runs add the expected answers once the run has ended."""

    task_id: str
    submitted: bool = False
    expected_outcome: Outcome | None = None
    expected_answer: JsonValue = None
    expected_sources: JsonValue = None


class RunStatus(Open):
    status: str | None = None
    tasks_submitted: int | None = None
    tasks_total: int | None = None
    accuracy: float | None = None
    cost_usd: float | None = None
    cost_score: float | None = None
    final_score: float | None = None
    tasks: tuple[TaskStatus, ...] = ()


class BlobInfo(Open):
    sha256: str
    size: int
    kind: str | None = None
    content_type: str | None = None


class TaskFiles(Open):
    task_id: str
    files: dict[str, str]  # path -> sha256


class RunManifest(Open):
    blobs: tuple[BlobInfo, ...] = ()
    tasks: tuple[TaskFiles, ...] = ()


class RunMeta(Frozen):
    commit: str | None = None
    dirty: bool | None = None
    note: str | None = Field(default=None, max_length=MAX_NOTE)


class FetchStats(Frozen):
    files: int
    from_zip: int
    one_by_one: int
    failed: int


class ArenaError(RuntimeError):
    status: int | None

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class PlatformUnreachable(ArenaError):
    def __init__(self, cause: Exception):
        super().__init__(f"cannot reach the platform ({type(cause).__name__}: {cause})")


# --- client -------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ArenaClient:
    http: httpx.Client
    cache: BlobCache
    locks: dict[str, threading.Lock] = field(default_factory=dict[str, threading.Lock], repr=False)
    locks_guard: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def lock_for(self, sha256: str):
        """One lock per blob: parallel tasks that need the same large file download it once."""
        with self.locks_guard:
            return self.locks.setdefault(sha256, threading.Lock())

    @classmethod
    def connect(cls, settings: Settings):
        http = httpx.Client(
            base_url=f"{settings.arena_base_url.rstrip('/')}/api/v1",
            headers={"Authorization": f"Bearer {settings.arena_api_key.get_secret_value()}"},
            timeout=60.0,
        )
        return cls(http=http, cache=BlobCache(root=settings.blob_cache))

    def send(self, rec: Recorder, method: str, path: str, body: JsonValue = None, acts: bool = False):
        """Send with retries on network errors, 5xx and 429 (honouring Retry-After); raise ArenaError on other 4xx.
        The retries span about 75 s: a gateway restart took about 7 s in the rehearsal. A request that ``acts``
        (a tool call, a submit) is retried only when it surely did not run: no connection, 429, 502 to 504."""
        last = ArenaError(f"{method} {path}: no attempt")
        for attempt in range(RETRIES + 1):
            delay = min(2.0 * (1 << attempt), 15.0)
            t0 = time.time()
            try:
                r = self.http.request(method, path, json=body)
            except httpx.TransportError as e:
                rec.log(
                    HttpFailed(
                        method=method,
                        path=path,
                        request=body,
                        error=ErrorInfo.from_exception(e),
                    )
                )
                last = PlatformUnreachable(e)
                if acts and not isinstance(e, httpx.ConnectError | httpx.ConnectTimeout):
                    raise last from e  # it may have arrived: never act twice
            else:
                ms = int((time.time() - t0) * 1000)
                response = rec.body(r.content, header(r, "content-type"))
                rec.log(
                    HttpExchange(
                        method=method,
                        path=path,
                        status=r.status_code,
                        ms=ms,
                        request=body,
                        response=response,
                    )
                )
                if r.status_code < 400:
                    return r
                message = f"{method} {path} -> {r.status_code}: {r.text[:300]}"
                if r.status_code == 429:
                    last = ArenaError(message, 429)
                    delay = max(delay, min(retry_after(header(r, "retry-after")), 60.0))
                elif r.status_code >= 500 and not (acts and r.status_code not in NOT_RUN):
                    last = ArenaError(message, r.status_code)
                else:
                    raise ArenaError(message, r.status_code)
            if attempt < RETRIES:
                time.sleep(delay)
        raise last

    # --- endpoints ---
    def tools(self, rec: Recorder):
        return ToolList.model_validate_json(self.send(rec, "GET", "/tools").content).tools

    def start_run(self, rec: Recorder, kind: RunKind, seed: int | None, meta: RunMeta):
        body: dict[str, JsonValue] = {
            "kind": kind,
            "meta": meta.model_dump(exclude_none=True),
        }
        if seed is not None:
            body["seed"] = seed
        return Run.model_validate_json(self.send(rec, "POST", "/runs", body).content)

    def active_final(self, rec: Recorder):
        """The final run that is still open (after a container restart), or None."""
        runs = RunList.model_validate_json(self.send(rec, "GET", "/runs").content).runs
        open_finals = [r for r in runs if r.kind == "final" and (r.status or "").lower() not in ENDED]
        if not open_finals:
            return None
        return Run.model_validate_json(self.send(rec, "GET", f"/runs/{open_finals[-1].run_id}").content)

    def run_status(self, rec: Recorder, run_id: str):
        return RunStatus.model_validate_json(self.send(rec, "GET", f"/runs/{run_id}").content)

    def call_tool(
        self,
        rec: Recorder,
        run_id: str,
        task_id: str,
        tool: str,
        args: dict[str, JsonValue],
    ):
        response = self.send(rec, "POST", f"/runs/{run_id}/tasks/{task_id}/tools/{tool}", args, acts=True)
        return ToolResult.model_validate_json(response.content)

    def run_manifest(self, rec: Recorder, run_id: str):
        return RunManifest.model_validate_json(self.send(rec, "GET", f"/runs/{run_id}/manifest").content)

    def blob(self, rec: Recorder, run_id: str, sha256: str, task_id: str | None = None):
        with self.lock_for(sha256):
            data = self.cache.get(sha256)
            if data is None:
                data = self.send(
                    rec,
                    "GET",
                    f"/runs/{run_id}/blobs/{sha256}" + (f"?task={task_id}" if task_id else ""),
                ).content
                self.cache.put(sha256, data)
        return data

    def fetch_task(self, rec: Recorder, run_id: str, task_id: str, files: dict[str, str]):
        """Put every file of one task (path -> sha256) into the cache: one ``workspace.zip``, then each file the zip
        left out (it omits large files) or holds in a changed state, one by one. A file that fails is logged and
        skipped: the task runs without it (``TaskSandbox.create`` logs it again)."""
        missing = {p: sha for p, sha in files.items() if self.cache.get(sha) is None}
        if missing:
            self.unzip_into_cache(rec, run_id, task_id, missing)
        left = {p: sha for p, sha in missing.items() if self.cache.get(sha) is None}
        failed = 0
        for sha in set(left.values()):
            try:
                self.blob(rec, run_id, sha, task_id)
            except (ArenaError, BlobMismatch):  # send() logs the request
                failed += 1
        return FetchStats(files=len(files), from_zip=len(missing) - len(left), one_by_one=len(left), failed=failed)

    def unzip_into_cache(self, rec: Recorder, run_id: str, task_id: str, wanted: dict[str, str]):
        """Cache each member of the task's zip whose bytes match the manifest; anything else is fetched later."""
        try:
            data = self.send(rec, "GET", f"/runs/{run_id}/tasks/{task_id}/workspace.zip").content
            archive = zipfile.ZipFile(io.BytesIO(data))
        except (ArenaError, zipfile.BadZipFile):
            return
        with archive:
            for info in archive.infolist():
                sha = wanted.get("/" + info.filename.lstrip("/"))
                if sha is None or info.file_size > ZIP_MEMBER_MAX:
                    continue
                with self.lock_for(sha), suppress(BlobMismatch):  # changed since the manifest: fetched by hash
                    if self.cache.get(sha) is None:
                        self.cache.put(sha, archive.read(info))

    def prefetch(self, rec: Recorder, manifest: RunManifest, run_id: str):
        """Every file of every task into the cache, tasks in parallel (``mcnulty collect``)."""

        def fetch(task: TaskFiles):
            return self.fetch_task(rec, run_id, task.task_id, task.files)

        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(fetch, manifest.tasks))

    def submit(self, rec: Recorder, run_id: str, task_id: str, submission: Submission):
        # answer null is meaningful (e.g. UNSUPPORTED), so only the sources drop their empty locators
        sources: list[JsonValue] = [s.model_dump(mode="json", exclude_none=True) for s in submission.sources]
        body: dict[str, JsonValue] = {"outcome": submission.outcome, "answer": submission.answer, "sources": sources}
        return SubmitResult.model_validate_json(
            self.send(rec, "POST", f"/runs/{run_id}/tasks/{task_id}/submit", body).content  # a repeat gets 409
        )

    def finish(self, rec: Recorder, run_id: str):
        return RunStatus.model_validate_json(self.send(rec, "POST", f"/runs/{run_id}/finish").content)


def git_meta(cwd: Path = REPO_ROOT):
    """``RunMeta`` with the commit and dirty flag of the checkout, or empty outside git."""
    git = shutil.which("git")
    if git is None:
        return RunMeta()
    try:
        # fixed arguments, no shell
        commit = subprocess.run([git, "rev-parse", "HEAD"], cwd=cwd, check=True, text=True, capture_output=True)  # noqa: S603
        status = subprocess.run([git, "status", "--porcelain"], cwd=cwd, check=True, text=True, capture_output=True)  # noqa: S603
    except (OSError, subprocess.CalledProcessError):
        return RunMeta()
    return RunMeta(commit=commit.stdout.strip(), dirty=bool(status.stdout.strip()))
