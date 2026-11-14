"""Task API client (``{ARENA_BASE_URL}/api/v1``) and its data models. Every request is logged to a Recorder."""

import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
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


class PrefetchStats(Frozen):
    blobs: int
    downloaded: int
    failed: int
    bytes: int


class ArenaError(RuntimeError):
    status: int | None

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class PlatformUnreachable(ArenaError):
    def __init__(self, cause: Exception):
        super().__init__(f"cannot reach the platform ({type(cause).__name__}: {cause})")


# --- client -------------------------------------------------------------------------------------------
class ArenaClient(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    http: httpx.Client
    cache: BlobCache

    @classmethod
    def connect(cls, settings: Settings):
        http = httpx.Client(
            base_url=f"{settings.arena_base_url.rstrip('/')}/api/v1",
            headers={"Authorization": f"Bearer {settings.arena_api_key.get_secret_value()}"},
            timeout=60.0,
        )
        return cls(http=http, cache=BlobCache(root=settings.blob_cache))

    def send(self, rec: Recorder, method: str, path: str, body: JsonValue = None):
        """Send with retries on network errors, 5xx and 429 (honouring Retry-After); raise ArenaError on other 4xx.
        The retries span about 75 s: a gateway restart took about 7 s in the rehearsal."""
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
                elif r.status_code >= 500:
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
        response = self.send(rec, "POST", f"/runs/{run_id}/tasks/{task_id}/tools/{tool}", args)
        return ToolResult.model_validate_json(response.content)

    def run_manifest(self, rec: Recorder, run_id: str):
        return RunManifest.model_validate_json(self.send(rec, "GET", f"/runs/{run_id}/manifest").content)

    def blob(self, rec: Recorder, run_id: str, sha256: str, task_id: str | None = None):
        data = self.cache.get(sha256)
        if data is None:
            data = self.send(
                rec,
                "GET",
                f"/runs/{run_id}/blobs/{sha256}" + (f"?task={task_id}" if task_id else ""),
            ).content
            self.cache.put(sha256, data)
        return data

    def prefetch(self, rec: Recorder, manifest: RunManifest, run_id: str):
        """Download every blob of the run that is not cached yet, in parallel. A blob that fails is logged and
        skipped: its task runs without that file (``TaskSandbox.create`` logs it again)."""
        owner = {sha: t.task_id for t in manifest.tasks for sha in t.files.values()}
        missing = [b for b in manifest.blobs if self.cache.get(b.sha256) is None]

        def fetch(blob: BlobInfo):
            try:
                self.blob(rec, run_id, blob.sha256, owner.get(blob.sha256))
            except (ArenaError, BlobMismatch):  # send() logs the request
                return False
            return True

        with ThreadPoolExecutor(max_workers=8) as ex:
            ok = list(ex.map(fetch, missing))
        return PrefetchStats(
            blobs=len(manifest.blobs),
            downloaded=sum(ok),
            failed=len(ok) - sum(ok),
            bytes=sum(b.size for b, done in zip(missing, ok, strict=True) if done),
        )

    def submit(self, rec: Recorder, run_id: str, task_id: str, submission: Submission):
        # answer null is meaningful (e.g. UNSUPPORTED), so only the sources drop their empty locators
        sources: list[JsonValue] = [s.model_dump(mode="json", exclude_none=True) for s in submission.sources]
        body: dict[str, JsonValue] = {"outcome": submission.outcome, "answer": submission.answer, "sources": sources}
        return SubmitResult.model_validate_json(
            self.send(rec, "POST", f"/runs/{run_id}/tasks/{task_id}/submit", body).content
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
