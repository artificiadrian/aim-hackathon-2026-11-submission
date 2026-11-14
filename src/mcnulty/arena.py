"""Task API client (``{ARENA_BASE_URL}/api/v1``) and its data models. Every request is logged to a Recorder."""

import hashlib
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from mcnulty.recorder import Frozen, HttpExchange, HttpFailed, Recorder
from mcnulty.settings import REPO_ROOT, Settings

type Outcome = Literal["OK", "DENIED_SECURITY", "NEEDS_CLARIFICATION", "UNSUPPORTED"]
type RunKind = Literal["dev", "final"]
MAX_NOTE = 500  # characters in a run note


class Open(BaseModel):
    """A platform response: tolerate fields the platform adds later."""

    model_config = ConfigDict(frozen=True, extra="allow")


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


class Source(Frozen):
    """A citation: the path plus the most specific locator for the file type (see docs/ARENA_DOCS.md)."""

    path: str
    page: int | None = None
    line: int | None = None
    record: str | None = None
    sheet: str | None = None
    row: int | None = None
    paragraph: int | None = None


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


class FileInfo(Open):
    path: str
    sha256: str
    size: int | None = None
    kind: str | None = None
    content_type: str | None = None
    readable: bool | None = None


class TaskManifest(Open):
    files: tuple[FileInfo, ...] = ()


class RunMeta(Frozen):
    commit: str | None = None
    dirty: bool | None = None
    note: str | None = Field(default=None, max_length=MAX_NOTE)


class PrefetchStats(Frozen):
    blobs: int
    downloaded: int
    bytes: int


class ArenaError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


# --- blob cache ---------------------------------------------------------------------------------------
class BlobCache(Frozen):
    """Content-addressed files: ``<root>/<sha256>``. A blob never goes stale, so runs and tasks share it."""

    root: Path

    def get(self, sha256: str):
        path = self.root / sha256
        if not path.exists():
            return None
        data = path.read_bytes()
        if (
            hashlib.sha256(data).hexdigest() != sha256
        ):  # damaged on disk: download again
            path.unlink(missing_ok=True)
            return None
        return data

    def put(self, sha256: str, data: bytes):
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ArenaError(f"blob {sha256[:12]} does not match its sha256")
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.root, prefix=f".{sha256[:16]}.", delete=False
        ) as f:
            f.write(data)
        Path(f.name).replace(
            self.root / sha256
        )  # atomic: parallel writers never see half a file


# --- client -------------------------------------------------------------------------------------------
class ArenaClient(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    http: httpx.Client
    cache: BlobCache
    retries: int = 3

    @classmethod
    def connect(cls, settings: Settings):
        http = httpx.Client(
            base_url=f"{settings.arena_base_url.rstrip('/')}/api/v1",
            headers={
                "Authorization": f"Bearer {settings.arena_api_key.get_secret_value()}"
            },
            timeout=60.0,
        )
        return cls(http=http, cache=BlobCache(root=settings.blob_cache))

    def send(self, rec: Recorder, method: str, path: str, body: JsonValue = None):
        """Send with retries on network errors, 5xx and 429 (honouring Retry-After); raise ArenaError on other 4xx."""
        last = ArenaError(f"{method} {path}: no attempt")
        for attempt in range(self.retries + 1):
            delay = 0.5 * 2**attempt
            t0 = time.time()
            try:
                r = self.http.request(method, path, json=body)
            except httpx.TransportError as e:
                rec.log(
                    HttpFailed(
                        method=method,
                        path=path,
                        request=body,
                        error=f"{type(e).__name__}: {e}",
                    )
                )
                last = ArenaError(
                    f"cannot reach the platform ({type(e).__name__}: {e})"
                )
            else:
                ms = int((time.time() - t0) * 1000)
                response = rec.body(r.content, r.headers.get("content-type", ""))
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
                    delay = max(delay, float(r.headers.get("retry-after", "1")))
                elif r.status_code >= 500:
                    last = ArenaError(message, r.status_code)
                else:
                    raise ArenaError(message, r.status_code)
            if attempt < self.retries:
                time.sleep(delay)
        raise last

    # --- endpoints ---
    def tools(self, rec: Recorder):
        data = self.send(rec, "GET", "/tools").json()
        return tuple(ToolSpec.model_validate(t) for t in data["tools"])

    def start_run(self, rec: Recorder, kind: RunKind, seed: int | None, meta: RunMeta):
        body: dict[str, JsonValue] = {
            "kind": kind,
            "meta": meta.model_dump(exclude_none=True),
        }
        if seed is not None:
            body["seed"] = seed
        return Run.model_validate(self.send(rec, "POST", "/runs", body).json())

    def run_status(self, rec: Recorder, run_id: str):
        return RunStatus.model_validate(self.send(rec, "GET", f"/runs/{run_id}").json())

    def call_tool(
        self,
        rec: Recorder,
        run_id: str,
        task_id: str,
        tool: str,
        args: dict[str, JsonValue],
    ):
        """``{ok, result}`` or ``{ok: false, error}``: tool-level errors are data, not exceptions."""
        return self.send(
            rec, "POST", f"/runs/{run_id}/tasks/{task_id}/tools/{tool}", args
        ).json()

    def run_manifest(self, rec: Recorder, run_id: str):
        return RunManifest.model_validate(
            self.send(rec, "GET", f"/runs/{run_id}/manifest").json()
        )

    def task_manifest(self, rec: Recorder, run_id: str, task_id: str):
        """A task's files in their current state (after writes and containment changes)."""
        return TaskManifest.model_validate(
            self.send(rec, "GET", f"/runs/{run_id}/tasks/{task_id}/manifest").json()
        )

    def blob(self, rec: Recorder, run_id: str, sha256: str, task_id: str | None = None):
        data = self.cache.get(sha256)
        if data is None:
            data = self.send(
                rec,
                "GET",
                f"/runs/{run_id}/blobs/{sha256}"
                + (f"?task={task_id}" if task_id else ""),
            ).content
            self.cache.put(sha256, data)
        return data

    def prefetch(self, rec: Recorder, manifest: RunManifest, run_id: str):
        """Download every blob of the run that is not cached yet, in parallel."""
        owner = {sha: t.task_id for t in manifest.tasks for sha in t.files.values()}
        missing = [b for b in manifest.blobs if self.cache.get(b.sha256) is None]
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(
                ex.map(
                    lambda b: self.blob(rec, run_id, b.sha256, owner.get(b.sha256)),
                    missing,
                )
            )
        return PrefetchStats(
            blobs=len(manifest.blobs),
            downloaded=len(missing),
            bytes=sum(b.size for b in missing),
        )

    def submit(self, rec: Recorder, run_id: str, task_id: str, submission: Submission):
        # answer null is meaningful (e.g. UNSUPPORTED), so only the sources drop their empty locators
        body = {
            "outcome": submission.outcome,
            "answer": submission.answer,
            "sources": [s.model_dump(exclude_none=True) for s in submission.sources],
        }
        return SubmitResult.model_validate(
            self.send(
                rec, "POST", f"/runs/{run_id}/tasks/{task_id}/submit", body
            ).json()
        )

    def finish(self, rec: Recorder, run_id: str):
        return RunStatus.model_validate(
            self.send(rec, "POST", f"/runs/{run_id}/finish").json()
        )


def git_meta(cwd: Path = REPO_ROOT):
    """``RunMeta`` with the commit and dirty flag of the checkout, or empty outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return RunMeta()
    return RunMeta(commit=commit.stdout.strip(), dirty=bool(status.stdout.strip()))
