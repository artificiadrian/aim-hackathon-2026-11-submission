"""Whole-CLI runs against the offline mock Task API (tests/mock_server.py). No model calls."""

import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))
import mock_server  # pyright: ignore[reportImplicitRelativeImport] - given organiser code, loaded from tests/

from mcnulty.blobs import BlobCache
from mcnulty.clients.arena import ArenaClient, ArenaError, RunMeta
from mcnulty.commands.run import RunArgs, start_or_resume
from mcnulty.record.events import Envelope, HttpExchange, RunFinished, TaskFinished
from mcnulty.record.log import EventLog, Recorder

FAKE_ARENA_KEY = "arena_testsecret_0123456789"


@pytest.fixture(scope="module")
def base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), mock_server.H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def run_cli(base_url: str, cwd: Path, *args: str):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ARENA_", "OPENROUTER_"))}
    env.update(
        ARENA_BASE_URL=base_url,
        ARENA_API_KEY=FAKE_ARENA_KEY,
        OPENROUTER_API_KEY="sk-or-test-not-used-0000",
        ARENA_BLOB_CACHE=str(cwd / "cache"),
    )
    return subprocess.run(  # noqa: S603  (our own CLI with test arguments)
        [str(Path(sys.executable).parent / "mcnulty"), *args],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def read_events(cwd: Path):
    (run_dir,) = (cwd / "runs").iterdir()
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return (
        run_dir,
        [TypeAdapter(Envelope).validate_json(line) for line in lines],
        "\n".join(lines),
    )


def test_collect_records_typed_events_without_keys(base_url: str, tmp_path: Path):
    r = run_cli(base_url, tmp_path, "collect", "--note", "probe")
    assert r.returncode == 0, r.stderr
    run_dir, events, raw = read_events(tmp_path)
    assert FAKE_ARENA_KEY not in raw
    paths = [e.event.path for e in events if isinstance(e.event, HttpExchange)]
    assert paths[:2] == ["/tools", "/runs"]
    assert paths[-1] == "/runs/mock/finish"
    assert not any(p.endswith("/submit") for p in paths), "collect submits nothing"
    assert {"run.json", "tools.json", "manifest.json", "finish.json"} <= {p.name for p in run_dir.iterdir()}
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["run_id"] == "mock"


def test_run_submits_every_task_with_its_task_id(base_url: str, tmp_path: Path):
    r = run_cli(base_url, tmp_path, "run", "--kind", "dev")
    assert r.returncode == 0, r.stderr
    _, events, _ = read_events(tmp_path)
    submits = {e.task_id for e in events if isinstance(e.event, HttpExchange) and e.event.path.endswith("/submit")}
    assert submits == {"t01", "t02"}
    finished = {e.task_id: e.event for e in events if isinstance(e.event, TaskFinished)}
    assert set(finished) == {"t01", "t02"}
    assert all(f.outcome == "UNSUPPORTED" and f.feedback for f in finished.values())
    assert isinstance(events[-1].event, RunFinished)


def test_invalid_arguments_start_no_run(base_url: str, tmp_path: Path):
    assert run_cli(base_url, tmp_path, "run", "--level", "4").returncode == 2
    assert run_cli(base_url, tmp_path, "run", "--match", "(").returncode == 2
    assert run_cli(base_url, tmp_path, "run", "--note", "x" * 501).returncode == 2
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def test_blob_cache_drops_damaged_files(tmp_path: Path):
    cache = BlobCache(root=tmp_path)
    data = b"hello"
    sha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    cache.put(sha, data)
    assert cache.get(sha) == data
    (tmp_path / sha).write_bytes(b"broken")
    assert cache.get(sha) is None


def test_a_tool_call_that_may_have_arrived_is_not_sent_again(tmp_path: Path):
    """Review OP6: a read timeout on a state-changing call must not repeat it (a second quarantine is a second act)."""
    calls: list[str] = []

    def handler(request: httpx.Request):
        calls.append(request.url.path)
        raise httpx.ReadTimeout("slow", request=request)

    arena = ArenaClient(
        http=httpx.Client(base_url="http://arena.test/api/v1", transport=httpx.MockTransport(handler)),
        cache=BlobCache(root=tmp_path / "blobs"),
    )
    rec = Recorder(events=EventLog(run_dir=tmp_path / "run"))
    with pytest.raises(ArenaError):
        arena.call_tool(rec, "r", "t01", "containment", {"ops": []})
    assert len(calls) == 1


def test_a_task_fetches_its_zip_then_each_file_the_zip_left_out_or_changed(tmp_path: Path):
    small, changed_old, large = b"rules", b'{"v": 1}', b"%PDF big"
    sha = {d: hashlib.sha256(d).hexdigest() for d in (small, changed_old, large)}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("AGENTS.md", small)
        z.writestr("plans/state.json", b'{"v": 2}')  # the zip shows the current state
        z.writestr("LARGE_FILES.txt", b"docs/big.pdf")
    blobs_asked: list[str] = []

    def handler(request: httpx.Request):
        if request.url.path.endswith("/workspace.zip"):
            return httpx.Response(200, content=buffer.getvalue())
        blobs_asked.append(request.url.path.rsplit("/", 1)[-1])
        data = {sha[changed_old]: changed_old, sha[large]: large}[blobs_asked[-1]]
        return httpx.Response(200, content=data)

    arena = ArenaClient(
        http=httpx.Client(base_url="http://arena.test/api/v1", transport=httpx.MockTransport(handler)),
        cache=BlobCache(root=tmp_path / "blobs"),
    )
    files = {"/AGENTS.md": sha[small], "/plans/state.json": sha[changed_old], "/docs/big.pdf": sha[large]}
    stats = arena.fetch_task(Recorder(events=EventLog(run_dir=tmp_path / "run")), "r", "t01", files)
    assert (stats.from_zip, stats.one_by_one, stats.failed) == (1, 2, 0)
    assert sorted(blobs_asked) == sorted([sha[changed_old], sha[large]])
    assert all(arena.cache.get(s) is not None for s in files.values())


def test_parallel_tasks_download_a_shared_large_file_once(tmp_path: Path):
    data = b"%PDF shared by every task"
    sha = hashlib.sha256(data).hexdigest()
    asked: list[str] = []

    def handler(request: httpx.Request):
        if request.url.path.endswith("/workspace.zip"):
            return httpx.Response(404)
        asked.append(request.url.path)
        return httpx.Response(200, content=data)

    arena = ArenaClient(
        http=httpx.Client(base_url="http://arena.test/api/v1", transport=httpx.MockTransport(handler)),
        cache=BlobCache(root=tmp_path / "blobs"),
    )
    rec = Recorder(events=EventLog(run_dir=tmp_path / "run"))
    threads = [
        threading.Thread(target=arena.fetch_task, args=(rec, "r", f"t{i:02d}", {"/docs/big.pdf": sha}))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(asked) == 1


def test_a_restarted_final_resumes_the_open_final_run(tmp_path: Path):
    """Review OP8: after a container restart, POST /runs answers 409 (final already used); go on with that run."""

    def handler(request: httpx.Request):
        if request.method == "POST":
            return httpx.Response(409, json={"error": "final run already used"})
        if request.url.path.endswith("/runs"):
            runs = [{"run_id": "run_old", "kind": "final", "status": "finished"}, {"run_id": "run_f", "kind": "final"}]
            return httpx.Response(200, json={"runs": runs})
        tasks = [{"task_id": "t01", "instruction": "x", "level": 1}]
        return httpx.Response(200, json={"run_id": "run_f", "kind": "final", "tasks": tasks})

    arena = ArenaClient(
        http=httpx.Client(base_url="http://arena.test/api/v1", transport=httpx.MockTransport(handler)),
        cache=BlobCache(root=tmp_path / "blobs"),
    )
    rec = Recorder(events=EventLog(run_dir=tmp_path / "run"))
    run = start_or_resume(rec, arena, RunArgs(kind="final"), RunMeta())
    assert (run.run_id, [t.task_id for t in run.tasks]) == ("run_f", ["t01"])
    with pytest.raises(ArenaError):
        start_or_resume(rec, arena, RunArgs(kind="dev"), RunMeta())
