"""Whole-CLI runs against the offline mock Task API (tests/mock_server.py). No model calls."""

import json
import os
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))
import mock_server  # pyright: ignore[reportImplicitRelativeImport] - given organiser code, loaded from tests/

from mcnulty.blobs import BlobCache
from mcnulty.record.events import Envelope, HttpExchange, RunFinished, TaskFinished

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
