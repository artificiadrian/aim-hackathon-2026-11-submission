"""The sandbox, both ways: model code works on plain files with the usual libraries (passes), and it cannot change the
workspace or its environment, see keys, use the network or run without limits (catches).

Runs only inside the Docker image (AGENTS.md, Commands); skipped elsewhere.
"""

import hashlib
import os
from pathlib import Path

import pytest

from mcnulty.arena import BlobCache
from mcnulty.recorder import EventLog, Recorder
from mcnulty.sandbox import SITECUSTOMIZE, TaskSandbox

pytestmark = pytest.mark.skipif(not SITECUSTOMIZE.exists(), reason="the sandbox runs only inside the Docker image")
CSV = b"id,actor,records\nR-1,ext,5\nR-2,int,7\n"


@pytest.fixture
def box(tmp_path: Path):
    blobs = BlobCache(root=tmp_path / "blobs")
    files = {"/AGENTS.md": b"# rules\n", "/logs/access/requests.csv": CSV}
    for data in files.values():
        blobs.put(hashlib.sha256(data).hexdigest(), data)
    rec = Recorder(events=EventLog(run_dir=tmp_path / "setup"))
    sandbox = TaskSandbox.create(rec, "t-test", {p: hashlib.sha256(d).hexdigest() for p, d in files.items()}, blobs)
    yield sandbox
    sandbox.remove()


@pytest.fixture
def rec(tmp_path: Path):
    return Recorder(events=EventLog(run_dir=tmp_path / "run"))


def run_ok(box: TaskSandbox, rec: Recorder, code: str):
    result = box.run(rec, code)
    assert result.exit_code == 0, result.stderr
    return result.stdout.strip()


# --- passes ---------------------------------------------------------------------------------------------
def test_passes_plain_files_at_workspace_paths(box: TaskSandbox, rec: Recorder):
    code = "import os; print(open('/AGENTS.md').read().strip(), open('logs/access/requests.csv').readline().strip())"
    assert run_ok(box, rec, code) == "# rules id,actor,records"


def test_passes_the_usual_libraries(box: TaskSandbox, rec: Recorder):
    code = (
        "import pandas as pd, openpyxl, docx, pypdf, odf, PIL, yaml, zoneinfo\n"
        "df = pd.read_csv('/logs/access/requests.csv')\n"
        "print(df.loc[df.actor == 'ext', 'records'].sum(), zoneinfo.ZoneInfo('Europe/Berlin'))"
    )
    assert run_ok(box, rec, code) == "5 Europe/Berlin"


def test_passes_scratch_kept_between_runs(box: TaskSandbox, rec: Recorder):
    run_ok(box, rec, "open('/tmp/notes.txt', 'w').write('step 1')")
    assert run_ok(box, rec, "print(open('/tmp/notes.txt').read())") == "step 1"


# --- catches --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "code",
    [
        "open('/AGENTS.md', 'a').write('x')",  # change a workspace file
        "open('/logs/new.txt', 'w').write('x')",  # add a file to the workspace
        "import os; os.remove('/logs/access/requests.csv')",
        "open('/usr/local/lib/python3.12/site-packages/evil.py', 'w').write('x')",  # change its own environment
        "open('/usr/local/lib/python3.12/site-packages/sitecustomize.py', 'w').write('x')",
    ],
)
def test_catches_writes_outside_scratch(box: TaskSandbox, rec: Recorder, code: str):
    result = box.run(rec, code)
    assert result.exit_code != 0
    assert "PermissionError" in result.stderr


def test_catches_no_keys_no_host_files_and_second_user(box: TaskSandbox, rec: Recorder, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-must-not-leak")
    paths = "'/app', '/proc/1', '/usr/local/bin/uv', '/sandbox'"
    code = f"import os; print(os.getuid(), sorted(os.environ), [os.path.exists(p) for p in ({paths})])"
    out = run_ok(box, rec, code)
    assert out.startswith("10001 ")
    assert "OPENROUTER" not in out
    assert out.endswith("[False, False, False, False]")
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-must-not-leak"


def test_catches_network(box: TaskSandbox, rec: Recorder):
    result = box.run(rec, "import urllib.request; urllib.request.urlopen('http://example.com', timeout=3)")
    assert "the sandbox has no network" in result.stderr


def test_catches_runaway_code(box: TaskSandbox, rec: Recorder):
    slow = box.run(
        rec,
        "import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'while True: pass'])\nwhile True: pass",
        timeout=2,
    )
    assert slow.timed_out
    assert slow.seconds < 5
    memory = box.run(rec, "x = bytearray(3_000_000_000)")
    assert "MemoryError" in memory.stderr
    big = box.run(rec, "open('/tmp/big', 'wb').write(b'0' * 100_000_000)")
    assert big.exit_code != 0


def test_passes_the_harness_keeps_its_network():
    """The network block is in the system Python only; the harness's venv does not load it."""
    import sys

    assert "sitecustomize" not in sys.modules


def test_catches_workspace_paths_that_leave_the_task_root(tmp_path: Path):
    blobs = BlobCache(root=tmp_path / "blobs")
    data = b"planted"
    sha = hashlib.sha256(data).hexdigest()
    blobs.put(sha, data)
    paths = ("/../../escaped-1.txt", "/logs/../../../escaped-2.txt", "/ok/../inside.txt")
    rec = Recorder(events=EventLog(run_dir=tmp_path / "run"))
    sandbox = TaskSandbox.create(rec, "t-escape", dict.fromkeys(paths, sha), blobs)
    try:
        assert not Path("/sandbox/escaped-1.txt").exists()
        assert not Path("/sandbox/tasks/escaped-2.txt").exists()
        assert not Path("/escaped-2.txt").exists()
        assert (sandbox.root / "inside.txt").read_bytes() == data
        events = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8")
        assert events.count('"outside_root"') == 2
    finally:
        sandbox.remove()
