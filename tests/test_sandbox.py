"""The sandbox, both ways: model code works on plain files with the usual libraries (passes), and it cannot change the
workspace or its environment, see keys, use the network or run without limits (catches).

Runs only inside the Docker image (AGENTS.md, Commands); skipped elsewhere.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest

from mcnulty.agent.sandbox import FSIZE, RECORD_MAX, RUNNER, SandboxPool, TaskSandbox, self_test
from mcnulty.blobs import BlobCache
from mcnulty.record.log import EventLog, Recorder

pytestmark = pytest.mark.skipif(not RUNNER.exists(), reason="the sandbox runs only inside the Docker image")
POOL = SandboxPool()
CSV = b"id,actor,records\nR-1,ext,5\nR-2,int,7\n"


@pytest.fixture
def box(tmp_path: Path):
    blobs = BlobCache(root=tmp_path / "blobs")
    files = {"/AGENTS.md": b"# rules\n", "/logs/access/requests.csv": CSV}
    for data in files.values():
        blobs.put(hashlib.sha256(data).hexdigest(), data)
    rec = Recorder(events=EventLog(run_dir=tmp_path / "setup"))
    sandbox = TaskSandbox.create(
        rec, POOL, "t-test", {p: hashlib.sha256(d).hexdigest() for p, d in files.items()}, blobs
    )
    yield sandbox
    sandbox.remove()


@pytest.fixture
def rec(tmp_path: Path):
    return Recorder(events=EventLog(run_dir=tmp_path / "run"))


def run_ok(box: TaskSandbox, rec: Recorder, code: str):
    result = box.run(rec, code)
    assert result.exit_code == 0, result.stderr.text
    return result.stdout.text.strip()


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
        "open('/usr/local/lib/python3.12/site-packages/ws/run.py', 'w').write('x')",
    ],
)
def test_catches_writes_outside_scratch(box: TaskSandbox, rec: Recorder, code: str):
    result = box.run(rec, code)
    assert result.exit_code != 0
    assert "PermissionError" in result.stderr.text


def test_catches_no_keys_no_host_files_and_second_user(
    box: TaskSandbox, rec: Recorder, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-must-not-leak")
    paths = "'/app', '/proc/1', '/usr/local/bin/uv', '/sandbox'"
    code = f"import os; print(os.getuid(), sorted(os.environ), [os.path.exists(p) for p in ({paths})])"
    out = run_ok(box, rec, code)
    assert out.split()[0].isdigit()
    assert int(out.split()[0]) >= 10001
    assert "OPENROUTER" not in out
    assert out.endswith("[False, False, False, False]")
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-must-not-leak"


def test_catches_network(box: TaskSandbox, rec: Recorder):
    result = box.run(rec, "import urllib.request; urllib.request.urlopen('http://example.com', timeout=3)")
    assert "the sandbox has no network" in result.stderr.text


def test_catches_runaway_code(box: TaskSandbox, rec: Recorder):
    slow = box.run(
        rec,
        "import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'while True: pass'])\nwhile True: pass",
        timeout=2,
    )
    assert slow.timed_out
    assert slow.seconds < 5
    memory = box.run(rec, "x = bytearray(3_000_000_000)")
    assert "MemoryError" in memory.stderr.text
    big = box.run(rec, "open('/tmp/big', 'wb').write(b'0' * 100_000_000)")
    assert big.exit_code != 0


def test_passes_the_harness_keeps_its_network():
    """The network block is installed only by ws.run inside the sandbox, never in the harness's process."""
    import sys

    assert "ws.run" not in sys.modules
    assert "ws.network" not in sys.modules


def test_catches_workspace_paths_that_leave_the_task_root(tmp_path: Path):
    blobs = BlobCache(root=tmp_path / "blobs")
    data = b"planted"
    sha = hashlib.sha256(data).hexdigest()
    blobs.put(sha, data)
    paths = ("/../../escaped-1.txt", "/logs/../../../escaped-2.txt", "/ok/../inside.txt")
    rec = Recorder(events=EventLog(run_dir=tmp_path / "run"))
    sandbox = TaskSandbox.create(rec, POOL, "t-escape", dict.fromkeys(paths, sha), blobs)
    try:
        assert not Path("/sandbox/escaped-1.txt").exists()
        assert not Path("/sandbox/tasks/escaped-2.txt").exists()
        assert not Path("/escaped-2.txt").exists()
        assert (sandbox.root / "inside.txt").read_bytes() == data
        events = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8")
        assert events.count('"outside_root"') == 2
    finally:
        sandbox.remove()


def test_passes_submit_from_code_with_numpy_values(box: TaskSandbox, rec: Recorder):
    code = (
        "import pandas as pd\n"
        "df = pd.read_csv('/logs/access/requests.csv')\n"
        "print('rows', len(df))\n"
        "submit(outcome='OK', answer=df.records.sum(), sources=[{'path': '/logs/access/requests.csv', 'line': 2}])"
    )
    result = box.run(rec, code)
    assert result.exit_code == 0, result.stderr.text
    assert result.stdout.text.startswith("rows 2\n")
    assert result.submitted is not None
    assert '"answer": 12' in result.submitted


def test_catches_a_print_flood_without_harness_memory(box: TaskSandbox, rec: Recorder):
    """A print loop stops at the file size limit; the harness keeps only head and tail (review SB1)."""
    result = box.run(rec, "while True: print('x' * 1000)", timeout=20)
    assert result.stdout.capped
    assert result.stdout.bytes <= FSIZE
    assert len(result.stdout.text) <= RECORD_MAX + 100
    assert "File too large" in result.stderr.text


def test_catches_a_detached_child(box: TaskSandbox, rec: Recorder):
    """A child in its own session must not hold the run open (review SB2)."""
    code = "import os, time\nif os.fork() == 0:\n    os.setsid()\n    time.sleep(600)\nprint('parent done')"
    result = box.run(rec, code, timeout=5)
    assert result.stdout.text.strip() == "parent done"
    assert result.seconds < 5


def test_passes_the_self_test(rec: Recorder):
    assert self_test(rec, POOL).startswith("sandbox ok ")


def test_catches_one_task_signalling_another(box: TaskSandbox, rec: Recorder, tmp_path: Path):
    """One uid per task (review SB4): kill(-1) reaches only the task's own processes."""
    other = TaskSandbox.create(rec, POOL, "t-other", {}, BlobCache(root=tmp_path / "none"))
    try:
        assert other.uid != box.uid
        result: list[str] = []

        def sleep_then_print():
            result.append(other.run(rec, "import time; time.sleep(3); print('alive')").stdout.text)

        sleeper = threading.Thread(target=sleep_then_print)
        sleeper.start()
        time.sleep(1)
        box.run(rec, "import os, signal; os.kill(-1, signal.SIGKILL)")
        sleeper.join()
        assert result == ["alive\n"]
    finally:
        other.remove()


def test_catches_a_full_tmp(box: TaskSandbox, rec: Recorder):
    code = "for i in range(17): open(f'/tmp/f{i}', 'wb').write(b'0' * (16 * 1024 * 1024 - 1))"
    assert box.run(rec, code).tmp_cleared
    assert run_ok(box, rec, "import os; print(os.listdir('/tmp'))") == "[]"


def test_passes_a_script_that_waits_for_a_slot_says_so(rec: Recorder, tmp_path: Path):
    pool = SandboxPool(slots=threading.BoundedSemaphore(1))
    sandbox = TaskSandbox.create(rec, pool, "t-slots", {}, BlobCache(root=tmp_path / "none"))
    try:
        pool.slots.acquire()
        assert not sandbox.run(rec, "print(1)", wait=0.2).started
        pool.slots.release()
        assert sandbox.run(rec, "print(1)").stdout.text == "1\n"
    finally:
        sandbox.remove()


def test_passes_workspace_files_hard_linked_from_the_cache(box: TaskSandbox):
    assert (box.root / "AGENTS.md").stat().st_nlink >= 2


HELPER_FILES = """
import openpyxl, docx, json
from odf.opendocument import OpenDocumentText
from odf.text import P, H
book = openpyxl.Workbook(); ws = book.active; ws.title = "Alerts"
ws.append(["Security alerts export"]); ws.append([]); ws.append(["id", "token", "status"])
ws.append(["A-1", "TOK-1", "open"]); ws.append(["A-2", "TOK-2", "closed"]); ws.row_dimensions[5].hidden = True
book.save("/tmp/a.xlsx")
d = docx.Document(); d.add_heading("Agreement", 1); d.add_paragraph(""); d.add_paragraph("Notify within 48 hours.")
d.save("/tmp/a.docx")
o = OpenDocumentText(); o.text.addElement(H(outlinelevel=1, text="Title")); o.text.addElement(P(text=""))
o.text.addElement(P(text="Signed by Ada.")); o.save("/tmp/a.odt")
json.dump({"items": [{"id": "X-1", "v": 1}, {"id": "X-2", "v": 2}]}, open("/tmp/a.json", "w"))
open("/tmp/a.jsonl", "w").write('{"a": 1}\\n{"agent": "devops-07"}\\n')
open("/tmp/r.csv", "w").write('request_id,actor\\nRQ-1,"ext, partner"\\nRQ-2,int\\n')
"""


def test_passes_the_helpers_cite_every_format_with_its_locator(box: TaskSandbox, rec: Recorder):
    run_ok(box, rec, HELPER_FILES)
    code = (
        "import json\n"
        "found = [cite('/tmp/a.xlsx', 'TOK-2', column='token'), cite('/tmp/a.docx', '48 hours'),"
        " cite('/tmp/a.odt', 'Ada'), cite('/tmp/a.json', 'X-2'), cite('/tmp/a.jsonl', 'devops-07'),"
        " cite('/tmp/r.csv', 'ext, partner'), cite('/logs/access/requests.csv', 'R-2', 'id')]\n"
        "print(json.dumps([[s.model_dump(exclude_none=True) for s in f] for f in found]))"
    )
    assert json.loads(run_ok(box, rec, code)) == [
        [{"path": "/tmp/a.xlsx", "sheet": "Alerts", "row": 5}],
        [{"path": "/tmp/a.docx", "paragraph": 2}],
        [{"path": "/tmp/a.odt", "paragraph": 2}],
        [{"path": "/tmp/a.json", "record": "X-2"}],
        [{"path": "/tmp/a.jsonl", "line": 2}],
        [{"path": "/tmp/r.csv", "record": "RQ-1"}],
        [{"path": "/logs/access/requests.csv", "record": "R-2"}],
    ]
    shown = run_ok(box, rec, "print(text('/tmp/a.xlsx'))")
    assert "sheet=Alerts row=5| A-2 | TOK-2 | closed (hidden row)" in shown
    hits = run_ok(box, rec, "print([str(h) for h in grep('devops-07', '/tmp')])")
    assert hits == """['/tmp/a.jsonl#line=2| {"agent": "devops-07"}']"""
    submitted = box.run(
        rec, "submit('OK', 1, sources=grep('devops-07', '/tmp/a.jsonl', '/tmp/a.json') + cite('/tmp/a.json', 'X-1'))"
    )
    assert submitted.submitted is not None
    assert json.loads(submitted.submitted)["sources"] == [
        {"path": "/tmp/a.jsonl", "line": 2},
        {"path": "/tmp/a.json", "record": "X-1"},
    ]
    assert (
        run_ok(box, rec, "r = sh('grep -c R- /logs/access/requests.csv'); print(r.stdout.strip(), r.exit_code)")
        == "2 0"
    )
    assert run_ok(box, rec, "print(sh('sleep 5', timeout=0.5).exit_code)") == "None"
