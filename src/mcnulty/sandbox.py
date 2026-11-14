"""Run model-written Python in a child process, with a parsed copy of the task workspace beside it.

The child runs ``sys.executable -I`` with an empty environment (no keys), its own temp folder as cwd, a wall-clock
timeout and CPU, memory and file-size limits. The script sees the workspace only as JSON files and the stdlib helper
module ``ws`` (``src/mcnulty/sandbox_ws.py``). In the final container it should also run as a second Unix user.
Note: macOS does not enforce ``RLIMIT_AS``; Linux does.
"""

import contextlib
import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from mcnulty.recorder import Frozen, Recorder, SandboxRan
from mcnulty.workspace.document import DocumentFile
from mcnulty.workspace.image_file import ImageFile
from mcnulty.workspace.json_file import JsonFile
from mcnulty.workspace.jsonl_file import JsonLinesFile
from mcnulty.workspace.parse import ParsedFile, UnreadableFile
from mcnulty.workspace.pdf_file import PdfFile
from mcnulty.workspace.table import TableFile
from mcnulty.workspace.text_file import TextFile
from mcnulty.workspace.unknown_file import BinaryFile
from mcnulty.workspace.workspace import Workspace

TIMEOUT = 20.0  # seconds, wall clock
CPU_SECONDS = 20
MEMORY = 1 << 30  # bytes of address space
FILE_SIZE = 64 << 20  # bytes per written file
OUTPUT_MAX = 4_000  # characters kept of stdout and of stderr
HELPER = Path(__file__).with_name("sandbox_ws.py")
BOOT = "import sys, runpy; sys.path.insert(0, '.'); import ws; runpy.run_path('main.py', {'ws': ws}, '__main__')"


class SandboxResult(Frozen):
    stdout: str
    stderr: str
    exit_code: int | None  # None when the run timed out
    seconds: float
    timed_out: bool


def limit_resources():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    with contextlib.suppress(ValueError):  # macOS refuses RLIMIT_AS in some processes and never enforces it
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_SIZE, FILE_SIZE))


def cut(text: str):
    """Head and tail of a long output, with the number of characters left out."""
    if len(text) <= OUTPUT_MAX:
        return text
    half = OUTPUT_MAX // 2
    return f"{text[:half]}\n... [{len(text) - OUTPUT_MAX} characters cut] ...\n{text[-half:]}"


def as_text(raw: bytes | str | None):
    if raw is None:
        return ""
    return raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")


def run_code(rec: Recorder, folder: Path, code: str):
    """Run ``code`` as ``main.py`` in ``folder`` (made by ``prepare``); log and return the result."""
    (folder / "main.py").write_text(code, encoding="utf-8")
    t0 = time.time()
    try:
        done = subprocess.run(  # noqa: S603 - running untrusted code in a limited child is the purpose
            [sys.executable, "-I", "-c", BOOT],
            cwd=folder,
            env={},
            capture_output=True,
            timeout=TIMEOUT,
            preexec_fn=limit_resources,
            stdin=subprocess.DEVNULL,
            check=False,  # a failed script is a normal result for the model to read
        )
        stdout, stderr, exit_code, timed_out = as_text(done.stdout), as_text(done.stderr), done.returncode, False
    except subprocess.TimeoutExpired as e:
        stdout, stderr, exit_code, timed_out = as_text(e.stdout), as_text(e.stderr), None, True
        stderr += f"\nTimeoutError: the script ran longer than {TIMEOUT:.0f} seconds and was stopped."
    result = SandboxResult(
        stdout=cut(stdout),
        stderr=cut(stderr),
        exit_code=exit_code,
        seconds=round(time.time() - t0, 3),
        timed_out=timed_out,
    )
    rec.log(SandboxRan(code=code, **result.model_dump()))
    return result


# --- workspace export ---------------------------------------------------------------------------------
def unit(source: BaseModel, text: str, **extra: object):
    return {"cite": source.model_dump(exclude_none=True), "text": text, **extra}


def export(parsed: ParsedFile):
    """One parsed file as plain JSON: its kind, its citable units and any extra facts."""
    match parsed:
        case TextFile():
            return {"kind": "text", "units": [unit(u.source, u.text) for u in parsed.lines], "info": {}}
        case JsonFile():
            units = [unit(u.source, u.text) for u in parsed.lines]
            return {"kind": "json", "units": units, "info": {"value": parsed.value}}
        case JsonLinesFile():
            units = [unit(u.source, u.text, value=u.value) for u in parsed.lines]
            return {"kind": "jsonl", "units": units, "info": {}}
        case TableFile():
            units = [
                unit(r.source, " | ".join(r.cells), cells=list(r.cells), hidden=r.hidden, table=t.name)
                for t in parsed.tables
                for r in t.rows
            ]
            tables = [{"name": t.name, "hidden": t.hidden, "rows": len(t.rows)} for t in parsed.tables]
            return {"kind": "table", "units": units, "info": {"tables": tables}}
        case DocumentFile():
            units = [unit(p.source, p.text) for p in parsed.paragraphs]
            return {"kind": "document", "units": units, "info": {"comments": list(parsed.comments)}}
        case PdfFile():
            units = [unit(p.source, p.text, image_count=p.image_count) for p in parsed.pages]
            return {"kind": "pdf", "units": units, "info": {}}
        case ImageFile():
            info = parsed.model_dump(mode="json", exclude={"kind", "path"})
            return {"kind": "image", "units": [], "info": info}
        case BinaryFile():
            return {"kind": "binary", "units": [], "info": {"sha256": parsed.sha256, "size": parsed.size}}
        case UnreadableFile():
            return {"kind": "unreadable", "units": [], "info": {"error": parsed.error.message}}


def prepare(workspace: Workspace, folder: Path):
    """Write ``ws.py`` and the parsed workspace (``workspace/index.json`` + one JSON per file) into ``folder``."""
    data = folder / "workspace"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HELPER, folder / "ws.py")
    index: list[dict[str, object]] = []
    for n, path in enumerate(sorted(workspace.files)):
        exported = {"path": path, **export(workspace.get(path))}
        name = f"{n:04d}.json"
        (data / name).write_text(json.dumps(exported, ensure_ascii=False), encoding="utf-8")
        index.append({"path": path, "kind": exported["kind"], "units": len(exported["units"]), "file": name})
    (data / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
