"""Run model-written Python on the task's real files, isolated. Works only inside the Docker image (Linux, root).

Layout in the container:

- ``/sandbox/base``: a copy of the container's own root filesystem (Python 3.12 with pandas, openpyxl, ... from the
  Dockerfile) without the harness, its keys' process data and uv: everything except ``EXCLUDED``. Made once per
  container in the writable layer: Docker's overlay filesystem copies a file up on every hard link, so linking from
  the image would write 320 MB per task.
- ``/sandbox/tasks/<task>``: one root per task, hard links to ``base`` (0.1 s, 4 MB) plus the task's files at their
  workspace paths. The code sees ``/AGENTS.md``, ``/logs/...`` as plain files.

Each run: ``chroot`` into the task root as uid 10001, cwd ``/``, an environment without keys, resource limits and a
wall-clock timeout. Workspace files and the system are root-owned and read-only; ``/tmp`` is the writable scratch
folder and is kept between the runs of a task. ``sitecustomize.py`` in the image blocks the network.

Output: stdout, stderr and ``submit(...)`` calls go to three anonymous files of the harness, never pipes. The file
size limit stops a print flood at ``FSIZE``, the harness reads only head and tail, and a child that leaves the process
group holds no pipe the harness waits on.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

from mcnulty.blobs import BlobCache
from mcnulty.frozen import Frozen
from mcnulty.record.events import FileSkipped, SandboxRan, Stream
from mcnulty.record.log import Recorder

SITECUSTOMIZE = Path("/usr/local/lib/python3.12/site-packages/sitecustomize.py")  # only in the Docker image
BASE = Path("/sandbox/base")
EXCLUDED = frozenset({"app", "sandbox", "proc", "sys", "dev", "tmp", "root", "home", "run", "mnt", "media", "boot"})
NOT_IN_SANDBOX = ("usr/local/bin/uv",)
TASKS = Path("/sandbox/tasks")
UID = 10001
TIMEOUT = 30.0  # seconds, wall clock
FSIZE = 16 * 1024 * 1024  # bytes per written file, also per output stream
LIMITS = (
    "--as=1500000000",  # bytes of address space
    "--cpu=30",  # CPU seconds
    f"--fsize={FSIZE}",
    "--nproc=256",  # processes of uid 10001, all tasks together
    "--nofile=256",
)
ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",  # noqa: S108 - the scratch folder inside the task's chroot
    "LANG": "C.UTF-8",
    "TZ": "UTC",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}
RECORD_MAX = 200_000  # bytes of each stream kept in the record and in memory (head and tail)
BASE_LOCK = threading.Lock()


class SandboxUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("the sandbox runs only inside the Docker image (see AGENTS.md, Commands)")


def ensure_base():
    """Copy the container's root filesystem, without ``EXCLUDED``, into the writable layer, once per container."""
    with BASE_LOCK:
        if BASE.exists():
            return
        if not SITECUSTOMIZE.exists():
            raise SandboxUnavailable
        partial = BASE.with_name("base.partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        top = sorted(str(p) for p in Path("/").iterdir() if p.name not in EXCLUDED)
        subprocess.run(["/bin/cp", "-a", *top, str(partial)], check=True)  # noqa: S603
        for name in NOT_IN_SANDBOX:
            (partial / name).unlink(missing_ok=True)
        (partial / "tmp").mkdir()
        (partial / "tmp").chmod(0o1777)
        partial.rename(BASE)


def read_stream(file: BinaryIO):
    """The head and tail of an output file (at most ``RECORD_MAX`` bytes in memory), its size and line count."""
    size = os.fstat(file.fileno()).st_size
    file.seek(0)
    lines = sum(chunk.count(b"\n") for chunk in iter(lambda: file.read(1 << 20), b""))
    file.seek(0)
    if size <= RECORD_MAX:
        text = file.read().decode("utf-8", errors="replace")
    else:
        half = RECORD_MAX // 2
        head = file.read(half)
        file.seek(size - half)
        tail = file.read()
        note = f"\n[... {size - 2 * half:,} bytes not kept ...]\n"
        text = head.decode("utf-8", errors="replace") + note + tail.decode("utf-8", errors="replace")
    return Stream(text=text, bytes=size, lines=lines, capped=size >= FSIZE)


def last_submission(file: BinaryIO):
    """The JSON of the last ``submit(...)`` call: one line per call, written by ``sandbox/sitecustomize.py``."""
    file.seek(0)
    calls = file.read().decode("utf-8", errors="replace").splitlines()
    return calls[-1] if calls else None


class TaskSandbox(Frozen):
    root: Path

    @classmethod
    def create(cls, rec: Recorder, task_id: str, files: dict[str, str], blobs: BlobCache):
        """A fresh root for one task with its files (path -> sha256) at their workspace paths, read-only.

        Paths are data: one that leaves the task root (``..``, or a link in the system folders) is skipped, since the
        harness writes as root.
        """
        ensure_base()
        root = TASKS / task_id
        shutil.rmtree(root, ignore_errors=True)
        TASKS.mkdir(parents=True, exist_ok=True)
        subprocess.run(["/bin/cp", "-al", str(BASE), str(root)], check=True)  # noqa: S603
        real_root = root.resolve()
        for path, sha in files.items():
            data = blobs.get(sha)
            if data is None:
                rec.log(FileSkipped(path=path, reason="not_downloaded"))
                continue
            relative = Path(path.lstrip("/"))
            folder = (root / relative).parent.resolve()  # follows links in the existing part of the path
            if not relative.name or not folder.is_relative_to(real_root):
                rec.log(FileSkipped(path=path, reason="outside_root"))
                continue
            folder.mkdir(mode=0o755, parents=True, exist_ok=True)
            target = folder / relative.name
            target.unlink(missing_ok=True)  # a system file at a workspace path: never write through its hard link
            target.write_bytes(data)
            target.chmod(0o444)
        return cls(root=root)

    def run(self, rec: Recorder, code: str, timeout: float = TIMEOUT):
        """Run ``code`` with ``python -I`` as uid 10001 in the task root; log and return the result."""
        command = [
            "/usr/bin/choom",
            "-n",
            "1000",  # out of memory: the kernel kills a script before the harness
            "--",
            "/usr/bin/prlimit",
            *LIMITS,
            "--",
            "/usr/sbin/chroot",
            f"--userspec={UID}:{UID}",
            str(self.root),
            "/usr/local/bin/python3",
            "-I",
            "-",
        ]
        t0 = time.time()
        with (
            tempfile.TemporaryFile() as out,
            tempfile.TemporaryFile() as err,
            tempfile.TemporaryFile() as calls,
        ):
            with subprocess.Popen(  # noqa: S603 - running untrusted code in the isolated child is the purpose
                command,
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=err,
                pass_fds=(calls.fileno(),),
                env={**ENV, "SUBMIT_FD": str(calls.fileno())},
                start_new_session=True,  # its own process group, so one kill ends its children too
            ) as child:
                try:
                    child.communicate(code.encode("utf-8"), timeout=timeout)
                    exit_code, timed_out = child.returncode, False
                except subprocess.TimeoutExpired:
                    exit_code, timed_out = None, True
                with suppress(ProcessLookupError):  # no process of the group is left
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            event = SandboxRan(
                code=code,
                stdout=read_stream(out),
                stderr=read_stream(err),
                submitted=last_submission(calls),
                exit_code=exit_code,
                seconds=round(time.time() - t0, 3),
                timed_out=timed_out,
            )
        rec.log(event)
        return event

    def remove(self):
        shutil.rmtree(self.root, ignore_errors=True)


def self_test(rec: Recorder):
    """One stdout line on whether model code can run here (only stdout comes back from the final container)."""
    t0 = time.time()
    try:
        box = TaskSandbox.create(rec, "self-test", {}, BlobCache(root=TASKS / ".self-test-blobs"))
        try:
            run = box.run(rec, "import pandas, os; print(os.getuid())", timeout=60)
        finally:
            box.remove()
    except Exception as e:
        return f"sandbox FAILED: {type(e).__name__}: {str(e)[:200]}"
    if run.exit_code != 0 or run.stdout.text.strip() != str(UID):
        return f"sandbox FAILED: exit {run.exit_code}: {run.stderr.text.strip()[-200:]}"
    return f"sandbox ok {time.time() - t0:.1f}s"
