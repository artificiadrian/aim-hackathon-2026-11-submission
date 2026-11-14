"""Run model-written Python on the task's real files, isolated. Works only inside the Docker image (Linux, root).

Layout in the container:

- ``/sandbox/image``: the sandbox root filesystem (Python 3.12, pandas, openpyxl, ...), built by the Dockerfile.
- ``/sandbox/base``: a copy of it in the container's writable layer, made once per container. Docker's overlay
  filesystem copies a file up on every hard link, so linking from the image would write 320 MB per task.
- ``/sandbox/tasks/<task>``: one root per task, hard links to ``base`` (0.1 s, 4 MB) plus the task's files at their
  workspace paths. The code sees ``/AGENTS.md``, ``/logs/...`` as plain files.

Each run: ``chroot`` into the task root as uid 10001, cwd ``/``, an environment without keys, resource limits and a
wall-clock timeout. Workspace files and the system are root-owned and read-only; ``/tmp`` is the writable scratch
folder and is kept between the runs of a task. ``sitecustomize.py`` in the image blocks the network.
"""

import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from mcnulty.arena import BlobCache
from mcnulty.recorder import Frozen, Recorder, SandboxRan

IMAGE = Path("/sandbox/image")
BASE = Path("/sandbox/base")
TASKS = Path("/sandbox/tasks")
UID = 10001
TIMEOUT = 30.0  # seconds, wall clock
LIMITS = (
    "--as=1500000000",  # bytes of address space
    "--cpu=30",  # CPU seconds
    "--fsize=67108864",  # bytes per written file
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
RECORD_MAX = 1_000_000  # characters of stdout and stderr kept in the record
BASE_LOCK = threading.Lock()


class SandboxUnavailable(RuntimeError):
    def __init__(self):
        super().__init__(f"no {IMAGE}: the sandbox runs only inside the Docker image (see AGENTS.md, Commands)")


class SandboxResult(Frozen):
    stdout: str
    stderr: str
    exit_code: int | None  # None when the run timed out
    seconds: float
    timed_out: bool


def ensure_base():
    """Copy the image's sandbox root into the writable layer, once per container."""
    with BASE_LOCK:
        if BASE.exists():
            return
        if not IMAGE.exists():
            raise SandboxUnavailable
        partial = BASE.with_name("base.partial")
        shutil.rmtree(partial, ignore_errors=True)
        subprocess.run(["/bin/cp", "-a", str(IMAGE), str(partial)], check=True)  # noqa: S603
        partial.rename(BASE)


def decode(raw: bytes):
    return raw.decode("utf-8", errors="replace")


class TaskSandbox(Frozen):
    root: Path

    @classmethod
    def create(cls, task_id: str, files: dict[str, str], blobs: BlobCache):
        """A fresh root for one task with its files (path -> sha256) at their workspace paths, read-only."""
        ensure_base()
        root = TASKS / task_id
        shutil.rmtree(root, ignore_errors=True)
        TASKS.mkdir(parents=True, exist_ok=True)
        subprocess.run(["/bin/cp", "-al", str(BASE), str(root)], check=True)  # noqa: S603
        for path, sha in files.items():
            data = blobs.get(sha)
            if data is None:
                continue  # not downloaded: the file is missing in the sandbox as on a broken share
            target = root / path.lstrip("/")
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            target.unlink(missing_ok=True)  # a system file at a workspace path: never write through its hard link
            target.write_bytes(data)
            target.chmod(0o444)
        return cls(root=root)

    def run(self, rec: Recorder, code: str, timeout: float = TIMEOUT):
        """Run ``code`` with ``python -I`` as uid 10001 in the task root; log and return the full output."""
        command = [
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
        with subprocess.Popen(  # noqa: S603 - running untrusted code in the isolated child is the purpose
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=ENV,
            start_new_session=True,  # its own process group, so a timeout kills every child too
        ) as child:
            try:
                out, err = child.communicate(code.encode("utf-8"), timeout=timeout)
                exit_code, timed_out = child.returncode, False
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                out, err = child.communicate()
                exit_code, timed_out = None, True
        result = SandboxResult(
            stdout=decode(out),
            stderr=decode(err),
            exit_code=exit_code,
            seconds=round(time.time() - t0, 3),
            timed_out=timed_out,
        )
        rec.log(
            SandboxRan(
                code=code,
                stdout=result.stdout[:RECORD_MAX],
                stderr=result.stderr[:RECORD_MAX],
                exit_code=result.exit_code,
                seconds=result.seconds,
                timed_out=result.timed_out,
            )
        )
        return result

    def remove(self):
        shutil.rmtree(self.root, ignore_errors=True)
