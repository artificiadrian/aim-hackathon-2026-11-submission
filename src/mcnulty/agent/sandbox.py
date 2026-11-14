"""Run model-written Python on the task's real files, isolated. Works only inside the Docker image (Linux, root).

Layout in the container:

- ``/sandbox/base``: a copy of the system folders of the container's root (``COPIED``: Python 3.12 with pandas,
  openpyxl, ... from the Dockerfile), without uv. An allow list: a folder the organisers mount (``/data``, ``/app``)
  never enters the sandbox. Made once per container in the writable layer: Docker's overlay filesystem copies a file
  up on every hard link, so linking from the image would write 320 MB per task.
- ``/sandbox/tasks/<task>``: one root per task, hard links to ``base`` (0.1 s, 4 MB) plus the task's files at their
  workspace paths, hard-linked from the blob cache when it is on the same filesystem (else copied).

Each run: ``chroot`` into the task root as the task's own uid (``SandboxPool``), cwd ``/``, an environment without
keys, resource limits and a wall-clock timeout. One uid per task: a task cannot signal, trace or starve another, and
every process it leaves behind is killed after the run. At most ``SCRIPTS_AT_ONCE`` scripts run at the same time
(2 CPUs); a script that waits too long does not start and says so. Workspace files and the system are root-owned and
read-only; ``/tmp`` is the writable scratch folder, kept between the runs of a task and emptied above ``TMP_MAX``.
``sitecustomize.py`` in the image blocks the network.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from mcnulty.blobs import BlobCache
from mcnulty.record.events import FileSkipped, SandboxRan, Stream
from mcnulty.record.log import Recorder

SITECUSTOMIZE = Path("/usr/local/lib/python3.12/site-packages/sitecustomize.py")  # only in the Docker image
BASE = Path("/sandbox/base")
COPIED = ("bin", "sbin", "lib", "lib32", "lib64", "libx32", "usr", "etc")  # those the root has; links stay links
NOT_IN_SANDBOX = ("usr/local/bin/uv",)
TASKS = Path("/sandbox/tasks")
FIRST_UID = 10001
UIDS = 64  # tasks with a sandbox at the same time
SCRIPTS_AT_ONCE = 3  # the final container has 2 CPUs; pandas imports alone take about 1 CPU second
SLOT_WAIT = 90.0  # seconds a script waits for a slot before it gives up
TMP_MAX = 256 * 1024 * 1024  # bytes in a task's /tmp before it is emptied (4 GB disk, 12 tasks at once)
TIMEOUT = 30.0  # seconds, wall clock
FSIZE = 16 * 1024 * 1024  # bytes per written file, also per output stream
LIMITS = (
    "--as=1500000000",  # bytes of address space
    "--cpu=30",  # CPU seconds
    f"--fsize={FSIZE}",
    "--nproc=64",  # processes of the task's uid
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


class SandboxUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("the sandbox runs only inside the Docker image (see AGENTS.md, Commands)")


class NoFreeUid(RuntimeError):
    def __init__(self):
        super().__init__(f"all {UIDS} sandbox uids are in use")


@dataclass(frozen=True, slots=True)
class SandboxPool:
    """What all task sandboxes of one process share: the base copy, the script slots and the uids."""

    slots: threading.BoundedSemaphore = field(default_factory=lambda: threading.BoundedSemaphore(SCRIPTS_AT_ONCE))
    base_lock: threading.Lock = field(default_factory=threading.Lock)
    free_uids: set[int] = field(default_factory=lambda: set(range(FIRST_UID, FIRST_UID + UIDS)))
    uid_lock: threading.Lock = field(default_factory=threading.Lock)

    def take_uid(self):
        with self.uid_lock:
            if not self.free_uids:
                raise NoFreeUid
            return self.free_uids.pop()

    def give_uid(self, uid: int):
        with self.uid_lock:
            self.free_uids.add(uid)


def ensure_base(pool: SandboxPool):
    """Copy the system folders (``COPIED``) into the writable layer, once per container."""
    with pool.base_lock:
        if BASE.exists():
            return
        if not SITECUSTOMIZE.exists():
            raise SandboxUnavailable
        partial = BASE.with_name("base.partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        top = [f"/{name}" for name in COPIED if Path(f"/{name}").exists() or Path(f"/{name}").is_symlink()]
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


def kill_uid(uid: int):
    """SIGKILL every process whose real uid is ``uid`` (children that left the process group too)."""
    for status in Path("/proc").glob("[0-9]*/status"):
        with suppress(OSError):
            uid_line = next(line for line in status.read_text().splitlines() if line.startswith("Uid:"))
            if int(uid_line.split()[1]) == uid:
                os.kill(int(status.parent.name), signal.SIGKILL)


def tree_bytes(folder: Path):
    """Bytes in ``folder`` without following links."""
    return sum((Path(top) / name).lstat().st_size for top, _, names in os.walk(folder) for name in names)


def place(source: Path, target: Path):
    """``target`` as a hard link to the cached blob, or a copy when the cache is on another filesystem."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    target.chmod(0o444)  # a link shares the cache file's mode: root-owned and read-only either way


@dataclass(frozen=True, slots=True)
class TaskSandbox:
    root: Path
    uid: int
    pool: SandboxPool

    @classmethod
    def create(cls, rec: Recorder, pool: SandboxPool, task_id: str, files: dict[str, str], blobs: BlobCache):
        """A fresh root for one task with its files (path -> sha256) at their workspace paths, read-only.

        Paths are data: one that leaves the task root (``..``, or a link in the system folders) is skipped, since the
        harness writes as root.
        """
        ensure_base(pool)
        root = TASKS / task_id
        shutil.rmtree(root, ignore_errors=True)
        TASKS.mkdir(parents=True, exist_ok=True)
        subprocess.run(["/bin/cp", "-al", str(BASE), str(root)], check=True)  # noqa: S603
        real_root = root.resolve()
        for path, sha in files.items():
            source = blobs.root / sha
            if not source.is_file():
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
            place(source, target)
        return cls(root=root, uid=pool.take_uid(), pool=pool)

    def run(self, rec: Recorder, code: str, timeout: float = TIMEOUT, wait: float = SLOT_WAIT):
        """Run ``code`` with ``python -I`` as the task's uid in the task root, when a script slot is free; log and
        return the result."""
        t0 = time.time()
        if not self.pool.slots.acquire(timeout=wait):
            event = not_started(code, time.time() - t0)
            rec.log(event)
            return event
        try:
            event = self.run_now(code, timeout)
        finally:
            self.pool.slots.release()
        rec.log(event)
        return event

    def run_now(self, code: str, timeout: float):
        command = [
            "/usr/bin/choom",
            "-n",
            "1000",  # out of memory: the kernel kills a script before the harness
            "--",
            "/usr/bin/prlimit",
            *LIMITS,
            "--",
            "/usr/sbin/chroot",
            f"--userspec={self.uid}:{self.uid}",
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
                kill_uid(self.uid)  # also children that left the process group
                child.wait()
            tmp = self.root / "tmp"
            tmp_cleared = tree_bytes(tmp) > TMP_MAX
            if tmp_cleared:
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir()
                tmp.chmod(0o1777)
            return SandboxRan(
                code=code,
                stdout=read_stream(out),
                stderr=read_stream(err),
                submitted=last_submission(calls),
                exit_code=exit_code,
                seconds=round(time.time() - t0, 3),
                timed_out=timed_out,
                tmp_cleared=tmp_cleared,
            )

    def remove(self):
        kill_uid(self.uid)
        shutil.rmtree(self.root, ignore_errors=True)
        self.pool.give_uid(self.uid)


def not_started(code: str, waited: float):
    empty = Stream(text="", bytes=0, lines=0, capped=False)
    return SandboxRan(
        code=code,
        stdout=empty,
        stderr=empty,
        submitted=None,
        exit_code=None,
        seconds=round(waited, 3),
        timed_out=False,
        started=False,
    )


def self_test(rec: Recorder, pool: SandboxPool):
    """One stdout line on whether model code can run here (only stdout comes back from the final container)."""
    t0 = time.time()
    try:
        box = TaskSandbox.create(rec, pool, "self-test", {}, BlobCache(root=TASKS / ".self-test-blobs"))
        try:
            run = box.run(rec, "import pandas, os; print(os.getuid())", timeout=60)
        finally:
            box.remove()
    except Exception as e:
        return f"sandbox FAILED: {type(e).__name__}: {str(e)[:200]}"
    if run.exit_code != 0 or not run.stdout.text.strip().isdigit():
        return f"sandbox FAILED: exit {run.exit_code}: {run.stderr.text.strip()[-200:]}"
    return f"sandbox ok {time.time() - t0:.1f}s"
