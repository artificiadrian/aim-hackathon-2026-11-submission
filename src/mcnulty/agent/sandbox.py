"""Run model-written Python on the task's real files, isolated. Works only inside the Docker image (Linux, root).

Layout in the container:

- ``/sandbox/base``: a copy of the system folders of the container's root (``COPIED``: Python 3.12 with pandas,
  openpyxl, ... from the Dockerfile), without uv. An allow list: a folder the organisers mount (``/data``, ``/app``)
  never enters the sandbox. Made once per container in the writable layer: Docker's overlay filesystem copies a file
  up on every hard link, so linking from the image would write 320 MB per task.
- ``/sandbox/tasks/<task>``: one root per task, hard links to ``base`` (0.1 s, 4 MB) plus the task's files at their
  workspace paths, hard-linked from the blob cache when it is on the same filesystem (else copied).

One Python session per task, the kernel (``ws.kernel``): ``chroot`` into the task root as the task's own uid
(``SandboxPool``), an environment without keys and resource limits. Each run is a cell of that session: variables
persist, earlier cells never run again. A cell has a wall-clock timeout and ``CPU_PER_CELL`` CPU seconds; a kernel
that dies (timeout, CPU, memory, crash) or stopped after ``IDLE_MAX`` seconds without a cell is started again at the
next cell, and that run says why (``SandboxRan.restarted``). One uid per task: a task cannot signal, trace or starve
another, and every other process of the uid is killed after each cell. At most ``SCRIPTS_AT_ONCE`` cells run at the
same time (2 CPUs); a cell that waits too long does not start and says so. Workspace files and the system are
root-owned and read-only; ``/tmp`` is the writable scratch folder, kept for the task and emptied above ``TMP_MAX``.
``ws.kernel`` blocks the network in Python.

Channel: a Unix socket pair (``ws.channel``). Per cell the harness sends the code with three anonymous files of its
own (stdout, stderr, ``submit(...)`` calls), never pipes, as file descriptors; the kernel answers with the exit code.
The file size limit stops a print flood at ``FSIZE``, the harness reads only head and tail, and a child that leaves
the process group holds no pipe the harness waits on. The harness never reads a file the task uid can change.
"""

import os
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from pydantic import ValidationError

from mcnulty.blobs import BlobCache
from mcnulty.frozen import Frozen
from mcnulty.record.events import FileSkipped, SandboxRan, Stream
from mcnulty.record.log import Recorder

KERNEL = Path("/usr/local/lib/python3.12/site-packages/ws/kernel.py")  # packages/sandbox, only in the Docker image
BASE = Path("/sandbox/base")
COPIED = ("bin", "sbin", "lib", "lib32", "lib64", "libx32", "usr", "etc")  # those the root has; links stay links
NOT_IN_SANDBOX = ("usr/local/bin/uv",)
TASKS = Path("/sandbox/tasks")
FIRST_UID = 10001
UIDS = 64  # tasks with a sandbox at the same time
SCRIPTS_AT_ONCE = 3  # the final container has 2 CPUs; pandas imports alone take about 1 CPU second
SLOT_WAIT = 90.0  # seconds a script waits for a slot before it gives up
TMP_MAX = 256 * 1024 * 1024  # bytes in a task's /tmp before it is emptied (4 GB disk, 12 tasks at once)
TIMEOUT = 30.0  # seconds, wall clock per cell
CPU_PER_CELL = 30  # CPU seconds per cell (the kernel lowers its soft limit before each cell)
CPU_PER_KERNEL = 900  # CPU seconds in the life of one kernel
IDLE_MAX = 120.0  # seconds without a cell before a kernel exits and frees its memory
IDLE_EXIT = 75  # the kernel's exit code when it stops for idleness (ws.kernel)
FSIZE = 16 * 1024 * 1024  # bytes per written file, also per output stream
LIMITS = (
    "--as=1500000000",  # bytes of address space
    f"--cpu={CPU_PER_CELL}:{CPU_PER_KERNEL}",
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


HEADER = 4  # bytes of the big-endian length before each message on the channel (``ws.channel``, same format)
ANSWER_MAX = 4096  # bytes of a kernel answer; a longer one is a broken channel


class Cell(Frozen):
    """One cell for the kernel; ``ws.models.Cell`` on the other side (the harness's venv has no ``ws``)."""

    cell: int
    code: str
    cpu: int


class CellDone(Frozen):
    """The kernel's answer after a cell; ``ws.models.CellDone`` on the other side."""

    cell: int
    exit_code: int


class ChannelClosed(EOFError):
    def __init__(self):
        super().__init__("the kernel closed the channel or sent a broken answer")


def send_cell(channel: socket.socket, cell: Cell, fds: list[int]):
    data = cell.model_dump_json().encode("utf-8")
    framed = len(data).to_bytes(HEADER, "big") + data
    sent = socket.send_fds(channel, [framed], fds)
    channel.sendall(framed[sent:])


def read_exact(channel: socket.socket, size: int):
    data = b""
    while len(data) < size:
        chunk = channel.recv(size - len(data))
        if not chunk:
            raise ChannelClosed
        data += chunk
    return data


def receive_answer(channel: socket.socket):
    size = int.from_bytes(read_exact(channel, HEADER), "big")
    if size > ANSWER_MAX:
        raise ChannelClosed
    return CellDone.model_validate_json(read_exact(channel, size))  # the kernel is untrusted: parse its answer


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
        if not KERNEL.exists():
            raise SandboxUnavailable
        partial = BASE.with_name("base.partial")
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        top = [f"/{name}" for name in COPIED if Path(f"/{name}").exists() or Path(f"/{name}").is_symlink()]
        # no ownership: the final container has no CHOWN capability (demo check 16:10); root owns the copy
        subprocess.run(["/bin/cp", "-a", "--no-preserve=ownership", *top, str(partial)], check=True)  # noqa: S603
        for name in NOT_IN_SANDBOX:
            (partial / name).unlink(missing_ok=True)
        (partial / "tmp").mkdir()
        (partial / "tmp").chmod(0o1777)
        (partial / "dev").mkdir()
        os.mknod(partial / "dev" / "null", stat.S_IFCHR | 0o666, os.makedev(1, 3))  # the only device in the sandbox
        (partial / "dev" / "null").chmod(0o666)  # mknod applies the umask
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
    """The JSON of the last ``submit(...)`` call: one line per call (``packages/sandbox``: ``ws.submit``)."""
    file.seek(0)
    calls = file.read().decode("utf-8", errors="replace").splitlines()
    return calls[-1] if calls else None


def kill_uid(uid: int, keep: int | None = None):
    """SIGKILL every process whose real uid is ``uid`` (children that left the process group too), but ``keep``."""
    for status in Path("/proc").glob("[0-9]*/status"):
        with suppress(OSError):
            uid_line = next(line for line in status.read_text().splitlines() if line.startswith("Uid:"))
            pid = int(status.parent.name)
            if int(uid_line.split()[1]) == uid and pid != keep:
                os.kill(pid, signal.SIGKILL)


def end_reason(exit_code: int):
    """Why a kernel ended, in words for the model."""
    if exit_code == IDLE_EXIT:
        return "idle"
    if exit_code < 0:
        return f"killed by {signal.Signals(-exit_code).name}"
    return f"exited with code {exit_code}"


@dataclass(slots=True)
class Session:
    """The task's kernel and the harness's end of its channel. Not frozen: it counts cells and restarts the kernel."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    process: subprocess.Popen[bytes] | None = None
    channel: socket.socket | None = None
    cells: int = 0
    lost: str | None = None  # why the kernel ended during the last cell; told at the next start


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
    idle: float = IDLE_MAX
    session: Session = field(default_factory=Session)

    @classmethod
    def create(
        cls,
        rec: Recorder,
        pool: SandboxPool,
        task_id: str,
        files: dict[str, str],
        blobs: BlobCache,
        idle: float = IDLE_MAX,
    ):
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
        return cls(root=root, uid=pool.take_uid(), pool=pool, idle=idle)

    def run(self, rec: Recorder, code: str, timeout: float = TIMEOUT, wait: float = SLOT_WAIT):
        """Run ``code`` as the next cell of the task's kernel, when a script slot is free; log and return the result."""
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
        session = self.session
        with session.lock:
            session.cells += 1
            cell = Cell(cell=session.cells, code=code, cpu=CPU_PER_CELL)
            t0 = time.time()
            with (
                tempfile.TemporaryFile() as out,
                tempfile.TemporaryFile() as err,
                tempfile.TemporaryFile() as calls,
            ):
                restarted: str | None = None
                exit_code: int | None = None
                timed_out = False
                for _ in range(2):  # a kernel that stopped for idleness just as the cell came: start it, send again
                    restarted = self.start_kernel(out, err) or restarted
                    exit_code, timed_out = self.exchange(cell, (out, err, calls), t0 + timeout)
                    if session.lost != "idle":
                        break
                alive = session.process.pid if session.process is not None else None
                kill_uid(self.uid, keep=alive)  # children the cell left behind; the kernel stays
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
                    cell=cell.cell,
                    restarted=restarted,
                )

    def start_kernel(self, out: BinaryIO, err: BinaryIO):
        """Start the kernel when none runs, with stdout and stderr on this cell's files (so start-up errors show
        there); why the last one ended, or None when it runs or this is the first."""
        session = self.session
        if session.process is not None and session.process.poll() is None:
            return None
        reason = session.lost
        if session.process is not None:  # it ended between cells (idle)
            reason = end_reason(session.process.returncode)
            self.stop_kernel()
        command = [
            "/usr/bin/choom",
            "-n",
            "1000",  # out of memory: the kernel kills the task's session before the harness
            "--",
            "/usr/bin/prlimit",
            *LIMITS,
            "--",
            "/usr/sbin/chroot",
            f"--userspec={self.uid}:{self.uid}",
            str(self.root),
            "/usr/local/bin/python3",
            "-I",
            "-m",
            "ws.kernel",  # blocks the network, adds the helpers, then runs the cells that come over the socket
        ]
        mine, theirs = socket.socketpair()
        with theirs:
            session.process = subprocess.Popen(  # noqa: S603 - running untrusted code in the isolated child is the purpose
                [*command, str(theirs.fileno()), str(self.idle)],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                pass_fds=(theirs.fileno(),),
                env=ENV,
                start_new_session=True,  # its own process group, so one kill ends its children too
            )
        session.channel = mine
        session.lost = None
        return reason

    def exchange(self, cell: Cell, files: tuple[BinaryIO, BinaryIO, BinaryIO], deadline: float):
        """Send the cell with its files and wait for the answer until ``deadline``: (exit code, timed out). A kernel
        that does not answer is stopped and ``session.lost`` says why."""
        session = self.session
        if session.channel is None:
            raise ChannelClosed
        try:
            session.channel.settimeout(max(0.01, deadline - time.time()))
            send_cell(session.channel, cell, [f.fileno() for f in files])
            done = receive_answer(session.channel)
            if done.cell == cell.cell:
                return done.exit_code, False
            reason = f"answered for cell {done.cell}, not {cell.cell}"
        except TimeoutError:
            self.stop_kernel()
            session.lost = "the last cell timed out"
            return None, True
        except (OSError, ChannelClosed, ValidationError) as e:
            reason = type(e).__name__
        exit_code = self.stop_kernel()
        session.lost = end_reason(exit_code) if exit_code is not None else reason
        return exit_code, False

    def stop_kernel(self):
        """Close the channel and kill every process of the uid; the kernel's exit code."""
        session = self.session
        if session.channel is not None:
            session.channel.close()
            session.channel = None
        kill_uid(self.uid)
        process, session.process = session.process, None
        return process.wait() if process is not None else None

    def remove(self):
        self.stop_kernel()
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
