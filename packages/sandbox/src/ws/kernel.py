"""A task's Python session: ``python3 -I -m ws.kernel <socket fd> <idle seconds>`` (``src/mcnulty/agent/sandbox.py``).

Each ``run_python`` call is one cell: the harness sends its code with three files (stdout, stderr, submit calls)
over the socket; the kernel puts them on fd 1, fd 2 and ``SUBMIT_FD``, runs the code in one namespace that lives as
long as the kernel, and answers with the exit code. Variables persist between cells; earlier cells never run again.
The kernel exits when no cell comes within the idle time (``IDLE_EXIT``) or when the socket closes.

Explicit instead of a start-up hook: it blocks the network, quiets pypdf, and holds the helpers in the namespace, so
no cell needs an import for them.
"""

import contextlib
import linecache
import logging
import os
import resource
import socket
import sys
import traceback

from ws.channel import ChannelClosed, receive, send
from ws.files import cite, columns, grep, read, text
from ws.hints import code_tip, error_hint
from ws.models import Cell, CellDone
from ws.network import block_network
from ws.shell import sh
from ws.submit import submit
from ws.tables import lenient_pandas, route_pandas, table


class NotPython(RuntimeError):
    def __init__(self, name: str):
        super().__init__(f"{name} is a tool, not a Python function: call the {name} tool itself, outside run_python")


def view_image(*_args: object, **_keywords: object):
    """The model's view_image tool called inside a cell (a NameError twice in eval 20261114T135828Z): say what to
    do instead."""
    raise NotPython("view_image")


HELPERS = {helper.__name__: helper for helper in (submit, text, table, cite, read, grep, columns, sh, view_image)}
IDLE_EXIT = 75  # no cell came within the idle time: a cell the harness sent at that moment never ran


def wide_pandas():
    """pandas prints 80 characters wide and elides columns ("id ... outcome"): cells then print the same table again
    and again. Called at the first cell that uses pandas: importing it costs about a second."""
    import importlib.util

    if importlib.util.find_spec("pandas") is None:
        return
    import pandas as pd

    for option, value in (("display.width", 250), ("display.max_columns", 50), ("display.max_colwidth", 80)):
        pd.set_option(option, value)
    route_pandas()
    lenient_pandas()


def uses_pandas(code: str):
    return "pandas" in code or "pd." in code or "table(" in code


def flush():
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError, ValueError):  # the cell closed it, or the file is full
            stream.flush()


def point_output(out: int, err: int, calls: int, submit_fd: int):
    """fd 1, fd 2 and the submit fd onto the given files; fresh ``sys.stdout``/``sys.stderr`` on them, since a cell
    may have closed or replaced Python's objects. Child processes (``sh``) write to the same files."""
    flush()
    os.dup2(out, 1)
    os.dup2(err, 2)
    os.dup2(calls, submit_fd)
    sys.stdout = open(1, "w", encoding="utf-8", errors="backslashreplace", closefd=False)  # noqa: SIM115
    sys.stderr = open(2, "w", encoding="utf-8", errors="backslashreplace", closefd=False)  # noqa: SIM115


def limit_cpu(seconds: int):
    """``RLIMIT_CPU`` counts the whole life of the process: allow ``seconds`` more than used so far (up to the hard
    limit). A cell over it gets ``SIGXCPU`` and the kernel ends."""
    used = resource.getrusage(resource.RUSAGE_SELF)
    soft = int(used.ru_utime + used.ru_stime) + 1 + seconds
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (soft if hard == resource.RLIM_INFINITY else min(soft, hard), hard))


def exit_code(stop: SystemExit):
    if stop.code is None:
        return 0
    if isinstance(stop.code, int):
        return stop.code
    print(stop.code, file=sys.stderr)
    return 1


def run_cell(cell: Cell, namespace: dict[str, object]):
    """Run one cell; tracebacks show its lines (``<cell n>``) and leave out the kernel's own frame."""
    name = f"<cell {cell.cell}>"
    linecache.cache[name] = (len(cell.code), None, cell.code.splitlines(keepends=True), name)
    tip = code_tip(cell.code)
    if tip is not None:
        print(tip, file=sys.stderr)
    try:
        exec(compile(cell.code, name, "exec"), namespace)  # noqa: S102 - running the model's code is the purpose
    except SystemExit as stop:
        return exit_code(stop)
    except BaseException as e:
        traceback.print_exception(type(e), e, e.__traceback__.tb_next if e.__traceback__ else None)
        hint = error_hint(e, namespace) if isinstance(e, Exception) else None
        if hint is not None:
            print(hint, file=sys.stderr)
        return 1
    return 0


def nowhere():
    """A write-only fd whose writes go nowhere: /dev/null, or a pipe nobody reads when the sandbox has no /dev/null
    (a container without the MKNOD capability); writes to the pipe fail and are dropped."""
    try:
        return os.open(os.devnull, os.O_WRONLY)
    except OSError:
        read_end, write_end = os.pipe()
        os.close(read_end)
        return write_end


def main():
    sock = socket.socket(fileno=int(sys.argv[1]))
    sock.set_inheritable(False)  # sh(...) children never get the channel
    sock.settimeout(float(sys.argv[2]))
    block_network()
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # a warning per unmapped font buries the real output
    pid = os.getpid()
    null = nowhere()
    submit_fd = os.dup(null)  # a fixed number for the submit file of each cell
    os.environ["SUBMIT_FD"] = str(submit_fd)
    namespace: dict[str, object] = {"__name__": "__main__", **HELPERS}
    pandas_ready = False
    while True:
        try:
            data, fds = receive(sock, fds=3)
        except TimeoutError:
            sys.exit(IDLE_EXIT)
        except ChannelClosed:
            return
        cell = Cell.model_validate_json(data)
        if len(fds) != 3:
            return
        out, err, calls = fds
        point_output(out, err, calls, submit_fd)
        for fd in fds:
            os.close(fd)
        os.chdir("/")
        if not pandas_ready and uses_pandas(cell.code):
            wide_pandas()
            pandas_ready = True
        limit_cpu(cell.cpu)
        code = run_cell(cell, namespace)
        flush()
        if os.getpid() != pid:  # a child the cell forked returned here: it must not act as the kernel
            os._exit(code)
        point_output(null, null, null, submit_fd)  # stray writes after the answer go nowhere
        send(sock, CellDone(cell=cell.cell, exit_code=code).model_dump_json().encode("utf-8"))


if __name__ == "__main__":
    main()
