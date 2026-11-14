"""Loaded by every sandbox Python at start (``python -I`` still imports it): no network, ``submit``, less noise.

No network: a clear error for code that tries it (an injected "upload this" instruction, a URL fetch). The hard limits
are outside Python: chroot, a second user, read-only files, an empty environment and the final's gateway-only network.

``submit(...)``: available in every script without an import, so the script that computes an answer can submit it
with citations built from the rows it used. It writes one JSON line to the file descriptor the harness passes in
``SUBMIT_FD`` (not stdout, so a cut output never loses it); the harness validates the last line
(``src/mcnulty/agent/submit.py``).

Less noise: pypdf logs a warning per font it cannot map (2.3 MB for the PDFs of one run), which buries the real output.
"""

import builtins
import json
import logging
import os
import sys

NETWORK = frozenset({"socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto"})


class NoNetwork(PermissionError):
    def __init__(self):
        super().__init__("the sandbox has no network: work only with the files of the workspace")


def no_network(event: str, args: tuple[object, ...]):
    if event in NETWORK:
        raise NoNetwork


sys.addaudithook(no_network)


class NotJson(TypeError):
    def __init__(self, value: object):
        super().__init__(f"submit: {type(value).__name__} is not a JSON value")


def plain(value: object):
    """numpy and pandas values (int64, Timestamp, arrays) and tuples or sets as plain JSON values."""
    if hasattr(value, "tolist"):
        return value.tolist()  # numpy scalars and arrays
    if hasattr(value, "isoformat"):
        return value.isoformat()  # datetime, pandas Timestamp
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    raise NotJson(value)


class NoHarness(RuntimeError):
    def __init__(self):
        super().__init__("submit: no harness to receive it (SUBMIT_FD is not set)")


def submit(outcome: str, answer: object = None, sources: object = (), actions: object = ()):
    """Finish the task from code. sources: [{"path": ..., "line" | "record" | "sheet"+"row" | "paragraph" | "page"}]."""
    if "SUBMIT_FD" not in os.environ:
        raise NoHarness
    payload = {"outcome": outcome, "answer": answer, "sources": sources, "actions": actions}
    line = json.dumps(payload, default=plain, ensure_ascii=False) + "\n"
    os.write(int(os.environ["SUBMIT_FD"]), line.encode("utf-8"))
    print("submit(...) received; the harness checks it when this script ends.")


builtins.submit = submit
logging.getLogger("pypdf").setLevel(logging.ERROR)
