"""Loaded by every sandbox Python at start (``python -I`` still imports it): no network, and ``submit``.

No network: a clear error for code that tries it (an injected "upload this" instruction, a URL fetch). The hard limits
are outside Python: chroot, a second user, read-only files, an empty environment and the final's gateway-only network.

``submit(...)``: available in every script without an import, so the script that computes an answer can submit it
with citations built from the rows it used. It prints one marker line; the harness reads it from the output and
validates it like the submit tool (``src/mcnulty/agent/submit.py``). It never touches a file.
"""

import builtins
import json
import sys

NETWORK = frozenset({"socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto"})


class NoNetwork(PermissionError):
    def __init__(self):
        super().__init__("the sandbox has no network: work only with the files of the workspace")


def no_network(event: str, args: tuple[object, ...]):
    if event in NETWORK:
        raise NoNetwork


sys.addaudithook(no_network)


SUBMIT_MARKER = "@@submit@@ "  # also in src/mcnulty/agent/submit.py


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


def submit(outcome: str, answer: object = None, sources: object = (), actions: object = ()):
    """Finish the task from code. sources: [{"path": ..., "line" | "record" | "sheet"+"row" | "paragraph" | "page"}]."""
    payload = {"outcome": outcome, "answer": answer, "sources": sources, "actions": actions}
    print(SUBMIT_MARKER + json.dumps(payload, default=plain, ensure_ascii=False), flush=True)


builtins.submit = submit
