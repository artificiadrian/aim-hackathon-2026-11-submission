"""``submit(...)`` for model-written scripts: the script that computes an answer submits it, with citations built
from the rows it used. It writes one JSON line to the file descriptor the harness passes in ``SUBMIT_FD`` (not
stdout, so a cut output never loses it); the harness validates the last line (``src/mcnulty/agent/submit.py``)."""

import datetime
import json
import os

from pydantic import BaseModel

from ws.models import Hit


class NotJson(TypeError):
    def __init__(self, value: object):
        super().__init__(f"submit: {type(value).__name__} is not a JSON value")


def plain(value: object) -> object:  # json.dumps(default=...) takes any value back
    """The helpers' models (a ``Hit`` as its source), numpy and pandas values (int64, Timestamp, arrays), dates, and
    tuples or sets as plain JSON values."""
    if isinstance(value, Hit):
        value = value.source  # a Hit cited as it is
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, datetime.date):
        return value.isoformat()  # also pandas Timestamp
    if isinstance(value, set | frozenset | tuple):
        return list(value)  # pyright: ignore[reportUnknownArgumentType] - any items; json.dumps checks them
    if type(value).__module__ == "numpy":
        import numpy as np

        return np.asarray(value).tolist()  # pyright: ignore[reportAny] - numpy's own nested lists
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
