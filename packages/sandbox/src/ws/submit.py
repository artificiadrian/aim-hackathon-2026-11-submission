"""``submit(...)`` for model-written scripts: the script that computes an answer submits it, with citations built
from the rows it used. It writes one JSON line to the file descriptor the harness passes in ``SUBMIT_FD`` (not
stdout, so a cut output never loses it); the harness validates the last line (``src/mcnulty/agent/submit.py``)."""

import datetime
import json
import os
from collections.abc import Iterable
from typing import cast

from pydantic import BaseModel, ValidationError

from ws.files import read, resolve
from ws.models import Hit, Source


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


def as_sources(value: object):
    """Every way a script passes citations as ``Source``: hits, sources, dicts, text (``Source.parse``), lists of
    them nested once (``[cite(...), cite(...)]``). What cannot be read comes back as a note."""
    sources: list[Source] = []
    notes: list[str] = []
    for item in parts(value):
        for part in parts(item):
            source = as_source(part)
            if source is None:
                notes.append(f"not a citation: {part!r}")
            else:
                sources.append(resolve(source))
    return sources, notes


def parts(value: object) -> list[object]:  # a script passes any value
    """A list or tuple as its items, anything else as a list of one."""
    return list(cast("Iterable[object]", value)) if isinstance(value, list | tuple) else [value]


def as_source(value: object):
    if isinstance(value, Hit):
        return value.source
    if isinstance(value, Source):
        return value
    if isinstance(value, str):
        return Source.parse(value)
    try:
        return Source.model_validate(value)
    except ValidationError:
        return None


def answer_values(answer: object) -> list[str]:  # recursive
    """The atoms of an answer as text (strings of two or more characters, numbers), any case."""
    if isinstance(answer, dict):
        return [v for item in parts(list(cast("dict[object, object]", answer).values())) for v in answer_values(item)]
    if isinstance(answer, list | tuple):
        items: list[object] = parts(cast("object", answer))
        return [v for item in items for v in answer_values(item)]
    if isinstance(answer, bool) or answer is None:
        return []
    text = str(answer).strip().casefold()
    return [text] if len(text) >= 2 else []


def citation_notes(sources: list[Source], answer: object):
    """What is wrong with the citations, checked against the files: units that do not exist, and none of the cited
    units holding any value of the answer. Empty when all is fine."""
    notes: list[str] = []
    texts: list[str] = []
    for source in sources:
        unit = read(source)
        if unit is None:
            notes.append(f"{source} does not exist (check the path and the locator; text(path) shows them)")
        elif source.locators:
            texts.append(unit.casefold())
    values = answer_values(answer)
    if values and texts and not any(v in t for v in values for t in texts):
        notes.append("none of the cited units holds any value of the answer: cite the units the answer comes from")
    return notes


def submit(outcome: str, answer: object = None, sources: object = (), actions: object = ()):
    """Finish the task from code. sources: what grep(...) and cite(...) return (hits or sources, lists of them),
    or their printed form ``/a.csv#record=R-7``. The citations are checked against the files; problems go to the
    harness as notes, which shows them once before it accepts."""
    if "SUBMIT_FD" not in os.environ:
        raise NoHarness
    cited, notes = as_sources(sources)
    notes += citation_notes(cited, answer)
    payload = {"outcome": outcome, "answer": answer, "sources": cited, "actions": actions, "notes": notes}
    line = json.dumps(payload, default=plain, ensure_ascii=False) + "\n"
    os.write(int(os.environ["SUBMIT_FD"]), line.encode("utf-8"))
    print("submit(...) received; the harness checks it when this script ends.")
