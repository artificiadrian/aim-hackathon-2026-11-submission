"""``submit(...)`` for model-written scripts: the script that computes an answer submits it, with citations built
from the rows it used. It writes one JSON line to the file descriptor the harness passes in ``SUBMIT_FD`` (not
stdout, so a cut output never loses it); the harness validates the last line (``src/mcnulty/agent/submit.py``)."""

import datetime
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

from pydantic import BaseModel, JsonValue, ValidationError

from ws.files import TABLES, files_under, parts, read, resolve
from ws.formats import JSON, units
from ws.models import Hit, Source, Unit


class NotJson(TypeError):
    def __init__(self, value: object):
        super().__init__(f"submit: {type(value).__name__} is not a JSON value")


def plain(value: object) -> object:  # json.dumps(default=...) takes any value back
    """The helpers' models (a ``Hit`` as its source), numpy and pandas values (int64, Timestamp, arrays, Series,
    Index; ``pd.NA`` as null), dates, and tuples or sets as plain JSON values."""
    if isinstance(value, Hit):
        value = value.source  # a Hit cited as it is
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, datetime.date):
        return value.isoformat()  # also pandas Timestamp
    if isinstance(value, set | frozenset | tuple):
        return list(value)  # pyright: ignore[reportUnknownArgumentType] - any items; json.dumps checks them
    if type(value).__module__.split(".")[0] in {"numpy", "pandas"}:
        tolist = cast("object", getattr(value, "tolist", None))  # arrays, scalars, Series, Index, extension arrays
        return cast("Callable[[], object]", tolist)() if callable(tolist) else None  # pd.NA and the like: no value
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


def answer_values(answer: JsonValue) -> list[str]:  # recursive
    """The atoms of an answer as text (strings of two or more characters, numbers), any case."""
    if isinstance(answer, dict):
        return [v for item in answer.values() for v in answer_values(item)]
    if isinstance(answer, list):
        return [v for item in answer for v in answer_values(item)]
    if isinstance(answer, bool) or answer is None:
        return []
    text = str(answer).strip().casefold()
    return [text] if len(text) >= 2 else []


HINT_VALUES = 20  # answer values looked up in the tables when no cited unit holds them
HINT_ROWS = 3  # a value in more rows of one file than this is not a lookup: no hint for that file


def cells_of(unit: Unit):
    """The cell values of a table row (csv, xlsx, ods) or the top-level values of a JSON object (json, jsonl)."""
    if unit.cells:
        return {c.strip().casefold() for c in unit.cells}
    try:
        value = JSON.validate_json(unit.text)
    except ValueError:
        return set[str]()
    if not isinstance(value, dict):
        return set[str]()
    return {str(v).strip().casefold() for v in value.values() if not isinstance(v, dict | list)}


def table_rows(values: list[str], root: Path):
    """For each value: the table rows (at most ``HINT_ROWS`` per file) that hold it as a whole cell."""
    found: dict[str, list[Source]] = {v: [] for v in values}
    for file in files_under(root):
        if file.suffix.lower() not in (*TABLES, ".json"):
            continue
        try:
            rows = units(str(file))
        except Exception:  # noqa: S112 - an unreadable file gives no hint
            continue
        per_file: dict[str, list[Source]] = {v: [] for v in values}
        for unit in rows:
            held = cells_of(unit)
            for v in values:
                if v in held:
                    per_file[v].append(resolve(unit.source))
        for v, sources in per_file.items():
            if 0 < len(sources) <= HINT_ROWS:
                found[v] += sources
    return found


WORKSPACE = Path("/")


def citation_notes(sources: list[Source], answer: JsonValue, root: Path = WORKSPACE):
    """What is wrong with the citations, checked against the files: units that do not exist, and answer values that
    no cited unit holds but a table row does (a lookup table the answer used and did not cite, or a wrong row).
    Values found in no table (counts, computed values) are not reported. Empty when all is fine."""
    notes: list[str] = []
    texts: list[str] = []
    for source in sources:
        unit = read(source)
        if unit is None:
            notes.append(f"{source} does not exist (check the path and the locator; text(path) shows them)")
        elif source.locators:
            texts.append(unit.casefold())
    values = list(dict.fromkeys(answer_values(answer)))
    missing = [v for v in values if not any(v in t for t in texts)][:HINT_VALUES]
    hints = [(v, rows) for v, rows in table_rows(missing, root).items() if rows]
    if hints:
        listed = "\n".join(f"  {v}: " + ", ".join(str(s) for s in rows) for v, rows in hints)
        head = "answer values that no cited unit holds, and the table rows that hold them:"
        notes.append(
            f"{head}\n{listed}\ncite each row your answer used (lookup tables too); leave out rows it did not use"
        )
    return notes


def submit(outcome: str, answer: object = None, sources: object = (), actions: object = ()):
    """Finish the task from code. sources: what grep(...) and cite(...) return (hits or sources, lists of them),
    or their printed form ``/a.csv#record=R-7``. The citations are checked against the files; problems go to the
    harness as notes, which shows them once before it accepts."""
    if "SUBMIT_FD" not in os.environ:
        raise NoHarness
    value = JSON.validate_json(json.dumps(answer, default=plain))  # any script value, parsed once into JSON
    cited, notes = as_sources(sources)
    notes += citation_notes(cited, value)
    payload = {"outcome": outcome, "answer": value, "sources": cited, "actions": actions, "notes": notes}
    line = json.dumps(payload, default=plain, ensure_ascii=False) + "\n"
    os.write(int(os.environ["SUBMIT_FD"]), line.encode("utf-8"))
    print("submit(...) received; the harness checks it when this script ends.")
