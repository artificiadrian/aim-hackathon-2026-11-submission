"""``submit(...)`` for model-written scripts: the script that computes an answer submits it, with citations built
from the rows it used. It writes one JSON line to the file descriptor the harness passes in ``SUBMIT_FD`` (not
stdout, so a cut output never loses it); the harness validates the last line (``src/mcnulty/agent/submit.py``)."""

import datetime
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

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


LOCATED_MAX = 3  # units of one bare-path file put in its place, per answer value


def located(source: Source, values: list[str]):
    """A bare path to a table or text file, as the units of that file that hold an answer value: a citation needs
    the unit (docs/ARENA_DOCS.md), and models cite the file they read (7 of 8 forced last steps were sent back for
    bare paths, docs/research/2026-11-14-prompt-review.md). The path stays when no unit holds a value, and for json
    (a path is a valid citation there) and other files (images, documents)."""
    suffix = Path(source.path).suffix.lower()
    if source.locators or suffix == ".json" or suffix not in (*TABLES, ".md", ".txt", ".log") or not values:
        return [source]
    try:
        found = units(source.path)
    except OSError:
        return [source]
    per_value = [[u.source for u in found if value in cells_of(u) or value in u.text.casefold()] for value in values]
    picked = [resolve(unit) for units_ in per_value if 0 < len(units_) <= LOCATED_MAX for unit in units_]
    return list(dict.fromkeys(picked)) or [source]


class Checked(BaseModel):
    """The citations after the certain fixes, and what is left for the model to fix."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    sources: tuple[Source, ...]
    added: tuple[Source, ...] = ()  # the one table row that holds an answer value no cited unit held
    dropped: tuple[Source, ...] = ()  # cited units that do not exist, left out while other citations remain
    notes: tuple[str, ...] = ()  # what only the model can fix: the harness sends the submission back once


def check_citations(sources: list[Source], answer: JsonValue, root: Path = WORKSPACE):
    """The citations checked against the files, fixed in code where the fix is certain (no extra model call): a cited
    unit that does not exist is left out while other citations remain, and an answer value that no cited unit holds
    is cited by its table row when exactly one row in the workspace holds it. The rest becomes notes: a missing unit
    that is the only citation, and values that several rows hold. Values in no table (counts, computed values) are
    not reported. Over-citing costs points (docs/ARENA_DOCS.md), so nothing is added on a guess."""
    values = list(dict.fromkeys(answer_values(answer)))
    texts: list[str] = []
    kept: list[Source] = []
    gone: list[Source] = []
    for source in [unit for s in sources for unit in located(s, values)]:
        unit = read(source)
        if unit is None:
            gone.append(source)
            continue
        kept.append(source)
        if source.locators:
            texts.append(unit.casefold())
    notes = [f"{s} does not exist (check the path and the locator; text(path) shows them)" for s in gone if not kept]
    missing = [v for v in values if not any(v in t for t in texts)][:HINT_VALUES]
    found = table_rows(missing, root)
    added = list(dict.fromkeys(rows[0] for rows in found.values() if len(rows) == 1 and rows[0] not in kept))
    hints = [(v, rows) for v, rows in found.items() if len(rows) > 1]
    if hints:
        listed = "\n".join(f"  {v}: " + ", ".join(str(s) for s in rows) for v, rows in hints)
        head = "answer values that no cited unit holds, and the table rows that hold them:"
        notes.append(
            f"{head}\n{listed}\ncite each row your answer used (lookup tables too); leave out rows it did not use"
        )
    return Checked(
        sources=(*kept, *added) if kept or added else tuple(gone),
        added=tuple(added),
        dropped=tuple(gone) if kept else (),
        notes=tuple(notes),
    )


def submit(outcome: str, answer: object = None, sources: object = (), actions: object = ()):
    """Finish the task from code. sources: what grep(...) and cite(...) return (hits or sources, lists of them),
    or their printed form ``/a.csv#record=R-7``. The citations are checked against the files; problems go to the
    harness as notes, which shows them once before it accepts."""
    if "SUBMIT_FD" not in os.environ:
        raise NoHarness
    value = JSON.validate_json(json.dumps(answer, default=plain))  # any script value, parsed once into JSON
    cited, notes = as_sources(sources)
    checked = check_citations(cited, value)
    for source in checked.added:
        print(f"submit: cited {source} too: it holds an answer value no cited unit held")
    for source in checked.dropped:
        print(f"submit: left out {source}: no such unit")
    payload = {
        "outcome": outcome,
        "answer": value,
        "sources": list(checked.sources),
        "actions": actions,
        "notes": [*notes, *checked.notes],
    }
    line = json.dumps(payload, default=plain, ensure_ascii=False) + "\n"
    os.write(int(os.environ["SUBMIT_FD"]), line.encode("utf-8"))
    print("submit(...) received; the harness checks it when this script ends.")
