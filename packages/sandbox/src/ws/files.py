"""Read, cite and grep workspace files of any format (``formats``); results are ``models`` types."""

import re
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import cast, final, override

from ws.formats import JSON, read_text, units, visible, with_decoded
from ws.models import Hit, Source, Unit

# system and scratch folders; also those only outside a chroot (the harness's open mode when capabilities are dropped)
SKIPPED = frozenset(
    {
        "bin",
        "sbin",
        "lib",
        "lib32",
        "lib64",
        "libx32",
        "usr",
        "etc",
        "tmp",
        "proc",
        "dev",
        "sys",
        "app",
        "sandbox",
        "run",
        "var",
        "root",
        "home",
        "opt",
        "boot",
        "srv",
        "mnt",
        "media",
        ".dockerenv",
    }
)


def resolve(source: Source):
    """A jsonl line cited by its record id (``id=M-1``) as the line that holds that object: jsonl is graded by line."""
    if source.record is None or not source.path.endswith(".jsonl") or not Path(source.path).is_file():
        return source
    pattern = re.compile(rf'"id"\s*:\s*"?{re.escape(source.record)}"?\s*[,}}]')
    found = next((u.source for u in units(source.path) if pattern.search(u.text)), None)
    return found or source


def read(source: Source):
    """The text of the unit a citation points at; None when the file has no such unit (or the file is missing)."""
    if not source.locators:
        return visible(read_text(source.path)) if Path(source.path).is_file() else None
    try:
        found = units(source.path)
    except OSError:
        return None
    return next((u.text for u in found if u.source == source), None)


def text(path: str, limit: int | None = 200):
    """The file as numbered units with their locators, e.g. ``row=12| ...``; at most ``limit`` units (None: all)."""
    found = units(path)
    shown = found if limit is None else found[:limit]
    left = len(found) - len(shown)
    more = f"\n[... {left} more units: text(path, limit=None), or grep]" if left else ""
    return Lines("\n".join(f"{u.source.where}| {with_decoded(u.text)}" for u in shown) + more)


@final
class Lines(str):
    """``text()``'s result: it prints as the text, and a for loop goes over its lines, not its characters (eval
    20261114T142425Z, t26 printed a policy one character per line)."""

    __slots__ = ()

    @override
    def __iter__(self):
        return iter(self.splitlines())


def cite(path: str, value: object = None, column: str | None = None):
    """Sources for the units of ``path`` that contain ``value`` (as text, any case); the whole file when None; for a
    list (tuple, set, pandas Series) of values, the units of each value.
    Units where ``value`` stands as a whole word win (``standby-01`` does not cite ``standby-010``); when there are
    none, units that contain it anywhere. ``column``: in a table (csv, xlsx, ods), only rows whose cell in that column
    equals ``value``; the header is the first row of each sheet that names the column. Never raises for a missing
    column: a citation helper must not stop the submit(...) it is part of; it matches the value anywhere and says so
    on stderr."""
    values = parts(value)
    if len(values) != 1 or values[0] is not value:  # several values: the units of each, once
        return list(dict.fromkeys(source for one in values for source in cite_value(path, one, column)))
    return cite_value(path, value, column)


def cite_value(path: str, value: object, column: str | None):
    found = units(path)
    if value is None or not found:
        return [Source(path=path)]
    wanted = str(value).casefold()
    headers = column_indexes(found, column) if column is not None else {}
    if column is not None and not headers:
        print(f"cite: no column {column!r} in {path}; matched {value!r} anywhere in each unit", file=sys.stderr)
    word = re.compile(rf"(?<!\w){re.escape(wanted)}(?!\w)", re.IGNORECASE)
    inside: list[Source] = []
    whole: list[Source] = []
    for unit in found:
        index = headers.get(unit.source.sheet)
        if index is not None:
            if index < len(unit.cells) and unit.cells[index].strip().casefold() == wanted:
                whole.append(unit.source)
        elif wanted in unit.text.casefold():
            inside.append(unit.source)
            if word.search(unit.text):
                whole.append(unit.source)
    return whole or inside


def parts(value: object) -> list[object]:  # a script passes any value
    """A list, tuple, set or pandas/numpy sequence (``df["_source"]``) as its items, anything else as a list of
    one."""
    if isinstance(value, list | tuple | set | frozenset):
        return list(cast("Iterable[object]", value))
    if type(value).__module__.split(".")[0] in {"numpy", "pandas"} and isinstance(value, Iterable):
        return list(value)
    return [value]


def column_indexes(found: list[Unit], column: str):
    """Sheet (None for csv) -> index of ``column``, from the first row of each sheet that names it."""
    headers: dict[str | None, int] = {}
    for unit in found:
        names = [c.strip().casefold() for c in unit.cells]
        if unit.source.sheet not in headers and column.casefold() in names:
            headers[unit.source.sheet] = names.index(column.casefold())
    return headers


def files_under(root: Path):
    if root.is_file():
        return [root]
    if root == Path("/"):  # a search from / skips the system folders and the scratch folder, without walking them
        tops = [p for p in root.iterdir() if p.name not in SKIPPED]
        return sorted(f for p in tops for f in ([p] if p.is_file() else p.rglob("*")) if f.is_file())
    return sorted(p for p in root.rglob("*") if p.is_file())


TABLES = (".csv", ".jsonl", ".xlsx", ".xlsm", ".ods")
COLUMNS_MAX = 240  # characters per header line
EXAMPLE_MAX = 160  # characters per example row


def header(found: list[Unit]):
    """Per sheet: the header (the first unit with two or more filled cells; title rows have one, jsonl: the keys of
    every object, ``json_keys``) and the first data row after it as an example of the values."""
    headers: dict[str | None, str] = {}
    examples: dict[str | None, str] = {}
    for unit in found:
        sheet = unit.source.sheet
        if sheet in examples:
            continue
        if sheet in headers:
            examples[sheet] = f"  e.g. {unit.text}"[:EXAMPLE_MAX]
        elif len([c for c in unit.cells if c.strip()]) >= 2:
            headers[sheet] = f"{unit.source.where}| " + " | ".join(c.strip() for c in unit.cells)
        elif unit.text.lstrip().startswith("{"):
            headers[sheet] = f"{unit.source.where}| keys: {json_keys(found)}"
            examples[sheet] = f"  e.g. {unit.text}"[:EXAMPLE_MAX]
    return [line for sheet, head in headers.items() for line in (head[:COLUMNS_MAX], examples.get(sheet, "")) if line]


def json_keys(found: list[Unit]):
    """The top-level keys of the JSON objects (one per unit), in first-seen order; a key that only some objects have
    with its count, e.g. ``request_id (735 of 900)``."""
    counts: Counter[str] = Counter()
    objects = 0
    for unit in found:
        try:
            value = JSON.validate_json(unit.text)
        except ValueError:
            continue
        if isinstance(value, dict):
            objects += 1
            counts.update(value.keys())  # a dict would add its values as counts
    return ", ".join(k if n == objects else f"{k} ({n} of {objects})" for k, n in counts.items())


def columns(*paths: str):
    """The header of every table file (csv, jsonl, xlsx, ods) in the given files and folders (default: all), one line
    per file and sheet, e.g. ``/a.xlsx sheet=alerts row=3| id | token | status``."""
    lines: list[str] = []
    for file in dict.fromkeys(f for path in paths or ("/",) for f in files_under(Path(path))):
        if file.suffix.lower() not in TABLES:
            continue
        try:
            found = units(str(file))
            found = found if file.suffix.lower() == ".jsonl" else found[:50]  # jsonl: the keys of every object
        except Exception:  # noqa: S112 - an unreadable file is left out of the overview
            continue
        lines += [f"{file} {line}" for line in header(found)]
    return "\n".join(lines)


HIT_MAX = 300  # characters of a hit's text: the part around the first match


def snippet(text: str, start: int):
    """The text around a match, at most ``HIT_MAX`` characters, with "..." where it is cut."""
    if len(text) <= HIT_MAX:
        return text
    begin = max(0, min(start - HIT_MAX // 3, len(text) - HIT_MAX))
    stop = begin + HIT_MAX
    return ("..." if begin else "") + text[begin:stop] + ("..." if stop < len(text) else "")


def grep(pattern: str, *paths: str, regex: bool = False, limit: int = 100):
    """Every unit that matches ``pattern`` (any case) in the given files and folders (default: the whole workspace),
    any format: a list of ``Hit`` (``hit.source`` goes into submit(...)), at most ``limit``. Each hit's text is the
    part of the unit around the match. A missing path, an unreadable file and the limit are told on stderr."""
    search = re.compile(pattern if regex else re.escape(pattern), re.IGNORECASE)
    for path in paths:
        if not Path(path).exists():
            print(f"grep: no file or folder {path} (paths are absolute, e.g. /logs)", file=sys.stderr)
    files = dict.fromkeys(f for path in paths or ("/",) for f in files_under(Path(path)))
    hits: list[Hit] = []
    for file in files:
        try:
            found = units(str(file))
        except Exception as e:
            print(f"grep: could not read {file} ({type(e).__name__})", file=sys.stderr)
            continue
        for unit in found:
            shown = with_decoded(unit.text)  # a match in decoded base64 counts
            match = search.search(shown)
            if match is None:
                continue
            hits.append(Hit(source=unit.source, text=snippet(shown, match.start())))
            if len(hits) >= limit:
                print(f"grep: stopped at {limit} hits; pass limit= or narrower paths for more", file=sys.stderr)
                return hits
    return hits
