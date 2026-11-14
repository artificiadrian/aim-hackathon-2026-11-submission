"""Read, cite and grep workspace files of any format (``formats``); results are ``models`` types."""

import re
import sys
from pathlib import Path

from ws.formats import CACHE, read_text, units
from ws.models import Hit, Source, Unit

SKIPPED = frozenset({"bin", "sbin", "lib", "lib32", "lib64", "libx32", "usr", "etc", "tmp", "proc", "dev"})


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
        return read_text(source.path) if Path(source.path).is_file() else None
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
    return "\n".join(f"{u.source.where}| {u.text}" for u in shown) + more


def cite(path: str, value: object = None, column: str | None = None):
    """Sources for the units of ``path`` that contain ``value`` (as text, any case); the whole file when None.
    ``column``: in a table (csv, xlsx, ods), only rows whose cell in that column equals ``value``; the header is the
    first row of each sheet that names the column. Never raises for a missing column: a citation helper must not
    stop the submit(...) it is part of; it matches the value anywhere and says so on stderr."""
    found = units(path)
    if value is None or not found:
        return [Source(path=path)]
    wanted = str(value).casefold()
    headers = column_indexes(found, column) if column is not None else {}
    if column is not None and not headers:
        print(f"cite: no column {column!r} in {path}; matched {value!r} anywhere in each unit", file=sys.stderr)
    sources: list[Source] = []
    for unit in found:
        index = headers.get(unit.source.sheet)
        if index is None:
            hit = wanted in unit.text.casefold()
        else:
            hit = index < len(unit.cells) and unit.cells[index].strip().casefold() == wanted
        if hit:
            sources.append(unit.source)
    return sources


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
    top = root == Path("/")  # a search from / skips the system folders and the scratch folder
    found = (p for p in root.rglob("*") if p.is_file() and CACHE not in p.parents)  # never the parse cache
    return sorted(p for p in found if not (top and p.parts[1] in SKIPPED))


TABLES = (".csv", ".jsonl", ".xlsx", ".xlsm", ".ods")
COLUMNS_MAX = 240  # characters per header line
EXAMPLE_MAX = 160  # characters per example row


def header(found: list[Unit]):
    """Per sheet: the header (the first unit with two or more filled cells; title rows have one, jsonl: the keys of
    the first object) and the first data row after it as an example of the values."""
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
            keys = re.findall(r'"([^"]+)"\s*:', unit.text)
            headers[sheet] = f"{unit.source.where}| keys: " + ", ".join(dict.fromkeys(keys))
            examples[sheet] = f"  e.g. {unit.text}"[:EXAMPLE_MAX]
    return [line for sheet, head in headers.items() for line in (head[:COLUMNS_MAX], examples.get(sheet, "")) if line]


def columns(*paths: str):
    """The header of every table file (csv, jsonl, xlsx, ods) in the given files and folders (default: all), one line
    per file and sheet, e.g. ``/a.xlsx sheet=alerts row=3| id | token | status``."""
    lines: list[str] = []
    for file in dict.fromkeys(f for path in paths or ("/",) for f in files_under(Path(path))):
        if file.suffix.lower() not in TABLES:
            continue
        try:
            found = units(str(file))[:50]
        except Exception:  # noqa: S112 - an unreadable file is left out of the overview
            continue
        lines += [f"{file} {line}" for line in header(found)]
    return "\n".join(lines)


def grep(pattern: str, *paths: str, regex: bool = False, limit: int = 100):
    """Every unit that matches ``pattern`` (any case) in the given files and folders (default: the whole workspace),
    any format: a list of ``Hit`` (``hit.source`` goes into submit(...)), at most ``limit``."""
    search = re.compile(pattern if regex else re.escape(pattern), re.IGNORECASE)
    files = dict.fromkeys(f for path in paths or ("/",) for f in files_under(Path(path)))
    hits: list[Hit] = []
    for file in files:
        try:
            found = units(str(file))
        except Exception:  # noqa: S112 - an unreadable file is skipped in a search
            continue
        for unit in found:
            if search.search(unit.text):
                hits.append(Hit(source=unit.source, text=unit.text[:300]))
                if len(hits) >= limit:
                    return hits
    return hits
