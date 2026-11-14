"""Read, cite and search workspace files of any format (``formats``); results are ``models`` types."""

import re
from pathlib import Path

from ws.formats import CACHE, units
from ws.models import Hit, Source

SKIPPED = frozenset({"bin", "sbin", "lib", "lib32", "lib64", "libx32", "usr", "etc", "tmp", "proc", "dev"})


class ColumnNotFound(KeyError):
    def __init__(self, column: str, path: str):
        super().__init__(f"no row of {path} names a column {column!r}; look with text(path) first")


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
    first row of each sheet that names the column."""
    found = units(path)
    if value is None or not found:
        return [Source(path=path)]
    wanted = str(value).casefold()
    sources: list[Source] = []
    headers: dict[str | None, int] = {}  # sheet (None for csv) -> index of the column
    for unit in found:
        if column is None:
            hit = wanted in unit.text.casefold()
        else:
            sheet = unit.source.sheet
            if sheet not in headers:
                names = [c.strip().casefold() for c in unit.cells]
                if column.casefold() in names:
                    headers[sheet] = names.index(column.casefold())
                continue
            index = headers[sheet]
            hit = index < len(unit.cells) and unit.cells[index].strip().casefold() == wanted
        if hit:
            sources.append(unit.source)
    if column is not None and not headers:
        raise ColumnNotFound(column, path)
    return sources


def files_under(root: Path):
    if root.is_file():
        return [root]
    top = root == Path("/")  # a search from / skips the system folders and the scratch folder
    found = (p for p in root.rglob("*") if p.is_file() and CACHE not in p.parents)  # never the parse cache
    return sorted(p for p in found if not (top and p.parts[1] in SKIPPED))


def grep(pattern: str, path: str = "/", regex: bool = False, limit: int = 100):
    """Every unit that matches ``pattern`` (any case) in every file under ``path``, any format: a list of ``Hit``
    (``hit.source`` goes into submit(...)), at most ``limit``."""
    search = re.compile(pattern if regex else re.escape(pattern), re.IGNORECASE)
    hits: list[Hit] = []
    for file in files_under(Path(path)):
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
