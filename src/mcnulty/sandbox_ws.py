"""``ws``: read the task workspace from a sandboxed script. Standard library only; copied as ``ws.py`` by sandbox.py.

Every unit (text line, JSON line, table row, document paragraph, PDF page) carries ``cite``: the citation of that
unit, a dict such as ``{"path": "/a.md", "line": 3}``, ready to put in the answer's ``sources``.
"""

import fnmatch
import json
import re
from functools import cache
from pathlib import Path

DATA = Path("workspace")


@cache
def _index():
    return {entry["path"]: entry for entry in json.loads((DATA / "index.json").read_text(encoding="utf-8"))}


@cache
def _load(path: str):
    entry = _index().get(path)
    if entry is None:
        raise FileNotFoundError(path)  # list paths with ws.files()
    return json.loads((DATA / entry["file"]).read_text(encoding="utf-8"))


def files(pattern: str = "*"):
    """Paths matching a glob pattern (``*`` also crosses ``/``), with kind and unit count: [(path, kind, units)]."""
    return [(p, e["kind"], e["units"]) for p, e in _index().items() if fnmatch.fnmatch(p, pattern)]


def kind(path: str):
    """text, json, jsonl, table, document, pdf, image, binary or unreadable."""
    return _load(path)["kind"]


def info(path: str):
    """Extra facts: json -> {"value"}, table -> {"tables"}, document -> {"comments"}, image -> size and hidden text."""
    return _load(path)["info"]


def units(path: str):
    """All citable units of a file: [{"cite", "text", ...}]; jsonl units add "value", table rows add "cells",
    "hidden" and "table" (sheet name)."""
    return _load(path)["units"]


def text(path: str):
    """The whole file as text (units joined by newlines)."""
    return "\n".join(u["text"] for u in units(path))


def locator(cite: dict):
    return ",".join(f"{k}={v}" for k, v in cite.items() if k != "path")


def show(path: str, start: int = 1, count: int = 80):
    """Print units start..start+count-1 (1-based) with their locators."""
    selected = units(path)[start - 1 : start - 1 + count]
    for u in selected:
        hidden = " [hidden]" if u.get("hidden") else ""
        print(f"[{locator(u['cite'])}]{hidden} {u['text']}")
    print(f"-- {path}: shown {len(selected)} of {len(units(path))} units")


def grep(pattern: str, glob: str = "*", ignore_case: bool = True, limit: int = 50):
    """Print and return units whose text matches a regex, across files matching ``glob``: [(cite, text)]."""
    regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    found = []
    for path, _, _ in files(glob):
        for u in units(path):
            if regex.search(u["text"]):
                found.append((u["cite"], u["text"]))
    for cite, line in found[:limit]:
        print(f"{cite['path']} [{locator(cite)}] {line[:300]}")
    if len(found) > limit:
        print(f"-- {len(found) - limit} more matches not printed")
    return found


def table(path: str, sheet: str | None = None, header: int | None = None):
    """Rows of a csv/xlsx/ods table (or one sheet) as dicts keyed by the header cells, plus "_cite" and "_hidden".
    ``header`` is the 1-based index of the header row in that table; by default the first row with the most filled
    cells among the first 10. Rows above the header are left out; rows after it are kept as they are (also totals)."""
    rows = [u for u in units(path) if sheet is None or u.get("table") == sheet]
    if not rows:
        return []
    if header is None:
        head = rows[:10]
        filled = [sum(1 for c in r["cells"] if c.strip()) for r in head]
        header = filled.index(max(filled)) + 1
    names = [c.strip() or f"col{i + 1}" for i, c in enumerate(rows[header - 1]["cells"])]
    out = []
    for r in rows[header:]:
        cells = r["cells"] + [""] * (len(names) - len(r["cells"]))
        out.append({**dict(zip(names, cells, strict=False)), "_cite": r["cite"], "_hidden": r.get("hidden", False)})
    return out


def records(path: str):
    """JSON Lines file as dicts: each line's object plus "_cite"; lines that are not JSON objects are skipped."""
    return [{**u["value"], "_cite": u["cite"]} for u in units(path) if isinstance(u.get("value"), dict)]


def value(path: str):
    """The parsed value of a .json file."""
    return info(path)["value"]
