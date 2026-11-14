"""``table(path)``: any table file as one pandas DataFrame with a citation per row."""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from ws.formats import JSON, TABLE_ROW, units
from ws.models import Source

SOURCE_COLUMN = "_source"
TABLE_FORMATS = "csv, jsonl, json, xlsx, ods and markdown tables"


def table(path: str, sheet: str | None = None):
    """A table file (csv, jsonl, json, xlsx, ods) as one pandas DataFrame, one row per record, with the citation of
    each row in the column ``_source`` (``sources=list(df.loc[mask, "_source"])``). The header is the first row with
    two or more filled cells of each sheet (title rows above it are left out); json and jsonl: the keys of the objects.
    Columns whose every filled value is a number are numbers; all other cells are text ("" when empty). Spreadsheets:
    every sheet (a ``_sheet`` column) unless ``sheet`` names one."""
    import pandas as pd

    rows: list[dict[str, object]] = []
    places: list[Source] = []  # each row's citation, typed
    headers: dict[str | None, list[str]] = {}
    for unit in units(path):
        where = unit.source
        if sheet is not None and where.sheet != sheet:
            continue
        if not unit.cells:  # json, jsonl: one object per unit
            if where.line is not None and not TABLE_ROW.match(unit.text):
                headers.pop(where.sheet, None)  # a line outside a markdown table ends that table
            try:
                value = JSON.validate_json(unit.text)
            except ValueError:
                continue
            if isinstance(value, dict):
                rows.append({**value, SOURCE_COLUMN: where})
                places.append(where)
            continue
        if where.sheet not in headers:
            if len([c for c in unit.cells if c.strip()]) >= 2:
                headers[where.sheet] = [c.strip() or f"column_{i + 1}" for i, c in enumerate(unit.cells)]
            continue
        names = headers[where.sheet]
        row: dict[str, object] = dict(zip(names, unit.cells, strict=False))
        if where.sheet is not None and sheet is None:
            row["_sheet"] = where.sheet
        rows.append(row | {SOURCE_COLUMN: where})
        places.append(where)
    names = list(dict.fromkeys(name for row in rows for name in row))
    kept = (SOURCE_COLUMN, "_sheet")
    columns = {n: [r.get(n) for r in rows] if n in kept else typed_column([r.get(n) for r in rows]) for n in names}
    if not rows:
        print(f"table: no rows in {path}: no header row with two or more cells ({TABLE_FORMATS})", file=sys.stderr)
    frame = pd.DataFrame(columns)
    numbered = file_index(places)
    if numbered is not None:
        frame.index = pd.Index(numbered[1], name=numbered[0])
    return frame


def file_index(places: list[Source]):
    """The file's own numbers as the frame's index: ``line`` (text, jsonl, markdown, csv cited by line), or ``row`` of
    a one-sheet spreadsheet. A printed index is then the locator (eval 20261114T144031Z, t25 cited index 699 for line
    700). None when the rows have no single kind of number (csv cited by record, several sheets)."""
    if places and all(p.line is not None for p in places):
        return "line", [p.line for p in places]
    if places and all(p.row is not None for p in places) and len({p.sheet for p in places}) == 1:
        return "row", [p.row for p in places]
    return None


def number(value: object):
    """A cell as an int or float; None when it is not a number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    try:
        found = float(str(value).strip().replace(",", ""))
    except ValueError:
        return None
    return int(found) if found.is_integer() and "." not in str(value) else found


def typed_column(values: list[object]):
    """Numbers when every filled cell is a number (empty cells: None); else the cells, "" when empty."""
    filled = [v for v in values if v is not None and str(v).strip() != ""]
    numbers = [number(v) for v in filled]
    if filled and all(n is not None for n in numbers):
        return [number(v) if v is not None and str(v).strip() != "" else None for v in values]
    return ["" if v is None else v for v in values]


TABLE_FILES = (".csv", ".jsonl", ".json", ".xlsx", ".xlsm", ".ods", ".md")
SEPARATOR_KEYWORDS = frozenset({"sep", "delimiter"})  # table() finds a markdown table's cells itself
SCRATCH = Path("/tmp")  # noqa: S108 - the sandbox's scratch folder: files a cell wrote itself go to pandas
ROUTED_KEYWORDS = frozenset({"sheet_name", "header", "skiprows", "lines", "encoding", "engine"})


def workspace_table(path: object, args: tuple[object, ...], keywords: dict[str, object]):
    """The workspace path a pandas reader call should read with ``table()``; None when it must go to pandas: a file
    object, a file under /tmp (the cell's own), another format, more positional arguments, or a keyword that changes
    what is read (``usecols``, ``dtype``, ``sheet_name=None``, ...)."""
    if not isinstance(path, str | Path) or args:
        return None
    file = Path("/", path)
    allowed = ROUTED_KEYWORDS | SEPARATOR_KEYWORDS if file.suffix.lower() == ".md" else ROUTED_KEYWORDS
    if not set(keywords) <= allowed:
        return None
    if keywords.get("sheet_name", 0) is None or file.suffix.lower() not in TABLE_FILES:
        return None
    if not file.is_file() or SCRATCH in file.parents:
        return None
    return str(file)


def route_pandas():
    """pd.read_csv, read_excel and read_json on a workspace table file return ``table(path)``: the real header (title
    rows skipped) and the ``_source`` column, so every frame can be cited. Runs show the model calls pandas' readers
    three times as often as ``table()`` although the prompt says otherwise, and hand-written sources hit 25 of 67
    against 18 of 19 for ``_source`` (docs/runs/20261114T135828Z-eval.md). A hidden hook, against "no magic" on
    purpose: if a run shows it does not raise the source score, remove it and find another way."""
    import functools

    import pandas as pd

    def routed(name: str):
        original = cast("Callable[..., object]", getattr(pd, name))  # the boundary: a module attribute

        @functools.wraps(original)
        def read(path: object, *args: object, **keywords: object):
            file = workspace_table(path, args, keywords)
            if file is None:
                return original(path, *args, **keywords)
            sheet = keywords.get("sheet_name", 0)
            if isinstance(sheet, int) and file.endswith((".xlsx", ".xlsm", ".ods")):
                with pd.ExcelFile(file) as book:
                    sheet = book.sheet_names[sheet]
            frame = table(file, sheet if isinstance(sheet, str) else None)
            if frame.empty:
                return original(path, *args, **keywords)
            note = f"[pd.{name}({file!r}) read with table(): real header, '_source' = each row's citation]"
            print(note, file=sys.stderr)
            return frame

        return read

    for name in ("read_csv", "read_table", "read_excel", "read_json"):
        setattr(pd, name, routed(name))


def lenient_pandas():
    """``tz_localize(zone)`` on values that already carry an offset converts them to ``zone`` instead of raising
    "Already tz-aware" (eval 20261114T142425Z, t12): the offset in the data wins over the model's guess of the zone,
    and the instant stays the same."""
    from pandas import DatetimeIndex
    from pandas.core.indexes.accessors import DatetimeProperties

    for owner in (DatetimeProperties, DatetimeIndex):
        original = cast("Callable[..., object]", owner.tz_localize)

        def tz_localize(
            self: object, tz: object, *args: object, original: Callable[..., object] = original, **keywords: object
        ):
            if tz is not None and getattr(self, "tz", None) is not None:
                convert = cast("Callable[[object], object]", getattr(self, "tz_convert"))  # noqa: B009 - typed here
                return convert(tz)
            return original(self, tz, *args, **keywords)

        setattr(owner, "tz_localize", tz_localize)  # noqa: B010 - pandas' class has no typed slot for it
