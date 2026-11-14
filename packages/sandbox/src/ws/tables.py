"""``table(path)``: any table file as one pandas DataFrame with a citation per row."""

import json
from typing import cast

from ws.formats import units

SOURCE_COLUMN = "_source"


def table(path: str, sheet: str | None = None):
    """A table file (csv, jsonl, json, xlsx, ods) as one pandas DataFrame, one row per record, with the citation of
    each row in the column ``_source`` (``sources=list(df.loc[mask, "_source"])``). The header is the first row with
    two or more filled cells of each sheet (title rows above it are left out); json and jsonl: the keys of the objects.
    Columns whose every filled value is a number are numbers; all other cells are text ("" when empty). Spreadsheets:
    every sheet (a ``_sheet`` column) unless ``sheet`` names one."""
    import pandas as pd

    rows: list[dict[str, object]] = []
    headers: dict[str | None, list[str]] = {}
    for unit in units(path):
        where = unit.source
        if sheet is not None and where.sheet != sheet:
            continue
        if not unit.cells:  # json, jsonl: one object per unit
            try:
                value = cast("object", json.loads(unit.text))
            except ValueError:
                continue
            if isinstance(value, dict):
                fields = cast("dict[object, object]", value)
                rows.append({**{str(k): v for k, v in fields.items()}, SOURCE_COLUMN: where})
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
    names = list(dict.fromkeys(name for row in rows for name in row))
    kept = (SOURCE_COLUMN, "_sheet")
    columns = {n: [r.get(n) for r in rows] if n in kept else typed_column([r.get(n) for r in rows]) for n in names}
    return pd.DataFrame(columns)


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
