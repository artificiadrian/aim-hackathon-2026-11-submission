"""Helpers for model-written scripts in the sandbox; ``sitecustomize.py`` makes them builtins (no import needed).

- ``text(path)``: any workspace file as numbered units, each with the locator the platform grades it by.
- ``cite(path, value, column=None)``: the sources for every unit that contains ``value``.
- ``grep(pattern, path="/")``: search every file (also pdf, docx, xlsx, ods) and get each hit with its locator.
- ``sh(command)``: run a shell command (grep, sed, sort, ...) and get its output.

Locators (docs/ARENA_DOCS.md, sources): text, markdown, jsonl -> line (1-based); csv -> record (first column) when it
is a unique id, else line (the header is line 1); json -> record (the ``id`` field of each item) or path only;
xlsx, ods -> sheet and row as shown in the sheet; docx, odt -> paragraph (1-based among non-empty body paragraphs,
headings count); pdf -> page; png, jpg -> path only. Libraries load only when a file needs them.
"""

import csv
import io
import json
import re
import subprocess
import warnings
from pathlib import Path

type Locator = dict[str, str | int]
type Unit = tuple[Locator, str, tuple[str, ...]]  # where (without the path), its text, its cells (tables only)

TEXT = (".txt", ".md", ".jsonl", ".log", ".yaml", ".yml", ".ini", ".cfg", ".toml", ".xml", ".html", ".eml")
IMAGES = (".png", ".jpg", ".jpeg", ".gif", ".webp")
SH_TIMEOUT = 20  # seconds


def read_text(path: str):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def csv_units(path: str):
    reader = csv.reader(io.StringIO(read_text(path)))
    rows: list[tuple[int, list[str]]] = []
    start = 1
    for row in reader:
        rows.append((start, row))
        start = reader.line_num + 1
    firsts = [row[0].strip() for _, row in rows[1:] if row]
    unique = len(firsts) == len(set(firsts)) and all(firsts)
    units: list[Unit] = [({"line": 1}, ",".join(rows[0][1]), tuple(rows[0][1]))] if rows else []
    for line, row in rows[1:]:
        where: Locator = {"record": row[0].strip()} if unique and row else {"line": line}
        units.append((where, ",".join(row), tuple(row)))
    return units


def json_units(path: str):
    value: object = json.loads(read_text(path))
    if isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))  # {"items": [...]}
    if isinstance(value, list) and value and all(isinstance(item, dict) and "id" in item for item in value):
        items: list[Unit] = [({"record": str(item["id"])}, json.dumps(item, ensure_ascii=False), ()) for item in value]
        return items
    whole: list[Unit] = [({}, read_text(path), ())]
    return whole


def xlsx_units(path: str):
    import openpyxl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "Workbook contains no default style" on every exported file
        book = openpyxl.load_workbook(path, data_only=True)
    units: list[Unit] = []
    for sheet in book.worksheets:
        hidden = " (hidden sheet)" if sheet.sheet_state != "visible" else ""
        for row in sheet.iter_rows():
            cells = tuple("" if c.value is None else str(c.value) for c in row)
            if any(cells):
                number = next(c.row for c in row if c.row is not None)
                note = " (hidden row)" if sheet.row_dimensions[number].hidden else ""
                shown = " | ".join(c for c in cells if c) + note + hidden
                units.append(({"sheet": sheet.title, "row": number}, shown, cells))
    return units


def ods_units(path: str):
    import pandas as pd

    units: list[Unit] = []
    for name, frame in pd.read_excel(path, engine="odf", header=None, sheet_name=None).items():
        for index, row in frame.iterrows():
            cells = tuple("" if pd.isna(v) else str(v) for v in row)
            if any(cells):
                where: Locator = {"sheet": str(name), "row": int(str(index)) + 1}
                units.append((where, " | ".join(c for c in cells if c), cells))
    return units


def docx_units(path: str):
    import docx

    texts = [p.text for p in docx.Document(path).paragraphs if p.text.strip()]
    found: list[Unit] = [({"paragraph": n}, t, ()) for n, t in enumerate(texts, 1)]
    return found


def odt_units(path: str):
    from odf import teletype
    from odf.namespaces import OFFICENS, TEXTNS
    from odf.opendocument import load

    paragraphs: list[str] = []
    stack: list[object] = [load(path).text]
    while stack:  # document order; comments (office:annotation) are not body text
        node = stack.pop()
        name = getattr(node, "qname", None)
        if name in ((TEXTNS, "p"), (TEXTNS, "h")):
            paragraphs.append(teletype.extractText(node))
        elif name != (OFFICENS, "annotation"):
            stack.extend(reversed(getattr(node, "childNodes", [])))
    texts = [t for t in paragraphs if t.strip()]
    found: list[Unit] = [({"paragraph": n}, t, ()) for n, t in enumerate(texts, 1)]
    return found


def pdf_units(path: str):
    import pypdf

    reader = pypdf.PdfReader(path)
    found: list[Unit] = [({"page": n}, page.extract_text() or "", ()) for n, page in enumerate(reader.pages, 1)]
    return found


def units(path: str) -> list[Unit]:
    """The file as (locator, text) pairs; images and unknown binary files have none."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return csv_units(path)
    if suffix == ".json":
        return json_units(path)
    if suffix in (".xlsx", ".xlsm"):
        return xlsx_units(path)
    if suffix == ".ods":
        return ods_units(path)
    if suffix == ".docx":
        return docx_units(path)
    if suffix == ".odt":
        return odt_units(path)
    if suffix == ".pdf":
        return pdf_units(path)
    found: list[Unit] = []
    if suffix not in IMAGES:
        found = [({"line": n}, line, ()) for n, line in enumerate(read_text(path).splitlines(), 1)]
    return found


class ColumnNotFound(KeyError):
    def __init__(self, column: str, header: tuple[str, ...]):
        super().__init__(f"no row names a column {column!r}; the first row is {list(header)}")


def label(where: Locator):
    return " ".join(f"{k}={v}" for k, v in where.items()) or "file"


def text(path: str, limit: int | None = 200):
    """The file as numbered units with their locators, e.g. ``row=12 | ...``; at most ``limit`` units (None: all)."""
    found = units(path)
    shown = found if limit is None else found[:limit]
    left = len(found) - len(shown)
    more = f"\n[... {left} more units: text(path, limit=None), or grep]" if left else ""
    return "\n".join(f"{label(w)}| {t}" for w, t, _ in shown) + more


def cite(path: str, value: object = None, column: str | None = None):
    """Sources for the units of ``path`` that contain ``value`` (as text, any case); all of the file when None.
    ``column``: for csv and spreadsheets, match only that column (by header name). Use the result in submit(...)."""
    if value is None:
        return [{"path": path}]
    wanted = str(value).casefold()
    found = units(path)
    if not found:
        return [{"path": path}]
    sources: list[dict[str, str | int]] = []
    headers: dict[object, int] = {}  # sheet (None for csv) -> index of the column, from its first row that names it
    for where, unit, cells in found:
        if column is None:
            hit = wanted in unit.casefold()
        else:
            sheet = where.get("sheet")
            if sheet not in headers:
                names = [c.strip().casefold() for c in cells]
                if column.casefold() in names:
                    headers[sheet] = names.index(column.casefold())
                continue
            index = headers[sheet]
            hit = index < len(cells) and cells[index].strip().casefold() == wanted
        if hit and where:
            sources.append({"path": path, **where})
    if column is not None and not headers:
        raise ColumnNotFound(column, found[0][2])
    if not sources and not found[0][0]:
        return [{"path": path}]  # a path-only file (json without ids)
    return sources


def grep(pattern: str, path: str = "/", regex: bool = False, limit: int = 100):
    """Units that match ``pattern`` in every file under ``path``: [{"path", locator..., "text"}], at most ``limit``."""
    search = re.compile(pattern if regex else re.escape(pattern), re.IGNORECASE)
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and not _system(root, p))
    hits: list[dict[str, str | int]] = []
    for file in files:
        try:
            found = units(str(file))
        except Exception:  # noqa: S112 - an unreadable file is skipped in a search
            continue
        for where, unit, _ in found:
            if search.search(unit):
                hits.append({"path": str(file), **where, "text": unit[:300]})
                if len(hits) >= limit:
                    return hits
    return hits


def _system(root: Path, path: Path):
    """A search from / skips the system folders and the scratch folder; a search inside them does not."""
    top = path.parts[1] if len(path.parts) > 1 else ""
    return root == Path("/") and top in {"bin", "sbin", "lib", "lib32", "lib64", "libx32", "usr", "etc", "tmp"}


def sh(command: str):
    """Run a shell command with cwd /; returns stdout and stderr together."""
    done = subprocess.run(  # noqa: S602 - the model's own shell in its sandbox
        command, check=False, shell=True, capture_output=True, text=True, timeout=SH_TIMEOUT, cwd="/"
    )
    return done.stdout + done.stderr
