"""Every workspace file as citable units, whatever its format. Libraries load only when a file needs them.

Locators (docs/ARENA_DOCS.md, sources): text, markdown, jsonl -> line (1-based); csv -> record (first column) when it
is a unique id, else line (the header is line 1); json -> record (the ``id`` field of each item) or path only;
xlsx, ods -> sheet and row as shown in the sheet; docx, odt -> paragraph (1-based among non-empty body paragraphs,
headings count); pdf -> page; png, jpg -> path only (no units).

Parsed files are cached under ``/tmp/.ws`` (kept for the task), so a second search does not parse every pdf again.
"""

import csv
import hashlib
import io
import json
import warnings
from pathlib import Path

from pydantic import TypeAdapter

from ws.models import Source, Unit

IMAGES = (".png", ".jpg", ".jpeg", ".gif", ".webp")
CACHE = Path("/tmp/.ws")  # noqa: S108 - the task's own scratch folder inside the sandbox
UNITS = TypeAdapter(list[Unit])


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
    found: list[Unit] = []
    for n, (line, row) in enumerate(rows):
        record = row[0].strip() if n > 0 and unique and row else None
        source = Source(path=path, record=record) if record else Source(path=path, line=line)
        found.append(Unit(source=source, text=",".join(row), cells=tuple(row)))
    return found


def json_units(path: str):
    value: object = json.loads(read_text(path))
    if isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))  # {"items": [...]}
    if isinstance(value, list) and value and all(isinstance(item, dict) and "id" in item for item in value):
        return [
            Unit(source=Source(path=path, record=str(item["id"])), text=json.dumps(item, ensure_ascii=False))
            for item in value
        ]
    return [Unit(source=Source(path=path), text=read_text(path))]


def xlsx_units(path: str):
    import openpyxl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "Workbook contains no default style" on every exported file
        book = openpyxl.load_workbook(path, data_only=True)
    found: list[Unit] = []
    for sheet in book.worksheets:
        hidden = " (hidden sheet)" if sheet.sheet_state != "visible" else ""
        for row in sheet.iter_rows():
            cells = tuple("" if c.value is None else str(c.value) for c in row)
            if any(cells):
                number = next(c.row for c in row if c.row is not None)
                note = " (hidden row)" if sheet.row_dimensions[number].hidden else ""
                text = " | ".join(c for c in cells if c) + note + hidden
                found.append(Unit(source=Source(path=path, sheet=sheet.title, row=number), text=text, cells=cells))
    return found


def ods_units(path: str):
    import pandas as pd

    found: list[Unit] = []
    for name, frame in pd.read_excel(path, engine="odf", header=None, sheet_name=None).items():
        for index, row in frame.iterrows():
            cells = tuple("" if pd.isna(v) else str(v) for v in row)
            if any(cells):
                source = Source(path=path, sheet=str(name), row=int(str(index)) + 1)
                found.append(Unit(source=source, text=" | ".join(c for c in cells if c), cells=cells))
    return found


def docx_units(path: str):
    import docx

    texts = [p.text for p in docx.Document(path).paragraphs if p.text.strip()]
    return [Unit(source=Source(path=path, paragraph=n), text=t) for n, t in enumerate(texts, 1)]


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
    return [Unit(source=Source(path=path, paragraph=n), text=t) for n, t in enumerate(texts, 1)]


def pdf_units(path: str):
    import pypdf

    pages = pypdf.PdfReader(path).pages
    return [Unit(source=Source(path=path, page=n), text=page.extract_text() or "") for n, page in enumerate(pages, 1)]


def line_units(path: str):
    lines = read_text(path).splitlines()
    return [Unit(source=Source(path=path, line=n), text=line) for n, line in enumerate(lines, 1)]


READERS = {
    ".csv": csv_units,
    ".json": json_units,
    ".xlsx": xlsx_units,
    ".xlsm": xlsx_units,
    ".ods": ods_units,
    ".docx": docx_units,
    ".odt": odt_units,
    ".pdf": pdf_units,
}


def parse(path: str) -> list[Unit]:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGES:
        return []
    return READERS.get(suffix, line_units)(path)


def units(path: str):
    """The file as citable units (cached per path, size and time of change)."""
    stat = Path(path).stat()
    key = hashlib.sha256(f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()).hexdigest()
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return UNITS.validate_json(cached.read_bytes())
    found = parse(path)
    try:
        CACHE.mkdir(exist_ok=True)
        cached.write_bytes(UNITS.dump_json(found))
    except OSError:  # a full or missing /tmp only costs a second parse
        pass
    return found
