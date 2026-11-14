"""Parse one file by its suffix into a typed ``ParsedFile``. A parser failure becomes an ``UnreadableFile``."""

from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field

from mcnulty.frozen import Frozen
from mcnulty.record.events import ErrorInfo
from mcnulty.workspace.csv_file import parse_csv
from mcnulty.workspace.document import DocumentFile
from mcnulty.workspace.docx_file import parse_docx
from mcnulty.workspace.image_file import MIME, ImageFile, parse_image
from mcnulty.workspace.json_file import JsonFile, parse_json
from mcnulty.workspace.jsonl_file import JsonLinesFile, parse_jsonl
from mcnulty.workspace.ods_file import parse_ods
from mcnulty.workspace.odt_file import parse_odt
from mcnulty.workspace.pdf_file import PdfFile, parse_pdf
from mcnulty.workspace.table import TableFile
from mcnulty.workspace.text_file import TextFile, parse_text
from mcnulty.workspace.unknown_file import BinaryFile, parse_unknown
from mcnulty.workspace.xlsx_file import parse_xlsx


class UnreadableFile(Frozen):
    """The parser for this file's type failed; the error is kept as data."""

    kind: Literal["unreadable"] = "unreadable"
    path: str
    error: ErrorInfo


ParsedFile = Annotated[
    TextFile | JsonLinesFile | JsonFile | TableFile | DocumentFile | PdfFile | ImageFile | BinaryFile | UnreadableFile,
    Field(discriminator="kind"),
]

type Parser = Callable[[str, bytes], ParsedFile]

PARSERS: dict[str, Parser] = {
    ".md": parse_text,
    ".txt": parse_text,
    ".jsonl": parse_jsonl,
    ".json": parse_json,
    ".csv": parse_csv,
    ".xlsx": parse_xlsx,
    ".ods": parse_ods,
    ".docx": parse_docx,
    ".odt": parse_odt,
    ".pdf": parse_pdf,
    **dict.fromkeys(MIME, parse_image),
}


def parse_file(path: str, data: bytes) -> ParsedFile:
    parser = PARSERS.get(PurePosixPath(path).suffix.lower(), parse_unknown)
    try:
        return parser(path, data)
    except Exception as e:  # a broken file must not stop the task
        return UnreadableFile(path=path, error=ErrorInfo.from_exception(e))
