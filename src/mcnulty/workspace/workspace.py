"""One task's workspace: its files (path -> blob hash), parsed on first use, and lookup by citation."""

import threading

from pydantic import BaseModel, PrivateAttr

from mcnulty.arena import BlobCache, Source
from mcnulty.recorder import ErrorInfo, Frozen
from mcnulty.workspace.document import DocumentFile, Paragraph
from mcnulty.workspace.image_file import ImageFile
from mcnulty.workspace.json_file import JsonFile
from mcnulty.workspace.jsonl_file import JsonLine, JsonLinesFile
from mcnulty.workspace.parse import ParsedFile, UnreadableFile, parse_file
from mcnulty.workspace.pdf_file import PdfFile, PdfPage
from mcnulty.workspace.table import TableFile, TableRow
from mcnulty.workspace.text_file import TextFile, TextLine
from mcnulty.workspace.unknown_file import BinaryFile

type Unit = TextLine | JsonLine | TableRow | Paragraph | PdfPage


class ParseCache(BaseModel):
    """Parsed files by (blob hash, path): one parse per blob and path, shared by every task. Thread-safe."""

    blobs: BlobCache
    _parsed: dict[tuple[str, str], ParsedFile] = PrivateAttr(default_factory=dict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def get(self, path: str, sha256: str):
        key = (sha256, path)  # the path is part of every Source, so it is part of the key
        with self._lock:
            hit = self._parsed.get(key)
        if hit is not None:
            return hit
        data = self.blobs.get(sha256)
        parsed = (
            parse_file(path, data)
            if data is not None
            else UnreadableFile(path=path, error=ErrorInfo.from_exception(FileNotFoundError(sha256)))
        )
        with self._lock:
            self._parsed[key] = parsed
        return parsed


def units(parsed: ParsedFile) -> tuple[Unit, ...]:
    match parsed:
        case TextFile() | JsonLinesFile() | JsonFile():
            return parsed.lines
        case TableFile():
            return tuple(row for table in parsed.tables for row in table.rows)
        case DocumentFile():
            return parsed.paragraphs
        case PdfFile():
            return parsed.pages
        case ImageFile() | BinaryFile() | UnreadableFile():
            return ()


class Workspace(Frozen):
    files: dict[str, str]  # path -> sha256, from the task manifest
    cache: ParseCache

    def get(self, path: str):
        return self.cache.get(path, self.files[path])

    def find(self, source: Source):
        """The unit a citation points to; the whole file for a path-only citation; None if it does not exist."""
        if source.path not in self.files:
            return None
        parsed = self.get(source.path)
        wanted = source.model_dump(exclude={"path"}, exclude_none=True)
        if not wanted:
            return parsed
        for unit in units(parsed):
            have = unit.source.model_dump(exclude={"path"}, exclude_none=True)
            if all(have.get(k) == v for k, v in wanted.items()):
                return unit
        return None
