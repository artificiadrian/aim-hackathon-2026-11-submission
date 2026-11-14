"""PDF: text per page, cited by page; embedded images counted (a page with an image but no text is a scan)."""

import io
from typing import Literal

import pypdf

from mcnulty.clients.arena import Source
from mcnulty.frozen import Frozen


class PdfPage(Frozen):
    source: Source  # path + page (1-based)
    text: str
    image_count: int


class PdfFile(Frozen):
    kind: Literal["pdf"] = "pdf"
    path: str
    pages: tuple[PdfPage, ...]


def image_count(page: pypdf.PageObject):
    """Images placed on the page (pypdf's documented ``page.images``; not decoded here)."""
    return len(page.images)


def parse_pdf(path: str, data: bytes):
    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = tuple(
        PdfPage(source=Source(path=path, page=n), text=page.extract_text() or "", image_count=image_count(page))
        for n, page in enumerate(reader.pages, 1)
    )
    return PdfFile(path=path, pages=pages)
