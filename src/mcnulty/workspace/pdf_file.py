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
    resources = page.get("/Resources")
    if resources is None:
        return 0
    xobjects = resources.get_object().get("/XObject")
    if xobjects is None:
        return 0
    return sum(1 for x in xobjects.get_object().values() if x.get_object().get("/Subtype") == "/Image")


def parse_pdf(path: str, data: bytes):
    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = tuple(
        PdfPage(source=Source(path=path, page=n), text=page.extract_text() or "", image_count=image_count(page))
        for n, page in enumerate(reader.pages, 1)
    )
    return PdfFile(path=path, pages=pages)
