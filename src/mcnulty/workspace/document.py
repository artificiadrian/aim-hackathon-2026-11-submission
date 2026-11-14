"""Word-processing documents (docx, odt) as paragraphs, plus reviewer comments kept apart from the body."""

from typing import Literal

from mcnulty.arena import Source
from mcnulty.recorder import Frozen


class Paragraph(Frozen):
    source: Source  # path + paragraph (1-based, non-empty body paragraphs; titles and headings count)
    text: str


class DocumentFile(Frozen):
    kind: Literal["document"] = "document"
    path: str
    paragraphs: tuple[Paragraph, ...]
    comments: tuple[str, ...]  # reviewer comments: not body text


def numbered(path: str, texts: list[str]):
    return tuple(Paragraph(source=Source(path=path, paragraph=n), text=t) for n, t in enumerate(texts, 1))
