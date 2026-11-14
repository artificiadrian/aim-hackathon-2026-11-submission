"""OpenDocument text: non-empty ``text:p`` and ``text:h`` in document order, and annotations (comments)."""

from lxml import etree

from mcnulty.workspace.document import DocumentFile, numbered
from mcnulty.workspace.odf import content, paragraphs_text, tag

PARAGRAPH_TAGS = {tag("text", "p"), tag("text", "h")}
ANNOTATION = tag("office", "annotation")


def body_text(element: etree._Element):
    """The element's text without annotations (comments) inside it."""
    parts = [element.text or ""]
    for child in element:
        if child.tag != ANNOTATION:
            parts.append(body_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def parse_odt(path: str, data: bytes):
    texts: list[str] = []
    comments: list[str] = []

    def walk(element: etree._Element):
        for child in element:
            if child.tag == ANNOTATION:
                comments.append(paragraphs_text(child))  # not body text, so not walked further
                continue
            if child.tag in PARAGRAPH_TAGS:
                text = body_text(child)
                if text.strip():
                    texts.append(text)
            walk(child)

    body = content(data).find(f"{tag('office', 'body')}/{tag('office', 'text')}")
    if body is not None:
        walk(body)
    return DocumentFile(path=path, paragraphs=numbered(path, texts), comments=tuple(comments))
