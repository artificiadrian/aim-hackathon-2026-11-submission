"""Word documents: non-empty body paragraphs in order, and reviewer comments."""

import io

import docx

from mcnulty.workspace.document import DocumentFile, numbered


def parse_docx(path: str, data: bytes):
    document = docx.Document(io.BytesIO(data))
    texts = [p.text for p in document.paragraphs if p.text.strip()]
    return DocumentFile(path=path, paragraphs=numbered(path, texts), comments=tuple(c.text for c in document.comments))
