"""Files of an unknown type: UTF-8 text is read as text lines; anything else is kept as an opaque binary."""

import hashlib
from typing import Literal

from mcnulty.frozen import Frozen
from mcnulty.workspace.text_file import parse_text


class BinaryFile(Frozen):
    kind: Literal["binary"] = "binary"
    path: str
    sha256: str
    size: int


def parse_unknown(path: str, data: bytes):
    try:
        return parse_text(path, data)
    except UnicodeDecodeError:
        return BinaryFile(path=path, sha256=hashlib.sha256(data).hexdigest(), size=len(data))
