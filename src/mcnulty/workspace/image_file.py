"""Images: cited by path only. Their text can be read only with the model's vision."""

import hashlib
from pathlib import PurePosixPath
from typing import Literal

from mcnulty.recorder import Frozen

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}


class ImageFile(Frozen):
    kind: Literal["image"] = "image"
    path: str
    mime: str
    sha256: str
    size: int


def parse_image(path: str, data: bytes):
    mime = MIME[PurePosixPath(path).suffix.lower()]
    return ImageFile(path=path, mime=mime, sha256=hashlib.sha256(data).hexdigest(), size=len(data))
