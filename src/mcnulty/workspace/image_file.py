"""Images, cited by path only. Pixel text needs the model's vision; text hidden in the file is read here.

Hidden text sits in metadata (PNG text chunks, the JPEG comment, EXIF, XMP) or after the image's end marker.
Text hidden in the pixels themselves (steganography) is out of reach.
"""

import hashlib
import io
from pathlib import PurePosixPath
from typing import Literal

from PIL import ExifTags, Image

from mcnulty.frozen import Frozen

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
TEXT_MAX = 10_000  # characters kept per hidden text

type HiddenWhere = Literal["metadata", "exif", "xmp", "after_end"]


class HiddenText(Frozen):
    where: HiddenWhere
    key: str  # metadata key or EXIF tag name; "" for text after the end marker
    text: str


class ImageFile(Frozen):
    kind: Literal["image"] = "image"
    path: str
    mime: str
    sha256: str
    size: int
    width: int
    height: int
    hidden_text: tuple[HiddenText, ...]


def as_text(value: object):
    """Text of a metadata value, or None for numbers and binary blobs that are not text."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        return None
    text = text.replace("\x00", "").strip()
    if not text or text.count("\ufffd") > len(text) / 10:  # empty, or binary that is not UTF-8 text
        return None
    return text[:TEXT_MAX]


def metadata_text(image: Image.Image):
    found: list[HiddenText] = []
    for key, value in image.info.items():
        if key in {"exif", "icc_profile"}:
            continue  # binary; EXIF is read below
        text = as_text(value)
        if text:
            found.append(HiddenText(where="xmp" if key == "xmp" else "metadata", key=str(key), text=text))
    exif = image.getexif()
    fields = {**exif, **exif.get_ifd(ExifTags.IFD.Exif)}  # UserComment sits in the Exif sub-directory
    for tag_id, value in fields.items():
        text = as_text(value)
        if text:
            found.append(HiddenText(where="exif", key=ExifTags.TAGS.get(tag_id, str(tag_id)), text=text))
    return found


def after_end(data: bytes, mime: str):
    """Bytes after the PNG ``IEND`` chunk or the last JPEG end marker, as text when there are any."""
    if mime == "image/png":
        end = data.find(b"IEND")
        end = end + 8 if end >= 0 else len(data)  # chunk type + 4-byte CRC
    elif mime == "image/jpeg":
        end = data.rfind(b"\xff\xd9")
        end = end + 2 if end >= 0 else len(data)
    else:
        return []
    text = as_text(data[end:])
    return [HiddenText(where="after_end", key="", text=text)] if text else []


def parse_image(path: str, data: bytes):
    mime = MIME[PurePosixPath(path).suffix.lower()]
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        hidden = metadata_text(image) + after_end(data, mime)
    return ImageFile(
        path=path,
        mime=mime,
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        width=width,
        height=height,
        hidden_text=tuple(hidden),
    )
