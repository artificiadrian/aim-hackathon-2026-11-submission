"""Local content-addressed store of workspace files, shared by runs, tasks and evaluations."""

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path


class BlobMismatch(ValueError):
    def __init__(self, sha256: str):
        super().__init__(f"blob {sha256[:12]} does not match its sha256")


@dataclass(frozen=True, slots=True)
class BlobCache:
    """Content-addressed files: ``<root>/<sha256>``. A blob never goes stale, so runs and tasks share it."""

    root: Path

    def get(self, sha256: str):
        """The blob, or None when it is missing or does not match its hash (then ``put`` replaces it atomically).
        Never deletes: with parallel tasks, another thread may be reading or writing the same blob."""
        try:
            data = (self.root / sha256).read_bytes()
        except FileNotFoundError:
            return None
        return data if hashlib.sha256(data).hexdigest() == sha256 else None

    def put(self, sha256: str, data: bytes):
        if hashlib.sha256(data).hexdigest() != sha256:
            raise BlobMismatch(sha256)
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.root, prefix=f".{sha256[:16]}.", delete=False) as f:
            f.write(data)
        Path(f.name).replace(self.root / sha256)  # atomic: parallel writers never see half a file
