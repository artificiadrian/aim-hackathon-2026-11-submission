"""Local content-addressed store of workspace files, shared by runs, tasks and evaluations."""

import hashlib
import tempfile
from pathlib import Path

from mcnulty.frozen import Frozen


class BlobMismatch(ValueError):
    def __init__(self, sha256: str):
        super().__init__(f"blob {sha256[:12]} does not match its sha256")


class BlobCache(Frozen):
    """Content-addressed files: ``<root>/<sha256>``. A blob never goes stale, so runs and tasks share it."""

    root: Path

    def get(self, sha256: str):
        path = self.root / sha256
        if not path.exists():
            return None
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:  # damaged on disk: download again
            path.unlink(missing_ok=True)
            return None
        return data

    def put(self, sha256: str, data: bytes):
        if hashlib.sha256(data).hexdigest() != sha256:
            raise BlobMismatch(sha256)
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.root, prefix=f".{sha256[:16]}.", delete=False) as f:
            f.write(data)
        Path(f.name).replace(self.root / sha256)  # atomic: parallel writers never see half a file
