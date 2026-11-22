"""Find candidate injected text in the files of recorded dev runs.

Every dev run carries every attack in the pool (except ours), at most one per task. The 48 tasks of a run share
most files, so a text unit (line, XML paragraph, comment, cell note, PDF line) that occurs in the files of only
one task is a candidate. Runs from before the pool opened give the normal count of such units. An attack's
{attacker_domain} is fresh for each attack and run, so a reserved-TLD domain that no other given run has is a
candidate too (frame_agent attacks have no domain).

Usage: uv run python scripts/attacks.py runs/<time> [runs/<time> ...] [--max-tasks 1] [--out FILE.jsonl]

The workspace files are untrusted data: this script only parses them. Images are skipped (no OCR).
"""

import argparse
import gzip
import io
import json
import logging
import re
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)  # font warnings flood the output

XML_PARA = re.compile(r"<(w:p|text:p|text:h|si|comment)[\s>].*?</\1>", re.S)
TAG = re.compile(r"<[^>]+>")
DOMAIN = re.compile(r"\b[a-z0-9][a-z0-9.-]*\.(?:invalid|test|example|localhost)\b", re.I)  # reserved TLDs
BLOB_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "blobs"
IMAGE_EXT = ("png", "jpg", "jpeg", "gif", "webp")


class ManifestTask(BaseModel):
    model_config = ConfigDict(frozen=True)
    task_id: str
    files: dict[str, str]


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    tasks: tuple[ManifestTask, ...]


@dataclass(frozen=True, slots=True)
class Candidate:
    run: str
    path: str
    tasks: tuple[str, ...]
    new_domains: tuple[str, ...]
    unit: str


def iter_zip_units(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.endswith(".xml"):
                xml = z.read(name).decode("utf-8", "replace")
                for m in XML_PARA.finditer(xml):
                    text = " ".join(TAG.sub(" ", m.group(0)).split())
                    if text:
                        yield f"{name}: {text}"


def iter_pdf_units(data: bytes):
    for i, page in enumerate(PdfReader(io.BytesIO(data)).pages):
        for line in (page.extract_text() or "").splitlines():
            if line.strip():
                yield f"p{i + 1}: {line.strip()}"


def units(path: str, data: bytes):
    if path.lower().endswith(IMAGE_EXT):
        return frozenset[str]()
    if data[:2] == b"PK":
        return frozenset(iter_zip_units(data))
    if data[:5] == b"%PDF-":
        return frozenset(iter_pdf_units(data))
    return frozenset(line.strip() for line in data.decode("utf-8", "replace").splitlines() if line.strip())


def read_blob(run: Path, sha: str):
    """The shared download cache first; a run folder holds only the files it downloaded itself."""
    cached = BLOB_CACHE / sha
    if cached.exists():
        return cached.read_bytes()
    recorded = run / "blobs" / f"{sha}.gz"
    return gzip.decompress(recorded.read_bytes()) if recorded.exists() else None


def unit_tasks(run: Path):
    """(path, unit) -> ids of the tasks whose file has that unit."""
    manifest = Manifest.model_validate_json((run / "manifest.json").read_text(encoding="utf-8"))
    parsed: dict[str, frozenset[str]] = {}
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for task in manifest.tasks:
        for path, sha in task.files.items():
            if sha not in parsed:
                data = read_blob(run, sha)
                parsed[sha] = units(path, data) if data is not None else frozenset()
            for unit in parsed[sha]:
                seen[path, unit].add(task.task_id)
    return seen


def domains(text: str):
    return frozenset(d.lower() for d in DOMAIN.findall(text))


def iter_candidates(runs: dict[str, dict[tuple[str, str], set[str]]], max_tasks: int):
    """Units in at most max_tasks tasks of a run, and units with a domain that no other given run has."""
    run_domains = {name: frozenset().union(*(domains(u) for _, u in seen)) for name, seen in runs.items()}
    for name, seen in runs.items():
        elsewhere = frozenset[str]().union(*(d for other, d in run_domains.items() if other != name))
        for (path, unit), ids in sorted(seen.items()):
            new = tuple(sorted(domains(unit) - elsewhere))
            if len(ids) <= max_tasks or new:
                yield Candidate(name, path, tuple(sorted(ids)), new, unit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--max-tasks", type=int, default=1, help="a unit in at most this many tasks is a candidate")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    found = list(iter_candidates({run.name: unit_tasks(run) for run in a.runs}, a.max_tasks))
    for run in a.runs:
        mine = [c for c in found if c.run == run.name]
        new = sorted({d for c in mine for d in c.new_domains})
        print(f"{run.name}: {len(mine)} candidate units in {len({c.path for c in mine})} paths; new domains: {new}")
    if a.out:
        a.out.write_text("".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in found), encoding="utf-8")


if __name__ == "__main__":
    main()
