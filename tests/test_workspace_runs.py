"""Every expected citation of the collected dev runs resolves to a parsed unit. Skipped without collected runs."""

from pathlib import Path

import pytest

from mcnulty.arena import BlobCache, RunManifest, RunStatus, Source
from mcnulty.settings import REPO_ROOT
from mcnulty.workspace.workspace import ParseCache, Workspace

RUNS = sorted(
    d for d in (REPO_ROOT / "runs").glob("*") if (d / "finish.json").exists() and (d / "manifest.json").exists()
)


@pytest.mark.skipif(not RUNS, reason="no collected runs in runs/")
@pytest.mark.parametrize("run_dir", RUNS, ids=[d.name for d in RUNS])
def test_expected_sources_resolve(run_dir: Path):
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text(encoding="utf-8"))
    finish = RunStatus.model_validate_json((run_dir / "finish.json").read_text(encoding="utf-8"))
    cache = ParseCache(blobs=BlobCache(root=REPO_ROOT / ".cache" / "blobs"))
    files = {t.task_id: t.files for t in manifest.tasks}
    misses: list[str] = []
    checked = 0
    for task in finish.tasks:
        workspace = Workspace(files=files[task.task_id], cache=cache)
        for raw in task.expected_sources if isinstance(task.expected_sources, list) else []:
            source = Source.model_validate(raw)
            checked += 1
            if workspace.find(source) is None:
                misses.append(f"{task.task_id} {source.model_dump(exclude_none=True)}")
    assert checked > 0
    assert not misses, f"{len(misses)} of {checked} expected sources do not resolve: {misses[:10]}"
