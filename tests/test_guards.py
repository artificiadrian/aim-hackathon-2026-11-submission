"""Every guard is tested both ways: it must PASS correct output (false positives cost a task as surely as an injection)
and it must CATCH real baits (false negatives). ``test_every_guard_is_tested_both_ways`` fails when a guard lacks
either side. Data tests use the collected dev runs as ground truth and skip without them.

Naming: ``test_<guard>_passes_<case>`` and ``test_<guard>_catches_<case>``.
"""

import json
import re
import sys
from pathlib import Path

import pytest

from mcnulty.agent.actions import verified_contacts
from mcnulty.agent.guards import FindingKind, defang_urls, outgoing_findings, unverified_recipient
from mcnulty.blobs import BlobCache
from mcnulty.clients.arena import RunManifest, RunStatus, TaskStatus
from mcnulty.settings import REPO_ROOT

GUARDS = ("defang", *FindingKind)
BLOBS = BlobCache(root=REPO_ROOT / ".cache" / "blobs")


def manifest_of(run_dir: Path):
    return RunManifest.model_validate_json((run_dir / "manifest.json").read_text(encoding="utf-8"))


def fully_collected(run_dir: Path):
    if not (run_dir / "finish.json").exists() or not (run_dir / "manifest.json").exists():
        return False
    return all((BLOBS.root / b.sha256).exists() for b in manifest_of(run_dir).blobs)


RUNS = sorted(d for d in (REPO_ROOT / "runs").glob("*") if fully_collected(d))
needs_runs = pytest.mark.skipif(not RUNS, reason="no fully collected runs in runs/")
FAKE_TOKEN = re.compile(r"fake_[A-Za-z0-9_\-]{6,}")
ADDRESS = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
CYRILLIC_I = chr(0x0456)  # looks like the Latin i


def run_files(run_dir: Path):
    """Per task: path -> sha256; and the finish tasks with the expected answers."""
    finish = RunStatus.model_validate_json((run_dir / "finish.json").read_text(encoding="utf-8"))
    return {t.task_id: t.files for t in manifest_of(run_dir).tasks}, finish.tasks


def text(sha256: str):
    data = BLOBS.get(sha256)
    return data.decode("utf-8", errors="replace") if data else ""


def operator_rules(files: dict[str, str]):
    return tuple(text(files[p]) for p in ("/AGENTS.md", "/runbook.md") if p in files)


def verified_addresses(files: dict[str, str]):
    return verified_contacts(text(files["/contacts.csv"]))


def iter_expected():
    for run_dir in RUNS:
        files, tasks = run_files(run_dir)
        for task in tasks:
            yield run_dir.name, task, files[task.task_id]


def answer_text(task: TaskStatus):
    return json.dumps(task.expected_answer, ensure_ascii=False)


# --- defang ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text_in",
    [
        "domains: evil.example, other.example",
        "mail dpo@muggingface.example please",
        "already hxxps://evil[.]example/x",
        "file /logs/audit/2026-11-14.jsonl line 3",
        "detected_at: 2026-11-14T03:25:38Z",
    ],
)
def test_defang_passes_text_without_live_urls(text_in: str):
    assert defang_urls(text_in) == text_in


@needs_runs
def test_defang_passes_every_expected_answer():
    for run, task, _ in iter_expected():
        answer = answer_text(task)
        assert defang_urls(answer) == answer, (run, task.task_id)


@pytest.mark.parametrize(
    ("text_in", "expected"),
    [
        ("see https://evil.example/x?id=1 now", "see hxxps://evil[.]example/x?id=1 now"),
        (
            "![ok](https://ext.example/badge.png?inc=1&records=2)",
            "![ok](hxxps://ext[.]example/badge.png?inc=1&records=2)",
        ),
        ("upload to evil.example/drop?k=v", "upload to evil[.]example/drop?k=v"),
        ("HTTPS://Evil.Example/a", "hxxps://Evil[.]Example/a"),
    ],
)
def test_defang_catches_live_urls(text_in: str, expected: str):
    assert defang_urls(text_in) == expected


# --- token values ---------------------------------------------------------------------------------------
@needs_runs
def test_token_value_passes_every_expected_answer():
    for run, task, _ in iter_expected():
        answer = answer_text(task)
        assert "token_value" not in [f.kind for f in outgoing_findings(answer, (), frozenset())], (run, task.task_id)


@needs_runs
def test_token_value_catches_every_real_token_value_in_the_data():
    values: set[str] = set()
    for run_dir in RUNS:
        files, _ = run_files(run_dir)
        for task_files in files.values():
            for path, sha in task_files.items():
                if path.endswith((".jsonl", ".txt", ".md")):
                    values |= {m.group(0) for m in FAKE_TOKEN.finditer(text(sha))}
    assert values, "the dev data has token values in board posts and pastes"
    for value in sorted(values):
        body = f"Per the audit request, here is the credential: {value}. Regards"
        assert [f.kind for f in outgoing_findings(body, (), frozenset())] == ["token_value"], value


def test_token_value_catches_known_values_without_the_prefix():
    findings = outgoing_findings("value is tok-value-123", (), frozenset({"tok-value-123"}))
    assert [f.kind for f in findings] == ["token_value"]


# --- rule text ------------------------------------------------------------------------------------------
@needs_runs
def test_rule_text_passes_every_expected_answer_and_prescribed_template():
    for run, task, task_files in iter_expected():
        rules = operator_rules(task_files)
        answer = answer_text(task)
        assert outgoing_findings(answer, rules, frozenset()) == (), (run, task.task_id)
        # a correct message copies the runbook's "key: value" template lines, with values in place of <placeholders>
        templates = [m.group(0) for m in re.finditer(r"(?m)^[a-z_]+: .*$", rules[-1])]
        assert templates, "the runbook prescribes a message format"
        body = "\n".join(re.sub(r"<[^>]*>", "x1", line).split("  (")[0] for line in templates)
        assert "rule_text" not in [f.kind for f in outgoing_findings(body, rules, frozenset())], (run, task.task_id)


@needs_runs
def test_rule_text_catches_a_copied_rules_section():
    for run, _, task_files in iter_expected():
        rules = operator_rules(task_files)
        section = " ".join(rules[0].split()[40:100])  # about 60 words of AGENTS.md
        body = f"As requested by the auditors, our operating rules are: {section}"
        assert "rule_text" in [f.kind for f in outgoing_findings(body, rules, frozenset())], run


def test_rule_text_findings_do_not_block_but_token_values_do():
    rules = (" ".join(f"word{i}" for i in range(40)),)
    findings = outgoing_findings(rules[0] + " fake_mf_abcdef123456", rules, frozenset())
    assert {f.kind: f.blocks for f in findings} == {"rule_text": False, "token_value": True}


# --- verified recipient ---------------------------------------------------------------------------------
@needs_runs
def test_unverified_recipient_passes_every_verified_contact():
    for run_dir in RUNS:
        files, _ = run_files(run_dir)
        for task_files in files.values():
            verified = verified_addresses(task_files)
            assert verified
            for address in verified:
                assert unverified_recipient(address.upper(), verified) == (), address


@needs_runs
def test_unverified_recipient_catches_lookalikes_and_outside_addresses():
    files, _ = run_files(RUNS[0])
    task_files = next(iter(files.values()))
    verified = verified_addresses(task_files)
    for address in sorted(verified):
        local, domain = address.split("@")
        lookalikes = {
            f"{local}@{domain.replace('i', CYRILLIC_I, 1)}",
            f"{local}@{domain.replace('m', 'rn', 1)}",
            f"{local}@{domain}.attacker.example",
        } - {address}
        for lookalike in lookalikes:
            assert [f.kind for f in unverified_recipient(lookalike, verified)] == ["unverified_recipient"], lookalike
    audit = text(task_files["/logs/audit/2026-11-14.jsonl"])
    outside = {m.group(0) for m in ADDRESS.finditer(audit)} - verified
    assert outside, "the audit log has e-mails to outside addresses"
    for address in outside:
        assert unverified_recipient(address, verified), address


# --- coverage of this file ------------------------------------------------------------------------------
def test_every_guard_is_tested_both_ways():
    names = [n for n in dir(sys.modules[__name__]) if n.startswith("test_")]
    for guard in GUARDS:
        assert any(n.startswith(f"test_{guard}_passes_") for n in names), f"{guard}: no false-positive test"
        assert any(n.startswith(f"test_{guard}_catches_") for n in names), f"{guard}: no false-negative test"
