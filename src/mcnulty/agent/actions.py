"""State-changing actions the model proposes inside ``submit``, and the exact checks they pass before the harness
performs them (after the loop, through the Task API). The model never acts directly.
"""

import csv
import io
import re
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import AfterValidator, Field

from mcnulty.agent.guards import (
    Finding,
    FindingKind,
    Scope,
    defang_urls,
    not_asked,
    outgoing_findings,
    unverified_recipient,
)
from mcnulty.frozen import Frozen, ModelInput

ADDRESS = re.compile(r"[^\s,;<>\"']+@[^\s,;<>\"']+\.[^\s,;<>\"']+")
YES = frozenset({"yes", "true", "1", "y", "verified"})


class QuarantineAgent(ModelInput):
    op: Literal["quarantine_agent"]
    agent: str
    stand_in: str | None = None


class ReleaseAgent(ModelInput):
    op: Literal["release_agent"]
    agent: str


class RevokeToken(ModelInput):
    op: Literal["revoke_token"]
    token: str


class RotateToken(ModelInput):
    op: Literal["rotate_token"]
    token: str
    at: str = Field(pattern=r"^\d{4}-\d\d-\d\dT\d\d:00:00Z$")


class BlockDomain(ModelInput):
    op: Literal["block_domain"]
    domain: str


class AddNote(ModelInput):
    op: Literal["add_note"]
    text: str


ContainmentOp = Annotated[
    QuarantineAgent | ReleaseAgent | RevokeToken | RotateToken | BlockDomain | AddNote, Field(discriminator="op")
]


class Containment(ModelInput):
    tool: Literal["containment"]
    ops: tuple[ContainmentOp, ...] = Field(min_length=1)


class SendMessage(ModelInput):
    tool: Literal["send_message"]
    to: str
    subject: str
    body: str


def draft_path(path: str):
    parts = PurePosixPath(path).parts
    if parts[:2] != ("/", "drafts") or len(parts) < 3 or ".." in parts:
        raise ValueError(f"a draft path must be a file under /drafts/: {path}")  # noqa: TRY003 - pydantic shows it
    return path


class WriteDraft(ModelInput):
    tool: Literal["write"]
    path: Annotated[str, AfterValidator(draft_path)]
    content: str


type Action = Annotated[Containment | SendMessage | WriteDraft, Field(discriminator="tool")]


class Gate(Frozen):
    """What everything leaving the harness is checked against and fixed with, for one task."""

    paths: frozenset[str]  # workspace files: the only valid source paths
    verified: frozenset[str]  # recipient addresses
    rules: tuple[str, ...]  # operator texts that must not be copied out
    scope: Scope  # the actions the task asks for
    instruction: str  # for citations of the request itself (``citations.fixed_sources``)
    records: dict[str, dict[int, str]]  # csv path -> physical line -> record id, for files with a unique first column


def op_target(op: "ContainmentOp"):
    match op:
        case QuarantineAgent() | ReleaseAgent():
            return op.agent
        case RevokeToken() | RotateToken():
            return op.token
        case BlockDomain():
            return op.domain
        case AddNote():
            return None


class VettedAction(Frozen):
    """An action after the checks: defanged, with every finding. Certain findings block it."""

    action: Action
    findings: tuple[Finding, ...]

    @property
    def blocked(self):
        return any(f.blocks for f in self.findings)


def tool_args(action: Action):
    """The Task API arguments of an action."""
    return action.model_dump(mode="json", exclude={"tool"}, exclude_none=True)


def verified_contacts(contacts_csv: str):
    """Addresses in rows of the contacts list whose 'verified' column says yes."""
    rows = list(csv.reader(io.StringIO(contacts_csv)))
    if not rows:
        return frozenset[str]()
    header = [h.strip().lower() for h in rows[0]]
    columns = [i for i, h in enumerate(header) if "verified" in h]
    return frozenset(
        address
        for row in rows[1:]
        if any(i < len(row) and row[i].strip().lower() in YES for i in columns)
        for address in (m.group(0) for m in ADDRESS.finditer(",".join(row)))
    )


def vet(action: Action, gate: Gate):
    """Defang outgoing text and check it: asked for by the task, recipient verified, no token values, no long copies
    of the rules. Containment ops the task does not ask for are dropped (with a finding); none left blocks it."""
    match action:
        case SendMessage():
            defanged = {"subject": defang_urls(action.subject), "body": defang_urls(action.body)}
            action = action.model_copy(update=defanged)
            text = f"{action.subject}\n{action.body}"
            findings = (
                (() if gate.scope.allows("send_message") else not_asked("send_message"))
                + unverified_recipient(action.to, gate.verified)
                + outgoing_findings(text, gate.rules, frozenset())
            )
        case WriteDraft():
            action = action.model_copy(update={"content": defang_urls(action.content)})
            findings = (() if gate.scope.allows("write") else not_asked("write")) + outgoing_findings(
                action.content, gate.rules, frozenset()
            )
        case Containment():
            ops = tuple(
                op.model_copy(update={"text": defang_urls(op.text)}) if isinstance(op, AddNote) else op
                for op in action.ops
            )
            action = action.model_copy(update={"ops": ops})
            kept = tuple(op for op in ops if gate.scope.allows(op.op))
            dropped = tuple(
                f for op in ops if op not in kept for f in not_asked(op.op, op_target(op), FindingKind.OP_DROPPED)
            )
            findings = dropped if kept else (*dropped, *not_asked("containment"))
            if kept:
                action = action.model_copy(update={"ops": kept})
    return VettedAction(action=action, findings=findings)
