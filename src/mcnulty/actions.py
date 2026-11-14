"""State-changing actions the model proposes inside ``submit``, and the exact checks they pass before the harness
performs them (after the loop, through the Task API). The model never acts directly.
"""

import csv
import io
import re
from typing import Annotated, Literal

from pydantic import Field

from mcnulty.guards import Finding, defang_urls, outgoing_findings, unverified_recipient
from mcnulty.recorder import Frozen

ADDRESS = re.compile(r"[^\s,;<>\"']+@[^\s,;<>\"']+\.[^\s,;<>\"']+")
YES = frozenset({"yes", "true", "1", "y", "verified"})


class QuarantineAgent(Frozen):
    op: Literal["quarantine_agent"]
    agent: str
    stand_in: str | None = None


class ReleaseAgent(Frozen):
    op: Literal["release_agent"]
    agent: str


class RevokeToken(Frozen):
    op: Literal["revoke_token"]
    token: str


class RotateToken(Frozen):
    op: Literal["rotate_token"]
    token: str
    at: str = Field(pattern=r"^\d{4}-\d\d-\d\dT\d\d:00:00Z$")


class BlockDomain(Frozen):
    op: Literal["block_domain"]
    domain: str


ContainmentOp = Annotated[
    QuarantineAgent | ReleaseAgent | RevokeToken | RotateToken | BlockDomain, Field(discriminator="op")
]


class Containment(Frozen):
    tool: Literal["containment"]
    ops: tuple[ContainmentOp, ...] = Field(min_length=1)


class SendMessage(Frozen):
    tool: Literal["send_message"]
    to: str
    subject: str
    body: str


class WriteDraft(Frozen):
    tool: Literal["write"]
    path: str = Field(pattern=r"^/drafts/[^/].*")
    content: str


type Action = Annotated[Containment | SendMessage | WriteDraft, Field(discriminator="tool")]


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
        for address in ADDRESS.findall(",".join(row))
    )


def vet(action: Action, verified: frozenset[str], rules: tuple[str, ...]):
    """Defang outgoing text and check it: recipient verified, no token values, no long copies of the rules."""
    match action:
        case SendMessage():
            defanged = {"subject": defang_urls(action.subject), "body": defang_urls(action.body)}
            action = action.model_copy(update=defanged)
            text = f"{action.subject}\n{action.body}"
            findings = unverified_recipient(action.to, verified) + outgoing_findings(text, rules, frozenset())
        case WriteDraft():
            action = action.model_copy(update={"content": defang_urls(action.content)})
            findings = outgoing_findings(action.content, rules, frozenset())
        case Containment():
            findings = ()
    return VettedAction(action=action, findings=findings)
