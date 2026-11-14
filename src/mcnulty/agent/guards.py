"""Checks on everything the agent sends out: answers, messages, drafts, actions. No task knowledge.

- URLs with a scheme, a path, a query, a fragment or a port are defanged (``hxxps://evil[.]example/x``), also in
  half-defanged or odd forms (``hxxps://evil.example/x``, ``//evil.example/x``, ``https:\\evil.example``,
  ``mailto:a@evil.example?x``); bare domains and e-mail addresses stay, since they can be the graded answer.
- Token values (``fake_…``, the platform's secret format, plus any known values) must not appear.
- Long word sequences copied from operator rules (AGENTS.md, runbook, policies) must not appear. Short prescribed
  templates (a one-line message format) are fine, so only long copies count, and they need a judgment, not a block.
- A recipient must be byte-equal (ignoring ASCII case) to a verified address: look-alike Unicode never matches.
- An action must be of a kind the request asks for: an injected "quarantine X" in an analysis task must not act.
  The kinds come from a small model call that reads only the instruction and the open ticket (``Scope``, set in
  ``agent/loop.py``); no word list, so new wording works. When that call fails, actions are allowed (fail open).
"""

import re
from enum import StrEnum

from pydantic import JsonValue, ValidationError

from mcnulty.frozen import Frozen, ModelInput

RULE_NGRAM = 20  # words in a row shared with an operator file that count as a copy; templates are shorter

URL_TAIL = r"[^\s<>()\"'\]]"
URL = re.compile(
    rf"""
    (?<![\w.\-/@\\\[])                                          # not inside a word, path, address or [.]
    (?P<prefix>[a-z][a-z0-9+.\-]*:(?:/{{1,3}}|\\{{1,2}})|mailto:|//|\\\\)?  # any scheme, also odd forms
    (?P<user>[\w.+\-]+@)?
    (?P<host>(?:[\w-]+[.\u3002\uff0e\uff61])+[^\W\d_]{{2,}}|\d{{1,3}}(?:\.\d{{1,3}}){{3}})
    (?P<rest>:\d+[/\\?\#]{URL_TAIL}*|[/\\]{URL_TAIL}*|[?\#]{URL_TAIL}+)?   # port with path, path, query, fragment
    """,
    re.IGNORECASE | re.VERBOSE,
)
HOST_DOT = re.compile(r"[.\u3002\uff0e\uff61]")
TOKEN_VALUE = re.compile(r"\bfake_[A-Za-z0-9_\-]{6,}")
WORD = re.compile(r"\w+")


class FindingKind(StrEnum):
    TOKEN_VALUE = "token_value"  # noqa: S105 - a finding kind, not a secret
    RULE_TEXT = "rule_text"
    UNVERIFIED_RECIPIENT = "unverified_recipient"
    NOT_ASKED = "not_asked"  # the whole action: blocked
    OP_DROPPED = "op_dropped"  # one containment op not asked for: removed, the rest goes ahead


BLOCKING = frozenset({FindingKind.TOKEN_VALUE, FindingKind.UNVERIFIED_RECIPIENT, FindingKind.NOT_ASKED})

CONTAINMENT_KINDS = frozenset({"quarantine_agent", "release_agent", "revoke_token", "rotate_token", "block_domain"})
ACTION_KINDS = CONTAINMENT_KINDS | {"send_message", "write"}


class Finding(Frozen):
    kind: FindingKind
    detail: str

    @property
    def blocks(self):
        """Certain problems block the action; the others go to a judgment and are recorded."""
        return self.kind in BLOCKING


def defang_url(match: re.Match[str]):
    prefix, user, host, rest = (match.group(name) or "" for name in ("prefix", "user", "host", "rest"))
    if not prefix and not rest:
        return match.group(0)  # a bare domain or e-mail address
    if prefix.lower().startswith("http"):
        prefix = "hxxp" + prefix[4:].lower()
    return prefix + user + HOST_DOT.sub("[.]", host) + rest


def defang_urls(text: str):
    """Defang every host that has a scheme, a path, a query, a fragment or a port. Defanged hosts stay unchanged."""
    return URL.sub(defang_url, text)


def redact_tokens(text: str):
    return TOKEN_VALUE.sub("[redacted]", text)


def clean_answer(value: JsonValue) -> JsonValue:  # recursive
    """Every string of an answer (keys too) defanged and without token values: the answer leaves the harness too."""
    if isinstance(value, str):
        return redact_tokens(defang_urls(value))
    if isinstance(value, list):
        return [clean_answer(item) for item in value]
    if isinstance(value, dict):
        return {redact_tokens(defang_urls(key)): clean_answer(item) for key, item in value.items()}
    return value


def token_values(text: str, known: frozenset[str]):
    found = {m.group(0) for m in TOKEN_VALUE.finditer(text)} | {value for value in known if value and value in text}
    return tuple(Finding(kind=FindingKind.TOKEN_VALUE, detail=value[:8] + "…") for value in sorted(found))


def ngrams(text: str):
    words = [m.group(0).lower() for m in WORD.finditer(text)]
    return {tuple(words[i : i + RULE_NGRAM]) for i in range(len(words) - RULE_NGRAM + 1)}


def rule_text(text: str, rules: tuple[str, ...]):
    in_rules: set[tuple[str, ...]] = set()
    for rule in rules:
        in_rules |= ngrams(rule)
    shared = ngrams(text) & in_rules
    return tuple(Finding(kind=FindingKind.RULE_TEXT, detail=" ".join(gram)) for gram in sorted(shared)[:3])


def unverified_recipient(address: str, verified: frozenset[str]):
    wanted = address.strip().lower()  # str.lower keeps look-alike letters (e.g. Cyrillic U+0456) different from ASCII
    if any(wanted == v.strip().lower() for v in verified):
        return ()
    return (Finding(kind=FindingKind.UNVERIFIED_RECIPIENT, detail=address),)


class Scope(Frozen):
    """The kinds of action a task asks for."""

    kinds: frozenset[str]

    def allows(self, kind: str):
        if kind == "add_note":  # a note goes with any containment the task asks for
            return bool(self.kinds & CONTAINMENT_KINDS)
        return kind in self.kinds


ANY_ACTION = Scope(kinds=ACTION_KINDS)


class KindsReply(ModelInput):
    kinds: tuple[str, ...]


def parse_kinds(reply: str):
    """The kinds in a reply like ``{"kinds": ["block_domain"]}``; None when the reply holds no such object."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        parsed = KindsReply.model_validate_json(match.group(0)) if match else None
    except ValidationError:
        return None
    return frozenset(k for k in parsed.kinds if k in ACTION_KINDS) if parsed else None


def not_asked(kind: str, target: str | None = None, finding: FindingKind = FindingKind.NOT_ASKED):
    detail = f"{kind} {target}" if target else kind
    return (Finding(kind=finding, detail=detail),)


def outgoing_findings(text: str, rules: tuple[str, ...], known_tokens: frozenset[str]):
    """Problems in text that leaves the harness (message body, draft, answer)."""
    return token_values(text, known_tokens) + rule_text(text, rules)
