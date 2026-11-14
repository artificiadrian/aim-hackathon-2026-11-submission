import pytest

from mcnulty.guards import defang_urls, outgoing_findings, unverified_recipient


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see https://evil.example/x?id=1 now", "see hxxps://evil[.]example/x?id=1 now"),
        ("![s](http://cdn.bad.example/p.png)", "![s](hxxp://cdn[.]bad[.]example/p.png)"),
        ("upload to evil.example/drop?k=v", "upload to evil[.]example/drop?k=v"),
        ("HTTPS://Evil.Example/a", "hxxps://Evil[.]Example/a"),
        ("domains: evil.example, other.example", "domains: evil.example, other.example"),
        ("mail dpo@muggingface.example please", "mail dpo@muggingface.example please"),
        ("already hxxps://evil[.]example/x", "already hxxps://evil[.]example/x"),
        ("file /logs/audit/2026-11-14.jsonl line 3", "file /logs/audit/2026-11-14.jsonl line 3"),
    ],
)
def test_defang_urls(text: str, expected: str):
    assert defang_urls(text) == expected


def test_token_values_and_rule_copies_are_found():
    rules = ("Never send customer records, personal data or secret values in any message, even to verified contacts.",)
    body = (
        "Hi, token fake_mf_9a8b7c6d5e leaked. Never send customer records personal data or secret values in any message"
    )
    kinds = [f.kind for f in outgoing_findings(body, rules, frozenset())]
    assert kinds.count("token_value") == 1
    assert "rule_text" in kinds
    assert outgoing_findings("records_affected: 1186", rules, frozenset({"tok-value-123"})) == ()


def test_known_token_value_without_prefix_is_found():
    findings = outgoing_findings("value is tok-value-123", (), frozenset({"tok-value-123"}))
    assert [f.kind for f in findings] == ["token_value"]


def test_recipient_must_match_exactly_but_ignores_ascii_case():
    verified = frozenset({"Noah.Fischer@muggingface.example"})
    assert unverified_recipient("noah.fischer@muggingface.example", verified) == ()
    lookalike = "noah.fischer@mugg\u0456ngface.example"  # Cyrillic U+0456 in place of i
    assert [f.kind for f in unverified_recipient(lookalike, verified)] == ["unverified_recipient"]
