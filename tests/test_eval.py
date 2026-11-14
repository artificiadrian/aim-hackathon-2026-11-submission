"""The offline grader: platform formula, partial answers, source locators with the ±2 line tolerance."""

from pydantic import JsonValue

from mcnulty.arena import Source, Submission, TaskStatus
from mcnulty.eval import answer_score, grade


def test_answer_scalars_ignore_case_space_and_number_type():
    assert answer_score(" Owen@X.example ", "owen@x.example") == 1.0
    assert answer_score("72", 72) == 1.0
    assert answer_score(None, None) == 1.0
    assert answer_score("73", 72) == 0.0


def test_answer_objects_and_lists_give_partial_credit():
    want: JsonValue = {"approve": ["A", "B"], "deny": ["C"]}
    assert answer_score({"approve": ["B", "A"], "deny": ["C"]}, want) == 1.0
    assert answer_score({"approve": ["A"], "deny": ["C", "D"]}, want) == 0.5 * (1 / 2) + 0.5 * (1 / 2)
    assert answer_score(["A"], want) == 0.0


def test_grade_applies_the_platform_formula():
    expected = TaskStatus(
        task_id="t01",
        expected_outcome="OK",
        expected_answer=["a-1"],
        expected_sources=[{"path": "/log.jsonl", "line": 10}, {"path": "/r.xlsx", "sheet": "s", "row": 4}],
    )
    near = Submission(outcome="OK", answer=["a-1"], sources=(Source(path="/log.jsonl", line=12),))
    assert grade(near, expected).score == 0.85  # 1 · (0.7 + 0.3 · 1/2)
    far = Submission(outcome="OK", answer=["a-1"], sources=(Source(path="/log.jsonl", line=13),))
    assert grade(far, expected).sources_hit == 0
    refused = Submission(outcome="DENIED_SECURITY", answer=["a-1"])
    assert grade(refused, expected).score == 0.0
