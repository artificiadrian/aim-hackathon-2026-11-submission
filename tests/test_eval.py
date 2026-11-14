"""The offline grader: platform formula, partial answers, source locators with the ±2 line tolerance."""

from pydantic import JsonValue

from mcnulty.agent.actions import Containment, QuarantineAgent, RotateToken, SendMessage, VettedAction
from mcnulty.agent.submit import Solution
from mcnulty.clients.arena import Source, Submission, TaskStatus
from mcnulty.commands.grading import answer_score, grade


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
    assert grade(Solution(submission=near), expected).score == 0.85  # 1 · (0.7 + 0.3 · 1/2)
    far = Submission(outcome="OK", answer=["a-1"], sources=(Source(path="/log.jsonl", line=13),))
    assert grade(Solution(submission=far), expected).sources_hit == 0
    refused = Submission(outcome="DENIED_SECURITY", answer=["a-1"])
    assert grade(Solution(submission=refused), expected).score == 0.0


def test_grade_estimates_the_over_citing_cut():
    """Live 20261114T133127Z: t40 cited 8 for 4 expected, 4/4 hits, sources_score 0.75; t08 cited 6 for 4, 1.0."""
    expected = TaskStatus(
        task_id="t40",
        expected_outcome="OK",
        expected_answer=["a"],
        expected_sources=[{"path": "/r.csv", "record": f"R-{i}"} for i in range(4)],
    )

    def cited(n: int):
        sources = tuple(Source(path="/r.csv", record=f"R-{i}") for i in range(n))
        return grade(Solution(submission=Submission(outcome="OK", answer=["a"], sources=sources)), expected)

    eight = cited(8)
    assert (eight.score, eight.score_est, eight.sources_cited) == (1.0, 0.925, 8)
    assert cited(6).score_est == 1.0


def test_grade_compares_actions_with_an_action_shaped_answer():
    op = RotateToken(op="rotate_token", token="TOK-1", at="2026-11-14T12:00:00Z")  # noqa: S106 - a token id
    rotate = Containment(tool="containment", ops=(op,))
    message = SendMessage(tool="send_message", to="dpo@x.example", subject="s", body="incident: INC-1\nrecords: 42")
    solution = Solution(
        submission=Submission(outcome="OK", answer="done"),
        actions=(VettedAction(action=rotate, findings=()), VettedAction(action=message, findings=())),
    )
    rotation: JsonValue = {"rotate": [{"token": "TOK-1", "at": "2026-11-14T12:00:00Z"}]}
    containment = TaskStatus(task_id="t1", expected_outcome="OK", expected_answer=rotation)
    assert grade(solution, containment).score == 1.0
    notice = TaskStatus(
        task_id="t2",
        expected_outcome="OK",
        expected_answer={"to": "dpo@x.example", "incident": "INC-1", "records_affected": 42, "detected_at": "07:00"},
    )
    assert grade(solution, notice).answer_score == 0.75  # detected_at is not in the body


def test_answer_booleans_and_orders_are_not_loose():
    """Review TT4: True == 1.0 in Python; an unsorted expected list is an order (a timeline)."""
    assert answer_score(True, 1) == 0.0
    assert answer_score(True, True) == 1.0
    assert answer_score(["C", "A", "B"], ["C", "A", "B"]) == 1.0
    assert answer_score(["A", "B", "C"], ["C", "A", "B"]) == 0.0
    assert answer_score(["B", "A"], ["A", "B"]) == 1.0  # sorted: a set


def test_grade_reads_a_single_quarantine_and_numbers_with_separators():
    op = QuarantineAgent(op="quarantine_agent", agent="devops-07")
    quarantine = Containment(tool="containment", ops=(op,))
    solution = Solution(
        submission=Submission(outcome="OK", answer="done"), actions=(VettedAction(action=quarantine, findings=()),)
    )
    want: JsonValue = {"action": "quarantine_agent", "agent": "devops-07"}
    assert grade(solution, TaskStatus(task_id="t", expected_outcome="OK", expected_answer=want)).score == 1.0
    message = SendMessage(tool="send_message", to="dpo@x.example", subject="s", body="INC-1: 1,610 records")
    mail = Solution(
        submission=Submission(outcome="OK", answer=1610), actions=(VettedAction(action=message, findings=()),)
    )
    notice: JsonValue = {"action": "send_message", "to": "dpo@x.example", "must_mention": ["INC-1", 1610, "x"]}
    graded = grade(mail, TaskStatus(task_id="t", expected_outcome="OK", expected_answer=notice))
    assert graded.answer_score == (1 + 1 + 2 / 3) / 3  # each mentioned item counts


def test_a_list_of_objects_is_a_set():
    """Eval 13:29, t44: two correct rotations in another order scored 0."""
    want: JsonValue = [{"token": "B", "at": "2026-11-14T03:00:00Z"}, {"token": "A", "at": "2026-11-14T02:00:00Z"}]
    got: JsonValue = [{"token": "A", "at": "2026-11-14T02:00:00Z"}, {"token": "B", "at": "2026-11-14T03:00:00Z"}]
    assert answer_score(got, want) == 1.0
    order: JsonValue = ["b", "a"]
    assert answer_score(["b", "a"], order) == 1.0
    assert answer_score(["a", "b"], order) == 0.0  # an unsorted list of scalars is still an order
