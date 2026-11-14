"""The hard spend cap stops model calls before any request is sent."""

from pathlib import Path

import httpx
import pytest

from mcnulty.clients.openrouter import TASK_SHARE, BudgetExhausted, CostTracker, OpenRouterClient, Usage
from mcnulty.record.log import EventLog, Recorder


def test_spend_cap_blocks_the_next_call(tmp_path: Path):
    sent: list[httpx.Request] = []
    http = httpx.Client(transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(500)))
    tracker = CostTracker(cap=0.01)
    tracker.add("m", Usage(prompt_tokens=1, completion_tokens=1, cost=0.02))
    llm = OpenRouterClient(http=http, base_url="http://x/api/v1", tracker=tracker)
    rec = Recorder(events=EventLog(run_dir=tmp_path))
    with pytest.raises(BudgetExhausted):
        llm.chat(rec, ())
    assert sent == []


def test_a_task_tracker_counts_its_own_spend_under_the_run_cap():
    run = CostTracker(cap=0.01)
    a, b = run.child(), run.child()
    a.add("m", Usage(prompt_tokens=100, cost=0.006))
    assert (a.total, b.total, run.total, run.calls) == (0.006, 0.0, 0.006, 1)
    assert not b.capped
    b.add("m", Usage(cost=0.005))
    assert a.capped
    assert b.capped


def test_a_task_stops_at_twice_its_fair_share():
    run = CostTracker(cap=1.5)
    task = run.child(TASK_SHARE * 1.5 / 48)
    task.add("m", Usage(cost=0.06))
    assert not task.capped
    task.add("m", Usage(cost=0.003))
    assert task.capped
    assert not run.capped


def test_usage_reads_chat_and_jev_token_counts():
    """Jev answers with input/output tokens; the record showed 0 for every Jev call (docs/research jev-placement)."""
    jev = Usage.model_validate({"input_tokens": 476, "output_tokens": 70, "cost": 0.00002})
    chat = Usage.model_validate({"prompt_tokens": 9, "completion_tokens": 3})
    assert (jev.prompt_tokens, jev.completion_tokens, chat.prompt_tokens) == (476, 70, 9)
