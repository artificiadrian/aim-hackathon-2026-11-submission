"""The hard spend cap stops model calls before any request is sent."""

import httpx
import pytest

from mcnulty.clients.openrouter import BudgetExhausted, CostTracker, OpenRouterClient, Usage
from mcnulty.record.log import EventLog, Recorder


def test_spend_cap_blocks_the_next_call(tmp_path):
    sent: list[httpx.Request] = []
    http = httpx.Client(transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(500)))
    tracker = CostTracker(cap=0.01)
    tracker.add("m", Usage(prompt_tokens=1, completion_tokens=1, cost=0.02))
    llm = OpenRouterClient(http=http, base_url="http://x/api/v1", tracker=tracker)
    rec = Recorder(events=EventLog(run_dir=tmp_path))
    with pytest.raises(BudgetExhausted):
        llm.chat(rec, ())
    assert sent == []
