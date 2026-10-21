import asyncio
from typing import TypedDict

import httpx
import pytest
from conftest import RecordingHandler
from langchain_core.runnables import RunnableLambda

from langchain_s1 import (
    GATE_EVENT,
    GateAction,
    NoulQuestion,
    S1APIError,
    S1Client,
    S1ConfigurationError,
    S1Decision,
    S1TimeoutError,
    S1UncertainDecisionError,
    choice_policy,
    noul_policy,
    s1_gate,
)
from langchain_s1.testing import MockSystemOne

POLICY = noul_policy("needs_summary", run_at=0.7, skip_below=0.3)


def noul_mock(p):
    return MockSystemOne(answer=lambda body: {"needs_summary": {"type": "noul", "noul": p}})


def decision_for(transport, **kw):
    client = S1Client("m", api_key="sk-secret-key", transport=transport, retry_backoff=0, **kw)
    return S1Decision(
        client,
        {"needs_summary": NoulQuestion("Is the conversation long enough to summarize?")},
        select_state=lambda s: s["text"],
    )


class Counter:
    def __init__(self):
        self.calls = 0

    def node(self, state):
        self.calls += 1
        return {"summary": "S"}


def skip_empty(state, outcome):
    return {}


def build(p_or_transport, counter, **kw):
    transport = noul_mock(p_or_transport) if isinstance(p_or_transport, float) else p_or_transport
    return s1_gate(
        RunnableLambda(counter.node, name="summarize"),
        decision_for(transport),
        policy=POLICY,
        skip_output=skip_empty,
        **kw,
    )


def test_run_executes_node_exactly_once():
    c = Counter()
    assert build(0.9, c).invoke({"text": "long"}) == {"summary": "S"}
    assert c.calls == 1


def test_skip_never_executes_node():
    c = Counter()
    seen = []
    gate = s1_gate(
        c.node,
        decision_for(noul_mock(0.1)),
        policy=POLICY,
        skip_output=lambda s, o: seen.append(o) or {},
    )
    assert gate.invoke({"text": "short"}) == {}
    assert c.calls == 0
    (outcome,) = seen
    assert outcome.action == "skip" and outcome.policy_result is GateAction.SKIP
    assert outcome.decision["needs_summary"].noul == 0.1


@pytest.mark.parametrize("mode,expected_calls", [("run", 1), ("skip", 0)])
def test_uncertain_policy(mode, expected_calls):
    c = Counter()
    build(0.5, c, on_uncertain=mode).invoke({"text": "?"})
    assert c.calls == expected_calls


def test_uncertain_raise():
    c = Counter()
    with pytest.raises(S1UncertainDecisionError) as e:
        build(0.5, c).invoke({"text": "?"})
    assert e.value.decision["needs_summary"].noul == 0.5
    assert c.calls == 0


def failing_transport(calls):
    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"detail": "down"})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("mode,expected_calls", [("run", 1), ("skip", 0)])
def test_inference_failure_policy(mode, expected_calls):
    c, http_calls = Counter(), []
    gate = build(failing_transport(http_calls), c, on_error=mode)
    gate.invoke({"text": "x"})
    assert c.calls == expected_calls
    assert len(http_calls) == 3  # inference retried (bounded), node not replayed


def test_inference_failure_raise():
    c, http_calls = Counter(), []
    with pytest.raises(S1APIError):
        build(failing_transport(http_calls), c).invoke({"text": "x"})
    assert c.calls == 0


def test_timeout_failure_skips():
    c = Counter()

    def handler(request):
        raise httpx.ReadTimeout("slow")

    seen = []
    gate = s1_gate(
        c.node,
        decision_for(httpx.MockTransport(handler), max_retries=1),
        policy=POLICY,
        skip_output=lambda s, o: seen.append(o) or {},
        on_error="skip",
    )
    gate.invoke({"text": "x"})
    assert c.calls == 0
    assert seen[0].reason == "inference_error" and isinstance(seen[0].error, S1TimeoutError)


def test_node_failure_is_not_retried():
    calls = []

    def boom(state):
        calls.append(1)
        raise RuntimeError("node failed")

    gate = s1_gate(boom, decision_for(noul_mock(0.9)), policy=POLICY, skip_output=skip_empty, on_error="run")
    with pytest.raises(RuntimeError):
        gate.invoke({"text": "x"})
    assert calls == [1]


def test_tracing_failure_does_not_replay_node():
    class Broken(RecordingHandler):
        raise_error = True

        def on_custom_event(self, *a, **k):
            raise RuntimeError("tracer down")

    c = Counter()
    with pytest.raises(RuntimeError, match="tracer down"):
        build(0.9, c, on_error="run").invoke({"text": "x"}, {"callbacks": [Broken()]})
    assert c.calls == 0


def test_invalid_policy_and_config():
    c = Counter()
    with pytest.raises(S1ConfigurationError, match="must return"):
        s1_gate(c.node, decision_for(noul_mock(0.9)), policy=lambda r: True, skip_output=skip_empty).invoke({"text": "x"})
    with pytest.raises(S1ConfigurationError):
        s1_gate(c.node, decision_for(noul_mock(0.9)), policy=POLICY, skip_output=skip_empty, on_error="retry")
    with pytest.raises(TypeError):
        s1_gate(c.node, decision_for(noul_mock(0.9)), policy=POLICY)  # type: ignore[call-arg]
    assert c.calls == 0


def test_tracing_records_decision_and_action_and_propagates_config():
    handler = RecordingHandler()
    c = Counter()
    build(0.9, c).invoke(
        {"text": "x"}, {"callbacks": [handler], "tags": ["app"], "metadata": {"tenant": "t1"}}
    )
    names = handler.names()
    assert names[0].startswith("s1_gate") and "S1Decision" in names and "summarize" in names
    for start in handler.starts:
        assert "app" in start["tags"] and start["metadata"]["tenant"] == "t1"
    gate_run = handler.starts[0]["run_id"]
    children = {s["name"]: s for s in handler.starts[1:]}
    assert children["S1Decision"]["parent_run_id"] == gate_run
    assert children["summarize"]["parent_run_id"] == gate_run
    assert "s1_decision" in children["S1Decision"]["tags"]
    (event,) = handler.events
    assert event["name"] == GATE_EVENT and event["run_id"] == gate_run
    assert event["data"]["action"] == "run" and event["data"]["reason"] == "policy"
    assert event["data"]["decision"]["answers"]["needs_summary"]["noul"] == 0.9
    assert "sk-secret-key" not in repr(event) and "sk-secret-key" not in repr(handler.starts)


def test_async_gate_run_skip_and_cancellation():
    c = Counter()
    assert asyncio.run(build(0.9, c).ainvoke({"text": "x"})) == {"summary": "S"}
    assert asyncio.run(build(0.1, c).ainvoke({"text": "x"})) == {}
    assert c.calls == 1

    async def main():
        started = asyncio.Event()

        async def handler(request):
            started.set()
            await asyncio.sleep(60)

        gate = build(httpx.MockTransport(handler), c, on_error="run")
        task = asyncio.ensure_future(gate.ainvoke({"text": "x"}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(main(), 5))
    assert c.calls == 1  # cancelled gate never ran the node


def test_async_tracing_event():
    handler = RecordingHandler()
    asyncio.run(build(0.1, Counter()).ainvoke({"text": "x"}, {"callbacks": [handler]}))
    assert handler.events[0]["data"]["action"] == "skip"


def test_gate_batch():
    c = Counter()
    gate = build(0.9, c)
    assert gate.batch([{"text": "a"}, {"text": "b"}]) == [{"summary": "S"}] * 2
    assert c.calls == 2


def test_choice_policy():
    mock = MockSystemOne(
        answer=lambda body: {
            "act": {"type": "choice", "choice": "escalate", "confidence": 0.6, "probabilities": {"escalate": 0.6, "ignore": 0.4}}
        }
    )
    from langchain_s1 import ChoiceQuestion

    decision = S1Decision(
        S1Client("m", transport=mock),
        {"act": ChoiceQuestion("What now?", {"escalate": "Escalate", "ignore": "Ignore"})},
        select_state=str,
    )
    result = decision.invoke("x")
    assert choice_policy("act", run=["escalate"], skip=["ignore"])(result) is GateAction.RUN
    assert choice_policy("act", run=["escalate"], skip=["ignore"], min_confidence=0.8)(result) is GateAction.UNCERTAIN


# -- LangGraph ----------------------------------------------------------------


def test_langgraph_gated_node_and_conditional_routing():
    graph_mod = pytest.importorskip("langgraph.graph")
    StateGraph, START, END = graph_mod.StateGraph, graph_mod.START, graph_mod.END

    class State(TypedDict, total=False):
        text: str
        summary: str
        route: str

    mock = MockSystemOne(
        answer=lambda body: {
            "needs_summary": {"type": "noul", "noul": 0.95 if "long" in body["state"] else 0.05},
        }
    )
    c = Counter()
    gate = s1_gate(
        RunnableLambda(c.node, name="summarize"),
        decision_for(mock),
        policy=POLICY,
        skip_output=skip_empty,
    )
    router = decision_for(mock)

    def route(state):
        return "gated" if router.invoke(state)["needs_summary"].noul >= 0.5 else "done"

    builder = StateGraph(State)
    builder.add_node("gated", gate)
    builder.add_conditional_edges(START, route, {"gated": "gated", "done": END})
    builder.add_edge("gated", END)
    graph = builder.compile()

    assert graph.invoke({"text": "a long conversation"}) == {"text": "a long conversation", "summary": "S"}
    assert graph.invoke({"text": "hi"}) == {"text": "hi"}
    assert c.calls == 1
