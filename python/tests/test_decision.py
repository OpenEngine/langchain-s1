import asyncio

import pytest
from conftest import RecordingHandler
from langchain_core.runnables import RunnableLambda

from langchain_s1 import (
    ChoiceQuestion,
    NoulQuestion,
    S1Client,
    S1ConfigurationError,
    S1Decision,
    S1Result,
)
from langchain_s1.testing import MockSystemOne

QUESTIONS = {
    "relevant": NoulQuestion("Is the latest message about billing?"),
    "route": ChoiceQuestion("Who should answer?", {"billing": "Billing", "general": "General"}),
}


def make_decision(mock, **client_kwargs):
    client = S1Client("jev-latest", transport=mock, retry_backoff=0, **client_kwargs)
    return S1Decision(client, QUESTIONS, select_state=lambda s: {"message": s["message"]})


def test_invoke_selects_state_and_returns_typed_result():
    mock = MockSystemOne()
    result = make_decision(mock).invoke({"message": "refund?", "secret": "do-not-send"})
    assert isinstance(result, S1Result)
    assert result["route"].choice == "billing"
    assert mock.bodies[0]["state"] == {"message": "refund?"}


def test_ainvoke():
    mock = MockSystemOne()
    result = asyncio.run(make_decision(mock).ainvoke({"message": "hi"}))
    assert result["relevant"].noul == 0.9


def test_batch_and_abatch():
    mock = MockSystemOne()
    decision = make_decision(mock)
    inputs = [{"message": f"m{i}"} for i in range(4)]
    results = decision.batch(inputs)
    assert len(results) == 4 and all(isinstance(r, S1Result) for r in results)
    results = asyncio.run(decision.abatch(inputs))
    assert len(results) == 4
    sent = sorted(b["state"]["message"] for b in mock.bodies)
    assert sent == sorted(["m0", "m1", "m2", "m3"] * 2)


def test_batch_return_exceptions():
    mock = MockSystemOne(path="/nope")
    results = make_decision(mock, max_retries=0).batch(
        [{"message": "a"}], return_exceptions=True
    )
    assert isinstance(results[0], Exception)


def test_config_tags_metadata_and_callbacks_propagate():
    handler = RecordingHandler()
    decision = make_decision(MockSystemOne())
    decision.invoke(
        {"message": "x"},
        {"callbacks": [handler], "tags": ["t1"], "metadata": {"k": "v"}, "run_name": "route_s1"},
    )
    (start,) = handler.starts
    assert start["name"] == "route_s1"
    assert "t1" in start["tags"] and start["metadata"]["k"] == "v"


def test_composes_in_chain_for_state_updates():
    chain = make_decision(MockSystemOne()) | RunnableLambda(lambda r: {"route": r["route"].choice})
    assert chain.invoke({"message": "x"}) == {"route": "billing"}


def test_same_decision_on_two_endpoints_by_config_only():
    vendor_a = MockSystemOne(path="/v1/systemone", required_headers={"authorization": "Bearer a"})
    vendor_b = MockSystemOne(
        path="/internal/s1",
        required_headers={"x-team-key": "b"},
        answer=lambda body: {
            "relevant": {"type": "noul", "noul": 0.1},
            "route": {
                "type": "choice",
                "choice": "general",
                "confidence": 0.8,
                "probabilities": {"billing": 0.2, "general": 0.8},
            },
        },
    )
    configs = [
        dict(base_url="https://a.example/v1", api_key="a", transport=vendor_a),
        dict(endpoint_url="http://localhost:7000/internal/s1", headers={"X-Team-Key": "b"}, transport=vendor_b),
    ]
    routes = []
    for cfg in configs:
        decision = S1Decision(S1Client("any-model", **cfg), QUESTIONS, select_state=lambda s: s)
        routes.append(decision.invoke("hello")["route"].choice)
    assert routes == ["billing", "general"]


def test_requires_explicit_state_selection():
    with pytest.raises(TypeError):
        S1Decision(S1Client("m"), QUESTIONS)  # type: ignore[call-arg]
    with pytest.raises(S1ConfigurationError):
        S1Decision(S1Client("m"), QUESTIONS, select_state="message")  # type: ignore[arg-type]
