"""Runnable tour of langchain-s1 against in-process mock endpoints (no credentials needed).

    python examples/quickstart.py

To use a real endpoint, drop ``transport=...`` and pass ``api_key=os.environ[...]``.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.runnables import RunnableLambda

from langchain_s1 import (
    ChoiceQuestion,
    NoulQuestion,
    S1Client,
    S1Decision,
    ScoreQuestion,
    noul_policy,
    s1_gate,
)
from langchain_s1.testing import MockSystemOne


def answer(body):
    text = body["state"]["last_message"]
    billing = "refund" in text
    return {
        "needs_reply": {"type": "noul", "noul": 0.1 if text == "thanks!" else 0.92},
        "route": {
            "type": "choice",
            "choice": "billing" if billing else "general",
            "confidence": 0.85,
            "probabilities": {"billing": 0.85, "general": 0.15}
            if billing
            else {"billing": 0.15, "general": 0.85},
        },
        "urgency": {
            "type": "score",
            "score": 1.2,
            "confidence": 0.6,
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4},
        },
    }


# Two interchangeable TypeSafe-compatible endpoints: only client config differs.
HOSTED = S1Client(
    "jev-latest",
    api_key="example-key",  # os.environ["TYPESAFE_API_KEY"] in real use
    transport=MockSystemOne(answer, required_headers={"authorization": "Bearer example-key"}),
)
LOCAL = S1Client(
    "my-local-s1",
    endpoint_url="http://localhost:8000/decide",
    headers={"X-Internal-Auth": "token"},
    transport=MockSystemOne(answer, path="/decide"),
)

QUESTIONS = {
    "needs_reply": NoulQuestion("Does the last message need a reply?"),
    "route": ChoiceQuestion(
        "Which team should handle the last message?",
        {"billing": "Payments, refunds, invoices", "general": "Anything else"},
    ),
    "urgency": ScoreQuestion("How urgent is the last message?", ["low", "medium", "high"]),
}


class State(TypedDict, total=False):
    last_message: str
    reply: str
    team: str


def make_decision(client: S1Client) -> S1Decision:
    # select_state controls exactly what is sent to the endpoint.
    return S1Decision(client, QUESTIONS, select_state=lambda s: {"last_message": s["last_message"]})


def draft_reply(state: State) -> State:
    return {"reply": f"Re: {state['last_message']}"}


def main(client: S1Client = HOSTED) -> None:
    decision = make_decision(client)

    # 1. Standalone decision
    result = decision.invoke({"last_message": "I want a refund"})
    print("P(needs reply) =", result["needs_reply"].noul)
    print("route =", result["route"].choice, result["route"].probabilities)
    print("urgency =", result["urgency"].score, "usage =", result.usage)

    # 2. Gated node: draft_reply runs only when S1 says a reply is needed.
    gated_reply = s1_gate(
        RunnableLambda(draft_reply, name="draft_reply"),
        decision,
        policy=noul_policy("needs_reply", run_at=0.7, skip_below=0.3),
        skip_output=lambda state, outcome: {},  # LangGraph node: empty update
        on_uncertain="run",
        on_error="skip",
    )
    print("gated:", gated_reply.invoke({"last_message": "thanks!"}))  # skipped -> {}

    # 3. Conditional graph routing (requires `pip install langgraph`)
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        print("langgraph not installed; skipping graph example")
        return

    def route(state: State) -> str:
        return decision.invoke(state)["route"].choice

    builder = StateGraph(State)
    builder.add_node("billing", lambda state: {"team": "billing"})
    builder.add_node("general", lambda state: {"team": "general"})
    builder.add_node("reply", gated_reply)
    builder.add_conditional_edges(START, route, ["billing", "general"])
    builder.add_edge("billing", "reply")
    builder.add_edge("general", "reply")
    builder.add_edge("reply", END)
    graph = builder.compile()
    print("graph:", graph.invoke({"last_message": "I want a refund"}))


if __name__ == "__main__":
    main(HOSTED)
    # 4. Switch to a custom endpoint: same decision and gate, different client config.
    main(LOCAL)
