# Python: Graph routing

A decision doubles as a router: `S1Decision` is a runnable, so it can be the
`path` function of a LangGraph conditional edge. No `langchain-s1` magic is needed —
just invoke the decision and return the chosen branch.

```python
from langgraph.graph import END, START, StateGraph
from langchain_s1 import S1Client, S1Decision, ChoiceQuestion

class State(TypedDict, total=False):
    last_message: str
    team: str
    reply: str

decision = S1Decision(
    S1Client("jev-latest", api_key="..."),
    {
        "route": ChoiceQuestion(
            "Which team should handle the last message?",
            {"billing": "Payments, refunds, invoices", "general": "Anything else"},
        )
    },
    select_state=lambda s: {"last_message": s["last_message"]},
)

def route(state):
    return decision.invoke(state)["route"].choice

builder = StateGraph(State)
builder.add_node("billing", billing_node)
builder.add_node("general", general_node)
builder.add_node("reply", gated_reply)          # an s1_gate from the previous page
builder.add_conditional_edges(START, route, ["billing", "general"])
builder.add_edge("billing", "reply")
builder.add_edge("general", "reply")
builder.add_edge("reply", END)
graph = builder.compile()

graph.invoke({"last_message": "I want a refund"})
```

!!! info
    The routing call costs one S1 request per graph invocation. If the same question
    is also answered by a node later in the graph, keep the two decisions separate
    (each `S1Decision` sends its own request) or restructure so the answer is
    computed once and shared via state.

## Combining routing and gating

The natural shape is:

1. **Route** on a `choice` answer (which team / which pipeline).
2. **Gate** the expensive node inside each branch with a `noul` answer (is it worth
   doing at all).

Both primitives are runnables, so they compose with `|` and drop into graphs as nodes:

```python
# Route first, then gate the reply node.
builder.add_conditional_edges(START, route, ["billing", "general"])
builder.add_edge("billing", gated_reply)
builder.add_edge("general", gated_reply)
builder.add_edge("gated_reply", END)
```

## Async and tracing

- Use `await graph.ainvoke(...)` for async; the decision and gate both support async.
- Every gate invocation emits an `s1_gate` custom event (see
  [Gated nodes](gates.md)), so traces show which branch ran, what the decision was, and
  why a node was skipped.

## Next

- [Configuration](configuration.md) — client and gate options.
- [Testing](testing.md) — run all of this against a mock endpoint in your test suite.
