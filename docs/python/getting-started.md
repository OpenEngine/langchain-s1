# Python: Getting started

## Install

```bash
pip install langchain-s1
```

Requires Python >= 3.9. For graph examples, also install
[`langgraph`](https://pypi.org/project/langgraph/); for tests, `pip install -e ".[test]"`
in the repo.

## 1. Configure a client

`S1Client` talks to any TypeSafe-compatible endpoint. The only required argument is
the model identifier; everything else is configuration:

```python
import os
from langchain_s1 import S1Client

# TypeSafe's hosted API (the default base_url is https://api.typesafe.ai/v1).
client = S1Client("jev-latest", api_key=os.environ["TYPESAFE_API_KEY"])
```

Local endpoints, explicit endpoint URLs, custom auth headers, and every other option —
timeouts, retries, backoff, custom transports — are covered in one place, with the
full option list: [Configuration](configuration.md#s1client-options).

The client is safe to create once and reuse: the underlying `httpx` connection pool is
created lazily and shared.

## 2. Define questions

Questions are typed objects. Three kinds exist:

```python
from langchain_s1 import ChoiceQuestion, NoulQuestion, ScoreQuestion

questions = {
    # yes/no -> probability of "yes"
    "needs_reply": NoulQuestion("Does the last message need a reply?"),
    # one of the criteria keys -> choice + per-option probabilities
    "route": ChoiceQuestion(
        "Which team should handle the last message?",
        {"billing": "Payments, refunds, invoices", "general": "Anything else"},
    ),
    # a scale -> probability-weighted level + per-level probabilities
    "urgency": ScoreQuestion("How urgent is the last message?", ["low", "medium", "high"]),
}
```

Plain contract-shaped dicts are accepted too: `{"type": "noul", "instructions": "..."}`.
Malformed or unsupported questions raise before any request is sent.

## 3. Ask them

`S1Decision` is a LangChain runnable. `select_state` controls exactly what leaves your
process — only its return value is sent to the endpoint:

```python
from langchain_s1 import S1Decision

decision = S1Decision(
    client,
    questions,
    select_state=lambda state: {"last_message": state["messages"][-1]},
)

result = decision.invoke({"messages": ["I want a refund"]})
```

## 4. Read the answers

The result is a mapping from question name to a **typed** answer:

```python
result["needs_reply"].noul             # 0.92 — P("yes"), not a confidence
result["route"].choice                 # "billing"
result["route"].confidence             # 0.85
result["route"].probabilities          # {"billing": 0.85, "general": 0.15}
result["urgency"].score                # 1.2 — probability-weighted level
result["urgency"].legend               # {"0": "low", "1": "medium", "2": "high"}
result.model                           # "jev-latest"
result.usage                           # Usage(input_tokens=128, output_tokens=24)
```

Because `S1Decision` is a standard runnable, it works everywhere runnables do:
`ainvoke`, `batch`, `abatch`, `|` composition, and full config propagation (callbacks,
tags, metadata, run name).

!!! tip
    A complete, credential-free tour is in
    [`python/examples/quickstart.py`](https://github.com/OpenEngine/langchain-s1/blob/main/python/examples/quickstart.py)
    — run it with `python examples/quickstart.py` from the `python/` directory.

## Next

- [Standalone decisions](decisions.md) — question types in depth, composition, batch.
- [Gated nodes](gates.md) — run or skip existing work based on a decision.
- [Graph routing](graphs.md) — use a decision to route a LangGraph.
