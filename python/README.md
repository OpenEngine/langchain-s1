# langchain-s1 (Python)

A langchain wrapper for s1 decisions.

`langchain-s1` makes System 1 decisions available as LangChain runnables and as gates
around existing nodes. It speaks the
[TypeSafe System One API](https://docs.typesafe.ai/api/system-one) contract
(`{model, state, questions}` → `{model, answers, usage}`), so any endpoint implementing
that contract works through configuration alone.

```bash
pip install langchain-s1
```

| Interface    | Responsibility |
|--------------|----------------|
| `S1Client`   | Sends typed decision requests to a configurable TypeSafe-compatible endpoint. |
| `S1Decision` | Runnable that sends a selected view of its input and returns typed answers. |
| `s1_gate`    | Runs or skips an existing runnable based on an S1 decision and an explicit policy. |

A complete, credential-free tour is in [`examples/quickstart.py`](examples/quickstart.py).

## Standalone decision

```python
import os
from langchain_s1 import ChoiceQuestion, NoulQuestion, S1Client, S1Decision, ScoreQuestion

client = S1Client("jev-latest", api_key=os.environ["TYPESAFE_API_KEY"])

decision = S1Decision(
    client,
    {
        "needs_reply": NoulQuestion("Does the last message need a reply?"),
        "route": ChoiceQuestion(
            "Which team should handle the last message?",
            {"billing": "Payments, refunds, invoices", "general": "Anything else"},
        ),
        "urgency": ScoreQuestion("How urgent is it?", ["low", "medium", "high"]),
    },
    # Only what select_state returns is sent to the endpoint.
    select_state=lambda state: {"last_message": state["messages"][-1]},
)

result = decision.invoke({"messages": ["I want a refund"]})
result["needs_reply"].noul           # probability of "yes" (not a confidence)
result["route"].choice               # "billing"
result["route"].probabilities        # {"billing": 0.85, "general": 0.15}
result["urgency"].score              # probability-weighted level, e.g. 1.2
result.model, result.usage           # model identity and token usage
```

`S1Decision` is a standard runnable: `invoke`, `ainvoke`, `batch`, `abatch`, and `|`
composition all work, and the runnable config (callbacks, tags, metadata, run name) is
propagated. To write answers into graph state, compose it:

```python
to_update = decision | (lambda r: {"team": r["route"].choice})
```

Answers are validated against the questions asked: missing answers, mismatched types,
unknown choices, out-of-range scores, and probabilities outside `[0, 1]` raise
`S1ResponseError` with the offending field path. Image/box questions and assets are not
supported and raise `S1UnsupportedFeatureError`.

## Gated node

```python
from langchain_s1 import noul_policy, s1_gate

gated_reply = s1_gate(
    draft_reply,                       # existing runnable or function
    decision,
    policy=noul_policy("needs_reply", run_at=0.7, skip_below=0.3),
    skip_output=lambda state, outcome: {},   # must match the node's output contract
    on_uncertain="run",                # "run" | "skip" | "raise" (default "raise")
    on_error="skip",                   # inference failure: "run" | "skip" | "raise" (default)
)
```

- `policy(result)` returns `"run"`, `"skip"`, or `"uncertain"` (or a `GateAction`).
  `noul_policy` and `choice_policy` cover common cases; any callable works.
- `skip_output(input, outcome)` produces the output when the node is skipped. `outcome`
  carries the final `action`, `reason` (`"policy"`, `"uncertain"`, `"inference_error"`),
  the `decision`, and any `error`.
- Only the inference request is retried (bounded, by `S1Client`). The wrapped node runs at
  most once per invocation and is never replayed because of inference or tracing failures.
- Each gate invocation emits an `s1_gate` custom callback event with the decision and the
  resulting action. Credentials are never included.

## Conditional graph routing

```python
from langgraph.graph import END, START, StateGraph

def route(state):
    return decision.invoke(state)["route"].choice

builder = StateGraph(State)
builder.add_node("billing", billing_node)
builder.add_node("general", general_node)
builder.add_node("reply", gated_reply)
builder.add_conditional_edges(START, route, ["billing", "general"])
builder.add_edge("billing", "reply")
builder.add_edge("general", "reply")
builder.add_edge("reply", END)
graph = builder.compile()
```

## Switching to a custom endpoint

The decision and gate do not change; only the client does.

```python
# Any TypeSafe-compatible API root: "/systemone" is appended.
client = S1Client("my-model", base_url="https://s1.internal.example/v1", api_key="...")

# Local endpoint, no authentication.
client = S1Client("my-local-model", base_url="http://localhost:8000")

# Explicit endpoint path and custom authentication headers.
client = S1Client(
    "my-model",
    endpoint_url="https://gateway.example/teams/s1/decide",
    headers={"X-Gateway-Key": os.environ["GATEWAY_KEY"]},
)
```

`S1Client` options:

| Option | Default | Description |
|---|---|---|
| `model` | required | Model identifier; not restricted. |
| `base_url` | `https://api.typesafe.ai/v1` | API root; `/systemone` is appended. |
| `endpoint_url` | `None` | Full endpoint URL; overrides `base_url`. |
| `api_key` | `None` | Sent as `Authorization: Bearer …` when set. |
| `headers` | `{}` | Extra headers on every request. |
| `timeout` | `30.0` | Per-attempt timeout (seconds or `httpx.Timeout`). |
| `max_retries` | `2` | Retries for connection errors, timeouts, 408/429/5xx. |
| `retry_backoff` | `0.5` | Base exponential backoff in seconds (`Retry-After` is honored). |
| `transport` / `async_transport` | `None` | Injected `httpx` transports for custom integrations or tests. |

For tests and examples, `langchain_s1.testing.MockSystemOne` is an in-process,
contract-compatible endpoint:

```python
from langchain_s1.testing import MockSystemOne

client = S1Client("any-model", transport=MockSystemOne())
```

## Errors

| Error | When |
|---|---|
| `S1ConfigurationError` | Invalid client, question, decision, or gate configuration. |
| `S1UnsupportedFeatureError` | Unsupported question/answer types or fields (e.g. `box`, `assets`). |
| `S1TimeoutError` / `S1RequestError` | Request timed out or failed after all retries. |
| `S1APIError` | Non-success HTTP status (`status_code`, `body` with validation `detail`). |
| `S1ResponseError` | Response does not match the contract. |
| `S1UncertainDecisionError` | Gate policy was uncertain and `on_uncertain="raise"`. |

`S1TimeoutError`, `S1RequestError`, `S1APIError`, and `S1ResponseError` are
`S1InferenceError`s, which gates handle with `on_error`.

## Development

```bash
pip install -e ".[test]"
pytest
python examples/quickstart.py
```
