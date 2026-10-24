# langchain-s1 (Python)

A langchain wrapper for s1 decisions.

`langchain-s1` makes System 1 decisions available as LangChain runnables and as gates
around existing nodes. It speaks the
[TypeSafe System One API](https://docs.typesafe.ai/api/system-one) contract
(`{model, state, questions}` → `{model, answers, usage}`), so any endpoint implementing
that contract works through configuration alone.

Guides, examples, configuration reference, and the API reference (generated from the
docstrings) live on the [docsite](https://openengine.github.io/langchain-s1/).
The package layout and data flow are described in [`ARCHITECTURE.md`](ARCHITECTURE.md).

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

An `S1Decision` is a LangChain runnable: it sends a selected view of its input (only
what `select_state` returns) to the endpoint and returns typed answers, keyed by
question name. The complete worked example — all three question types, `select_state`,
and reading results — is in the
[Getting started guide](https://openengine.github.io/langchain-s1/python/getting-started/):

```python
result = decision.invoke({"messages": ["I want a refund"]})
result["route"].choice   # "billing"
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
    draft_reply,   # existing runnable or function
    decision,
    policy=noul_policy("needs_reply", run_at=0.7, skip_below=0.3),
    skip_output=lambda state, outcome: {},
    on_uncertain="run",
    on_error="skip",
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

A decision doubles as a router for a LangGraph conditional edge: invoke it in the
`path` function and return the chosen branch. The complete example is in the
[Graph routing guide](https://openengine.github.io/langchain-s1/python/graphs/) on the
docsite.

## Switching to a custom endpoint

The decision and gate do not change; only the client does. Local endpoints, explicit
endpoint paths, and custom auth headers — the canonical setup examples, with the full
`S1Client` option list (generated from the docstring) — are in the
[Configuration guide](https://openengine.github.io/langchain-s1/python/configuration/).

For tests and examples, `langchain_s1.testing.MockSystemOne` is an in-process,
contract-compatible endpoint:

```python
from langchain_s1.testing import MockSystemOne

client = S1Client("any-model", transport=MockSystemOne())
```

## Errors

All errors derive from `S1Error`; the inference errors (the `S1InferenceError`
subtree) are the ones a gate's `on_error` handles. The full reference — hierarchy,
when each one fires, and what never leaks — is in the
[Errors guide](https://openengine.github.io/langchain-s1/python/errors/) on the docsite.

## Development

```bash
pip install -e ".[test]"
pytest
python examples/quickstart.py
```
