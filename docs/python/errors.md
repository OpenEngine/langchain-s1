# Python: Errors

All errors derive from `S1Error`. The inference errors (the `S1InferenceError`
subtree) are the ones a gate's `on_error` handles.

| Error | Base | When |
|---|---|---|
| `S1ConfigurationError` | `S1Error`, `ValueError` | Invalid client, question, decision, or gate configuration — detected locally, before any request. |
| `S1UnsupportedFeatureError` | `S1Error` | An unsupported question/answer type or field (e.g. `box`, image questions, `assets`). |
| `S1RequestError` | `S1InferenceError` | Connection failure or failure after all retries. Carries `attempts`. |
| `S1TimeoutError` | `S1RequestError` | Every attempt timed out. |
| `S1APIError` | `S1InferenceError` | Non-success HTTP status. Carries `status_code` and `body` (JSON-decoded when possible, truncated at 2048 chars, kept **out** of the message). |
| `S1ResponseError` | `S1InferenceError`, `ValueError` | The response does not match the contract — missing answers, mismatched types, unknown choices, out-of-range scores, probabilities outside [0, 1]. Carries `path` (the offending field, e.g. `answers.route.choice`). |
| `S1UncertainDecisionError` | `S1Error` | A gate policy returned `uncertain` and `on_uncertain="raise"`. Carries `decision`. |

```
S1Error
├── S1ConfigurationError
├── S1UnsupportedFeatureError
├── S1InferenceError            # what gates' on_error handles
│   ├── S1RequestError
│   │   └── S1TimeoutError
│   ├── S1APIError
│   └── S1ResponseError
└── S1UncertainDecisionError
```

## Handling

Catch narrowly where you can act on the detail:

```python
from langchain_s1 import S1APIError, S1ResponseError, S1TimeoutError

try:
    result = client.decide(state, questions)
except S1APIError as e:
    if e.status_code == 401:
        ...  # bad credentials
    # e.body has the endpoint's detail (never in e.args / str(e))
except S1ResponseError as e:
    ...  # e.path points at the offending field
except S1TimeoutError:
    ...  # every attempt timed out
```

And let gates absorb inference failures instead of crashing the graph:

```python
s1_gate(node, decision, policy=policy, skip_output=lambda s, o: {},
        on_error="skip")   # S1InferenceError -> GateOutcome(reason="inference_error")
```

When a node is skipped due to failure, `skip_output` receives the error in
`outcome.error` and the reason in `outcome.reason` (`"inference_error"`).

## What never leaks

- Credentials (API key, header values) never appear in error messages, `repr`s, or
  tracing events.
- URLs in messages are redacted (userinfo/query/fragment removed).
- Error bodies are truncated and kept on `e.body`, out of the message text that
  reaches tracing.

## Next

- [Testing](testing.md) — force these failures deterministically with `MockSystemOne`.
- [API reference](api.md) — full signatures and docstrings.
