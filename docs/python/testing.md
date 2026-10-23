# Python: Testing

`langchain_s1.testing.MockSystemOne` is an in-process, contract-compatible System One
endpoint. It is an `httpx.MockTransport`, so you plug it into a real `S1Client` — the
same code path (headers, retries, validation) runs, without network or credentials.

```python
from langchain_s1 import S1Client
from langchain_s1.testing import MockSystemOne

client = S1Client("any-model", transport=MockSystemOne())
```

## Custom answers

Pass an `answer(request_body) -> {question_name: answer}` function to control the
responses. This is the pattern used by
[`examples/quickstart.py`](https://github.com/OpenEngine/langchain-s1/blob/main/python/examples/quickstart.py):

```python
def answer(body):
    text = body["state"]["last_message"]
    return {
        "needs_reply": {"type": "noul", "noul": 0.1 if text == "thanks!" else 0.92},
        "route": {
            "type": "choice",
            "choice": "billing" if "refund" in text else "general",
            "confidence": 0.85,
            "probabilities": {"billing": 0.85, "general": 0.15},
        },
    }

client = S1Client("any-model", transport=MockSystemOne(answer))
```

Without one, every question gets a deterministic default
([`default_answer`](api.md#langchain_s1.testing.default_answer)): `noul` → 0.9,
`choice` → first option at 0.9, `score` → top level at 1.0.

## Assertions on the wire

The mock records every request:

```python
mock = MockSystemOne(answer)
client = S1Client("any-model", transport=mock)

decision = S1Decision(client, questions, select_state=lambda s: {"last_message": s["last_message"]})
decision.invoke({"last_message": "I want a refund"})

body = mock.bodies[0]
assert body["model"] == "any-model"
assert body["state"] == {"last_message": "I want a refund"}  # select_state did its job
assert set(body["questions"]) == {"needs_reply", "route"}
```

## Auth and paths

```python
# Enforce an Authorization header; missing/wrong value -> 401 (S1APIError).
mock = MockSystemOne(answer, required_headers={"authorization": "Bearer test-key"})

# Serve a non-default path.
mock = MockSystemOne(answer, path="/decide")
client = S1Client("any-model", endpoint_url="http://localhost/decide", transport=mock)
```

## In pytest

```python
def test_gate_skips_thanks():
    decision = make_decision()   # S1Decision over a MockSystemOne transport
    gated = s1_gate(draft_reply, decision, policy=noul_policy("needs_reply", run_at=0.7, skip_below=0.3),
                    skip_output=lambda s, o: {}, on_uncertain="run")
    assert gated.invoke({"last_message": "thanks!"}) == {}

def test_gate_runs_refunds():
    ...
    assert "reply" in gated.invoke({"last_message": "I want a refund"})
```

The repo's own test suite (`python/tests/`) is a working reference: it covers request
bodies, auth headers, retries, response validation, gate outcomes, and tracing events.

## Next

- [Errors](errors.md) — force failures with a mock and see how gates handle them.
- [Configuration](configuration.md) — transports are a client option.
