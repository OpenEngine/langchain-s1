# Python: Gated nodes

`s1_gate` wraps an existing runnable so it only runs when an S1 decision — plus an
explicit policy — says it should. This is the main cost lever: the expensive node
(drafting a reply, calling a big model, writing to a system) runs only when the cheap
decision justifies it.

```python
from langchain_core.runnables import RunnableLambda
from langchain_s1 import S1Decision, noul_policy, s1_gate

def draft_reply(state):
    return {"reply": f"Re: {state['last_message']}"}

decision = S1Decision(
    client,
    {"needs_reply": NoulQuestion("Does the last message need a reply?")},
    select_state=lambda s: {"last_message": s["last_message"]},
)

gated_reply = s1_gate(
    RunnableLambda(draft_reply, name="draft_reply"),
    decision,
    policy=noul_policy("needs_reply", run_at=0.7, skip_below=0.3),
    skip_output=lambda state, outcome: {},   # what a LangGraph node should return when skipped
    on_uncertain="run",
    on_error="skip",
)

gated_reply.invoke({"last_message": "I want a refund"})   # -> draft_reply runs
gated_reply.invoke({"last_message": "thanks!"})           # -> skipped, returns {}
```

## The pieces

The full `s1_gate(...)` signature, with defaults, is in
[Configuration](configuration.md#s1_gate-options) and the
[API reference](api.md#langchain_s1.gate.s1_gate).

### Policies

A policy maps an `S1Result` to `"run"`, `"skip"`, or `"uncertain"` (or a `GateAction`);
any callable works. The two built-ins, with their exact threshold rules, are
[`noul_policy`](api.md#langchain_s1.gate.noul_policy) and
[`choice_policy`](api.md#langchain_s1.gate.choice_policy):

```python
noul_policy("needs_reply", run_at=0.7, skip_below=0.3)
choice_policy("route", run={"billing"}, skip={"general"}, min_confidence=0.6)
policy = lambda r: "run" if r["urgency"].score >= 1.5 else "skip"   # your own
```

### `skip_output`

When the node is skipped, `skip_output(input, outcome)` produces the output instead.
It must match the node's output contract — for a LangGraph node that is usually an
empty state update:

```python
skip_output=lambda state, outcome: {}
```

The `outcome` is a `GateOutcome` describing *why* the node was skipped: `action`,
`reason` (`"policy"`, `"uncertain"`, `"inference_error"`), `decision`, `policy_result`,
and `error` — the fields are documented on the class in the
[API reference](api.md#langchain_s1.gate.GateOutcome).

### Fallbacks: `on_uncertain` and `on_error`

| Situation | Default | Alternatives |
|---|---|---|
| Policy says `uncertain` | `"raise"` — `S1UncertainDecisionError` | `"run"` (fail open), `"skip"` (fail closed) |
| Inference failed after all retries | `"raise"` — the `S1InferenceError` | `"run"` (fail open), `"skip"` (fail closed) |

```python
# Fail open: always answer, even when unsure or offline.
s1_gate(draft_reply, decision, policy=policy,
        skip_output=lambda s, o: {}, on_uncertain="run", on_error="run")

# Fail closed: when in doubt, skip.
s1_gate(draft_reply, decision, policy=policy,
        skip_output=lambda s, o: {}, on_uncertain="skip", on_error="skip")
```

## Guarantees

- The wrapped node runs **at most once** per gate invocation. Only the inference
  request is retried (by `S1Client`); the node is never replayed because of
  inference or tracing failures.
- Each gate invocation emits an `s1_gate` custom callback event carrying the decision
  and the resolved action — inspectable in any LangChain callback handler.
  Credentials are never included.

```python
from langchain_core.callbacks import BaseCallbackHandler

class GateWatcher(BaseCallbackHandler):
    def on_custom_event(self, name, data, **kwargs):
        if name == "s1_gate":
            print(data["action"], data["reason"])   # e.g. "skip" "policy"
```

## Next

- [Graph routing](graphs.md) — route a whole LangGraph with a decision.
- [Configuration](configuration.md) — every option, in one place.
