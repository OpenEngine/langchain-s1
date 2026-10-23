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

```python
s1_gate(
    node,          # the existing runnable (or any callable) to gate
    decision,      # an S1Decision — or any runnable returning an S1Result
    *,
    policy,        # S1Result -> "run" | "skip" | "uncertain"
    skip_output,   # (input, outcome) -> output, when the node is skipped
    on_uncertain="raise",   # what "uncertain" resolves to
    on_error="raise",       # what an inference failure resolves to
    name=None,              # run name for tracing
)
```

### Policies

A policy maps an `S1Result` to `"run"`, `"skip"`, or `"uncertain"` (or a `GateAction`).
Two built-ins cover the common cases; any callable works.

```python
# noul: threshold band
noul_policy("needs_reply", run_at=0.7, skip_below=0.3)
# P(yes) >= 0.7  -> run
# P(yes) <  0.3  -> skip
# otherwise      -> uncertain

# choice: which choices run, which skip
choice_policy("route", run={"billing"}, skip={"general"}, min_confidence=0.6)
# choice in run, confidence >= min_confidence -> run
# choice in skip                              -> skip
# anything else                               -> uncertain

# your own
policy = lambda r: "run" if r["urgency"].score >= 1.5 else "skip"
```

### `skip_output`

When the node is skipped, `skip_output(input, outcome)` produces the output instead.
It must match the node's output contract — for a LangGraph node that is usually an
empty state update:

```python
skip_output=lambda state, outcome: {}
```

The `outcome` is a `GateOutcome` describing *why* the node was skipped:

| Field | Values |
|---|---|
| `action` | the resolved action: `"run"`, `"skip"`, or `"raise"` |
| `reason` | `"policy"`, `"uncertain"`, or `"inference_error"` |
| `decision` | the `S1Result` (`None` on inference failure) |
| `policy_result` | the raw policy verdict, if one was reached |
| `error` | the exception, when inference failed |

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
