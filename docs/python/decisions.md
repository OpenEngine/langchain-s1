# Python: Standalone decisions

An `S1Decision` asks named questions about a selected view of its input and returns an
`S1Result`. All questions are sent in **one** request, so asking three things costs
one inference. The complete worked example (client, all three question types, reading
results) is in [Getting started](getting-started.md); the shape that matters on this
page is the `select_state` bridge:

```python
decision = S1Decision(
    client,
    {"needs_reply": NoulQuestion("Does the last message need a reply?")},
    select_state=lambda state: {"last_message": state["messages"][-1]},
)

result = decision.invoke({"messages": ["I want a refund"]})
```

## `select_state`: what leaves your process

`select_state(input) -> str | dict | list` is the only bridge between your data and the
wire. Everything it does not return is never sent — keep it minimal, and strip anything
sensitive.

```python
# Only the last message goes out; the conversation history stays local.
select_state=lambda state: {"last_message": state["messages"][-1]}

# A plain string state is fine too.
select_state=lambda state: state["email_body"]
```

## The three question types

### `NoulQuestion` — yes/no

```python
NoulQuestion("Does the last message need a reply?")
NoulQuestion(
    "Is this message abusive?",
    criteria={"true": "Profanity, threats, or harassment", "false": "Everything else"},
)
```

Answer: `NoulAnswer` with a single field, `noul` — the **probability of "yes"**.
It is a probability, not a confidence: `0.92` means the model says "yes" with
probability 0.92.

### `ChoiceQuestion` — one of many

```python
ChoiceQuestion(
    "Which team should handle the last message?",
    {"billing": "Payments, refunds, invoices", "general": "Anything else"},
)
```

`criteria` maps choice names to descriptions (1–255 choices). Answer: `ChoiceAnswer`
with `choice` (a criteria key), `confidence`, and `probabilities` (per option).

### `ScoreQuestion` — a scale

```python
ScoreQuestion("How urgent is the last message?", ["low", "medium", "high"])
```

`criteria` lists level descriptions from level 0 upward (1–10 levels). Answer:
`ScoreAnswer` with `score` (the probability-weighted level, e.g. `1.2`),
`confidence`, `probabilities` (keyed by level number), and `legend`
(level number -> description).

## Reading results

The result is an `S1Result`: a mapping from question name to its typed answer (the
field-by-field tour is in [Getting started](getting-started.md#4-read-the-answers)).
It behaves like a dict of answers, and `to_dict()` returns the raw contract shape:

```python
for name in result:
    print(name, result[name].to_dict())

result.to_dict()   # {"model", "answers", "usage"}
```

## Runnable behavior

`S1Decision` is a standard LangChain runnable:

```python
# Async
result = await decision.ainvoke(state)

# Batch (each item is one request; select_state applies per item)
results = await decision.abatch([state_a, state_b, state_c])

# Compose: write the answer into downstream state
to_update = decision | (lambda r: {"team": r["route"].choice})
pipeline = decision | (lambda r: r["route"].choice) | some_other_runnable
```

Runnable config (callbacks, tags, metadata, run name) is propagated; pass `name=` to
`S1Decision(...)` to label the run in tracing.

!!! warning
    Answers are validated against the questions you asked. A response that is missing
    an answer, has a mismatched type, an unknown choice, an out-of-range score, or
    probabilities outside [0, 1] raises `S1ResponseError` naming the offending field —
    see [Errors](errors.md).

## Next

- [Gated nodes](gates.md) — let a decision decide whether existing work runs at all.
- [Configuration](configuration.md) — client options in full.
