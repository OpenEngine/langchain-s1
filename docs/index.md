# langchain-s1

LangChain wrappers for [System 1 decisions](https://docs.typesafe.ai/api/system-one),
in Python and TypeScript.

A System 1 ("S1") model answers a few fast, cheap questions about a piece of state in
one request — *"does this need a reply?"*, *"which team should handle this?"*,
*"how urgent is this?"*. `langchain-s1` turns those answers into LangChain primitives:

| Primitive | What it is |
|---|---|
| `S1Client` | A typed HTTP client for **any** endpoint speaking the System One contract, with timeouts, retries, and credential redaction. |
| `S1Decision` | A LangChain *runnable* that asks named questions about a selected view of its input and returns typed answers. |
| `s1_gate` | Wraps an existing runnable so it runs — or is skipped — based on a decision and an explicit policy. |

## Packages

| Package | Install | Documentation |
|---|---|---|
| Python | `pip install langchain-s1` | [Getting started](python/getting-started.md) · [API reference](python/api.md) |
| TypeScript | `npm install langchain-s1` | [Guide](typescript/guide.md) · [API reference](typescript/api.md) |

## The contract in one exchange

Everything routes through one request/response pair. Any endpoint that speaks it —
TypeSafe's hosted API, a self-hosted model, an internal gateway — works through
configuration alone:

```json
// request: POST {base_url}/systemone
{
  "model": "jev-latest",
  "state": { "last_message": "I want a refund" },
  "questions": {
    "needs_reply": { "type": "noul", "instructions": "Does the last message need a reply?" },
    "route": {
      "type": "choice",
      "instructions": "Which team should handle the last message?",
      "criteria": { "billing": "Payments, refunds, invoices", "general": "Anything else" }
    }
  }
}

// response
{
  "model": "jev-latest",
  "answers": {
    "needs_reply": { "type": "noul", "noul": 0.92 },
    "route": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.85,
      "probabilities": { "billing": 0.85, "general": 0.15 }
    }
  },
  "usage": { "input_tokens": 128, "output_tokens": 24 }
}
```

`langchain-s1` validates that response field by field and hands you typed answers:

```python
result = decision.invoke({"last_message": "I want a refund"})
result["needs_reply"].noul   # 0.92 — probability of "yes"
result["route"].choice       # "billing"
result["route"].probabilities
```

## Where to start

- **New here?** [Python getting started](python/getting-started.md) — install, configure a
  client, run your first decision in five lines.
- **Cut costs?** [Gated nodes](python/gates.md) — run the expensive step only when S1 says
  it is worth it.
- **Wiring a graph?** [Conditional routing](python/graphs.md).
- **What can I configure?** [Configuration](python/configuration.md) — every client,
  policy, and gate option.
- **No credentials?** [Testing](python/testing.md) — an in-process mock endpoint.
- **How is it built?** [Architecture](architecture.md) — the code structure of the
  Python package, diagram included.
- **Browsing symbols?** [Python API reference](python/api.md) and
  [TypeScript API reference](typescript/api.md), generated from source on every build.
