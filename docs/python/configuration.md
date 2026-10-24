# Python: Configuration

Everything is configuration: the same `S1Decision` and `s1_gate` code runs against
TypeSafe's hosted API, a local model, or an internal gateway — only the client
changes.

## `S1Client` options

The full option list, with defaults, is the
[`S1Client` entry in the API reference](api.md#langchain_s1.client.S1Client) —
generated from the docstring, so it never drifts from the code. The three common
setups, all configuration only:

```python
import os
from langchain_s1 import S1Client

# Hosted, bearer auth, longer timeout.
client = S1Client(
    "jev-latest",
    api_key=os.environ["TYPESAFE_API_KEY"],
    timeout=60.0,
)

# Local endpoint, no auth, no retries (fail fast in dev).
client = S1Client("my-local-s1", base_url="http://localhost:8000", max_retries=0)

# Internal gateway with a custom header and an explicit endpoint path.
client = S1Client(
    "my-model",
    endpoint_url="https://gateway.example/teams/s1/decide",
    headers={"X-Gateway-Key": os.environ["GATEWAY_KEY"]},
)
```

### Endpoint resolution

1. If `endpoint_url` is set, it is used as-is.
2. Otherwise `base_url` (default `https://api.typesafe.ai/v1`) has `/systemone`
   appended.

`repr(client)` never includes credentials or header values, and URLs in error
messages are redacted (userinfo, query, and fragment stripped).

## Question construction

| Question | Required | Optional | Constraints |
|---|---|---|---|
| `NoulQuestion(instructions)` | non-empty `instructions` (str, object, or array) | `criteria`: `{"true": ..., "false": ...}` | criteria keys must be exactly `true`/`false` |
| `ChoiceQuestion(instructions, criteria)` | both | — | `criteria`: 1–255 name -> description pairs |
| `ScoreQuestion(instructions, criteria)` | both | — | `criteria`: 1–10 level descriptions, index = level |

Contract-shaped dicts are accepted anywhere a typed question is:
`{"type": "choice", "instructions": "...", "criteria": {...}}`. Unsupported types
(`box`, image questions) and extra fields raise `S1UnsupportedFeatureError`
immediately, before any request.

## `S1Decision` options

| Option | Description |
|---|---|
| `client` | The `S1Client` to use. |
| `questions` | Non-empty mapping of question name -> question. |
| `select_state` | `(input) -> str \| dict \| list`. The only thing sent to the endpoint. |
| `name` | Run name for tracing (default `"S1Decision"`). |

## `s1_gate` options

| Option | Default | Description |
|---|---|---|
| `node` | *(required)* | The runnable (or callable) to gate. |
| `decision` | *(required)* | An `S1Decision` — or any runnable returning an `S1Result`. |
| `policy` | *(required)* | `S1Result -> "run" \| "skip" \| "uncertain"`. |
| `skip_output` | *(required)* | `(input, GateOutcome) -> output`. Must match the node's output contract. |
| `on_uncertain` | `"raise"` | What an uncertain policy resolves to: `"run"`, `"skip"`, or `"raise"`. |
| `on_error` | `"raise"` | What an inference failure resolves to: `"run"`, `"skip"`, or `"raise"`. |
| `name` | `s1_gate(<node name>)` | Run name for tracing. |

### Policy options

`noul_policy(question, run_at=, skip_below=)` and
`choice_policy(question, run=, skip=, min_confidence=0.0)` map an answer to run/skip.
The exact threshold rules are in the
[API reference](api.md#langchain_s1.gate.noul_policy).

## Environment variables

There are no magic environment variables — pass credentials explicitly
(`api_key=os.environ["TYPESAFE_API_KEY"]`) so configuration stays visible and testable.

## Next

- [Testing](testing.md) — configure a client against the in-process mock.
- [Errors](errors.md) — what each failure looks like.
