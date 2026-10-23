# TypeScript: Guide

`langchain-s1` on npm is the TypeScript twin of the Python package: the same
primitives, the same System One contract
(`{model, state, questions}` → `{model, answers, usage}`), so anything written here
translates directly to the [Python guide](../python/getting-started.md).

```bash
npm install langchain-s1
```

## Current surface

The TypeScript package is still a thin port — it currently ships the version constant,
and the API surface (`S1Client`, `S1Decision`, `s1_gate`, the question/answer types,
and the error hierarchy) lands alongside it. The API reference on this site is
**generated from the package source on every build**, so it grows in lockstep with the
port: nothing is maintained by hand.

## What it will look like

The Python API is the specification. The intended correspondence:

| Python | TypeScript (planned) |
|---|---|
| `S1Client` | `S1Client` — same options (`model`, `base_url`/`endpoint_url`, `api_key`, `headers`, `timeout`, retries) |
| `NoulQuestion` / `ChoiceQuestion` / `ScoreQuestion` | `NoulQuestion` / `ChoiceQuestion` / `ScoreQuestion` |
| `S1Decision` (runnable) | `S1Decision` — a LangChain.js `Runnable` |
| `s1_gate` + `noul_policy` / `choice_policy` | `s1Gate` + `noulPolicy` / `choicePolicy` |
| `S1Result`, `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`, `Usage` | the same named types |
| `S1*` errors | the same error hierarchy |
| `MockSystemOne` | a fetch-mock equivalent for tests |

Until the port lands, use the Python package — the contract it speaks is
language-neutral.

## API reference

[TypeDoc](https://typedoc.org/) generates the reference from `typescript/src` on every
build. See the [TypeScript API reference](api.md).
