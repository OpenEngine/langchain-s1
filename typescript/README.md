# langchain-s1 (TypeScript)

A langchain wrapper for s1 decisions.

```bash
npm install langchain-s1
```

The TypeScript package mirrors the [Python package](../python/README.md): the same
primitives (`S1Client`, `S1Decision`, `s1Gate`) speaking the
[TypeSafe System One API](https://docs.typesafe.ai/api/system-one) contract
(`{model, state, questions}` → `{model, answers, usage}`).

The API reference is generated from this package's source with
[TypeDoc](https://typedoc.org/) on every build and published on the
[docsite](https://openengine.github.io/langchain-s1/typescript/api/).

## Development

```bash
npm ci
npm run build
npx typedoc --out docs-api   # local API reference
```
