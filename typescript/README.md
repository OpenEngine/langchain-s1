# langchain-s1 (TypeScript)

A langchain wrapper for s1 decisions.

```bash
npm install langchain-s1
```

The TypeScript port of the [Python package](../python/README.md) is in progress: the
same primitives (`S1Client`, `S1Decision`, `s1Gate`) will speak the
[TypeSafe System One API](https://docs.typesafe.ai/api/system-one) contract
(`{model, state, questions}` → `{model, answers, usage}`). Today the package ships
only the `VERSION` constant — the planned surface is described in the
[TypeScript guide](https://openengine.github.io/langchain-s1/typescript/guide/).

The API reference is generated from this package's source with
[TypeDoc](https://typedoc.org/) on every build and published on the
[docsite](https://openengine.github.io/langchain-s1/typescript/api/).

## Development

```bash
npm ci
npm run build
npx typedoc --out docs-api   # local API reference
```
