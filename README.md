# langchain-s1
A langchain wrapper for s1 decisions

## Documentation

The docsite — usage guides, configuration reference, examples, and the auto-generated
API reference for both packages — is published on GitHub Pages:
<https://openengine.github.io/langchain-s1/>. It is rebuilt from source on every push
to `main` (`.github/workflows/docsite.yml`); the Python reference comes from the
docstrings via mkdocstrings, the TypeScript reference from TypeDoc.

## Packages

- `python/` — the `langchain-s1` PyPI package
- `typescript/` — the `langchain-s1` npm package

## Releasing

Releases are managed by [release-please](https://github.com/googleapis/release-please).
Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `feat!:`) on `main`.
release-please opens a release PR per package (`python`, `typescript`) that bumps the version and
updates its `CHANGELOG.md`. Merging that PR tags the release (`python-vX.Y.Z` / `typescript-vX.Y.Z`),
creates a GitHub release, and publishes the package:

- Python → PyPI via trusted publishing (configure the `pypi` environment and a PyPI trusted publisher for `release-please.yml`).
- TypeScript → npm via trusted publishing (configure the `npm` environment and an npm trusted publisher for package `langchain-s1`, workflow `release-please.yml`, environment `npm`).
