# langchain-s1
A langchain wrapper for s1 decisions

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
- TypeScript → npm via trusted publishing (configure an npm trusted publisher for `release-please.yml`).
