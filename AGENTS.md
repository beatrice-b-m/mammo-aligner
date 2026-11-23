# AGENTS.md

Guidance for agents and contributors working in this repository.

## Scope

`mammo-aligner` is a minimal library for checking the orientation of mammography
images and realigning them, and coordinates on them, into a common frame. It
targets synthetic 2D (S2D), full-field digital mammography (FFDM), and digital
breast tomosynthesis (DBT) images. Keep it small, clear, and focused on that
job; anything outside it (dataset loaders, preprocessing, model code) belongs
elsewhere.

## Working rules

- **Commits**: track changes in informatively scoped commits — one logical
  change per commit, with a message that says what changed and why.
- **Docs stay in sync**: when behavior or the public API changes, update the
  README and docstrings in the same commit.
- **Tests**: keep them minimal and focused on behavior and edge cases (round
  trips, flips, unknown or malformed metadata, bounds). Do not hardcode
  internal semantics or in-development conventions into tests.
- **Simplicity**: prefer a small public surface and few dependencies. Add a
  module, class, or dependency only when it earns its place.

## Environment

The project uses [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync          # create the environment and install dev dependencies
uv run pytest    # run the tests
```
