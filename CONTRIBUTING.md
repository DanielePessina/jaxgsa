# Contributing to gsax

Thanks for your interest in improving gsax! This guide covers the local setup
and the checks a pull request is expected to pass.

## Development setup

gsax uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
git clone https://github.com/DanielePessina/gsax.git
cd gsax
uv sync --extra dev
```

## Before opening a pull request

Run the same checks CI runs, and make sure they all pass:

```bash
uv run ruff format .          # format
uv run ruff check .           # lint
uv run ty check src/gsax      # type-check
uv run pytest                 # tests
```

Please also:

- Add or update tests for any behaviour you change.
- Keep public APIs documented with Google-style docstrings.
- Update `CHANGELOG.md` under the unreleased section when user-facing behaviour
  changes.
- If you touch the public API, run `uv run python scripts/check_api_docs_coverage.py`
  so the reference docs stay complete.

## Reporting issues

Please open an issue with a minimal reproducible example, the gsax version, and
your Python/JAX versions.
