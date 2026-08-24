# Contributing to jaxgsa

Thanks for your interest in improving jaxgsa. This guide covers the local setup
and the checks your pull request must pass.

## Development setup

jaxgsa uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
git clone https://github.com/DanielePessina/jaxgsa.git
cd jaxgsa
uv sync --extra dev
```

## Before opening a pull request

Run the same checks CI runs, and make sure they all pass:

```bash
uv run ruff format .          # format
uv run ruff check .           # lint
uv run ty check src/jaxgsa      # type-check
uv run pytest                 # tests
```

Please also:

- Add or update tests for any behaviour you change.
- Keep public APIs documented with Google-style docstrings.
- Update `CHANGELOG.md` under the unreleased section when user-facing behaviour
  changes.
- If you touch the public API, run
  `uv run python scripts/check_api_docs_coverage.py`. This keeps the reference
  docs complete.

## Adding a method

A new sensitivity method touches eight places. Do all of them in one pull
request. The **Guarded** column says what happens if you forget: a guarded
touchpoint fails a test, a convention touchpoint fails nothing.

| # | Touchpoint | What to add | Guarded by |
|---|---|---|---|
| 1 | `src/jaxgsa/<method>/__init__.py` | `SPEC = register(MethodSpec(...))` at module level | `tests/test_registry.py` — it walks the source tree, so an unregistered package fails |
| 2 | `src/jaxgsa/<method>/_analyze.py` | The analysis function the spec names | Convention. Name the module what you like; the registry holds the function |
| 3 | `src/jaxgsa/<method>/_result.py` | The result class, with a `_schema = ResultSchema(...)` | `tests/test_result_schema.py` |
| 4 | `src/jaxgsa/__init__.py` | Import the package, add its name to `__all__` | `tests/test_registry.py` and `scripts/check_api_docs_coverage.py` |
| 5 | `docs/guide/methods.md` | One row in the table under `### Method capabilities` | `tests/test_docs_matrix.py` — the Own design, Correlated, Categorical and Bootstrap CI cells are read back against the spec; Reports is only checked for being non-empty |
| 6 | `docs/api/index.md` | One row in the method table, plus a reference heading | `tests/test_docs_matrix.py` and `scripts/check_api_docs_coverage.py` |
| 7 | `tests/_result_fixtures.py` | A builder in `BUILDERS`, then run `uv run scripts/dump_result_schema.py` to regenerate `tests/data/result_dataset_schema.json` | `tests/test_result_schema.py` |
| 8 | `scripts/baseline_dump.py` | A runner in `DESIGN_METHODS` (own sampler) or `GIVEN_DATA_METHODS` (given data), then regenerate the stored baseline | `tests/test_baseline_check.py` |

Also update `CHANGELOG.md`, and read these two notes before you write the
result class:

- Declare a capability only if the method proves it. `tests/test_registry.py`
  runs the method on a correlated problem and on a categorical problem. An
  `"accepts"` claim passes only when the call completes.
- Pick the right axis for every field. A field that is as long as the
  parameter list but is not indexed by parameter takes `axes="index"`, not
  `axes="param"`. See `Axes` in `src/jaxgsa/_core/result.py`.

## Reporting issues

Please open an issue with a minimal reproducible example, the jaxgsa version, and
your Python/JAX versions.
