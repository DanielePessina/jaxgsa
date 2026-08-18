# Numerical baseline

`baseline-0.8.0.json` is a fixed-seed record of what jaxgsa computed before
the 0.9 refactors started. It exists for gate 5 of the review protocol in
`PLAN-V1.0.md` section 3.1.

**A changed number is a wiring error, not a tolerance issue.** The refactors
that this file guards are declared "plumbing only": they move code, rename
fields, and share helpers. None of them may change an index. There is no
tolerance in the comparison, and adding one would defeat the purpose. If a
number moves, find the mis-wired call before you touch this file.

## How to use it

Re-run the check after every batch:

```
uv run scripts/baseline_check.py
```

It re-runs the same analyses and diffs them against the stored file. It prints
every changed field with the old value, the new value and the difference, and
it exits non-zero on any change. A clean run takes about one minute.

Write a new baseline only when a change to a number is intended and reviewed:

```
uv run scripts/baseline_dump.py
```

Confirm that the dump is reproducible with `--twice`. That builds it two times
in one process and fails if the two differ.

## What is covered

Five problems, chosen for the shapes and the input features that the 0.9
refactors touch:

| Case | Problem | Output shape |
| --- | --- | --- |
| `ishigami_scalar` | Ishigami, 3 uniform inputs | `(N,)` |
| `sobol_g_multi` | Sobol g-function, 8 uniform inputs | `(N, 2)` |
| `ishigami_series` | Ishigami, widened over time | `(N, 3, 2)` |
| `ishigami_correlated` | Ishigami with a declared correlation | `(N,)` |
| `categorical_mixed` | 1 uniform and 1 three-level categorical input | `(N,)` |

All thirteen methods run against every case: `borgonovo`, `dgsm`, `efast`,
`hdmr`, `hsic`, `kucherenko`, `morris`, `optimal_transport`, `pawn`, `pce`,
`shapley`, `sobol`, `vkoga`. A method that refuses a case is recorded as
`{"status": "raised", "exception": "ValueError"}`. That is data too: a
refactor that removes a gate shows up as a status change. The exception
message is deliberately not recorded, because wording is allowed to change.

Each case also records the Monte Carlo design and the model output it was
built from, so a change in the samplers is caught as well as a change in the
estimators.

## What is in the file

- `header`: versions, platform, x64 flag, git commit, seed. Never compared.
- `results`: one entry per case, holding every field of every result
  dataclass. Traversal uses `dataclasses.fields`, so a field added later is
  picked up without editing the script.

Arrays of at most 1024 elements are written out element by element, as Python
float `repr`, which round-trips exactly. A larger array is stored as a
SHA-256 digest of its bytes. The comparison is bit-for-bit either way; only
the diff report is coarser for the large ones.

The file contains bare `NaN` tokens, which Python's `json` reads and writes
but a strict JSON parser rejects. The `NaN` entries are the unused cells of
the `S2` matrices. The check treats `NaN` as equal to `NaN`.

## Determinism

Every method in 0.8.0 reproduces bit-for-bit at a fixed seed. Verified on
2026-08-18 with `uv run scripts/baseline_dump.py --twice`. The exclusion list
`EXCLUDED_FIELDS` in `scripts/baseline_dump.py` is therefore empty. If a
future method is not reproducible, add its field there with a comment saying
why, rather than dropping it quietly.

The baseline is machine-specific. It was recorded on an Apple M1 Pro, CPU
backend, single precision (`jax_enable_x64` is off). Compare on the same
machine and the same JAX version. A different CPU or a different JAX release
can change the last bits for reasons that have nothing to do with the
refactor.
