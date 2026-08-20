# Numerical baseline

`baseline-1.0.0.json` is the file the check runs against. It is a fixed-seed
record of what jaxgsa computes. It exists so that a change with no
behavioural target can be proved not to have moved a number — see
`docs/adr/0009-a-change-with-no-behavioural-target.md`.

`baseline-0.8.0.json` is kept beside it as the earlier record. Nothing reads
it; it is there so the one place a number was allowed to move stays auditable.

**A changed number is a wiring error, not a tolerance issue.** Most of the
work this file guards is declared "plumbing only": it moves code, renames
fields, and shares helpers. None of that may change an index. There is no
tolerance in the comparison, and adding one would defeat the purpose. If a
number moves, find the mis-wired call before you touch this file.

## The reviewed exceptions

Three changes have been allowed to move numbers. All are recorded here because
the whole value of this file is that a moved number is a defect until someone
writes down why it is not.

### 1. Sobol standardizes its outputs

`sobol.analyze` used to leave the output as given unless you passed
`prenormalize=True`. The Saltelli/Jansen `S1` and every estimator's `S2` are
uncentred products, so a non-zero output mean injects an error proportional to
that mean. SALib has always standardized unconditionally
(`SALib/analyze/sobol.py`, `Y = (Y - Y.mean()) / Y.std()`); jaxgsa made that
optional and defaulted it off.

183 sobol values moved. The proof that this is the intended fix and not a
wiring error is the asymmetry: `S1` moved by up to **3.6e-1**, while `ST`
moved by **1.2e-7**. Jansen's total-order estimator is a difference and is
therefore already shift-invariant, so it should see nothing but float32
re-association from the changed operand magnitudes. It does not.

Validated against the closed-form Ishigami values rather than against this
file, because this file recorded the behaviour being corrected.

### 2. VKOGA and PAWN draw independent RNG streams

VKOGA derived per-stream seeds by arithmetic (`seed + 1 + i`, `seed + 7919`).
Offset seeds are not independent streams. It now spawns children from one
`numpy.random.SeedSequence` root. PAWN's harness carried a matching arbitrary
`+1`.

Moved: VKOGA's six index fields, and `pawn.pawn_conf`. **`pawn.pawn` did not
move**, and VKOGA's `gamma`, `ridge`, `n_centers`, `rmse` and `cv_rmse` are
bit-identical — the surrogate is untouched and the movement is the integration
reseed alone. The new values sit inside the old estimator's own key-to-key
spread, measured over 8 keys.

### 3. DGSM selects the autodiff mode by shape (2026-08-20)

`dgsm` used to hard-code `jax.jacrev`. It now selects `jax.jacfwd` when the
output slices outnumber the inputs (`T*K > D`) and `jax.jacrev` otherwise.
This closes `docs/adr/0005-autodiff-mode-selection.md`. The two modes compute
the same Jacobian; only the order of the float arithmetic differs.

Moved: 24 dgsm values, all in `ishigami_series` — the one case with
`T*K = 6 > D = 3`, so the one case where the mode changed. Only `sigma`
(12 of 18 elements, deltas about 1e-8 absolute) and `lower_bound` (12 of 18,
deltas about 1e-10) moved. `nu`, `upper_bound` and `var_y` are bit-identical,
and every other case is untouched. This is exactly the movement the old
`jacobian_of` docstring predicted when the flip was first measured and
deferred.

The same regeneration folds in the 24 schema additions from the 1.0 features
PR as the new recorded surface: `optimal_transport` gained `S1`, `S1_conf`
and `above_dummy`, and `pawn` gained `n_valid_bins`, on each of the six
cases. Those are added fields, not moved numbers.

### 4. PCE explained_variance measured two different things (2026-08-20)

`explained_variance` divided the surrogate's variance under the input
measure, from Parseval on the coefficients, by the sample variance of `Y`.
Those are two different measures. At finite `N` the empirical Gram is not the
identity, so Parseval overstates the surrogate's variance on the sample and
the ratio could pass 1. The numerator is now the sample variance of the
fitted values, so both sides measure the same rows, and the field is a
coefficient of determination.

The old definition was wrong rather than merely differently scaled, and the
baseline recorded that wrongness as ground truth: the `gaussian_mixed` entry
held `1.0284`, a share of explained variance above 1, which is not reachable.

Moved: 20 values, every one an `explained_variance`, on `pce` and on
`shapley`, which carries the PCE backend's diagnostic through unchanged. All
move downward. `gaussian_mixed` `1.0284 -> 0.9917`, `ishigami_scalar`
`0.47374 -> 0.47261`, `ishigami_series` the same shift across its six slices,
`sobol_g_multi` `0.89231 -> 0.87967` across its two. No `S1`, `ST`, `S2` or
`Sh` value moved, because the Shapley normalizer reads
`explained_variance` only as a finiteness mask.

Because a corrected R-squared cannot exceed 1, the Shapley overfit warning
that fired on `explained_variance` crossing a threshold became unreachable
for the PCE backend. It now reads `loo_rmse` against `std(Y)` instead. The
HDMR backend keeps the old check, whose `sum(V_u) / Var(Y)` genuinely can
exceed 1.

## The float32 batch-width exception

Kernel work is not plumbing. Batching an estimator over more output slices at
once is the whole speedup, and XLA schedules a float32 reduction differently
at a different batch width, so the last bits move. That is a property of the
hardware, not a choice in the code.

Regenerating `baseline-1.0.0.json` absorbed exactly one such change: 44 sobol
values, every delta 1 to 4 units in the last place of float32, from the
bootstrap resampler moving to a chunked `vmap` over slices. Two facts made it
reviewable rather than waved through. Feeding the *old* resampler from the
new flattened arrays is bit-for-bit identical, so the layout change is not
the cause; and the only batch width that reproduces the old bits is one
holding a single slice, which is the same as not batching at all.

One of the 44 is a fix. `sobol_g_multi.sobol.S2` moved because the bootstrap
point estimate now runs the same kernel as the plain path. Before, the two
disagreed by 1.4e-7 for one design, so an interval was centred on a number
`analyze(num_resamples=0)` never reported.

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
