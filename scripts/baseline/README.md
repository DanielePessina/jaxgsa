# Numerical baseline

`baseline-1.0.0.json` is the file the check runs against. It is a fixed-seed
record of what jaxgsa computes. It exists so that a change with no
behavioural target — a rename, a shared helper, a reordered call — can be
proved not to have moved a number.

`baseline-0.8.0.json` is kept beside it as the earlier record. Nothing reads
it; it is there so the one place a number was allowed to move stays auditable.

**A changed number is a wiring error, not a tolerance issue.** Most of the
work this file guards is declared "plumbing only": it moves code, renames
fields, and shares helpers. None of that may change an index. There is no
tolerance in the comparison, and adding one would defeat the purpose. If a
number moves, find the mis-wired call before you touch this file.

## The reviewed exceptions

Several changes have been allowed to move numbers. All are recorded here
because the whole value of this file is that a moved number is a defect
until someone writes down why it is not.

Every count below was re-measured by diffing the two committed JSON files
element by element with `scripts/baseline_check.py`'s own `compare()`
function (the same code the check itself runs), not estimated. A count is
"raw scalar array cells": one number for a plain field, and one number per
differing cell of an array field. `S2` and `S2_conf` are stored as full
`(D, D)` matrices with the upper triangle mirrored into the lower one before
this release (see the Sobol `S2` fix below), so a change to one off-diagonal
value there is counted twice — once at `[i, j]` and once at `[j, i]`. That
convention is stated once here rather than repeated per entry.

### 1. Sobol standardizes its outputs (commit `784746a`)

`sobol.analyze` used to leave the output as given unless you passed
`prenormalize=True`. The Saltelli/Jansen `S1` and every estimator's `S2` are
uncentred products, so a non-zero output mean injects an error proportional to
that mean. SALib has always standardized unconditionally
(`SALib/analyze/sobol.py`, `Y = (Y - Y.mean()) / Y.std()`); jaxgsa made that
optional and defaulted it off.

**677 sobol values moved**, not 183: `S1` 39, `ST` 34, `S2` 156 (78 unique,
mirrored), `S1_conf` 78, `ST_conf` 58, `S2_conf` 312 (156 unique, mirrored).
The stated 183 undercounted because it did not include the confidence-interval
fields, which move for the same reason the point estimate does — the bootstrap
resamples the same centred output. The proof that this is the intended fix and
not a wiring error is the asymmetry: `S1` moved by up to **3.6e-1**, while `ST`
moved by **1.2e-7**. Jansen's total-order estimator is a difference and is
therefore already shift-invariant, so it should see nothing but float32
re-association from the changed operand magnitudes. It does not.

Validated against the closed-form Ishigami values rather than against this
file, because this file recorded the behaviour being corrected.

### 2. VKOGA and PAWN draw independent RNG streams (commit `784746a`)

VKOGA derived per-stream seeds by arithmetic (`seed + 1 + i`, `seed + 7919`).
Offset seeds are not independent streams. It now spawns children from one
`numpy.random.SeedSequence` root. PAWN's harness carried a matching arbitrary
`+1`. Bundled into the same commit as item 1 above, so the same regeneration
covers both.

**Moved: 210 VKOGA values** (`S_C`, `S_IU`, `S_TC`, `S_TU`, `S_U`, `variance`,
35 each on average across the six cases) **and 84 `pawn.pawn_conf` values**
(the confidence-interval bounds, 2 per parameter per case). **`pawn.pawn` did
not move**, and VKOGA's `gamma`, `ridge`, `n_centers`, `rmse` and `cv_rmse` are
bit-identical — the surrogate is untouched and the movement is the integration
reseed alone. The new values sit inside the old estimator's own key-to-key
spread, measured over 8 keys.

### 3. HDMR's fit path was unified (commit `315effe`, 2026-08-19)

Not recorded here until now — the gap this file's own rule warns against. The
commit's own message named the cause and a bound ("223 values move ... none
exceeding 2.4e-5 relative to its own scale ... old and new implementations
agree to 8e-15 under `jax_enable_x64`") but that reasoning never reached this
file. This entry also introduced the `gaussian_mixed` case (a 2-sigma
truncated Gaussian, an unbounded Gaussian, and a uniform control), which is
why the moved-value count below is confined to the four cases that already
existed.

**1214 raw hdmr values moved, plus one array recorded as a changed hash**
(`sobol_g_multi.hdmr._fit.C2`, too large — `(2, 25, 28)` — to store
element-wise): `S` 118, `ST` 40, `Sa` 120, `Sb` 120, `rmse` 9, and the rest in
the internal `_fit.C1`/`_fit.C2`/`_fit.f0` coefficient arrays that back them
(688 cells across the four cases, plus the one hashed array). The commit
message's "223" evidently counted only the user-facing `S`/`ST`/`Sa`/`Sb`/
`rmse` fields and not the internal `_fit` coefficients, which this file's own
comparison does not distinguish from any other field. Re-measured maximum
absolute deltas on the user-facing fields: `S` 5.6e-6, `ST` 7.9e-6, `Sa`
9.4e-6, `Sb` 1.13e-5 — all far below any tolerance that would matter to a
caller, and consistent with the commit's own "conditioning, not a different
computation" claim. `_fit.C1` moved by up to 6.7e-4, `_fit.C2` by up to
4.1e-3 — both internal, both undocumented for a caller.

### 4. DGSM selects the autodiff mode by shape (commit `6c2a151`, 2026-08-20)

`dgsm` used to hard-code `jax.jacrev`. It now selects `jax.jacfwd` when the
output slices outnumber the inputs (`T*K > D`) and `jax.jacrev` otherwise.
The two modes compute the same Jacobian; only the order of the float
arithmetic differs.

Moved: 24 dgsm values, all in `ishigami_series` — the one case with
`T*K = 6 > D = 3`, so the one case where the mode changed. Only `sigma`
(12 of 18 elements, deltas about 1e-8 absolute) and `lower_bound` (12 of 18,
deltas about 1e-10) moved. `nu`, `upper_bound` and `var_y` are bit-identical,
and every other case is untouched. This is exactly the movement the old
`jacobian_of` docstring predicted when the flip was first measured and
deferred. Re-measured directly against the two committed JSON files: 24 is
correct as written here.

The same regeneration folds in the 24 schema additions from the 1.0 features
PR as the new recorded surface: `optimal_transport` gained `S1`, `S1_conf`
and `above_dummy`, and `pawn` gained `n_valid_bins`, on each of the six
cases. Those are added fields, not moved numbers.

### 5. PCE explained_variance measured two different things (2026-08-20)

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

Regenerating `baseline-1.0.0.json` at commit `c7d0269` absorbed exactly one
such change: **186 sobol values**, every delta 1 to 4 units in the last place
of float32, from the bootstrap resampler moving to a chunked `vmap` over
slices — not 44. The 44 undercounted the same way item 1 above did: it
covered only `S2` (80 raw cells, 40 unique — `S2` is a mirrored `(D, D)`
matrix at this point in the code, see the Sobol `S2` note below) and missed
the confidence-interval fields the same regeneration also touched:
`S2_conf` 52 raw cells (26 unique), `S1_conf` 29, `ST_conf` 25. Two facts made
the `S2` part reviewable rather than waved through. Feeding the *old*
resampler from the new flattened arrays is bit-for-bit identical, so the
layout change is not the cause; and the only batch width that reproduces the
old bits is one holding a single slice, which is the same as not batching at
all.

One of the 40 unique `S2` values is a fix. `sobol_g_multi.sobol.S2` moved
because the bootstrap point estimate now runs the same kernel as the plain
path. Before, the two disagreed by 1.4e-7 for one design, so an interval was
centred on a number `analyze(num_resamples=0)` never reported.

## The 1.0.0 review-fix pass (2026-08-21)

The pre-release review (`REVIEW-1.0.md`) found several places where jaxgsa
computed the wrong number. Fixing them was the point of this regeneration —
unlike every entry above, most of this batch is not plumbing, and moving a
number was the goal, not an accident. `header.jaxgsa_version` also corrected
itself from a stale `"0.8.0"` to `"1.0.0"` simply by re-running the dump
against the installed package, and the coverage table above now lists all six
cases (see item 3).

**3097 raw values moved, all attributable, none left over.** By method:

| method | moved | cause |
| --- | --- | --- |
| `hdmr` | 1002 | H4 (relative backfit stop rule) + M8 (`S` measured against the fitted expansion, not `Y`) |
| `pce` | 731 | Section 5 (`coeffs = solve(gram, Phi.T @ Y)` rewrite; float32 reassociation only) |
| `sobol` | 486 | `S2` symmetrisation (averaging the two triangles instead of mirroring one) |
| `optimal_transport` | 471 | M3 (Sinkhorn `epsilon` scaled by `V`, not per-class max) + M4 (per-parameter dummy) + Section 5 (exact 1-D W2, float32 centering) |
| `vkoga` | 217 | H3 (per-outer-point inner block for `S_TU`) + M5 (marginal-dependent component basis) + M6 (`_RIDGE_GRID` floor) |
| `shapley` | 126 | Downstream of the HDMR M8 fix (`backend="hdmr"` reads `Sa`/`Sb`) and the PCE Section 5 rewrite (`backend="pce"`) |
| `borgonovo` | 35 | Cross-cutting bootstrap plumbing (`interval()`/`bootstrap_draws()` adoption) and the D4 partition consumer fix; float32 reassociation only |
| `efast` | 29 | H2 (`/ N**2` moved inside the square to avoid the int32 overflow); float32 reassociation only, none of the baseline's `n_per_curve` values are near the overflow threshold itself |

`kucherenko`, `morris`, `dgsm`, `hsic`, `pawn` and `pce.explained_variance`
did not move: the last was already corrected at commit `229d8b0` (item 5
above), and this pass's PCE change does not touch it (verified: no
`explained_variance` line appears in the diff for any case). Schema changes:
0 — every field this pass added or removed had already been reconciled by an
earlier commit in the same series, so this regeneration is values only.

Two methods moved for reasons no finding names directly, and both are
explained rather than left as an open question:

- **`sobol`** moved *only* in `S2`/`S2_conf` (checked directly: no `S1` or
  `ST` line appears in the diff). That is exactly what the symmetrisation
  predicts — `S1` and `ST` do not read `S2` — and confirms the fix did not
  leak into the two point estimates the review's H-items were most worried
  about.
- **`borgonovo`** and **`efast`** moved by 1-4 units in the last place of
  float32 (the same magnitude class as the batch-width exception above), not
  by a value large enough to change a conclusion. Both are consequences of
  changes made for other reasons — a shared bootstrap helper for the former,
  reordered arithmetic to dodge an int32 overflow for the latter — landing in
  code that also runs these two methods. Re-measured directly: `borgonovo`
  max delta 5.96e-8 (`S1_conf`), `efast` max delta 5.96e-8 (`S1`/`ST`), both
  at the scale of the value they perturb (≤ 1 in the last significant digit
  printed).

Every method whose numbers were meant to move by more than rounding was
checked against the magnitude the finding predicted: `hdmr` (H4+M8) moved up
to 0.080 (`ST`), inside the review's "6e-3 independent to 4e-2 correlated"
range for an independent case and consistent with H4's larger, case-dependent
swings for a correlated one; `vkoga` moved up to 0.063 (`S_IU`, `S_TU`),
matching H3's and M5's measured bias corrections; `optimal_transport` moved
up to 0.011 (`above_dummy`), the M4 per-column floor taking effect.

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

Six problems, chosen for the shapes and the input features that the 0.9 and
1.0 refactors touch:

| Case | Problem | Output shape |
| --- | --- | --- |
| `ishigami_scalar` | Ishigami, 3 uniform inputs | `(N,)` |
| `sobol_g_multi` | Sobol g-function, 8 uniform inputs | `(N, 2)` |
| `ishigami_series` | Ishigami, widened over time | `(N, 3, 2)` |
| `ishigami_correlated` | Ishigami with a declared correlation | `(N,)` |
| `categorical_mixed` | 1 uniform and 1 three-level categorical input | `(N,)` |
| `gaussian_mixed` | 1 truncated Gaussian (2 sigma), 1 unbounded Gaussian, 1 uniform | `(N,)` |

`gaussian_mixed` was added at commit `315effe` (2026-08-19; see item 3 above).
Every other case is uniform-marginal, so before it the Gaussian transforms
were never exercised by this file at all: the truncated marginal takes PCE
down its Legendre path and DGSM through the finite-element Poincare solve,
and the unbounded marginal is the only thing that makes
`morris._squash_open_sides` do any work.

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
