# Code findings from the v1.0 docs pass (2026-08-20)

## 0. CI is red across the whole stack, and master is green

This one blocks the merge, so it goes first. It was found by checking CI
rather than by the docs pass.

`master` passes CI (last five runs green, newest `03ee1f3`). Every PR in the
stack fails, as far down as #59, with the SAME nine tests:

    tests/test_borgonovo.py::TestGridTiling::test_tiled_grid_is_bit_identical
      [2, 3, 4, 7, 8, 16, 50, 99]
    tests/test_hsic.py::TestIndicesCore::test_is_vmappable

So the stack introduced them. Merging it as it stands turns master red.

They pass on an Apple M1 Pro and fail on the x86 Linux runner, which is why
local runs are green. Both tests assert bit-identity between two paths that
the algorithm says cannot differ.

- Borgonovo: 12 of 45 elements differ, max absolute 2.98e-08 on values near
  0.21, so roughly one float32 ULP. The test's docstring states the
  reasoning plainly: every grid point sums over its own class members and
  nothing else, so no tiling reorders a reduction. That is true of the
  algorithm. It is not true of the compiled code, because XLA is free to
  vectorize a reduction differently at a different trip count, and the tile
  width is the trip count.
- HSIC: max absolute 8.88e-06, max relative 2.43e-04, against a tolerance of
  rtol=1e-4. That is about a hundred times the Borgonovo gap, so it may be a
  different cause and should not be assumed to be the same one.

Two ways to resolve, and the choice is a real decision about what the
library promises, so it is left alone here:

1. Accept that bit-identity across tile widths is not portable, and compare
   with a tight tolerance plus a comment naming compiler vectorization as
   the reason. Cheapest, and honest, but it weakens a guarantee someone
   deliberately wrote `assert_array_equal` for, with the reasoning recorded
   in the docstring.
2. Make the kernels genuinely bit-identical on every target, by fixing the
   reduction order rather than leaving it to XLA. Keeps the promise, costs
   speed, and needs measurement before anyone commits to it.

The HSIC failure deserves its own look before either is applied to it.

Note for whoever reads the CI dashboard: #67's `baseline` job failing is a
different thing and is expected. It compares against the PR's base and
reports the 22 dgsm values that the sanctioned jacfwd regeneration moves on
purpose. #69's baseline job passes.


The docs pass ran every code example against the branch. That found bugs the
test suite does not cover, because `examples/` and the doc pages are not
tested. Listed worst first. None of these are fixed on the docs branch; the
docs branch only fixes docs and the broken scripts noted at the end.

## 1. Sobol bootstrap is 2.6x slower on scalar output, for nothing

Commit `c7d0269` "perf: batch the Sobol bootstrap over output slices" (on
`perf/resample-kernels`, inside the existing #40-#59 chain) made the scalar
case much slower and did not make the wide case faster.

Measured on an Apple M1 Pro, Ishigami, N=1024 base samples, R=1000
resamples, first-order only, best of four runs. Script:
`scratchpad/boot_shapes.py`.

| output width | before `c7d0269` | v1 stack tip | change |
| --- | --- | --- | --- |
| T*K = 1 | 7.35 ms | 19.15 ms | 2.6x slower |
| T*K = 20 | 95.53 ms | 91.33 ms | 4% faster |
| T*K = 200 | 934.18 ms | 967.53 ms | 3.5% slower |

Bisect (same script, one number per branch, T*K=1):
`master` 7.78, `s7-estimator` 7.41, `k1-hsic` 7.45, **`k2-resample`
(c7d0269) 19.02**, `w2-b-bootstrappers` 18.91, `w34-consolidate` 19.35,
`perf/kernels-and-precision` 19.79, v1 tip 19.62.

So the v1.0 stack (#62-#67) did not cause it and did not fix it. It has been
sitting in the stack since the resample-kernel work.

Reading: slice batching bounds peak memory, which is worth having. It is not
a speed win at any width measured, so the commit's "perf:" framing overclaims.
The cost lands entirely on the narrow case that never needed bounding.

Suggested fix, and it is the rule #62 already froze everywhere else: when the
resolved chunk width covers every slice, take the unchunked path instead of
running a one-iteration chunk loop. That is exactly what PCE now does when
`batch_size >= N`. Doing it here would restore about 7.4 ms on scalar output
and change nothing else.

Cost of the fix: it touches `sobol/_bootstrap.py`, which sits at the bottom
of the PR stack, so landing it means rebasing everything above it.

## 1b. Two estimators disagree between the scalar and 3-D point-estimate paths

Found while testing the fix for item 1, and unrelated to it. Feeding one
output as `(N,)` and as `(N, 1, 1)` should give the same point estimate. For
four of the six estimators it does, bit for bit. Two do not:

| estimator | max abs difference in S1 |
| --- | --- |
| `janon-monod` | 3.278e-07 |
| `azzini-rosati` | 1.073e-06 |

Confirmed present before the bootstrap fix, so it is pre-existing, and the
interval endpoints agree exactly for every estimator. It is float32
reassociation between two kernels rather than a wrong formula, and 1e-6 is
below the noise floor of any design that would be used in practice. Recorded
because a user comparing the two layouts could otherwise think one is wrong,
and because `test_single_slice_bootstrap_matches_the_mapped_one` documents
the same fact in the suite.

## 2. PCE `explained_variance` returns 1.0709 on Ishigami

Order 8, N=2000, and the same value in float32 and float64, so it is not a
precision artefact. A share of variance above 1 should not be possible.
Found by the API-pages agent, which documented the field without printing the
number. Worth a look before 1.0 ships a diagnostic that can read above 1.

## 3. The float64-truncation warning fires on nearly every doc example

Pass a plain NumPy `Y` with x64 off, which is the default path for anyone
copying an example, and every method warns. The warning is correct but it
makes the first-run experience noisy for a reader following the docs. Either
soften it for the common case or say plainly in one shared place how to
silence it. The docs now handle it per page, which is a workaround, not a fix.

## 4. `SobolSamples.save()` fails with a NumPy-internal error

If the parent directory does not exist it raises `FileNotFoundError` from
inside NumPy's `_savez`. A clear error naming the missing directory would be
kinder. The save/load page now tells readers to create the directory first.

## Already fixed on the docs branch

- `examples/batch_reactor_gsa.py`, `dynamic_gsa.py`, `benchmark_all.py` and
  `morris_gsa.py` called `num_resamples=`, retired in the vocabulary freeze.
  All four raised `TypeError` on 1.0. Renamed to `n_bootstrap`; all nine
  example scripts now run.
- `benchmark_salib.py` had the same break, plus its timed calls printed the
  verbose block inside the timing loop, which costs milliseconds and forces a
  device sync. Renamed, and the timed calls now pass `verbose=False`.
- The marimo artifacts under `docs/public/notebooks/` and
  `examples/__marimo__/` embedded the old broken source. Regenerated.

## 5. The CHANGELOG describes breaks that never happened

The "Unreleased" section documents the whole 0.9-plus-0.10 development arc,
so some entries compare against unreleased intermediate states instead of
against shipped 0.8.0. A user upgrading 0.8 to 1.0 never saw those states,
so those entries are wrong for them. The release ships this section as its
notes, so it needs a pass.

Verified against `git show master:src/...` (released 0.8.0). `keep_replicates`
and `CIInfo` have ZERO occurrences in 0.8.0 source, so anything describing a
change to either is intra-arc churn, not a break.

| CHANGELOG line | Claim | Reality in 0.8.0 |
| --- | --- | --- |
| 74 | `CIInfo.n_resamples` renamed to `CIInfo.n_bootstrap` | `CIInfo` did not exist. Nothing was renamed. |
| 126 | `keep_replicates` "sat in" other positions | The keyword did not exist. It never sat anywhere. |
| 182 | Stored S2 bootstrap draws are now symmetrized | There were no stored draws to change. |
| 52 | Sobol's bootstrap path and Morris both capped an explicit chunk value | Only Morris did. Sobol capped at R, the resample count, with no memory term. |

The docs pass fixed the migration page, which had inherited all four. The
CHANGELOG itself is untouched, because it belongs to the release PR and
editing it here would conflict. Whoever does that pass should re-derive every
Unreleased entry against `master`, not against the previous entry.

## Still open for the release PR

`pyproject.toml` says `version = "0.8.0"` and `CHANGELOG.md` heads its
section "Unreleased (0.10.0)", while the decision was to go straight to
1.0.0 and the docs say 1.0 throughout. Both need the bump.
