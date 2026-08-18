# Execution plan: jaxgsa 0.8.0 to 1.0

Status: ready to start. Drafted 2026-08-18.

This document merges `AUDIT-DECISIONS.md` (21 fixes to existing code) and
`ROADMAP-1.0.md` (new methods and verification policy) into one ordered plan.

Where the two source documents disagree with the source code, this document
follows the code. Section 2 lists every such correction. Read section 2 before
you start any task: eight statements in the source documents are wrong, and two
of them change the work.

---

## 1. Decisions that shape this plan

Four decisions were taken before the plan was written.

1. **Version 0.9 carries fixes only.** All 21 audit decisions, the three
   roadmap "fix first" defects, and the test cleanup. No new method. It is a
   breaking release with a migration guide. New methods then build on a settled
   API instead of a moving one.
2. **The release ladder shifts by one.** The roadmap put given-data methods in
   0.9. They move to 0.10, gradients move to 0.11, and 1.0 keeps coverage and
   stability.
3. **Oracles run locally, never in CI and never in the package.** Any language
   is allowed for a local check. What lands in the repository is the *number*
   the oracle produced, typed into a test as a literal, with the oracle, its
   version, and the script that produced it recorded next to it. This turns
   every T2 and T3 oracle into a T1 literal for CI purposes.
4. **Detail is front-loaded.** Release 0.9 is planned task by task. Later
   releases are planned tranche by tranche, and each is expanded when reached.

### 1.1 The release ladder

| Release | Content | Breaking |
| --- | --- | --- |
| **0.9** | 21 audit decisions, 3 roadmap defects, test cleanup, packaging | Yes |
| **0.10** | Methods that work on data you already have | No |
| **0.11** | Methods that use gradients | No |
| **1.0** | Coverage, stability, release engineering | No |

---

## 2. Corrections to the source documents

A verification pass on 2026-08-18 checked every file and line reference in both
documents against the working tree at `03ee1f3`. Most references are exact. The
following are not.

### 2.1 Two corrections that change the work

**C1. D9 cannot be implemented as written.** The audit says to raise at the top
of `borgonovo.analyze` when `degenerate_bandwidth * h_full < grid_step`. That
test cannot run there. `h_full` is computed per output column *inside the jitted
kernel* at `borgonovo/_analyze.py:267`, from `jnp.std(y_r)`. `grid_step` is also
data-dependent, at `:240`. Neither value exists at the top of `analyze`.

This is the same objection D9 itself raises when it refuses to warn about
`degenerate_tol`. Three routes are open. Section 5, batch 4 states which one to
take and why.

**C2. `on_invalid="raise"` is a behaviour change, not a new argument.** The
roadmap says only `sobol` and `morris` handle non-finite output. Four modules
do: `sobol`, `morris`, `kucherenko`, and `efast` (warn-only). Three of them
*already* drop bad rows by default. Making `"raise"` the default changes what
those three do today, so it belongs in the breaking release, not in 0.10 with
the rest of the policy work. Section 5, batch 7 splits it accordingly.

### 2.2 Corrections that do not change the work

| Claim | Correction |
| --- | --- |
| D2: `poincare_constant` and `marginal_variance` are both public | Only `poincare_constant` is exported. `dgsm/__init__.py:23,26` exports `axis_constants` and `poincare_constant`. D2 applies to one signature, not two. |
| D10: the unreachable branch is at `_sampling.py:151-153` | It is at `:158-162`. Lines 151-153 are the `D == 1` early exit, which is reachable. |
| D12: `slice_chunk_size` "is validated and then ignored" | It is not validated either. The validation block at `pawn/_analyze.py:363-368` covers `statistic`, `n_bins` and `conf_level` only. |
| Section 5: `SobolSamples.sample_ids` has zero readers in `src/` | It has zero *algorithmic* readers. It is read by the `.npz` payload at `sobol/_sampling.py:311` and restored at `:331`. The stated ordering constraint gets stronger, not weaker. |
| Section 5: the Sobol `slice_chunk_size` test passes `num_resamples=0`, and that argument reaches only the bootstrap path | It reaches both paths. `_analyze_no_bootstrap` validates it at `:248-249` and chunks at `:251`. The test does exercise chunking; it asserts a shape and never compares chunked against unchunked. The gap is real, the reason given for it is not. |
| Section 5: 31 test files, ~14 renamed-keyword tests, 6 paired tests | 33 files. 7 verbatim `test_old_chunk_size_kwarg_raises` copies plus 7 paired "accepted" tests, so 14 in the cluster. The total is right by coincidence. |
| D8: `tests/test_pce.py:138-162` has the gap | The gap is at `:140-150` and `:152-162`, and only `ST` is genuinely unasserted. `test_s1_x3_near_zero` at `:164-167` already covers the near-zero S1 entry. |
| D7: Sobol guard at `sobol/_result.py:116`; D4: clamp at `optimal_transport/_analyze.py:344`; D13: scalar squeeze at `sobol/_analyze.py:215-217`; roadmap 3.1: `_sampling.py:230` and `_analyze.py:188` | Off by one or two. Real lines: `_result.py:115`, `_analyze.py:342`, `_analyze.py:216-218`, `_sampling.py:224` and `_analyze.py:190`. |
| D21: 58 `pytest.warns(UserWarning` assertions | 57. The 35 `warnings.warn` sites and the zero categories are exact. |
| D20: "four duplicated helpers" in the benchmark harness | Four real duplicate pairs exist, but the generic traversal to mirror lives in `examples/benchmark_all.py:269-270`, not in `src/`. The target is the sibling harness. |

### 2.3 Coupling the source documents miss

These matter when tasks are split across agents.

1. **D1 changes `_normalized_input_to_dict`.** The helper at `problem.py:312-337`
   currently *returns the public TypedDicts* D1 deletes. Its body and return
   annotation must change. The decision text does not say this.
2. **D1 has a hashability constraint.** The spec tuple is jit-cache metadata on
   a hashable `Problem` (`problem.py:446`), and `_categorical_payload`
   (`:382-392`) relies on tuple-of-tuples. `CategoricalSpec.probs` and `.labels`
   must be tuples, not lists.
3. **D16 has a third unpack site.** `hdmr/_analyze.py:406` unpacks all twelve
   values. The audit names only `:87` and `:686`. `:406` is the one that would
   break silently on a reorder.
4. **D13's premise holds for one of two branches.** `_bootstrap_ci_endpoints`
   returns a tuple. Only the quantile branch (`_core/bootstrap.py:57-58`) builds
   a stacked `(2, ...)` array internally; the Gaussian branch (`:64-68`) never
   does. The refactor must stack in the caller.
5. **D10 must not poison the kernel cache.** `_get_efast_kernel` is memoised on
   `(N, M, omega_0, batched)` at `efast/_analyze.py:87`. A frequency-plan object
   holding a NumPy array must stay out of that key.
6. **Three unflagged file collisions.** D3 collides with D9 in
   `borgonovo/_analyze.py:573-618`, with D11 in `dgsm.analyze`, and softly with
   D12 in `pawn/_analyze.py:370-391`. The batching in section 5 accounts for all
   three.
7. **D15 and the `coeffs_flat` deletion touch the same two lines**
   (`pce/_analyze.py:163` and `:345`). Do them as one edit.

---

## 3. Verification protocol

This is the part that must not be relaxed. Every batch passes the same gates.

### 3.1 Per-batch gates

A batch is not finished until all six pass.

1. **Local gate.** `uv run ruff format`, `uv run ruff check`,
   `uv run ty check src/jaxgsa` on edited files, `uv run pytest -q` green.
   Baseline is **979 tests**; state the new count and account for the difference.
2. **Correctness review.** One review agent that did not write the code reads
   the diff against the decision text and the source paper. It reports
   confirmed defects only, with a failure scenario for each.
3. **Verification review.** A second, independent agent audits the *tests*, not
   the code. It answers three questions per new or changed test: what tier is
   this, does the docstring say so, and is it a mirror? A mirror is a finding.
4. **Blast-radius review.** A third agent greps for every call site the change
   should have touched and reports the ones it did not. This is the gate that
   catches half-applied refactors such as D3 across 48 call sites.
5. **Numerical invariance.** For any change the decision text calls "plumbing
   only" (D3, D4, D5, D13, D16, D17), assert that indices are bit-identical
   before and after on a fixed seed. A changed number means a wiring error, not
   a tolerance problem.
6. **Documentation.** The doc update ships in the same commit as the code, per
   `AUDIT-DECISIONS.md` section 8. A docs reviewer checks the plain-language
   rules and that no stale API name survives.

Then: draft pull request, and `/code-review ultra` before it is marked ready.

### 3.2 How agents are used

- **Implementation.** One agent per batch, in its own git worktree. Never the
  main checkout. Batches inside a tranche that touch disjoint files run in
  parallel; the tables in section 5 mark which.
- **Review.** The three review agents in gates 2 to 4 run in parallel and must
  be fresh agents, not the implementer and not forks of it. An agent that wrote
  the code cannot audit its own blast radius.
- **Conflict rule.** Two agents must never hold the same file. Section 5's
  batching is built from the verified collision list in section 2.3, so follow
  the batching rather than the decision numbers.

### 3.3 Oracle policy

Per the decision in section 1: oracles are local, in any language, and their
output is committed as literals.

**Every oracle-derived literal carries a provenance block** in the test
docstring: tier, oracle name, exact version, the date it was run, and the path
to the script under `scripts/oracles/` that produced it. A number with no
provenance block is not an oracle, and gate 3 rejects it.

Verified available on this machine, 2026-08-18:

| Oracle | Tier | Status |
| --- | --- | --- |
| `scipy.stats.chatterjeexi` | T2 | scipy 1.18.0 present. The project pin was raised to `scipy>=1.15` in tranche 0, so the function is always available. |
| SALib `analyze.rsa` | T2 | 1.5.2, already a dev extra. |
| OpenTURNS | T2 | 1.27, wheel installs clean. All four Sobol estimators, `RankSobolSensitivityAlgorithm` and `HSICEstimatorGlobalSensitivity` are in the main namespace. |
| POT `ot.emd` | T2 | 0.9.7, already a dev extra. |
| dcor | T2 | 0.7. |
| UQpy `GeneralisedSobolSensitivity` | T2 | 4.2.1, but needs Python 3.12 and `setuptools<81`. Use its two pure-array methods and feed them our own A/B/C blocks. |
| R `sensitivity` | T3 | 1.31.0 installed. `sobolrank`, `sobolshap_knn`, `shapleyPermEx`, `PoincareOptimal` available. |
| R `gsaot` | T3 | 1.1.1 installed. |
| SAFEpython | T3 | 0.2.0rc1 resolves. GPL-3, so subprocess only, and do not read its source while implementing. |
| R `sensobol` | T3 | 1.2.0, installed 2026-08-18. Needed for BCa intervals and the dummy-parameter floor. |
| ATHENA | T2 | **Dead end.** Its dependency chain cannot be resolved on any supported Python. Active subspaces must use its T0 oracles instead, which are strong. |

Because oracles stay out of the package, `pyproject.toml` gains no `oracles`
extra. Put the local environment recipes in `scripts/oracles/README.md`.

---

## 4. Tranche 0: clear the ground

Do this first. It is half a day and it removes noise from every later diff.

| Task | Detail |
| --- | --- |
| **T0.1 Prune worktrees** | Eleven worktrees exist under `.claude/worktrees/`, most from merged pull requests (#31 to #34), plus two detached review checkouts. Remove the merged ones and their branches. |
| **T0.2 Commit the two planning documents** | `AUDIT-DECISIONS.md`, `ROADMAP-1.0.md` and this file are untracked. Commit them so later work can cite a revision. |
| **T0.3 The audit's suggested first batch** | Five independent one-file changes: rename the shadowing eFAST fixture (`tests/test_efast.py:423-424`), delete `_PCEFit.coeffs_flat`, delete `validate_correlation` and repoint its 11 tests at `canonicalize_correlation`, fix `CITATION.cff` (drop `version` and `date-released`), declare `numpy>=2` in `pyproject.toml`. |
| **T0.4 Baseline** | Record the test count (979), the benchmark numbers, and a fixed-seed index dump for every method. Gate 5 compares against this file for the rest of 0.9. |

`_PCEFit.coeffs_flat` in T0.3 must be deleted **together with** D15's field
addition, or the two edits collide on the same lines. If batch 5 is not ready,
leave `coeffs_flat` to batch 5 instead.

---

## 5. Release 0.9: the fixes

Seven batches. The order is forced by the collision list, not by decision
number. Batches marked "parallel" touch disjoint files and may run at the same
time.

### Batch 1 — test cleanup and the layout-pinning rewrite (parallel with 2)

This comes first because `AUDIT-DECISIONS.md` section 7 requires it: done in
this order, the D1 dataclass migration touches no test.

| Task | Where |
| --- | --- |
| Rewrite the 10 spec-tuple assertions through `_normalized_input_to_dict` | `tests/test_problem.py:108,109,115,116,120,123,140,141`; `tests/test_categorical.py:76,90` |
| Delete the 14 renamed-keyword tests | 7 copies of `test_old_chunk_size_kwarg_raises`, 7 paired "accepted" tests, across 7 files |
| Delete or replace the 5 mirror tests | `test_borgonovo.py:381`, `test_pce.py:308-325`, `test_hdmr_streaming.py:225-237`, `test_vkoga.py:144-151`, `test_dgsm.py:61` |
| Delete the 6 `isinstance(result, XResult)` assertions | across `tests/` |
| Add the missing Sobol chunking test | assert chunked output equals unchunked, with `num_resamples > 0` and `K > 1` |
| **D8**: add the `else` branches | `tests/test_efast.py:446`, `tests/test_pce.py:145` and `:157` |

Gate 3 applies to this batch reflexively: the replacement tests must state
their tier.

### Batch 2 — packaging and warnings (parallel with 1)

| Task | Where |
| --- | --- |
| **D19.3**: derive the sidebar order from `config.ts` | `scripts/check_vitepress_pager.py:17-42`, 27 hardcoded slugs |
| **D21**: add `JaxgsaWarning(UserWarning)`, export it, pass it at all 35 sites | 18 files; docs at `README.md:562`, `docs/guide/methods.md:213,283,912,980`, `docs/api/vkoga.md:49,107`, `docs/api/kucherenko.md:67` |
| **D21 guard**: a test that walks the source and fails on any `warnings.warn` without a category | new test |
| `np.unique` rewrite of `_stable_unique_rows` | `_core/sampling.py:215-248`; needs `numpy>=2` from T0.3 |
| VKOGA power vector in the loop state | `vkoga/_engine.py:205,237` |
| `_set_memory_budget` signature; teardowns use the public getter | `_core/batching.py:43`, two test files |
| `cdf_to_unit_interval` comment or clip | `_core/transforms.py:42-45,57` |
| Morris `n_blocks_dropped` staleness in `downsample` | `morris/_sampling.py:180` |

The `np.unique` rewrite is the one item here with a numerical consequence.
Gate 5 applies: unique-row output must be identical, including row order.

### Batch 3 — the two shape refactors, in order

**D5 before D4** (same file, same kernels), then **D3**, which is the largest
change in the release.

| Step | Task |
| --- | --- |
| 3a | **D5**: `build_partition_groups` returns `group_levels`; delete the hand-rebuild at `optimal_transport/_analyze.py:669-673`. Three callers. |
| 3b | **D4**: `counts` becomes unconditionally `(R, Dg, Mg)`; delete `_replicate_slice`; the clamp at `optimal_transport/_analyze.py:342` becomes `dm // M`. |
| 3c | **D3**: `YLayout` enum replaces the two squeeze booleans, **and** the two `is_scalar` sites at `sobol/_analyze.py:201` and `efast/_analyze.py:178`. 15 `_prepare_Y` sites, 48 `_squeeze_output_axes` sites, 13 modules. Zero test references, so this touches no test. |

Gate 4 is the critical one for 3c: 48 call sites is where a partial refactor
hides. Gate 5 applies to all three steps — every index must be bit-identical.

### Batch 4 — module correctness fixes (parallel internally after batch 3)

D3 must land first: it collides with D9, D11 and D12.

| Task | Note |
| --- | --- |
| **D13**: three accumulators instead of nine | `sobol/_analyze.py:341-419`. Stack in the caller; `_bootstrap_ci_endpoints` builds the layout in only one of its two branches. |
| **D10 / roadmap 3.1**: one `_frequency_plan(D, n_per_curve, M)` | Called from `efast/_sampling.py:224` and `efast/_analyze.py:190`. Keep `omega_0` a concrete int in the cache key. Absorb `_min_n_per_curve`; delete the unreachable branch at `:158-162`. Add a test that asserts the two sides read the same band — the defect the current tests structurally cannot catch. |
| **D11**: exclusive argument-group resolution | `dgsm/_analyze.py:225-254`. Fix the `Raises:` list at `:212-218`. |
| **D12**: implement PAWN `slice_chunk_size` | `pawn/_analyze.py:198-200`, copy the Sobol or Borgonovo pattern. Add a chunk-invariance test. Remove the "signature parity" wording at `:336`. |
| **D9**: Borgonovo bandwidth | **See C1.** The check as written is not reachable. Take route (a): compute the host-side full-sample bandwidth and grid step once, before the jitted kernel, and raise there — it costs one `std` and one `linspace` on the host. Routes (b) return a NaN from inside the kernel and let the existing range check fail, or (c) document the failure mode and raise nothing. Route (a) keeps the "fail up front" principle the decision is built on. Also collapse the two resolvers at `:790-861`, and add the `degenerate_tol` bias sentence. |
| **D7**: drop the redundant guards | `sobol/_result.py:115`, `morris/_analyze.py:374` |
| **D17**: `chol_full` on `_ConditionalPlan` | `_core/copula.py:80-106`, append the field; two call sites at `kucherenko/_sampling.py:166` and `vkoga/_analyze.py:228` |

### Batch 5 — the fit paths

| Task | Note |
| --- | --- |
| **D14**: one Cholesky-based leverage formula | `pce/_engine.py:229-242` and `pce/_analyze.py:230-246`. Correct the memory estimate at `:274` from `3 * n_terms` to `2 * n_terms`. Do **not** touch the different formula at `:495`. |
| **D15**: `streamed: bool` on `PCEResult` and `HDMRResult` | Replaces the monkeypatch-and-count tests at `tests/test_pce_streaming.py:141,155,171` and `tests/test_hdmr_streaming.py:195,206,221`. Weaken the call-count assertion at `test_pce_streaming.py:144`. |
| Delete `_PCEFit.coeffs_flat` | Same two lines as D15. One edit. |
| **D16**: `NamedTuple` for the HDMR static data; one order-map helper | `hdmr/_analyze.py:41,87,406,686`; `_engine.py:411-422` and `_stream.py:438-446`. Remember `:406`. |
| **D18**: collapse the HSIC dispatch; index-adjusted quantile | `hsic/_analyze.py:47-194`. Document that `batch_size` bounds working memory, not the kernel matrix. |

D14 moves the streaming threshold, which is exactly why D15's `streamed` flag
lands in the same batch: without it, a fit that silently stops streaming passes
every behavioural test.

### Batch 6 — the public API break

Last, so that everything above is already stable.

| Step | Task |
| --- | --- |
| 6a | **D2**: `poincare_constant` stops naming the private tuple. One signature, not two. |
| 6b | **D1**: `UniformSpec`, `GaussianSpec`, `CategoricalSpec` frozen dataclasses replace both the tuple and the three TypedDicts. ~10 call sites in `src/`, ~60 construction sites across 19 test files, plus `docs/api/index.md:11-16`, `README.md:581` and two benchmark files. Keep `probs`/`labels` as tuples; rewrite `_normalized_input_to_dict`. |
| 6c | **D6**: freeze 11 result dataclasses. `HDMRResult` stays mutable, with the one-line `cached_property` comment. Document the three ways to get a modified copy. |
| 6d | Delete `SobolSamples.sample_ids`, including the `.npz` payload at `sobol/_sampling.py:311,331` and its four test readers. |
| 6e | Migration guide `docs/guide/migration-0.8-to-0.9.md`, modelled on `migration-0.4.md`. `CHANGELOG.md` entry marking 0.9.0 breaking. |

### Batch 7 — the failed-evaluation policy (roadmap 3.3)

Split by C2, because the default change is breaking and the rest is not.

| Step | Task | Release |
| --- | --- | --- |
| 7a | Shared `on_invalid={"raise","propagate","drop"}` helper in `_core/`, generalising `sobol._drop_nonfinite`. Report count, row indices, and design position — `nan_counts` today reports NaNs in the *output indices* and discards row identity at `sobol/_analyze.py:104`. | 0.9 |
| 7b | Flip the default to `"raise"` in the four modules that handle non-finite output today (`sobol`, `morris`, `kucherenko` currently drop; `efast` currently warns). This is the breaking part. | 0.9 |
| 7c | Thread `on_invalid` through the remaining nine `analyze()` entry points: `pce`, `hdmr`, `optimal_transport`, `hsic`, `pawn`, `borgonovo`, `dgsm`, `shapley`, `vkoga`. Note three are named `analyze_pce`, `analyze_hdmr`, `analyze_vkoga` and a grep for `def analyze(` misses them. | 0.10 |

T4 oracle, correctly: inject NaNs into a model with known analytic indices and
assert each of the three settings behaves as specified.

### 0.9 exit criteria

- All seven batches merged, each through the six gates.
- Test count stated and reconciled against 979.
- Fixed-seed index dump identical to the T0.4 baseline, except where a decision
  says a number changes (D14's streaming threshold is the only one).
- Migration guide published; `docs/.vitepress` builds; pager check green.
- Benchmark tables re-run and unchanged (D20 says they stand; confirm it).

---

## 6. Release 0.10: methods that work on data you already have

Detail is expanded when 0.9 closes. Order within the tranche:

| Order | Item | Why here | Oracle |
| --- | --- | --- | --- |
| 1 | **Rank-based estimators** (Chatterjee; Gamboa et al.) | Best value per line in the whole plan, and it needs nothing from the other items | T2 `scipy.stats.chatterjeexi` locally, then T1 literals. T0 Ishigami. Cite the *Bernoulli* DOI, never the withdrawn arXiv, and cite the variance erratum. |
| 2 | **`on_invalid` step 7c** | Finishes what 0.9 started before new modules add more entry points | T4 behavioural |
| 3 | **Confidence intervals in the remaining eight modules**, with percentile, basic and BCa | Every later method wants intervals; build the machinery once | T4 coverage simulation. `sensobol` locally for BCa — needs installing. |
| 4 | **Dummy-parameter significance test** | Shares the interval machinery; turns a ranking into a decision | T4 by construction; `sensobol::sobol_dummy` locally |
| 5 | **More Sobol estimators** (Jansen first-order, Janon-Monod, Martinez, Mauntz-Kucherenko, Azzini-Rosati) | Independent | T2 OpenTURNS, four-way, locally |
| 6 | **Regional sensitivity analysis** | Shares the PAWN kernel; also consumes the 0.9 non-finite indicator | Pick the formulation first: SAFEpython threshold-and-KS, or SALib binned Cramer-von Mises. They cannot both be the oracle. |
| 7 | **HSIC p-values and the distance-correlation kernel** | Builds on D18's collapsed dispatch | T2 OpenTURNS both p-value routes; `dcor` for distance correlation. Route distance kernels away from the Gamma null. |

---

## 7. Release 0.11: methods that use gradients

This is the tranche that carries the project's stated position. Ship it as one
story with one benchmark.

| Order | Item | Note |
| --- | --- | --- |
| 1 | **Active subspaces** | Twenty lines plus tests, and the T0 oracles are unusually strong: `C = a a^T` exactly for a linear model, `A(Sigma + mu mu^T)A` for a Gaussian quadratic, and `trace(C) = sum nu_i` ties it to DGSM. ATHENA is unavailable, and does not matter. |
| 2 | **Crossed DGSM** | `jax.hessian` makes it nearly free. Advertise it as a bound on **superset importance**, not on the second-order index. |
| 3 | **Poincare: external oracle and the PCE route** | R `sensitivity::PoincareOptimal` is installed. Fix the two citation errors in the existing docstring. |
| 4 | **Forward versus reverse mode selection** | T4: the two paths must agree. |
| 5 | **d(index)/d(theta)** | The flagship. Medium effort, most of it API and honest limits. T0 linear-Gaussian derivatives, verified to 5e-11. |

---

## 8. Release 1.0: coverage and stability

| Item | Note |
| --- | --- |
| **kNN Shapley effects from given data** | The real capability gap. Implement the `knn` variant only. Benchmark the dimension ceiling before claiming anything: Broto's own Corollary 1 contradicts the dimension-independence claim repeated in later work. |
| **Optimal-transport parity with gsaot** | Exact solver via POT, irrelevance threshold sharing the dummy-parameter interface, sensitivity maps, user-supplied cost. Drop `entropic_bound` and `higher_order_terms` from the plan. |
| **Generalised Sobol indices** | The trace form is forced, not conventional. T2 UQpy locally, through its pure-array methods. |
| **More input distributions** | Internal `Marginal` protocol plus adapters. No new required dependency. |
| **Linear-Gaussian fixture fixes** | `tests/_linear_gaussian.py`: state the unit-diagonal assumption or use the general form, document the mixed convention, replace the per-input `solve` loop with one Cholesky. It is the strongest oracle in the repository and it is currently correct only by accident. |
| **Release engineering** | SPEC 0 support windows, SPEC 7 `rng`, NEP 23 deprecations, `py.typed` (already shipped), Trusted Publishing, conda-forge, Zenodo DOI, `CITATION.cff` `preferred-citation`. |

---

## 9. Open questions

Three were answered on 2026-08-18 and are recorded here as settled. The rest
need an answer before the tranche that depends on them. None blocks 0.9.

### Settled

| # | Question | Answer |
| --- | --- | --- |
| Q1 | D9: which of the three routes? | **Route (a)**: compute the full-sample bandwidth and grid step on the host, before the jitted kernel, and raise there. It costs one `std` and one `linspace` and keeps the "fail up front" principle. |
| Q2 | Does `scipy>=1.10` rise, or does the Chatterjee oracle test skip? | **Raise the pin.** `scipy>=1.15` is declared as of tranche 0, so `scipy.stats.chatterjeexi` is always available and needs no skip. |
| Q4 | Install R `sensobol` locally? | **Yes.** `sensobol` 1.2.0 installed 2026-08-18 alongside `sensitivity` 1.31.0 and `gsaot` 1.1.1. All three T3 oracles are now available on the development machine. |

### Open

| # | Question | Needed by |
| --- | --- | --- |
| Q3 | Regional sensitivity analysis: SAFEpython threshold form, or SALib binned form? They are not interchangeable and the choice fixes the oracle. | 0.10 item 6 |
| Q5 | Non-Gaussian copulas: Clayton, Frank, Gumbel, t. GlobalSensitivity.jl has them and it is the gap a reviewer notices first. In or out of 1.0? | 1.0 |
| Q6 | "Pick-freeze": keep the blanket substitution, or allow the term where it is technically correct? A sweep of 13 occurrences is still pending. | 1.0 docs |

---

## 10. Master order

```
T0  ground clearing
     |
0.9  B1 tests ----+
     B2 packaging-+--> B3 shapes (D5 -> D4 -> D3)
                            |
                            +--> B4 module fixes (D13,D10,D11,D12,D9,D7,D17)
                            +--> B5 fit paths (D14,D15,D16,D18)
                                      |
                                      +--> B6 API break (D2 -> D1 -> D6 -> sample_ids -> migration)
                                      +--> B7 on_invalid 7a,7b
     |
0.10 rank-based -> 7c -> intervals -> dummy -> estimators -> RSA -> HSIC
     |
0.11 active subspaces -> crossed DGSM -> Poincare oracle -> mode selection -> d(index)/d(theta)
     |
1.0  kNN Shapley -> OT parity -> generalised Sobol -> distributions -> fixture -> release engineering
```

Six orderings are forced and every one of them is verified: tests before D1,
D2 before D1, `numpy>=2` before `np.unique`, `sample_ids` before the payload,
D3 before D13, D5 before D4. Three more come from section 2.3: D3 before D9,
D11 and D12.
