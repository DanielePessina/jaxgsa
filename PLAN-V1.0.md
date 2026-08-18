# Execution plan: jaxgsa 0.8.0 to 1.0

Status: in progress. Drafted 2026-08-18, 0.9 re-cut the same day after
batches 1 and 2 shipped.

This document merges `AUDIT-DECISIONS.md` (21 fixes to existing code) and
`ROADMAP-1.0.md` (new methods and verification policy) into one ordered plan.

**Release 0.9 fixes defects and breaks nothing.** Seven of the audit's twenty-one
decisions improve only the shape of the code, and they are deferred to a
release that needs that shape. Section 1.2 gives the reasoning and section 5.1
lists them.

Where the two source documents disagree with the source code, this document
follows the code. Section 2 lists every such correction. Read section 2 before
you start any task: eleven statements in the source documents are wrong, and
two of them change the work.

---

## 1. Decisions that shape this plan

Five decisions shape this plan. The first was revised on 2026-08-18, after
batches 1 and 2 had shipped; section 1.2 records why.

1. **Version 0.9 fixes what is broken, and breaks one thing on purpose.** It
   carries the defects a user can hit today, the latent ones that would produce
   a silently wrong number after any nearby edit, and two resource wins. It
   does **not** carry the structural work: no new spec dataclasses, no layout
   enum, no frozen results, no `.npz` break. Section 5.1 lists what was cut and
   why.

   The one deliberate break is `config.set_memory_budget`, which now reads its
   value in megabytes instead of bytes, on the user's instruction. That is a
   silent reinterpretation by a factor of a million, so a guard refuses
   bytes-shaped values passed without a unit and names both fixes. Everything
   else in the release is additive or invisible.
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
5. **A change with no behavioural target must earn its place.** Restructuring
   is not free: it costs a full review cycle, and it can introduce the very
   defects it was meant to prevent. Section 1.2 gives the evidence from this
   repository. Where a decision improves only the shape of the code, it waits
   until a release needs that shape for something else.

### 1.1 The release ladder

| Release | Content | Breaking |
| --- | --- | --- |
| **0.9** | Correctness, latent silent-wrongness, two resource wins | One function: `config.set_memory_budget` |
| **0.10** | Methods that work on data you already have | No |
| **0.11** | Methods that use gradients | No |
| **1.0** | Coverage, stability, release engineering | No |

---

### 1.2 Why 0.9 was re-cut

The original plan put all 21 audit decisions in 0.9 as one breaking release.
Batches 1 and 2 shipped under that plan. Their outcome is the reason for the
change.

**What the completed work shows.** The items that paid were the ones with a
behavioural target: `JaxgsaWarning`, the 3.4x faster row deduplication, the
Morris `downsample` bug, and the test sweep, which found unasserted near-zero
entries, a Sobol coverage hole, and a test that compared SciPy against SciPy.

The two changes that produced **new** defects were both pure refactors. The
`np.unique` rewrite was a performance *regression* until it was re-measured,
and the rewritten Sobol test covered the wrong code path. Three independent
review agents were needed to catch them. That is the cost of a change with no
behavioural target: the same gate cycle, and a defect rate of its own.

**Three further reasons.**

1. **The breaking-change budget is better spent elsewhere.** A user pays a
   migration cost either way. Spending it on internal ergonomics buys them
   nothing.
2. **D1 pays for itself later, and only once.** Release 1.0 adds input
   distributions (section 8), which needs an internal `Marginal` protocol.
   That is when the specification representation genuinely has to change.
   Doing it now means doing it twice.
3. **Timing.** `jaxonomy` was created on 2026-07-06 and already has more
   downloads per month than jaxgsa. The roadmap's own judgement is that the
   first-mover position is real but not permanent. Internal refactoring is the
   most expensive way to spend that window.

---

## 2. Corrections to the source documents

A verification pass on 2026-08-18 checked every file and line reference in both
documents against the working tree at `03ee1f3`. Most references are exact. The
following are not.

### 2.1 Two corrections that change the work

**C1. D9 cannot be implemented at all, and the decision behind it is wrong.**

*Superseded on 2026-08-18 by measurement. The paragraphs below record why the
audit's fix was unbuildable; the deeper finding is that it should not be built.*

D9 rests on the claim that a kernel narrower than one grid step "guarantees"
a failed computation. It does not. Two conditions must both hold before the
integral breaks: a conditioning class must actually be **degenerate**, so the
floor is applied at all, and the resulting spike must land **on a grid point**,
so the trapezoid rule sees it.

Measured on a fixture built to have a genuinely degenerate class, where one
grid step is `0.108 * h_full`:

| `degenerate_bandwidth` | spike on a grid point | spike off the boundary |
| --- | --- | --- |
| 0.100 | delta 0.6721 | 0.7336 |
| 0.010 | delta 0.9433 | 0.5982 |
| 0.001 | **fails**, delta 4.01 | 0.5982 |
| 1e-05 | — | 0.5982 |

So a kernel one tenth of a grid step returns a valid answer, and off the
boundary the estimate is stable five orders of magnitude below the step.
Whether it breaks is a property of where the class sits relative to the grid,
which is **data, not configuration**. No configuration-only check can be a
true precondition.

A first attempt shipped the audit's rule and refused four configurations that
return bit-identical, correct results — including `degenerate_bandwidth=0.1`,
the very fraction `"auto"` uses internally. That was caught in review before
it left the branch.

**What ships instead.** No up-front raise. The existing out-of-range error now
builds its advice from what the kernel actually did: whether a class was
floored, which column, the floor it used against the real grid step, and the
fraction that would fix it. Cost was never the obstacle — a host-side
degeneracy scan measured 282 ms against `analyze`'s 5037 ms — it simply would
not have been correct.

The original objection, still true: The audit says to raise at the top
of `borgonovo.analyze` when `degenerate_bandwidth * h_full < grid_step`. That
test cannot run there. `h_full` is computed per output column *inside the jitted
kernel* at `borgonovo/_analyze.py:267`, from `jnp.std(y_r)`. `grid_step` is also
data-dependent, at `:240`. Neither value exists at the top of `analyze`.

This is the same objection D9 itself raises when it refuses to warn about
`degenerate_tol`.

**C2. `on_invalid="raise"` is a behaviour change, not a new argument.** The
roadmap says only `sobol` and `morris` handle non-finite output. Four modules
do: `sobol`, `morris`, `kucherenko`, and `efast` (warn-only). Three of them
*already* drop bad rows by default. Making `"raise"` the default changes what
those three do today. It is the **one** deliberate behaviour change in 0.9,
and it is a correctness fix: silently dropping rows changes what the estimator
computes, so it must not be the default. Everything else in the release is
additive or invisible.

### 2.2 Corrections that do not change the work

| Claim | Correction |
| --- | --- |
| D2: `poincare_constant` and `marginal_variance` are both public | Only `poincare_constant` is exported. `dgsm/__init__.py:23,26` exports `axis_constants` and `poincare_constant`. D2 applies to one signature, not two. |
| D10: the unreachable branch is at `_sampling.py:151-153` | It is at `:158-162`. Lines 151-153 are the `D == 1` early exit, which is reachable. |
| D12: `slice_chunk_size` "is validated and then ignored" | It is not validated either. The validation block at `pawn/_analyze.py:363-368` covers `statistic`, `n_bins` and `conf_level` only. |
| Section 5: `SobolSamples.sample_ids` has zero readers in `src/` | It has zero *algorithmic* readers. It is read by the `.npz` payload at `sobol/_sampling.py:311` and restored at `:331`. The stated ordering constraint gets stronger, not weaker. |
| Section 5: the Sobol `slice_chunk_size` test passes `num_resamples=0`, and that argument reaches only the bootstrap path | It reaches both paths, but it **means a different thing in each**, which the audit and an earlier draft of this table both missed. In `_analyze_no_bootstrap` it chunks **output columns**. In `_analyze_bootstrap` it chunks **resamples**, and the point estimates there come from `jit_ft`/`jit_so` per slice and do not depend on it at all. So a bootstrap-only invariance test asserts nothing about `S1`, `ST` or `S2` — only about the `_conf` fields. A real test needs both a `num_resamples=0` pair and a `num_resamples>0` pair. The gap was real; neither stated reason for it was. |
| Section 5: 31 test files, ~14 renamed-keyword tests, 6 paired tests | 33 files. 7 verbatim `test_old_chunk_size_kwarg_raises` copies plus 7 paired "accepted" tests, so 14 in the cluster. The total is right by coincidence. |
| D8: `tests/test_pce.py:138-162` has the gap | The gap is at `:140-150` and `:152-162`. The PCE `ST` pair has **no** near-zero analytical entry at all — the smallest Ishigami total index is `ST[x3] = 0.2437` — so there was never an entry to add an `else` for. Its guard was dead code and was removed, so every entry is now asserted unconditionally. The real gap in that pair was the `S1` `else`. |
| Audit section 5 line numbers for two mirrors | The Borgonovo mirror is at `tests/test_borgonovo.py:368`, not `:381` (`:381` is an unrelated test). The DGSM mirror assert is at `tests/test_dgsm.py:62`, not `:61` (`:61` is the `poincare_constant` call). |
| Section 5: the renamed-keyword cluster is 14 tests | 13 in practice. `tests/test_pawn.py::test_slice_chunk_size_kwarg_accepted` was **kept**: PAWN has no chunk-invariance sibling, because `slice_chunk_size` is a documented no-op there until D12 implements it. Deleting it would have left the keyword with no coverage at all. Batch 4 replaces it with a real invariance test. |
| D7: Sobol guard at `sobol/_result.py:116`; D4: clamp at `optimal_transport/_analyze.py:344`; D13: scalar squeeze at `sobol/_analyze.py:215-217`; roadmap 3.1: `_sampling.py:230` and `_analyze.py:188` | Off by one or two. Real lines: `_result.py:115`, `_analyze.py:342`, `_analyze.py:216-218`, `_sampling.py:224` and `_analyze.py:190`. |
| D21: 35 `warnings.warn` sites and 58 `pytest.warns(UserWarning` assertions | 34 sites and 57 assertions. A plain grep gives 35 because `shapley/_analyze.py:56` mentions `warnings.warn` inside a comment. Confirmed by an AST walk, which is what the D21 guard test now uses. Shapley has 2 real sites, not 3. The "zero pass a category" claim was exact. |
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
- **Conflict rule.** Two agents must never hold the same file. Give each agent
  an explicit list of the files it owns, and tell it to report rather than edit
  anything outside that list. That rule is what let batches 1 and 2 run
  concurrently in one worktree without a single collision.

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

## 4. Tranche 0: clear the ground (done)

Shipped as `dd9ebc4`, `3af9635` and `0140832`.

| Task | Detail |
| --- | --- |
| **T0.1 Prune worktrees** | Eleven worktrees exist under `.claude/worktrees/`, most from merged pull requests (#31 to #34), plus two detached review checkouts. Remove the merged ones and their branches. |
| **T0.2 Commit the two planning documents** | `AUDIT-DECISIONS.md`, `ROADMAP-1.0.md` and this file are untracked. Commit them so later work can cite a revision. |
| **T0.3 The audit's suggested first batch** | Five independent one-file changes: rename the shadowing eFAST fixture (`tests/test_efast.py:423-424`), delete `_PCEFit.coeffs_flat`, delete `validate_correlation` and repoint its 11 tests at `canonicalize_correlation`, fix `CITATION.cff` (drop `version` and `date-released`), declare `numpy>=2` in `pyproject.toml`. |
| **T0.4 Baseline** | Record the test count (979), the benchmark numbers, and a fixed-seed index dump for every method. Gate 5 compares against this file for the rest of 0.9. |

Eleven worktrees and seven merged branches were removed. Four branches with
commits in no other branch, whose pull requests were closed unmerged, were
kept: `feat/shapley-correlative` (#21), `worktree-emulate-batching` (#25),
`worktree-feat+hdmr-s2-s3` (#22), `worktree-borgonovo-delta`.

`_PCEFit.coeffs_flat` was deleted here, so D15's field addition in batch 4
lands on clean lines.

---

## 5. Release 0.9: correctness

Two batches are done. Two remain. Nothing here breaks an API.

### 5.1 What was cut, and why

These decisions improve the shape of the code. None of them fixes anything a
user can hit. They are not cancelled; they wait for a release that needs them.

| Decision | Why it waits |
| --- | --- |
| **D1** spec tuple to dataclasses, **D2** public signature rule | The largest single item in the audit: ~10 modules, ~60 test construction sites, plus docs and benchmarks. Release 1.0 adds input distributions, which needs an internal `Marginal` protocol carrying the inverse CDF, the polynomial recurrence, the Poincare constant and the copula transform. That is when this representation has to change. Fold D1 and D2 into that work and do it once. |
| **D3** `YLayout` enum | 15 `_prepare_Y` sites, 48 `_squeeze_output_axes` sites, 13 modules, no known defect. The stated motivation is that the fourth boolean combination is silently ignored at `_core/validation.py:402`. That is worth one assertion, not a 48-site refactor. Keep the assertion in mind for whoever is next in that file. |
| **D4** partition counts shape, **D5** partition group levels | Real latent risk, but the pair rewrites `_core/partition.py` and both consumers. The clamp at `optimal_transport/_analyze.py:342` masks an out-of-range index; before deferring, confirm whether it can actually trigger. If it can, that half is a correctness item and moves into 5.4. |
| **D6** freeze eleven result classes | The genuine footgun is one class: assigning to `MorrisResult.space` relabels the arrays without converting them. Fixing eleven classes to close one hole is the wrong ratio. Freeze `MorrisResult` alone if it proves to bite. |
| **D13** nine Sobol accumulators to three | Tidiness. No defect. |
| Delete `SobolSamples.sample_ids` and the generic `.npz` payload | Breaks saved design files for no user-visible gain. |
| The 0.8-to-0.9 migration guide | Not needed: there is nothing to migrate. |

**What this changes about the order of work.** Every forced ordering that
involved D1 or D3 dissolves. D9, D11 and D12 no longer collide with anything,
so batch 3 can be split freely across agents.

### 5.2 Batch 1 — tests (done)

Shipped as `49c04e0`. The layout-pinning rewrite still stands on its own merit:
no test now spells the private specification layout, which is worth having
whenever D1 does land.

### 5.3 Batch 2 — warnings, packaging, performance (done)

Shipped as `2cbaa44`. `JaxgsaWarning`, the 3.4x faster row deduplication, the
Morris `downsample` fix, the pager checker, and the `cdf_to_unit_interval`
comment.

### 5.4 Batch 3 — the defects a user can hit

The core of the release. These are independent and may run in parallel.

| Task | The defect |
| --- | --- |
| **Failed-evaluation policy** (roadmap 3.3) | The biggest item in either document. A non-finite model output flows straight into the indices in nine of thirteen modules. `sobol`, `morris` and `kucherenko` silently drop rows; `efast` warns and continues. Add one shared `on_invalid={"raise","propagate","drop"}` policy in `_core/`, generalising `sobol._drop_nonfinite`, and thread it through all thirteen `analyze()` entry points. Report the count, the row indices and their position in the design: `nan_counts` today counts NaNs in the *output indices* and throws row identity away at `sobol/_analyze.py:104`. Default `"raise"`. Three of the entry points are named `analyze_pce`, `analyze_hdmr`, `analyze_vkoga`, so a grep for `def analyze(` misses them. |
| **D11** DGSM argument groups | `analyze(problem, X=X, Y=Y, dfdx=J)` takes the precomputed branch and silently ignores `X`. The check that validates `X` against the problem's bounds runs only in the other branch, so a user who passed inputs and believed they were checked had them discarded unchecked. Resolve both groups once at the top, raise on both-given, neither-given, or partly-filled. Fix the `Raises:` list at `:212-218`. |
| **DGSM batch callables** | Found while building the baseline: `dgsm.analyze` takes a one-sample `(D,) -> ...` function. A batch callable dies with `IndexError` from deep inside. Add the check next to D11's resolution. |
| **D12** PAWN `slice_chunk_size` | Declared, documented as "accepted for signature parity", and not even validated. PAWN nests two `vmap` calls, so the whole `(T*K, D, n_bins)` computation materialises at once — on the time-series case the project advertises with a dedicated example. Copy the Sobol or Borgonovo pattern. Replace the placeholder test at `tests/test_pawn.py:46` with a real chunk-invariance test. Remove the "signature parity" wording. |
| **D9** Borgonovo bandwidth | **Done, but not as the audit specified — see correction C1.** The premise is false: a bandwidth below one grid step fails only when a class is degenerate *and* its spike lands on a grid point, which is data, not configuration. No up-front raise ships. The out-of-range error instead builds its advice from what the kernel did. The two resolvers are collapsed and the `degenerate_tol` bias is documented, as planned. |
| **D10** eFAST frequency plan | The sampler assigns complementary frequencies in `[1, omega_0 // (2*M)]`; the analyzer reads `arange(omega_0 // 2)`. Two separately written bounds that agree today only because one range contains the other. One `_frequency_plan(D, n_per_curve, M)` called from both sides. Keep `omega_0` a concrete int in the kernel cache key. Absorb `_min_n_per_curve`; delete the unreachable branch at `_sampling.py:158-162`. Add a test that the two sides read the same band — the current tests recompute the formula and structurally cannot catch this. |
| **D7** redundant guards | Two lines. The Sobol one silently exports nothing when `S2_conf` is present without `S2`, while `__repr__` still advertises the field. |

### 5.5 Batch 4 — latent silent-wrongness, and two resource wins

Each of these produces a wrong number rather than an error if someone edits
nearby code. All are cheap.

| Task | Why |
| --- | --- |
| **D16** HDMR static data | Twelve values unpacked positionally at three sites, two of them with blind placeholders. Reordering gives silently wrong indices, not an error. Return a `NamedTuple`, fix the `-> tuple` annotation, and extract the order map written verbatim in both `_engine.py:411-422` and `_stream.py:438-446`. Remember the third unpack site at `_analyze.py:406`. |
| **D17** `chol_full` on the plan | Two call sites build a `_ConditionalPlan` and then call `np.linalg.cholesky` on the same matrix, carrying the two as unlinked values. One function takes the parameter count from one and the index geometry from the other; a mismatched pair gives wrong indices, not an error. The raw call also bypasses `_safe_cholesky`, which exists because the module accepts matrices that are only just positive definite. |
| **D14** PCE leave-one-out | Computed twice, two different ways, with a comment asserting they agree. Compute leverage once from a Cholesky factor of the Gram matrix. The memory estimate charges three `N x n_terms` arrays where two are needed, so streaming engages at the wrong point. This is the one item in the release that deliberately moves a number. |
| **D15** `streamed` flag | Honest observability, and it replaces tests that monkeypatch a module function and count calls — which freeze the import style and the loop structure. Needed alongside D14, or a fit that silently stops streaming passes every test. |
| **D18** HSIC dispatch and bandwidth | **Done.** Collapsed the two-by-two dispatch across five helpers, deleted the unreachable `raise`, and replaced the upper-triangle index machinery with an index-adjusted quantile. Measured: `3.57 * N^2` down to `1.00 * N^2`, saving 164 MiB at `N = 4096`. `batch_size` bounds working memory, not the kernel matrix, and now says so. **Correction:** the quantile position this document originally gave, `q = (N + (N^2 - N)/2) / N^2`, is wrong — it lands half a slot short and biases toward the lower order statistic at even counts, a 15% relative error at `N = 4`. The exact position is `q = (N^2 + N - 1) / (2 * (N^2 - 1))`, which reproduces the strict-upper-triangle median to zero error for every `N` from 4 to 1024. |

### 5.6 Optional

**D20**: the benchmark tables stand as published; only the harness needs
tidying. Do it when someone next touches benchmarks.

### 0.9 exit criteria

- Batches 3 and 4 merged, each through the six gates.
- Test count stated and reconciled.
- Fixed-seed baseline identical, except D14's streaming threshold.
- `CHANGELOG.md` entry for 0.9.0 marking it non-breaking, with the single
  `on_invalid` default change called out plainly.

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
| **More input distributions**, carrying **D1** and **D2** | Internal `Marginal` protocol plus adapters for `scipy.stats` and optionally `numpyro`. No new required dependency. This is where the input-specification representation changes: the protocol has to carry the inverse CDF, the density, moments, support bounds, the orthogonal polynomial recurrence, the Poincare constant and the conditional-copula transform — none of which any distribution library supplies. Do D1's public `UniformSpec` / `GaussianSpec` / `CategoricalSpec` dataclasses here, in the same pass, and D2's public-signature rule with them. This is the release's one breaking change and it earns the migration guide. Keep `probs` and `labels` as tuples: the specification is jit-cache metadata on a hashable `Problem`. |
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
T0  ground clearing                                          done
0.9 B1 tests            done
    B2 warnings/perf    done
    B3 user-visible defects   NaN policy, D11, DGSM callable, D12, D9, D10, D7
    B4 latent + resource      D16, D17, D14+D15, D18
     |
0.10 rank-based -> intervals -> dummy -> estimators -> RSA -> HSIC p-values
     |
0.11 active subspaces -> crossed DGSM -> Poincare oracle -> mode selection
     -> d(index)/d(theta)
     |
1.0  kNN Shapley -> OT parity -> generalised Sobol
     -> distributions (carrying D1 and D2) -> fixture -> release engineering
```

Only two orderings remain forced, and both are inside batch 4: D14 before or
with D15, because D14 moves the point at which streaming engages and D15 is
what proves it still engages; and the order map in D16 must be extracted
before either copy is edited.

Every other ordering in the original plan existed because of D1 or D3. Both
are deferred, so batch 3 and batch 4 are free to run in parallel, and every
task inside batch 3 is independent of every other.

### What deferring costs

Nothing in 0.10 or 0.11 depends on D1, D3, D4, D5, D6 or D13. The methods in
those releases add new modules and read existing results; none of them
rewrites the shape plumbing. The one real cost is that each new method written
before D3 threads `squeeze_time` and `squeeze_output` by hand, the same way
the existing thirteen do. That is a known, copyable pattern, and the baseline
harness catches it if a new method gets it wrong.
