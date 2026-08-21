# Audit decisions

Status: agreed 2026-08-18; most items implemented for 1.0.0. See
section 10 for D4, D5, D6, `sample_ids`, and the D9 reversal.

## 1. What this document is

This document records **21 decisions** about how to change existing jaxgsa code.
Each decision came from a code audit of the whole repository, followed by a
review session that settled the open questions.

Each entry states what changes, why, and where. Each entry is small enough to
implement on its own.

### What this document is not

This document does **not** plan the road to version 1.0. It does not choose new
methods, name papers, or set verification oracles. That work belongs to
`ROADMAP-1.0.md`, which is a separate document with a separate purpose.

The two documents touch in one place only. `ROADMAP-1.0.md` defines a tier
system for verification oracles, from T0 (a closed-form answer) to T4 (internal
consistency only). Decision 8 below applies that existing policy to a test that
currently breaks it. No other decision here depends on the roadmap, and nothing
here changes it.

A later document is expected to merge both: first the fixes recorded here, then
the new methods from the roadmap.

## 2. Terms used here

Define these once, then reuse them. They appear throughout.

| Term | Meaning |
| --- | --- |
| **Spec tuple** | The internal 6-slot tuple that describes one input parameter's distribution. Private, named `_NormalizedInputSpec`. |
| **Y layout** | Which shape the user passed for model outputs: `(N,)`, `(N, K)`, or `(N, T, K)`. Here `N` is samples, `T` is time steps, `K` is outputs. |
| **Output slice** | One `(t, k)` pair. A time-series analysis has `T x K` output slices. |
| **Conditioning class** | A bucket of samples grouped by an input's value. Borgonovo and optimal transport both use these. |
| **Leverage** | The influence of one sample on its own fitted value. Used to compute leave-one-out error without refitting. |
| **Search curve** | The sinusoidal path through parameter space that eFAST walks. One curve per parameter. |
| **Streaming fit** | The low-memory fit path in PCE and HDMR. It processes data in batches instead of all at once. |

## 3. Ground rules

Four rules were agreed first. They decide the shape of everything below.

1. **Breaking the public API is allowed.** Version 0.9 is a breaking release.
   Treat it the way 0.4 was treated, including a migration guide.
2. **Breaking saved `.npz` design files is allowed.** No backward-compatible
   reader is required.
3. **Prefer one data structure over two.** Where the same idea has two
   representations, keep one.
4. **A test must justify its presence.** A test that must be edited whenever the
   source changes, with no behaviour change, asserts implementation rather than
   behaviour. Delete or rewrite it.

## 4. Decisions

### Group A: input specifications

#### D1. Replace the spec tuple with one public dataclass family

**Decision.** Delete the private spec tuple. Delete the public input TypedDicts
(`UniformInputSpec`, `GaussianInputSpec`, `CategoricalInputSpec`). Replace both
with **one** family of frozen dataclasses, exported from the package root:

```python
UniformSpec(low, high)
GaussianSpec(mean, variance, low=None, high=None)
CategoricalSpec(probs, labels)
```

`Problem.input_specs` returns these. `Problem.from_dict` accepts these.

**Why.** The current tuple uses the same slots for different meanings. Slots 1
and 2 hold `(low, high)` for a uniform parameter and `(mean, variance)` for a
Gaussian one. A categorical parameter fills them with dummy zeros. The type
cannot state which combinations are legal, so the code carries two
`assert ... is not None` guards and the same apologetic comment in two files.
Positional indexing leaks into ten modules.

Keeping both the TypedDicts and the dataclasses would leave two public
vocabularies for one idea, with names one word apart. One family avoids that.

**Note on the Gaussian bounds.** `low` and `high` stay nullable. Truncation is
not a yes-or-no fact: a Gaussian can be open on both sides, on one side, or on
neither. All four states are reachable and all four are handled, including the
one-sided case in `dgsm/_poincare.py`.

**Where.** `problem.py` owns the change. Roughly ten call sites need one line
each: `_core/transforms.py`, `_core/sampling.py`, `_core/samples.py`,
`pce/_analyze.py`, `morris/_sampling.py`, `sobol/_sampling.py`,
`dgsm/_poincare.py`, `pawn/_analyze.py`.

**Keep.** `_normalize_input_spec` and `_normalized_input_to_dict` stay as
internal helpers. The `.npz` round trip uses them. They stop being backed by
public named types.

`Problem.from_dict` keeps its name. Its "dict" is the outer mapping from
parameter name to spec, not the spec itself.

#### D2. State one rule for public signatures

**Decision.** A public function must not name a private type in its signature.
`dgsm.poincare_constant` and `dgsm.marginal_variance` currently take the private
spec tuple. After D1 they take the public dataclass.

**Why.** `poincare_constant` is exported and documented, but a user cannot
legitimately call it. They must either reach through `Problem.input_specs` or
hand-build a tuple from a source comment.

**Order.** Apply this before D1. It removes those two signatures from D1's
public surface.

### Group B: shapes and state

#### D3. Replace the two squeeze booleans with one layout value

**Decision.** `_prepare_Y` returns a `YLayout` enum instead of two booleans:

```python
class YLayout(Enum):
    SCALAR = 1        # user passed (N,)
    MULTI_OUTPUT = 2  # user passed (N, K)
    TIME_SERIES = 3   # user passed (N, T, K)
```

`_squeeze_output_axes` maps the layout to an index tuple through a three-entry
table.

**Why.** Two booleans encode three real states. The fourth combination is
accepted and silently ignored at `_core/validation.py:402`. The pair is threaded
through ten analysis modules and stored on the PCE fit object, far from where it
was computed.

**Scope note.** This decision owns **all** Y-layout representation. That includes
the third encoding, `is_scalar = Y.ndim == 1`, which exists in **two** places:
`sobol/_analyze.py:201` and `efast/_analyze.py:178`. No other decision may touch
layout handling. This matters because D13 rewrites nearby Sobol code.

#### D4. Give the partition counts array one shape

**Decision.** `counts` in `_core/partition.py` becomes unconditionally
`(R, Dg, Mg)`, matching the first three axes of `cls_idx`. Broadcast once when
the array is built.

**Why.** Today the first two axes are each either their full size or 1, giving
four possible shapes where only two are meaningful. Three places emulate
broadcasting by hand, including an index clamp at
`optimal_transport/_analyze.py:344` that silently returns a wrong answer instead
of raising when an index is out of range.

The stated reason for the trick does not survive arithmetic. The `cls_idx` array
in the same tuple is already materialised at full size, and a full `counts` is
about `1/Pg` of it. For `R=100, D=20, N=10k, M=100` that is roughly 0.8 MB
against 80 MB.

**Effect.** `_replicate_slice` deletes entirely. The clamp becomes plain
`dm // M`. Four docstrings lose a caveat.

**Care.** Keep the row mask when rewiring, or cross-validation will train on
held-out rows. Assert `counts.shape == cls_idx.shape[:3]` during the migration.
This is shape plumbing only, so any change in a computed number means a wiring
error, not a tolerance issue.

#### D5. Carry the per-group level lists out of the partition builder

**Decision.** `build_partition_groups` returns
`(groups, group_levels, col_order)`. Build `group_levels` inside the same two
branches that build `groups`.

**Why.** The middle return value is currently `group_dims`, which every external
caller discards. Meanwhile `optimal_transport/_analyze.py:669-673` rebuilds the
level lists by hand, relying on a comment that says the group order "mirrors
`build_partition_groups`". If that order ever changed, the mismatch would not
raise. The optimal-transport kernel would solve the wrong classes and return
indices that are quietly too small.

**Rejected alternative.** Do **not** make the group a `NamedTuple` carrying the
levels. A `NamedTuple` is a JAX pytree. The groups are passed as jit arguments in
both optimal transport and Borgonovo, so a Python list field would become traced
values in the jit cache key.

#### D6. Freeze the result classes

**Decision.** Add `frozen=True` to eleven result dataclasses. Leave
`HDMRResult` mutable. `KucherenkoResult` is already frozen.

**Why.** Results are values. Nothing should edit one after an analysis produces
it. `MorrisResult` shows the cost of mutability: it carries a `space` label
saying whether its measures are in unit or physical coordinates, and assigning
to that field relabels the arrays without converting them. The guard against
double conversion then refuses to run.

**Why HDMR is excluded.** `HDMRResult` is the only class using
`cached_property`, at lines 222 and 251, for the lazy `S2` and `S3` interaction
tensors. `cached_property` stores its value by writing to the instance
dictionary, which a frozen dataclass blocks. Making those tensors eager would
make every HDMR analysis pay for arrays most callers never read. Add a one-line
comment recording this.

**Migration for users.** Freezing does not remove any capability.
`to_physical_units()` already returns a new object. To change a result, use
`dataclasses.replace()`. To relabel a plot, rename on the exported dataset with
`ds.rename(...)`, or pass your own labels to the plotting call. Document these
three paths in that order.

#### D7. Delete the redundant guard clauses on result fields

**Decision.** Simplify two guards to match what the producers actually do. Do
**not** add `__post_init__` validation.

```python
# sobol/_result.py:116
if self.S2_conf is not None:              # drop "and self.S2 is not None"

# morris/_analyze.py:374
if mu_conf is not None:                   # drop the other two checks
```

**Why.** The three Morris confidence arrays are set together at
`morris/_analyze.py:365-369`, so a triple check tests a two-state fact. The Sobol
guard silently exports nothing when `S2_conf` is present without `S2`, while
`__repr__` still advertises the field.

Validation at construction was considered and rejected. The library cannot
produce these states. Adding checks to five classes to catch a mistake only a
hand-built result can make trades code for very little. Deleting the guards
removes code instead.

### Group C: correctness and honesty

#### D8. Assert the near-zero entries in accuracy tests

**Decision.** Every accuracy test that compares indices against analytical values
must assert something for **all** entries, including near-zero ones. Where an
estimator cannot meet a tight bound, record a loosened absolute tolerance with a
comment naming the bias. Do not leave an entry unasserted.

**Why.** `tests/test_efast.py:442-450` guards on `if analytical[i] > 0.01:` with
no `else` branch. Every near-zero total-order entry of the 8-dimensional Sobol
G-function is therefore checked by nothing. The neighbouring `test_s1` has the
`else` branch. `tests/test_pce.py:138-162` has the same gap.

This follows the existing tier policy in `ROADMAP-1.0.md`. That document ranks a
closed-form answer as the strongest oracle (T0), instructs authors to use the
strongest tier available, and requires that weaker verification be stated in the
test. A test that claims a closed-form check while skipping part of the vector
meets none of those conditions.

#### D9. Raise when a Borgonovo bandwidth cannot be integrated

**Decision.** Raise a `ValueError` at the top of `borgonovo.analyze` when
`degenerate_bandwidth * h_full < grid_step`. Name both knobs in the message. Do
not clip the value.

**Why.** The delta index is a half L1 distance between densities, so a result
outside `[0, 1]` is a failed computation, not a poor estimate. The code already
refuses such results. A kernel narrower than one grid step guarantees that
failure, because the numerical integral cannot resolve the spike.

The `"auto"` setting is bounded below by the grid step for exactly this reason. A
user-supplied float is documented as applied exactly, with no grid-step bound, so
it can configure a computation that is certain to fail. Failing up front is
better than failing three steps later.

Do not clip. The existing error message states the principle: a clipped value
would look plausible and still be wrong.

**Also.** `degenerate_tol` and `degenerate_bandwidth` can disagree in a second,
milder way. Raising `degenerate_tol` above the floor fraction narrows the kernel
of the affected classes, which inflates delta for the classes you distrusted.
That computation is valid, only biased, so it does not raise. Add one sentence to
the `degenerate_tol` docstring. Do not warn: an accurate warning depends on data
inside a jitted kernel, and a configuration-only warning would be a false alarm
in most runs.

**Also.** Collapse `_resolve_bandwidth` and `_resolve_degenerate_bandwidth`
(lines 790-860) into one sentinel resolver. They are the same string-or-float
ladder with different words.

#### D10. Compute the eFAST frequency plan once

**Decision.** Add one helper, `_frequency_plan(D, n_per_curve, M)`, returning
both the focal frequency and the assigned complementary frequencies. Call it from
both `sample` and `analyze`. Store nothing new on `EFASTSamples`.

**Why.** `sample` computes the focal frequency, builds the design from it, then
discards it. `analyze` recomputes the same expression. Worse, `analyze` picks its
own complementary band, `arange(omega_0 // 2)`, while the sampler assigns
frequencies in `[1, omega_0 // (2*M)]`. Those are two separately written bounds.
They agree today only because one range happens to contain the other.

If either changes, `analyze` reads a different frequency bin than the one the
design put the signal in. There is no exception and no shape error, only wrong
indices that look plausible. The current tests cannot catch this, because they
recompute the same formula.

**Also.** The helper absorbs `_min_n_per_curve`, which encodes the same relation
a third time, and lets you delete the defensive branch at `_sampling.py:151-153`
whose own docstring calls it unreachable.

#### D11. Reject over-specified DGSM calls

**Decision.** Resolve the argument groups once, at the top of `dgsm.analyze`,
before any computation. Raise when arguments from both groups are given, when
neither is, or when one group is only partly filled. Name the conflicting or
missing argument. Leave both branch bodies unchanged.

**Why.** `dgsm.analyze` has two valid call styles: give it a model and inputs, or
give it precomputed outputs and derivatives. The current dispatch is first-match,
not exclusive, so passing arguments from both groups silently drops some of them.

The worst case is `analyze(problem, X=X, Y=Y, dfdx=J)`. It takes the precomputed
branch and ignores `X`. The check that validates `X` against the problem's bounds
and shape runs only in the other branch. A user who passed inputs and believed
they were checked had them discarded unchecked.

#### D12. Implement PAWN's output-slice chunking

**Decision.** Make `slice_chunk_size` work in `pawn.analyze`. Chunk the outer
`vmap` over output columns, matching Sobol, Borgonovo, HDMR, and eFAST.

**Why.** The parameter is declared, documented as "accepted for signature
parity", and appears nowhere else in the file. It is validated and then ignored.

PAWN nests two `vmap` calls, the outer one over output columns, so the whole
`(T*K, D, n_bins)` computation is materialised in one call. Large `T*K` is the
time-series case, which the project advertises with a dedicated example. The
configuration where the missing chunking hurts is one the project recommends.

The name is not ambiguous. It means the same thing in four sibling modules.

#### D13. Give the Sobol bootstrap path one set of accumulators

**Decision.** Replace the nine parallel accumulator lists in
`_analyze_bootstrap` with three. Accumulate one `(2, D)` array per output slice
directly from `_bootstrap_ci_endpoints`.

**Why.** Nine lists must each receive exactly one append per output slice, in the
same order, with three of them appended only under a condition. Nothing enforces
that except the lines sitting near each other. Six of the nine exist only to be
restacked into a layout that `_bootstrap_ci_endpoints` already built and then
split apart.

**Do not** also unify the point estimates across the bootstrap and non-bootstrap
paths. That changes numbers. **Do not** route the scalar fast path through shared
assembly: it squeezes by construction at `sobol/_analyze.py:215-217`, which is
correct, and calling the shared squeeze on its output would index into the
parameter axis.

**Order.** Apply D3 first. Both touch `sobol/_analyze.py:199-238`.

### Group D: fit paths

#### D14. Compute the PCE leave-one-out error in one place

**Decision.** Both fit paths compute leverage from a **Cholesky factor** of the
Gram matrix, which is `(n_terms, n_terms)`. Delete the second copy of the
formula. Update the memory estimate in the same change.

**Why.** Leverage is computed twice, in two different ways. The single-pass path
builds an intermediate of shape `(n_terms, N)`, one column per sample, and
`loo_error` then takes its transpose, materialising a third array that scales
with `N`. The streaming path cannot afford anything that large, so it carries the
small Gram inverse and re-implements the formula inline. A comment asserts that
the two agree.

**Why a Cholesky factor rather than an explicit inverse.** The default ridge is
`1e-8`, deliberately small. PCE Gram matrices become badly conditioned as the
polynomial order rises, and forming an explicit inverse worsens that. A degraded
leave-one-out value would feed back into automatic order selection.

**Accepted consequence.** The memory estimate currently charges three arrays of
size `N x n_terms`, one of them for the intermediate being removed. Correcting it
to two changes the point at which the streaming fit engages, so some fits that
stream today will stay single-pass. That is a more accurate estimate, and the
corrected check verifies the real requirement.

#### D15. Record which fit path ran

**Decision.** Add `streamed: bool` to `PCEResult` and `HDMRResult`.

**Why.** Two reasons.

First, this is honest observability. A user whose fit took much longer than
expected has a real reason to ask whether the memory budget engaged.

Second, it removes a test that freezes the source. Today the streaming tests
replace a function on the module object and count calls. That silently forbids
changing the import style in the production module: a local import, an alias, a
closure, or inlining would fail the test with no behaviour change. The call count
also encodes the exact loop structure.

**Also.** Weaken the surviving structural assertion. Assert the fit streamed,
not that a function was called exactly eight times. Without some such check, a
future change to the memory estimate could stop the streaming path from engaging
at all, and every behavioural test would still pass, because both paths agree by
design. D14 changes that estimate.

#### D16. Name the HDMR static data, and share the order map

**Decision.** Two changes, both small.

1. `_get_hdmr_static_data` returns a `NamedTuple` instead of a bare 12-value
   tuple. Fix its `-> tuple` annotation.
2. Extract the per-term order map, which is written twice, into one helper.

**Why.** The twelve values are unpacked positionally, twice with blind
placeholders, at `hdmr/_analyze.py:87` and `:686`. Reordering the tuple produces
silently wrong indices rather than an error.

The order map, which records each term's order, basis count, and critical F
value, is written verbatim in both `_engine.py:411-422` and `_stream.py:438-446`.
The second admits in a comment that it mirrors the first. They must stay
bit-identical or an exact-equality test fails.

**Deliberately out of scope.** Do not thread the new object through
`_make_hdmr_kernel`, `_f_test`, `_fit_hdmr_streamed`, and `_full_fit_bytes`.
Those long positional argument lists are unpleasant, but HDMR is the most
numerically delicate module here, and the kernel closure captures concrete Python
integers that must not become traced. The naming change already removes the
failure that bites.

**Correction.** An earlier draft warned that the returned object must use tuple
fields to protect a cache. That was wrong. Nothing is keyed on the returned
object, and the current tuple already contains two NumPy arrays.

#### D17. Attach the full Cholesky factor to the conditional plan

**Decision.** Add `chol_full` to `_ConditionalPlan`, computed with the module's
own `_safe_cholesky`. Append the field; do not insert it.

**Why.** Two call sites build the plan and then immediately call
`np.linalg.cholesky` on the same matrix. They carry the two results as unlinked
values, and one function takes the parameter count from one and the index
geometry from the other. A mismatched pair produces wrong indices rather than an
error.

The raw call also bypasses `_safe_cholesky`, which exists because the module
deliberately accepts matrices that are only just positive definite.

### Group E: reduce the memory HSIC uses

#### D18. Collapse the HSIC kernel dispatch and fix the median heuristic

**Decision.** Two changes in one function.

1. Resolve the squared bandwidth first, then decide whether to build the kernel
   in blocks. This replaces a two-by-two dispatch across five helpers.
2. Replace the upper-triangle index machinery with an index-adjusted quantile:

```python
q = (N + (N**2 - N) / 2) / N**2
sigma_sq = jnp.maximum(jnp.quantile(dists_sq, q), 1e-20)
```

**Why the dispatch collapse.** Two independent questions — how the bandwidth is
chosen, and whether the matrix is built in blocks — are multiplied into four
paths. One branch contains an unreachable `raise` that exists only to satisfy the
type checker. The median path exists in two incompatible forms, one taking a
square root that the other re-squares, which is why the test comparing the two
can only assert a loose tolerance.

**Why the quantile.** On the default path the function allocates roughly three
times `N^2` before the block builder runs: the distance matrix, two index arrays,
and a copy of the upper triangle. The index arrays exist only to skip the
diagonal zeros. Those zeros are the smallest entries, so their effect can be
absorbed into the quantile position instead.

**Accepted tolerance.** `jnp.median` and `jnp.quantile` can differ in how they
interpolate at even element counts. Test the equivalence with a relative
tolerance, not exact equality.

**Separate issue, do not bundle.** `batch_size` does not bound peak memory here
even after this change, because the builder concatenates back to a full `N x N`
matrix. Say so in the docstring. Subsampling the median heuristic would fix it
but changes results, so leave it until someone needs it.

### Group F: packaging and reporting

#### D19. Declare NumPy, and other packaging fixes

**Decision.** Three independent items.

1. **Declare `numpy>=2`** in `pyproject.toml`. NumPy is a direct import in about
   twenty modules but is not declared. It reaches users only through JAX and
   SciPy, and both allow NumPy 1.x. The lockfile pins 2.4.2, which hides the gap
   from CI but not from anyone installing from PyPI. This must land before or
   with the `np.unique` rewrite, which depends on NumPy 2 semantics.
2. **Remove `version` and `date-released` from `CITATION.cff`.** The file says
   0.5.0; the package is 0.8.0. Nothing updates it and nothing checks it. Both
   fields are optional, and GitHub and Zenodo take the version from the release
   tag when they are absent.
3. **Derive the sidebar order in `scripts/check_vitepress_pager.py`** from
   `docs/.vitepress/config.ts` instead of restating 27 slugs. The script already
   parses that file for one other field. A checker that compares against a
   hand-copy of its own input can only detect that the copy is stale.

#### D20. Keep the published benchmark numbers

**Decision.** No change to the published speedup tables. Clean up the harness
only.

**Why.** The audit suspected the benchmark script stopped its timer before the
device finished, which would overstate the speedups. Investigation showed it does
not. Every Sobol array is blocked, and `_count_nans` calls `int()` on device
arrays inside `analyze`, which forces synchronisation before the function
returns. The HDMR fields that are not blocked come from the same compiled kernels
as the fields that are, so waiting on one waits on the others. The tables were
generated with exactly the current behaviour.

**Cleanup only.** Make the hand-written blocking mirror the generic
dataclass-field traversal, so the two harnesses cannot drift. Remove the four
duplicated helpers, including a hand-written problem definition that hardcodes
one benchmark's parameter names instead of reading them from the problem object.

*(This is numbered as a decision because it reverses an audit finding. It
requires almost no work.)*

### Group G: warnings

#### D21. Add one warning class

**Decision.** Add `JaxgsaWarning(UserWarning)`, export it from the package root,
and pass `category=JaxgsaWarning` at all 35 warning sites. Do not subdivide it
further.

**Why.** No warning in the package passes a category, so all 35 default to
`UserWarning`. The only way to tell a jaxgsa warning from a NumPy or JAX one is
the message text, and that is inconsistent: six different prefixes are in use,
and two sites have no prefix at all. Filtering on the most common prefix silences
28 of 35 and misses eFAST, PAWN, and DGSM entirely. Filtering on `UserWarning`
also silences NumPy, SciPy, and JAX. No filter selects exactly these warnings.

This matters because several warnings fire once per call inside loops.

**Compatibility.** Subclassing `UserWarning` keeps all 58 existing
`pytest.warns(UserWarning, ...)` assertions passing, and keeps the documented
behaviour true.

**Guard.** Add a test that walks the source and asserts every `warnings.warn`
call passes a category, so new modules cannot drift back.

## 5. Test cleanup

A separate sweep reviewed all 31 test files against ground rule 4. Full detail is
in the sweep notes. Four clusters matter.

| Cluster | What to do |
| --- | --- |
| **Renamed-keyword tests** | Delete about 14 tests. `test_old_chunk_size_kwarg_raises` exists five times verbatim, plus three variants. Each asserts that Python raises `TypeError` for an undeclared keyword, which is a language guarantee. Six paired "kwarg accepted" tests assert only a shape and are already covered by sibling tests that assert the keyword changes nothing. |
| **Mirror tests** | Delete or replace with literals. The test retypes the source's own formula and asserts the two agree. The clearest case is named `test_matches_salib_formula` but never imports SALib; it retypes an expression character for character from the source. |
| **Layout-pinning tests** | Rewrite, do not delete. Eight assertions index into the spec tuple. Route them through `_normalized_input_to_dict`, which returns named keys and which production already maintains. |
| **Type-check tests** | Delete. `isinstance(result, XResult)` is guaranteed by the return annotation, and every such assertion is followed by tests that use the result. |

**Order matters.** Rewrite the layout-pinning tests **before** D1. Done in that
order, the dataclass migration touches no tests.

**Two seams close with their tests.** `SobolSamples.sample_ids` has zero readers
in `src/`; it is written, saved, reloaded, and never used. `validate_correlation`
has zero production callers. Both stay alive only because tests read them.

**One test hides a gap.** The Sobol test that appears to cover
`slice_chunk_size` passes `num_resamples=0`, and that parameter only reaches the
bootstrap path. No Sobol test asserts that chunked output equals unchunked
output. Add one.

**`tests/test_categorical.py` is fine.** It is 1159 lines, but it uses shared
builders and nine parametrised blocks. Its length is scope, not duplication.

## 6. Mechanical items

These need no discussion. Each is small and independent.

| Item | Change |
| --- | --- |
| `_stable_unique_rows` | Replace the per-row Python loop with `np.unique`. It currently runs up to 4.2 million interpreter iterations, twice on the Sobol path, building three redundant representations of one fact. Needs D19 item 1. |
| VKOGA greedy loop | Carry the power vector in the loop state instead of recomputing it twice per iteration. Results are unchanged. |
| `_PCEFit.coeffs_flat` | Delete. It is written and never read, and it keeps a large array alive in the module whose purpose is bounding memory. |
| `validate_correlation` | Delete. Repoint its tests at `canonicalize_correlation`, which runs the same code. |
| `cdf_to_unit_interval` | The comment promises clipping to the open unit interval, but the uniform branch does not clip while both Gaussian branches do. Decide which is right and fix the comment. Do **not** merge this function with its float64 sibling: the split is deliberate and documented, because the float64 path's precision is load-bearing for Morris. |
| `tests/test_efast.py:424` | Rename the fixture. It shadows a session fixture of the same name that holds a Sobol result, so removing the local one would silently test the wrong estimator. |
| `_set_memory_budget` | Drop the `None` from its signature once the two test teardowns use the public getter. |
| Morris `n_blocks_dropped` | Keep the field. Fix only the staleness: `downsample` carries the count forward, so a downsampled design reports a "requested" trajectory count the user never asked for. |
| `pawn.analyze` docstring | After D12, remove the "signature parity" wording. |

## 7. Order of work

Only six orderings are forced. Everything else is independent.

| Do this first | Then this | Reason |
| --- | --- | --- |
| Rewrite layout-pinning tests | D1 | The migration then touches no tests. |
| D2 (public signature rule) | D1 | Removes two signatures from D1's public surface. |
| D19 item 1 (`numpy>=2`) | `np.unique` rewrite | The rewrite depends on NumPy 2 semantics. |
| Delete `sample_ids` | Generic `.npz` payload | Otherwise the payload code needs an exclusion list it would then delete. |
| D3 (`YLayout`) | D13 (Sobol accumulators) | Both touch the same block. D3 owns layout. |
| D5 (partition levels) | D4 (partition counts) | Same file, same kernels. |

### Suggested first batch

Five independent, low-risk changes in five different files: rename the shadowing
eFAST fixture, delete `_PCEFit.coeffs_flat`, delete `validate_correlation`, fix
`CITATION.cff`, declare `numpy>=2`.

Do **not** start with D1 or D3. They have the largest reach, and D1 has an unmet
prerequisite.

## 8. Documentation to update

Every change below has a documentation consequence. Update the docs in the same
commit as the code. Source files only; `docs/.vitepress/dist/` is build output.

| Decision | Update |
| --- | --- |
| D1, D2 | `docs/api/problem.md`, `docs/api/dgsm.md`, `docs/guide/getting-started.md`, and every README example that builds a problem from dicts. Add a 0.8-to-0.9 migration guide, modelled on `docs/guide/migration-0.4.md`. |
| D6 | `docs/api/morris.md` and each result page: state that results are frozen, and give the three ways to get a modified copy. |
| D9 | Docstrings in `borgonovo/_analyze.py`, plus the delta section of `docs/guide/methods.md`. State the new error and the `degenerate_tol` bias. |
| D11 | `docs/api/dgsm.md`: correct the `Raises` list, which currently documents only one of the error cases. |
| D12 | `docs/api/pawn.md`: `slice_chunk_size` now does something. |
| D14, D15 | `docs/api/pce.md`, `docs/api/hdmr.md`: document `streamed`, and note that the point at which streaming engages has moved. |
| D18 | `docs/api/hsic.md`: state plainly that `batch_size` bounds the working memory, not the kernel matrix. |
| D19 | `CITATION.cff`; installation notes if they list dependencies. |
| D21 | `README.md:562` and `docs/guide/methods.md` lines 213, 283, 912, 980, plus `docs/api/vkoga.md` and `docs/api/kucherenko.md`. These currently name `UserWarning`. The statements stay true, but they should name `JaxgsaWarning` and show how to filter it. |

Add a `CHANGELOG.md` entry for 0.9.0 marking it a breaking release.

All documentation must follow the project's technical-writing guidance: plain
language, one idea per sentence, terms defined on first use, and steps numbered
where a reader must follow them in order.

## 9. Corrections to the audit report

Three claims in the audit report are wrong. They are corrected here so the error
does not propagate.

1. **Morris `n_blocks_dropped` is not dead.** The audit said no production path
   reaches the warning that consumes it. One does:
   `tests/test_to_morris.py:291-301` builds a real design, converts it, and the
   warning fires. Keep the field.
2. **The `gram_inv_PhiT` keyword is not a test hook.** No test passes it. Its one
   caller reuses a factorisation it already computed, avoiding a second solve.
   D14 still stands: the object passed should be the small factor, not the large
   product.
3. **Poincaré bounds are already implemented.** `dgsm/_poincare.py` computes them
   per marginal, and `DGSMResult.upper_bound` already reports the resulting bound
   on the total index. Any plan that treats them as new work is wrong. What is
   genuinely missing is an external oracle for the truncated-Gaussian branch,
   which is the only case without a closed form.

## 10. Implementation status (1.0.0)

This document was drafted 2026-08-18 and marked "agreed, not implemented." The
1.0.0 release pass implemented the items below. Status line 3 is stale; read
this section for the current state.

### D4. Partition counts array

Implemented. `_core/partition.py`'s `counts` is unconditionally `(R, Dg, Mg)`.
`_replicate_slice` is deleted. The consumer half in `optimal_transport` landed
in a follow-up commit: `c0829e8` (core), `d885e85` (optimal_transport).

### D5. Partition level lists

Implemented in the same pass. `build_partition_groups` returns
`(groups, group_levels, col_order)`. `optimal_transport/_analyze.py` no longer
rebuilds the level lists by hand. Commits: `c0829e8`, `d885e85`.

### D6. Freeze the result classes

Implemented, but wider than originally decided. The original text excludes
`HDMRResult` because `cached_property` was believed to conflict with a frozen
dataclass. That belief was wrong: `cached_property` writes directly to
`instance.__dict__`, bypassing `__setattr__`, so it works unchanged on a
frozen dataclass. Verified for `HDMRResult.S2`/`S3` before freezing.

All thirteen result classes are now `frozen=True`:
`SobolResult` and `KucherenkoResult` were frozen before this pass
(`ef26acb`, and an earlier commit respectively); `DeltaResult`, `DGSMResult`,
`EFASTResult`, `HDMRResult`, `HSICResult`, `MorrisResult`, `OTResult`,
`PAWNResult`, `PCEResult`, `ShapleyResult`, `VKOGAResult` were frozen in this
commit. No `__post_init__` exists on any result class, so no
`object.__setattr__` migration was needed. No code in `src/` mutates a result
field after construction (the only in-package rebinding uses
`dataclasses.replace`, which frozen dataclasses support unchanged).

### `SobolSamples.sample_ids`

Implemented. Deleted entirely — field, docstring, `_extra_arrays`,
`_from_payload`, `sample()`, `downsample()`. Commit: `ef26acb`.

### D9. Raise up front on an unintegrable Borgonovo bandwidth

**Deliberately reversed for 1.0.0.** The code still range-checks the returned
`delta` after the fact (`borgonovo/_analyze.py:911-924`) instead of raising
before the computation runs, and is tested that way at
`tests/test_borgonovo.py:670`. This is not an oversight: the post-hoc check
already refuses an out-of-range result, which is the property D9 wanted, and
a pre-flight raise would need to re-derive the grid-step comparison outside
the jitted kernel for every call, including the common case where it never
fires. The 1.0.0 review reproduced the discrepancy and the decision was to
keep the code as it stands and correct this document instead of the code.

## 11. Considered and rejected

Recorded so they are not reopened without new information.

| Proposal | Why rejected |
| --- | --- |
| Merge the two marginal-CDF implementations | The split is deliberate and documented. The float64 path exists because its precision is load-bearing for Morris elementary effects; merging would move every column to the host at six call sites. |
| Change the Borgonovo bandwidth assignment to a floor | The docstring states the value is applied exactly. The change would alter results in the exact region cited as motivation, which makes it a change of behaviour, not a simplification. |
| Add `__post_init__` validation to result classes | The library cannot produce the invalid states. Deleting the redundant guards (D7) achieves the same benefit by removing code. |
| Treat a declared identity correlation as no correlation | `sampling.correlate()` documents that an identity matrix is a valid declaration meaning random re-pairing. The two states are deliberately distinct. |
| Collapse the four validation guard functions | The shared logic is already factored out. What remains is prose naming different recovery routes. |
| Merge `poincare_constant` and `marginal_variance` | Both are public. Merging removes duplicated branching but no state. |
| Replace `to_physical_units()` with a units argument | Freezing the class (D6) closes the same hole without removing a documented method. |
| Split HDMR's static data across all downstream signatures | See D16. The naming change removes the failure that bites; the rest is comfort. |
