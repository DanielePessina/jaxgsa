# jaxgsa domain context

The shared vocabulary for this codebase. One term, one meaning, one spelling.

This file is normative for the public interface. A keyword that appears in two
methods must mean the same thing in both, and must be named the same in both.
`tests/test_vocabulary.py` reads these rules off the method registry and fails
when a signature drifts from them.

---

## Domain terms

**Problem** — the input specification: names, marginal distributions, optional
correlation, optional output names. A plain value object. It is deliberately
*not* a JAX pytree, so passing one into a jitted function never turns its
marginal parameters into tracers. Differentiation with respect to marginal
parameters goes through **Theta** instead.

**Theta** — a mapping pytree of marginal parameters, consumed by
`SobolSamples.transform(theta)`. This is the differentiation surface: it is
opt-in and exists only where gradients are wanted.

**Input spec** — one parameter's marginal distribution, as a frozen dataclass:
`UniformSpec`, `GaussianSpec` or `CategoricalSpec`. `InputSpec` is the union.

**Design-based method** — needs a specific sample layout that it generates
itself (Sobol saltelli, Morris trajectories, eFAST curves, Kucherenko
conditional). Takes `(sampling_result, Y)`.

**Given-data method** — works from any `(X, Y)` pair the caller already has.
Takes `(problem, X, Y)`.

**Output slice** — one scalar output over the sample axis. A `Y` of shape
`(N, T, K)` has `T*K` output slices, each analysed independently. This is the
axis the atomic-kernel-plus-vmap rule maps over.

**Sample row** — one row of the sample axis, length `N`. The unit that invalid
handling drops and that most bootstraps resample.

**Invalid unit** — the thing a method can drop without corrupting its design:
`ROW` for given-data methods, `BASE_POINT` for Kucherenko, `TRAJECTORY` for
Morris, `CURVE` for eFAST. eFAST's curve cannot be dropped at all — removing a
point does not shrink the sample, it changes what the estimator computes.

**Pure core** — a method's `indices()`: takes the design object or
`(problem, X, Y)`, returns a bare tuple of arrays, and survives `jit`, `vmap`
and `jacrev`. Every branch is on a shape or a Python scalar, never on data.
No result class, no diagnostics, no host read of an array value.

**Preamble** — the shared policy layer every `analyze()` runs before its
kernel: scalar validation, invalid handling, zero-variance detection. It is
what makes `analyze()` untraceable and `indices()` traceable, and the split
between them is deliberate.

---

## Parameter vocabulary

### The three batching axes

These are orthogonal. A method takes the ones that apply to it and no others;
a keyword that does nothing is worse than an absent one.

| Keyword | Unit | Meaning |
|---|---|---|
| `batch_size` | sample rows | How many rows of the sample axis to process per device call. Bounds the working set of a row-wise computation: a surrogate fit, a surrogate evaluation, a Jacobian, a kernel-matrix build. |
| `slice_chunk_size` | output slices | How many `(t, k)` output slices to vmap per device call. Bounds the working set when `T*K` is large. |
| `resample_chunk_size` | bootstrap replicates | How many bootstrap replicates to vmap per device call. Only for methods whose replicate axis, not their slice axis, is the memory bound. |

All three accept `int | None`, where `None` means "derive one from
`jaxgsa.config.get_memory_budget()`". A hard-coded element budget is a defect,
not a default.

### Bootstrap and confidence intervals

| Keyword | Type | Rule |
|---|---|---|
| `n_bootstrap` | `int`, default `0` | Number of bootstrap replicates. `0` means no interval. The single spelling — not `num_resamples`. |
| `conf_level` | `float`, default `0.95` | Two-sided confidence level. |
| `ci_method` | `"quantile" \| "gaussian"`, default `"quantile"` | How endpoints are formed. Every method that offers `n_bootstrap` offers this. |
| `key` | `Array \| None`, default `None` | A JAX PRNG key. Required when `n_bootstrap > 0`. The single spelling — not `seed: int`. |
| `keep_replicates` | `bool`, default `False` | Keyword-only, and the **last** keyword in the signature. Retains the per-replicate values on the result. |

`key` rather than `seed` because a key can be split. An `int` seed forces
per-call reseeding, which silently correlates nested or repeated bootstraps —
a correctness problem, not a style preference. Callers who have an integer
write `jax.random.key(0)`.

Borgonovo is the one method where `n_bootstrap` defaults to non-zero, because
its bias correction needs the replicates. That default is documented on the
method.

### Surrogate-backed methods

`n_bootstrap` defaults to `0` on `pce`, `hdmr`, `vkoga` and `shapley` and must
stay there: each replicate refits a surrogate, so the cost is a different order
of magnitude from a row resample.

### Entry points

Every method package exports `analyze`. The function is *defined* as `analyze`
too — an internal name like `analyze_pce` makes `grep "def analyze("` lie about
how many entry points exist.

Every design-based method names its first argument `sampling_result`. Every
given-data method takes `(problem, X, Y)`. DGSM is the documented exception: it
takes a callable to differentiate, so its signature is
`(problem, fn, X, *, Y, dfdx, ...)`.

### Precision

jaxgsa computes in whatever precision JAX is configured for and infers dtype
from the caller's arrays. It does **not** set `jax_enable_x64`, and it ships no
wrapper around `jax.enable_x64()` — that context manager is already the
primitive, it is thread-local, and it works with jaxgsa as-is.

What the library owes the caller: never silently destroy precision. Passing a
float64 array while x64 is off truncates it to float32, so say so once.

---

## Policies

**Warn versus error.** Raise when a provable property is violated or a
computation is undefined. Warn when a result is degraded but still valid. Never
drop data silently. `analyze()` is cheap to re-run, so it is the right place to
be strict.

**Verification oracles, T0-T4.** Use the strongest tier available.
T0 closed form, T1 published literals, T2 permissive-licence library, T3
copyleft run out-of-process, T4 internal consistency. A method must not ship at
T4 alone unless there is a recorded reason no external oracle exists. **A test
that retypes the source's own formula is not an oracle. It is a mirror.**
Record the tier in the test docstring.

**Atomic kernel, then vmap.** Write one kernel for one output slice. vmap it
over a chunk of slices. Loop over chunks on the host. Size the chunk from the
memory budget. Never one vmap over everything.

**Ragged chunks.** Pad the trailing chunk back to the full width and slice the
answer back, so a kernel compiles once rather than twice. See
`efast/_analyze.py`, `morris/_analyze.py`, `dgsm/_analyze.py` for the pattern.

**Numerical baselines.** A changed number is a wiring error, not a tolerance
issue — with one reviewed exception, recorded in `scripts/baseline/README.md`.

**Tests justify their presence.** A test that must be edited whenever the
source changes, with no behaviour change, asserts implementation rather than
behaviour. Delete or rewrite it.
