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
parameters goes through **Theta** instead. See
`docs/adr/0013-problem-is-not-a-pytree.md`.

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
and differentiation in **at least one mode**. Every branch is on a shape or a Python scalar, never on data.
No result class, no diagnostics, no host read of an array value.

Two limits are structural rather than incidental, and a core is allowed to
refuse rather than pretend:

- **Categorical inputs.** A categorical partition pads its classes to
  `counts.max()`, which is a *shape* read off the data. `jit` requires static
  output shapes, so no amount of moving code fixes it — a traceable version
  would need a pad width the data does not supply. A core that cannot support
  them raises, pointing at `analyze`, which does.
- **Dropped rows.** `analyze` decides which rows survive by looking at their
  values, so the row count is data-dependent. That is why `analyze` is not
  traceable and `indices()` is, and why the split exists at all.
- **Reverse mode, for one method.** `hdmr.indices` supports `jacfwd` but not
  `jacrev`: its backfitting stops early through a `lax.while_loop`, which JAX
  refuses to differentiate in reverse. Supporting it would mean a fixed-trip
  `lax.scan` of `maxiter` iterations, making every fit pay the iterations the
  early stop saves. A test pins the refusal, so the exemption fails if the
  loop is ever rewritten.

Two methods have no core. `kucherenko` and `vkoga` are host NumPy and SciPy
end to end; see `docs/adr/0015-pure-core-exemptions.md`. Every method
*declares* its status, which is the property 1.0 freezes — not that every
method has the same one.

**Preamble** — the shared policy layer every `analyze()` runs before its
kernel: scalar validation, invalid handling, zero-variance detection. It is
what makes `analyze()` untraceable and `indices()` traceable, and the split
between them is deliberate.

---

## Parameter vocabulary

### Batching: two questions, not three axes

Every method is built the same way. Find the **atomic kernel** — the
computation for one unit, usually one output slice. `vmap` it over many units.
Loop over groups of units on the host.

That gives exactly two things a caller might need to make smaller, and each has
one keyword:

| Keyword | Question it answers |
|---|---|
| `slice_chunk_size` | How many atomic kernels to `vmap` at once. Lower it to run more, smaller vmaps. |
| `resample_chunk_size` | The same question for a bootstrap replicate axis, when replicates rather than slices are the vectorised dimension. |
| `batch_size` | How to subdivide the work *inside* one atomic kernel, for when a single kernel invocation is itself too large. |

**A method takes a keyword when its shape calls for one, and not otherwise.**
That is testable rather than a matter of taste:

- `sobol` takes only `slice_chunk_size`. One slice is a few reductions over
  `N`, so subdividing rows within a slice would buy nothing.
- `hsic` takes `batch_size` because one slice materialises an `(N, N)` kernel
  matrix, which can exceed memory on its own. Row-blocking bounds the build
  without changing the result.
- `pce` and `dgsm` take only `batch_size`. Their real work is not per-slice at
  all — one multi-RHS solve, one Jacobian — so the slice axis is never the
  bound.
- `hdmr` takes both, and they must **compose**. Its B-spline bases are built
  from `X` alone and shared across every slice, so they are large in `N` and
  independent of `T*K`; the per-slice fit is large in `T*K`. Two bounds, two
  keywords. A path that honours one and silently ignores the other is a defect.

The "subdivide inside" case also covers a precomputation shared across kernels,
not only the kernel body — HDMR's bases are the example.

All three accept `int | None`, where `None` means "derive one from
`jaxgsa.config.get_memory_budget()`". A hard-coded element budget is a defect,
not a default.

### Bootstrap and confidence intervals

| Keyword | Type | Rule |
|---|---|---|
| `n_bootstrap` | `int`, default `0` | Number of bootstrap replicates. `0` means no interval. The single spelling — not `num_resamples`. |
| `conf_level` | `float`, default `0.95` | Two-sided confidence level. |
| `ci_method` | `"quantile" \| "gaussian"`, default `"quantile"` | How endpoints are formed. Every method that offers `n_bootstrap` offers this. |
| `key` | `Array \| None`, default `None` | A JAX PRNG key. Required **wherever the method draws randomness** — a bootstrap, a permutation test, a Monte-Carlo integral — not only when `n_bootstrap > 0`. HSIC and VKOGA need one despite declaring no bootstrap. The single spelling — not `seed: int`. |
| `keep_replicates` | `bool`, default `False` | Keyword-only, and the **last named** parameter. Retains the per-replicate values on the result. "Last named" rather than "last" because Python requires `**kwargs` to be syntactically final, and `shapley.analyze` forwards `**backend_kwargs`. |

`key` rather than `seed` because a key can be split. An `int` seed forces
per-call reseeding, which silently correlates nested or repeated bootstraps —
a correctness problem, not a style preference. Callers who have an integer
write `jax.random.key(0)`.

**`n_bootstrap` defaults to `0` on every method, with no exceptions.** The
plainest possible call must work, and a non-zero default plus a required `key`
would make `borgonovo.analyze(problem, X, Y)` an error. Borgonovo's bias
correction does need replicates, so it warns when `bias_correct` is asked for
with `n_bootstrap == 0` rather than silently returning the uncorrected,
positively-biased delta.

**Never derive a key from a constant.** A method that falls back to
`jax.random.key(0)` when none is given reintroduces exactly the silent
correlated reseeding this contract removes. Raise instead. For the same reason,
per-stream keys come from `jax.random.split` or `fold_in`, never from an
integer offset like `seed + 1` or `seed + 7919`.

### Surrogate-backed methods

`pce`, `hdmr`, `vkoga` and `shapley` do offer `n_bootstrap`, and it must stay
defaulted to `0`: each replicate refits a surrogate, so the cost is a different
order of magnitude from a row resample. An on-by-default interval would make a
routine call an order of magnitude slower for a caller who never asked.

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
float64 array while x64 is off truncates it to float32, so say so once. The
measurements behind this are in
`docs/adr/0014-float32-default-no-x64-wrapper.md`.

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
Record the tier in the test docstring. The full tier definitions, the mirror
rule and the licence rules are in `docs/adr/0001-verification-oracle-tiers.md`
and `docs/adr/0003-copyleft-oracles-and-licences.md`.

**Parity against another engine is run locally, not in CI.** The committed test
checks a *recorded literal*, and the script that produced it lives in
`scripts/oracles/` with the engine version recorded beside the numbers. Be
honest about what that buys: a live comparison catches a regression on either
side, a recorded literal only catches ours. That is the intended trade — the
other engine changing is not a jaxgsa bug — but a recorded T2 number is weaker
evidence than a live one, and the docstring should say which it is.

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
