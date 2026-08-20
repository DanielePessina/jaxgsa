# Migration guide

Two upgrades in jaxgsa's history break working code. This page covers both.

- Coming from 0.5 to 0.9? Read [From 0.8 to 1.0](#from-0-8-to-1-0). That is
  where almost all the churn is.
- Coming from 0.3? Read [From 0.3 to 0.4](#from-0-3-to-0-4) first. It renames
  the package itself, so nothing else applies until you have done it.

1.0 freezes the public interface. After it, a keyword does not move again.

## From 0.8 to 1.0

1.0 gives one idea one name everywhere, deletes keywords that could not do
what their name promised, and changes several defaults. There is no alias
and no deprecation window. An old spelling raises `TypeError` and names the
keyword, so the traceback points at the line to change.

```
TypeError: analyze() got an unexpected keyword argument 'num_resamples'
```

Three of the changes move numbers. They are flagged below. Read those before
you compare a 1.0 result against one you stored under 0.8.

### Your scripts print now

Every one of the thirteen `analyze()` functions and all four samplers take
`verbose: bool = True`. The default is on.

Not all of that is new. Under 0.8 `sobol.sample` and `morris.sample` already
printed a summary by default. What changed is that no `analyze()` printed
anything before, and `efast.sample` and `kucherenko.sample` were silent too.
So a script that only sampled with Sobol or Morris sees more output than
before, and a script that analysed anything sees output where there was
none:

```python
import jaxgsa
from jaxgsa.benchmarks import ishigami

sr = jaxgsa.sobol.sample(ishigami.PROBLEM, 40960, seed=0)
result = jaxgsa.sobol.analyze(sr, ishigami.evaluate(sr.samples))
```

```
jaxgsa.sobol.sample: D=3, mode=second-order, base_n=8192, requested_runs>=40960, n_runs=65536, n_expanded=65536, duplicates_removed=0 (0.0%), scramble=True
jaxgsa.sobol.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: independent
    output: N=65536 runs, T=1 x K=1 output slice
    invalid: none found in 8192 Saltelli groups (policy 'raise')
  timing:
    estimators (includes compile on the first call): 0.6112 s
    slice_chunk_size: 1 (resolved from the memory budget)
    estimator: saltelli-jansen
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.5572
    2. x2  ST=0.4421
    3. x3  ST=0.2435
```

Your indices do not change. Only stdout does, and the timing line moves run to
run. Pass `verbose=False` to get the old silence back:

```python
result = jaxgsa.sobol.analyze(sr, Y, verbose=False)
```

Two places where this matters more than it looks. Inside a timing loop the
print costs real milliseconds, and it forces a device synchronization, so
leave it off when you benchmark. Inside a pipeline that parses stdout, the
summary is new text you did not ask for.

The pure `indices()` cores never print. They take no `verbose` keyword at all.

### Keyword renames

Every row is a clean break. Change the spelling and the call behaves exactly
as it did.

| Was | Now | Where |
| --- | --- | --- |
| `num_resamples` | `n_bootstrap` | `sobol.analyze`, `morris.analyze` |
| `seed=<int>` | `key=<jax.Array>` | `analyze` on pawn, borgonovo, optimal_transport, hsic, vkoga |
| `chunk_size` | `resample_chunk_size` | `morris.analyze` |
| `samples=` | `sampling_result=` | `efast.analyze` |
| `standardize` | `standardize_outputs` | `optimal_transport.analyze`, `optimal_transport.indices` |
| `correlation_kind` | `correlation_type` | `Problem(...)`, `Problem.from_dict(...)` |
| `kind` | `correlation_type` | `Problem.with_correlation(...)` |

`seed` became `key` on `analyze` because a JAX key splits and an integer does
not. Under 0.8 a method that needed two independent streams derived them by
adding a constant to the seed, which does not give independence. VKOGA did
exactly that, and its indices moved when it was fixed. Where you passed
`seed=0`, pass `jax.random.key(0)`:

```python
import jax

result = jaxgsa.pawn.analyze(problem, X, Y, n_bootstrap=200, key=jax.random.key(0))
```

`sample()` keeps `seed`, and keeps accepting an `int`, a
`numpy.random.Generator`, or `None`. Design generation runs host-side through
`scipy.stats.qmc`, which has no JAX PRNG interface. The split between the two
is real, and now it is documented rather than accidental.

### Defaults that changed

**`borgonovo.analyze` no longer bootstraps.** `n_bootstrap` was `100`; it is
now `0`, like every other method. Together with the `key` requirement, the old
default would have made `jaxgsa.borgonovo.analyze(problem, X, Y)` an error out
of the box. To keep intervals, ask for them:

```python
result = jaxgsa.borgonovo.analyze(
    problem, X, Y, n_bootstrap=100, key=jax.random.key(0)
)
```

**`bias_correct` is tri-state.** `None` (the default) applies the Plischke
correction when there are replicates and does nothing when there are not.
`True` asks for it and warns if `n_bootstrap` is `0`. `False` never applies
it. The first default call per process that resolves to "corrected" emits one
`JaxgsaWarning` saying which delta it returned. Pass `bias_correct=True` or
`False` explicitly to silence it. The uncorrected estimate is biased upward,
because a KDE separation is a distance and sampling noise can only add to it.

**`kucherenko.sample(seed=...)` defaults to `None`.** It was `seed: int = 0`.
Pass `seed=0` to reproduce the old design bit for bit. The sampler also
rejects one setting that never did anything: a seed with `scramble=False` now
raises, because the seed only feeds the Owen scrambling.

```
ValueError: jaxgsa.kucherenko.sample: seed has no effect with scramble=False.
The unscrambled Sobol' sequence is deterministic, so the seed would do nothing.
Use scramble=True, or drop the seed.
```

### One batching contract

Four rules now hold on every method, and tests enforce them.

`batch_size` sizes row blocks, clamped to `N`. That is all it does. It never
picks a different algorithm. Under 0.8, an explicit PCE `batch_size` forced
the streamed fit even when the value exceeded `N`. Now `batch_size < N`
streams and `batch_size >= N` is one full block, which is the single-pass fit,
even if that pass exceeds the memory budget. To force the PCE streamed fit,
pass a value smaller than your row count.

`None` on a batching keyword means "derive the width from the memory budget".
DGSM used to read `None` as one batch of every row. It now derives a width
from `jaxgsa.config.get_memory_budget()` using a real bytes model, a few
Jacobian-sized transients per row at `T*K*D` floats each, on both the autodiff
path and the precomputed-`dfdx` path. At ordinary sizes the derived width is
still one block. `dgsm.indices` also raises `ValueError` on `batch_size=0` or
a negative value, where it used to read them as "one batch".

An explicit chunk value always wins. In 0.8 `morris.analyze` narrowed your
`chunk_size` with `min(chunk_size, num_resamples, mem_cap)`, so a memory
estimate could silently shrink the width you asked for. The renamed
`resample_chunk_size` now honors your value, capped only at the resample
count. You need no migration here unless you relied on that cap to shrink a
value you knew was too large.

**`hsic.analyze` loses `batch_size` with no replacement.** Delete the
argument. The keyword row-blocked one kernel build while the resident kernel
stack, about `(2D + 1) * N^2` floats, stayed whole, so it never bounded peak
memory. If your sample does not fit, lower `N`, which HSIC tolerates well
because it converges quickly in `N`, or screen parameters first.

### `prenormalize` is gone

It meant four different things across the five methods that took it, and on
two of them it meant nothing at all.

| Method | Was | Now |
| --- | --- | --- |
| `sobol` | `prenormalize: bool = False` | removed; the standardization always runs |
| `efast` | `prenormalize: bool = False` | removed; measured as a no-op at 6e-16 |
| `hdmr` | `prenormalize: bool = False` | removed; measured as a no-op at 1e-6 |
| `morris` | `prenormalize: bool = False` | renamed `standardize_outputs` |
| `dgsm` | absent | new `standardize_outputs: bool = False` |
| `hsic` | `prenormalize: bool = False` | removed, no replacement |

On `morris` and `dgsm` the keyword earns its place, because those two return
dimensional quantities. Under $Y \to aY + b$, Morris's `mu`, `mu_star` and
`sigma` scale by $a$, DGSM's `sigma` scales by $a$ and its `nu` by $a^2$.
`standardize_outputs=True` reports them in units of the output standard
deviation, so output slices of different magnitude become comparable. DGSM's
`upper_bound` and `lower_bound` are ratios and do not move, and its reported
`var_y` becomes 1.

On `efast` and `hdmr` the keyword did nothing measurable, because their
indices are ratios. HSIC's `bandwidth` is now a multiplier on the median
heuristic, and the heuristic carries the scale of `Y`, so its indices are
already invariant under $Y \to aY + b$. A no-op keyword on a frozen interface
is worse than no keyword, because a caller sets it and believes it acted.

`optimal_transport` keeps the behavior, still defaulting to `True`, under the
name `standardize_outputs`. There it does real work, because the method builds
distances out of `Y` itself rather than a ratio.

One consequence: HDMR now fits on your output scale, so `predict()` and `rmse`
need no inverse transform. No numbers move, because `prenormalize` defaulted
to `False` anyway.

### Changes that move numbers

**Sobol standardizes the outputs, always.** This is a bug fix, and it is the
one to read. The Sobol'-Mauntz first-order estimator and every second-order
estimator are *uncentred* products, so a non-zero output mean adds an error
term proportional to that mean. On Ishigami at N=4096 with an output offset of
1e4, `S1` came back as `[6.26, 0.434, 1.71]` against the analytic
`[0.314, 0.442, 0.000]`. Float64 gave `[6.27, 0.433, 1.72]`, so this was
estimator bias, not rounding.

`sobol.analyze` and `sobol.indices` now standardize every output slice to mean
0 and unit standard deviation over the sample axis before the estimators run.
SALib has always done this. `S1` and `S2` point estimates and intervals move.
`ST` moves only in the last bits of a float32 result, because the Jansen
total-order estimator is a difference and was already shift-invariant.

Be clear about the size of the win when the output mean is *small*. Ishigami's
own mean is 3.5, and there the change is close to a wash: largest `S1` error
0.106 against 0.123 at N=1024, and 0.0017 against 0.0017 at N=16384. What the
fix removes is an error term proportional to the output mean, whose size was
otherwise unpredictable. At that same N=1024 with the 1e4 offset the largest
`S1` error was 50.8; it is now 0.106.

**DGSM picks its autodiff mode from the output shape.** `dgsm.analyze` and
`dgsm.indices` used to call `jax.jacrev` always. They now call `jax.jacfwd`
when the output slices outnumber the inputs (`T*K > D`) and `jax.jacrev`
otherwise. There is no keyword, because the right mode follows from two
numbers the library already has. A time-series DGSM call gets cheaper in
proportion to `T*K / D`. The two modes compute the same Jacobian, so only the
order of the float arithmetic differs. Where the mode changed, `sigma` and
`lower_bound` can move at float32 precision, around 1e-8. The verbose summary
names the mode that ran.

**VKOGA's random streams are independent now.** Its index estimator seeded its
quasi-Monte-Carlo draws with `seed + 1 + i` and `seed + 7919`. Streams that
differ by a constant are not independent. The estimators are host-side scipy
and cannot split a JAX key, so they now spawn one `numpy.random.SeedSequence`
child per draw. Every VKOGA index moves by the size of its own Monte-Carlo
noise. The fitted surrogate does not move: `gamma`, `ridge` and the greedy
centres are bit-for-bit identical.

### Removed and retyped attributes

`SobolResult.nan_counts` is removed. Read `result.invalid` instead, which the
verbose summary also prints.

`Problem.input_specs` returns named dataclasses. Under 0.8 each entry was a
positional 6-tuple, `(kind, a, b, low, high, categorical_data)`, where the
meaning of `a` and `b` depended on `kind`. 1.0 returns one of `UniformSpec`,
`GaussianSpec` or `CategoricalSpec`, each carrying only the fields that apply
to it:

```python
>>> p = jaxgsa.Problem.from_dict({
...     "a": (0, 1),
...     "b": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
...     "c": {"dist": "categorical", "labels": ["x", "y"], "probs": [0.5, 0.5]},
... })
>>> for spec in p.input_specs:
...     print(spec)
UniformSpec(low=0.0, high=1.0)
GaussianSpec(mean=0.0, variance=1.0, low=None, high=None)
CategoricalSpec(probs=(0.5, 0.5), labels=('x', 'y'))
```

Code that indexed or unpacked the tuple breaks. Replace `spec[0] == "uniform"`
with an `isinstance(spec, UniformSpec)` check, and read the field by name.

`Problem` also rejects a non-string or duplicate parameter name at
construction, with a `ValueError` that names the fix. Under 0.8 a duplicate
name failed later, inside a `Theta` lookup or a dataset export.

### `set_memory_budget` reads megabytes, not bytes

0.8 took a byte count, and the parameter was named `budget_bytes`. 1.0 reads
megabytes by default and accepts `unit=` of `"b"`, `"kb"`, `"mb"`, `"gb"` or
`"tb"`. `get_memory_budget()` still answers in bytes.

```python
jaxgsa.config.set_memory_budget(512)             # 512 MiB, the default
jaxgsa.config.set_memory_budget(2, unit="gb")    # 2 GiB
jaxgsa.config.get_memory_budget()                # 2147483648, always bytes
```

The dangerous case would be a 0.8 call that stays syntactically valid and
silently means a million times more memory. It does not: a value too large to
be a plausible MB figure raises and tells you both readings.

```python
>>> jaxgsa.config.set_memory_budget(536870912)   # a 0.8-era byte count
ValueError: set_memory_budget now reads its value in megabytes by default, and
536870912 is too large to be a plausible MB figure. It looks like a byte count
written for the old bytes-only signature. Say which you mean:
set_memory_budget(536870912, unit='b') for the old meaning, or
set_memory_budget(512) for the same budget in MB.
```

### New in 1.0 that you may want while you are here

None of these break anything. They are worth a look during the upgrade because
they replace patterns people wrote by hand under 0.8.

- **Pure `indices()` cores on ten more methods.** `efast`, `pawn`,
  `morris`, `hsic`, `borgonovo`, `optimal_transport`, `dgsm`, `pce`, `hdmr`
  and `shapley` join `sobol`, which makes eleven of the thirteen. A core takes
  the design object (or `problem, X,
  Y`) and returns a bare tuple of arrays. It survives `jit`, `vmap` and
  differentiation, so an index can sit inside a larger JAX computation.

  ```python
  S1, ST, S2 = jaxgsa.sobol.indices(samples, Y)
  grad = jax.jacrev(lambda y: jaxgsa.pce.indices(problem, X, y)[0].sum())(Y)
  ```

  `kucherenko` and `vkoga` have no core and say so, because both are host
  NumPy and SciPy end to end. Two limits are structural: a core refuses
  categorical inputs, because a categorical partition pads to `counts.max()`,
  a shape read off the data; and `hdmr.indices` supports `jacfwd` but not
  `jacrev`, because its backfitting stops early through a `lax.while_loop`,
  which JAX will not differentiate in reverse.

- **Confidence intervals on six more methods.** `dgsm`, `kucherenko`, `pce`,
  `hdmr`, `vkoga` and `shapley` take `n_bootstrap` now. Eleven of thirteen
  methods offer an interval; `efast` and `hsic` do not. The default is `0`
  everywhere, so nothing costs more unless you ask. That default matters most
  for the four surrogate-backed methods, which refit their surrogate on every
  replicate.

- **`result.ci` records how an interval was made.** A bare `*_conf` array does
  not say whether it is a 95% or a 68% interval, which endpoint rule drew it,
  or how many resamples it rests on. The new `CIInfo` carries `level`,
  `method`, `n_bootstrap` and `replicates`, so a plot can label its error bars.
  The draws themselves are kept only when you pass the new
  `keep_replicates=True`, because they are large: 1000 resamples of a
  `(T=100, K=5, D=20)` index array is 80 MB. For Sobol `S2` the stored draws
  follow the reported convention, symmetric with a NaN diagonal.

- **`ci_method` reaches every bootstrapping method.** pawn, borgonovo and
  optimal_transport were hard-wired to percentile endpoints while recording
  `"quantile"`. All three now accept `"quantile"` or `"gaussian"`.

- **`jaxgsa.pce.effective_order(problem, n_samples, *, order, fit_ratio)`**
  answers what order a PCE fit will actually use, with no fit and no side
  effect. PCE reduces the requested order when the design matrix would be
  underdetermined. Under 0.8 you learned that from a warning during the fit.

- **`EFASTSamples.save()` and `.load()`.** It was the last design object
  without persistence. Same single compressed NPZ layout as the others.

- **`PAWNResult.n_valid_bins`.** A conditioning bin with fewer than two
  samples gives no KS value and used to be dropped without a word. The field
  counts contributing bins per parameter. When a parameter keeps fewer than
  half its bins, one warning names it.

- **`OTResult.S1`, `S1_conf` and `above_dummy`.** `S1` is the given-data
  first-order Sobol index, rescaled onto the population-variance (ddof=0)
  convention that `jaxgsa.borgonovo` uses, so the two share one definition.
  `above_dummy` is `max(ot - ot_dummy, 0)`, and is `None` unless you passed
  `dummy=True`.

- **`jaxgsa.__version__`**, and **`Theta`** re-exported from `jaxgsa` and
  `jaxgsa.sobol`, so gradient code can type its `transform(theta)` without
  reaching into a private module.

- **Warnings name their method.** Every warning now starts
  `jaxgsa.<method>:`. Five prefix styles were in use before, including bare
  `PAWN:`, bare `eFAST:`, and none at all.

## From 0.3 to 0.4

Historical. Skip this section unless you are upgrading from 0.3. Every "0.3"
snippet below uses `import gsax`, the original name, so the before and after
are faithful.

0.4 moved commands into method namespaces and moved operations on fitted
surrogates onto their result objects.

### Install the renamed package

0.4 renames the distribution and the import package from `gsax` to `jaxgsa`.
The old name is frozen at `0.3.0b1` on PyPI and receives no further releases.
There is no compatibility shim.

```sh
pip uninstall gsax      # remove the old package
pip install jaxgsa      # install the new one
```

```python
import gsax             # 0.3
import jaxgsa           # 0.4
```

### Replace root-level shortcuts with namespace calls

The package root exports `Problem`, the input specification types, and the
method namespaces. If your code calls anything in the left column, replace it
with the call in the right column.

| 0.3 | 0.4 |
| --- | --- |
| `gsax.sample(...)` | `jaxgsa.sobol.sample(...)` |
| `gsax.analyze(...)` | `jaxgsa.sobol.analyze(...)` |
| `gsax.sample_mc(...)` | `jaxgsa.sampling.monte_carlo(...)` |
| `gsax.sample_efast(...)` | `jaxgsa.efast.sample(...)` |
| `gsax.analyze_efast(...)` | `jaxgsa.efast.analyze(...)` |
| `gsax.sample_morris(...)` | `jaxgsa.morris.sample(...)` |
| `gsax.analyze_morris(...)` | `jaxgsa.morris.analyze(...)` |
| `gsax.analyze_dgsm(...)` | `jaxgsa.dgsm.analyze(...)` |
| `gsax.analyze_hsic(...)` | `jaxgsa.hsic.analyze(...)` |
| `gsax.analyze_pawn(...)` | `jaxgsa.pawn.analyze(...)` |
| `gsax.analyze_borgonovo(...)` | `jaxgsa.borgonovo.analyze(...)` |
| `gsax.analyze_optimal_transport(...)` | `jaxgsa.optimal_transport.analyze(...)` |
| `gsax.analyze_shapley(...)` | `jaxgsa.shapley.analyze(...)` or `result.shapley()` |
| `gsax.enable_compilation_cache(...)` | `jaxgsa.config.enable_compilation_cache(...)` |

If you called `sample_mc(N=...)`, rename the argument. `monte_carlo` uses
`n=...`.

### Update the Sobol workflow

Before:

```python
samples = gsax.sample(problem, n_samples=4096, seed=42)
Y = model(samples.samples)
result = gsax.analyze(samples, Y)
```

After:

```python
samples = jaxgsa.sobol.sample(problem, n_samples=4096, seed=42)
Y = model(samples.samples)
result = jaxgsa.sobol.analyze(samples, Y)
```

If your code names the types, the sampling result type is
`jaxgsa.sobol.SobolSamples` and the analysis result type is
`jaxgsa.sobol.SobolResult`.

### Rename the design row-count fields

Both fields were renamed on `SobolSamples` and `MorrisSamples`.

| 0.3 | 0.4 |
| --- | --- |
| `samples.n_total` | `samples.n_runs` |
| `samples.expanded_n_total` | `samples.n_expanded` |

`n_runs` is the number of unique rows you evaluate, one model run per row.
`n_expanded` is the size of the full design layout before deduplication.

### Update the eFAST workflow

`efast.sample` renamed its second parameter from `N` to `n_per_curve`. It
returns a typed `EFASTSamples` object instead of a bare array. `efast.analyze`
takes that object first. The `M` and `problem` parameters are gone, because
both travel inside the design object and can no longer be mismatched between
sampling and analysis.

Before:

```python
X = gsax.sample_efast(problem, 4096, M=4, seed=42)
Y = model(X)
result = gsax.analyze_efast(problem, Y, M=4)
```

After:

```python
samples = jaxgsa.efast.sample(problem, n_per_curve=4096, M=4, seed=42)
Y = model(samples.samples)
result = jaxgsa.efast.analyze(samples, Y)
```

In 1.0 that last call takes `sampling_result=` if you name the argument.

`EFASTSamples` carries `samples`, `n_per_curve`, `M`, `problem`, and an
`n_runs` property. `n_runs` is `n_per_curve * D`, matching the package-wide
meaning: unique rows you run the model on.

Then check your design size against the stricter bound. 0.3 required only
`n_per_curve > 4*M^2`. 0.4 requires `n_per_curve >= 4*M^2*(D-1) + 1`, which
grows with the number of parameters. Below that bound there are not enough
frequencies to give every non-focal parameter a distinct one. 0.3 wrapped them
cyclically, so two parameters shared a frequency and a phase. That made them
identical along the search curve and silently biased the indices. Such designs
now raise `ValueError`. To fix it, raise `n_per_curve` or lower `M`.

### Rename the batching parameters

0.4 introduced one vocabulary for the two kinds of batching. `batch_size`
means rows of X/Y per batch. `slice_chunk_size` means output slices (`T * K`
columns) per batch. See [One batching contract](#one-batching-contract) for
what 1.0 added to those rules.

| 0.3 | 0.4 |
| --- | --- |
| `gsax.analyze(..., chunk_size=...)` | `jaxgsa.sobol.analyze(..., slice_chunk_size=...)` |
| `gsax.analyze_efast(..., chunk_size=...)` | `jaxgsa.efast.analyze(..., slice_chunk_size=...)` |
| `gsax.analyze_hdmr(..., chunk_size=...)` | `jaxgsa.hdmr.analyze(..., slice_chunk_size=...)` |
| `gsax.analyze_pawn(..., chunk_size=...)` | `jaxgsa.pawn.analyze(..., slice_chunk_size=...)` |
| `gsax.analyze_borgonovo(..., chunk_size=...)` | `jaxgsa.borgonovo.analyze(..., slice_chunk_size=...)` |
| `gsax.analyze_optimal_transport(..., chunk_size=...)` | `jaxgsa.optimal_transport.analyze(..., slice_chunk_size=...)` |
| `gsax.analyze_dgsm(..., chunk_size=...)` | `jaxgsa.dgsm.analyze(..., batch_size=...)` |
| `gsax.analyze_hsic(..., chunk_size=...)` | `jaxgsa.hsic.analyze(...)` and drop it; 1.0 removed the keyword |

`morris.analyze` kept a `chunk_size` in 0.4, because there it bounds bootstrap
resamples per batch, which is neither rows nor output slices. 1.0 renamed it
`resample_chunk_size`.

### Streaming surrogate fits

`jaxgsa.pce.analyze` and `jaxgsa.hdmr.analyze` gained a `batch_size` parameter
and automatic streaming. When the estimated memory of the single-pass fit
exceeds the active budget, the fit streams over row batches. The streamed fit
is mathematically exact. It accumulates the same Gram matrices and moments as
the in-memory path, and PCE leave-one-out diagnostics stay exact through a
second pass. Only the floating-point summation order differs.

```python
import jaxgsa

jaxgsa.config.set_memory_budget(256)  # MiB since 0.9
result = jaxgsa.pce.analyze(problem, X, Y, order=4)           # streams if needed
result = jaxgsa.hdmr.analyze(problem, X, Y, batch_size=8192)  # streams if 8192 < N
```

The budget sizes every automatic batching decision: surrogate `predict`, HDMR
output-slice chunking, and the streaming fits. See the
[configuration guide](/guide/configuration) for details.

### Move PCE and HDMR prediction onto the result

Analysis stays namespace-based, but prediction is a result method. Replace
`emulate_pce(result, X_new)` and `emulate_hdmr(result, X_new)` with
`result.predict(X_new)`:

```python
pce_result = jaxgsa.pce.analyze(problem, X, Y, order=4)
Y_pred = pce_result.predict(X_new)

hdmr_result = jaxgsa.hdmr.analyze(problem, X, Y, maxorder=2)
Y_pred = hdmr_result.predict(X_new)
```

Both accept `batch_size=...` for bounded-memory prediction.

HDMR exposes structural interaction arrays directly:

```python
hdmr_result.S1  # (..., D)
hdmr_result.S2  # (..., D, D)
hdmr_result.S3  # (..., D, D, D)
```

### Derive Shapley effects from a fitted result

There is no standalone Shapley pipeline. Fit the surrogate you want, then
derive Shapley effects from that result:

```python
pce_result = jaxgsa.pce.analyze(problem, X, Y, order=4)
effects = pce_result.shapley()

hdmr_result = jaxgsa.hdmr.analyze(problem, X, Y, maxorder=2)
structural = hdmr_result.shapley()
correlation_aware = hdmr_result.shapley(include_correlative=True)
```

That makes one fit serve prediction, diagnostics, Sobol-style indices and
Shapley effects, without fitting the same surrogate twice.

If you need only the effects, `jaxgsa.shapley.analyze` wraps the two steps. It
is literally `jaxgsa.pce.analyze(...).shapley()`, or the HDMR equivalent, with
no separate pipeline behind it:

```python
effects = jaxgsa.shapley.analyze(problem, X, Y, backend="pce", order=4)
effects = jaxgsa.shapley.analyze(
    problem, X, Y, backend="hdmr", include_correlative=True
)
```

### Reshape your model outputs

0.4 accepts three output layouts and no others.

- `(N,)` for one scalar output.
- `(N, K)` for multiple outputs.
- `(N, T, K)` for time-varying multiple outputs.

The sample axis must be first and the output axis must be last. jaxgsa takes
the shape you give it and never infers or transposes an axis, so a 2-D `Y` is
always `(N, K)`. If you have one time-varying output, pass `(N, T, 1)`.

If you set `problem.output_names`, its length must match `K`.

### Widen pre-computed DGSM Jacobians

The `dfdx` contract of `jaxgsa.dgsm.analyze` narrowed. In 0.3, singleton axes
were paired loosely: `(N,)` outputs were accepted with an `(N, 1, D)`
Jacobian, and `(N, 1)` outputs with an `(N, D)` Jacobian. Both tolerances are
gone. `dfdx.ndim` must equal `Y.ndim + 1`, with the leading axes matching `Y`
exactly and the trailing axis of length `D`.

- `(N,)` outputs require `(N, D)`.
- `(N, K)` outputs require `(N, K, D)`.
- `(N, T, K)` outputs require `(N, T, K, D)`.

### Move saved designs to the NPZ format

Sobol designs use one NPZ file:

```python
samples.save("runs/design")
samples = jaxgsa.sobol.SobolSamples.load("runs/design")
```

The `.npz` suffix is optional. CSV, text, pickle, Excel, and Parquet
persistence were removed, along with the pandas dependency. If you relied on
one of those, regenerate the design and save it as NPZ.

`MorrisSamples` gained the same `save(path)` and `load(path)` pair, using the
identical single-NPZ format and metadata schema:

```python
samples = jaxgsa.morris.sample(problem, n_trajectories=64, seed=42)
samples.save("runs/morris_design")
samples = jaxgsa.morris.MorrisSamples.load("runs/morris_design")
```

`EFASTSamples` got the same pair in 1.0, which completes the set.
