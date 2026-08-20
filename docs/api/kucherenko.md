# Kucherenko (dependent-input Sobol' indices)

```python
jaxgsa.kucherenko.sample(problem, n_samples, *, scramble=True, seed=None,
                         verbose=True) -> KucherenkoSamples

jaxgsa.kucherenko.analyze(samples, Y, *, n_bootstrap=0, conf_level=0.95,
                          ci_method="quantile", key=None, on_invalid="raise",
                          verbose=True, keep_replicates=False) -> KucherenkoResult
```

`jaxgsa.kucherenko` estimates the Sobol' indices generalised to dependent
inputs (Kucherenko, Tarantola & Annoni, 2012). It runs the estimators on your
actual model outputs, evaluated on a conditional-copula design. No surrogate is
fitted anywhere. That makes it the design-based counterpart to
[`jaxgsa.vkoga`](/api/vkoga), which reaches the same two quantities by querying
a fitted kernel surrogate instead.

The trade is straightforward. Kucherenko costs `base_n * (2D + 1)` model
evaluations and has no surrogate error. VKOGA costs one `(X, Y)` sample of any
size and inherits whatever the surrogate gets wrong.

## A run

```python
import numpy as np
import jaxgsa

problem = jaxgsa.Problem(names=("x1", "x2", "x3"), bounds=((-np.pi, np.pi),) * 3)
rho = np.array([[1.0, 0.7, 0.0], [0.7, 1.0, 0.0], [0.0, 0.0, 1.0]])
corr_problem = problem.with_correlation(rho)

ks = jaxgsa.kucherenko.sample(corr_problem, 4096, seed=0)
X = ks.samples
Y = np.sin(X[:, 0]) + 7.0 * np.sin(X[:, 1]) ** 2 + 0.1 * X[:, 2] ** 4 * np.sin(X[:, 0])

res = jaxgsa.kucherenko.analyze(ks, Y)
print(np.asarray(res.S1).round(4))
print(np.asarray(res.ST).round(4))
```

```
jaxgsa.kucherenko.sample: D=3, base_n=4096, n_blocks=7, n_runs=28672, dependence=copula-conditional, scramble=True
jaxgsa.kucherenko.analyze
  problem: D=3 (x1, x2, x3)
    marginals: uniform=3
    correlation: correlated (Gaussian copula)
    output: N=28672 runs, T=1 x K=1 output slice
    invalid: none found in 4096 base points (policy 'raise')
  timing:
    compute: 0.02966 s
    design: copula-conditional (2D+1 = 7 blocks of 4096 base points)
  results: top 3 of 3 parameters by ST
    1. x1  ST=0.4246
    2. x2  ST=0.4234
    3. x3  ST=0.2434

[0.3308 0.523  0.0025]
[0.4246 0.4234 0.2434]
```

Read `x2` first: `S1 = 0.52` against `ST = 0.42`. Under independence that
ordering is impossible. Here it happens because `S1` collects everything `x2`
explains including what arrives through its 0.7 correlation with `x1`, while
`ST` counts only what `x2` alone can explain. Then read `x3`, which is the
opposite case: `S1 = 0.002` and `ST = 0.24`, a parameter with no first-order
effect and a large interaction effect.

Both `sample` and `analyze` print because `verbose=True` is the default on both.

## Index reference

Every index has shape `(..., D)`. The leading axes follow the shape contract:
`(D,)` for `Y` of shape `(n_runs,)`, `(K, D)` for `(n_runs, K)`, `(T, K, D)` for
`(n_runs, T, K)`.

| Index | Definition | Reading |
| --- | --- | --- |
| `S1` | $V(E(Y \mid X_i)) / V(Y)$ | Correlation-inclusive first order. What $X_i$ explains through itself plus what it explains through its coupling. Equals VKOGA's `S_TC`. |
| `ST` | $E(V(Y \mid X_{\sim i})) / V(Y)$ | Correlation-exclusive total. What only $X_i$ can explain. Equals VKOGA's `S_TU`. |

Under independent inputs both reduce exactly to the classic Sobol' `S1` and
`ST`. Under correlation `ST >= S1` no longer holds, as the run above shows.

## The design

`sample` builds `base_n * (2D + 1)` rows, where `base_n` is `n_samples` rounded
up to the next power of two. The Sobol' sequence keeps its balance guarantees
only at powers of two, so 4096 stays 4096 and 4097 becomes 8192.

The design holds one joint block, then two blocks per parameter. The first keeps
$X_i$ and redraws the rest from $p(\mathbf{X}_{\sim i} \mid X_i)$. The second
keeps the rest and redraws $X_i$ from $p(X_i \mid \mathbf{X}_{\sim i})$. Both
conditionals are closed-form Gaussians in the latent copula space.

The dependence comes from `problem.correlation`. This sampler is deliberately
exempt from the correlated-design error that `sobol`, `morris` and `efast`
raise, because conditioning on the declared copula is the whole point of the
method. With no declared correlation the design is exactly the Saltelli
column-swap scheme and the analysis reproduces the classic Sobol' indices.

| Argument | Default | What it changes |
| --- | --- | --- |
| `n_samples` | required | Base points per block, at least 2, rounded up to a power of two. It is the only knob on accuracy: everything else in the design is fixed by `D`. |
| `scramble` | `True` | Owen-scrambles the Sobol' sequence. Keep it on, because it is what makes different seeds give statistically independent designs. Off, the sequence is deterministic and repeated runs are identical. |
| `seed` | `None` | `int`, `np.random.Generator`, or `None`. It seeds the scrambling only, so it does nothing with `scramble=False` and passing both raises `ValueError`. `seed=0` reproduces the design earlier versions produced by default. |
| `verbose` | `True` | Prints the one-line design summary shown above. |

`sample` raises `ValueError` when the problem declares a categorical parameter
(the conditional copula needs continuous marginals; use
[`jaxgsa.optimal_transport`](/api/optimal-transport),
[`jaxgsa.borgonovo`](/api/borgonovo), [`jaxgsa.pawn`](/api/pawn), or the
Saltelli-based Sobol pipeline), when the problem has fewer than two parameters,
and when `n_samples < 2`.

## Analysis

`analyze` applies the single-loop estimators of the 2012 paper: the paired
product over the shared-$X_i$ rows for `S1`, and the Jansen squared difference
over the shared-$\mathbf{X}_{\sim i}$ rows for `ST`. The exact formulas are in
the `jaxgsa.kucherenko._analyze` module docstring.

The resampling unit for `n_bootstrap` is one **base point**, which carries the
joint row and all `2D` conditional rows drawn around it. Base points are
i.i.d., so resampling them is a clean bootstrap of the design. Resampling
individual rows would not be, because dropping part of a conditional block
leaves the estimator reading rows that no longer line up. `S1` and `ST` get
intervals; `variance` does not, since it is the shared denominator of both and
its uncertainty is already inside the two index intervals.

`on_invalid` works on the same unit. A base point appears once in the joint
block and once in each of the `2D` conditional blocks, so one non-finite value
removes all `2D + 1` of its rows. `"raise"` is the default, `"drop"` analyzes
the surviving base points and warns, `"propagate"` lets the value reach the
indices. Base points are not contiguous in the array: base point `k` occupies
rows `k`, `N + k`, `2N + k` and so on, and `result.invalid` reports them. See
[Failed model runs](/guide/methods#failed-model-runs).

`analyze` raises `ValueError` on a `Y` that violates the shape contract, on an
unknown `on_invalid`, on a non-finite sample under `"raise"`, when fewer than 2
base points survive a drop, and on the usual bootstrap-argument checks
(`n_bootstrap < 0`, `conf_level` outside `(0, 1)`, an unknown `ci_method`, or
`n_bootstrap > 0` with no `key`).

It warns on an output slice with zero variance. Its indices come back as NaN.
Drop that slice, or widen the input ranges so the output varies.

## KucherenkoResult

| Field | Meaning |
| --- | --- |
| `S1`, `ST` | The two index arrays, `(..., D)`. |
| `S1_conf`, `ST_conf` | `(2, ..., D)` intervals, `None` when `n_bootstrap=0`. |
| `variance` | Output variance under the joint input measure, per output slice. |
| `is_correlated` | Whether the problem declared a dependence. |
| `invalid` | What the non-finite check found and which policy ran. |
| `ci` | Confidence level, endpoint rule, resample count, and the draws when `keep_replicates=True`. |

`result.to_dataset(time_coords=None)` gives the labeled xarray view of `S1`,
`ST` and `variance`.

## KucherenkoSamples

`samples` is the `(base_n * (2D + 1), D)` design in physical units, stored as
`2D + 1` stacked blocks of `base_n` rows: the joint block first, then one
conditional block per parameter for `S1`, then one per parameter for `ST`.
`n_runs`, `base_n`, `n_params` and `problem` describe it.

Every conditional draw is a distinct continuous point, so this design has no
duplicate rows: `n_expanded == n_runs`, `expanded_to_unique` is the identity,
and `expand_outputs(Y)` returns `Y` unchanged. The shared
`UniqueDesignSamples` base is here for the NPZ format and the metadata schema,
not for deduplication.

`KucherenkoSamples.save(path)` and `KucherenkoSamples.load(path)` use the same
one-file NPZ format as the other design classes. The stored problem metadata
carries the correlation matrix, so a loaded design analyzes identically.

## Reference

- Kucherenko, S., Tarantola, S. & Annoni, P. (2012). Estimation of global sensitivity indices for models with dependent variables. *Computer Physics Communications*, 183(4), 937-946.

See the [Kucherenko example](/examples/kucherenko), [Methods](/guide/methods),
and the [API overview](/api/).
