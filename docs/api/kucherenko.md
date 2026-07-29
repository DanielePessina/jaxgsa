# Kucherenko (Dependent-Input Sobol' Indices)

`jaxgsa.kucherenko` estimates the Sobol' indices generalised to dependent
inputs (Kucherenko, Tarantola & Annoni, 2012) by evaluating your actual model
on a conditional-copula design. It follows the usual sample / evaluate /
analyze split:

```python
ks = jaxgsa.kucherenko.sample(problem, 4096, seed=0)
Y = model(ks.samples)                    # (n_runs, ...) model evaluations
result = jaxgsa.kucherenko.analyze(ks, Y)
```

Public objects:

- `jaxgsa.kucherenko.sample`
- `jaxgsa.kucherenko.analyze`
- `jaxgsa.kucherenko.KucherenkoSamples`
- `jaxgsa.kucherenko.KucherenkoResult`

## Index reference

Both indices have shape `(..., D)` under the usual output contract: `(D,)`
for `(N,)` outputs, `(K, D)` for `(N, K)`, `(T, K, D)` for `(N, T, K)`.

| Index | Definition | Reading |
| --- | --- | --- |
| `S1` | $V(E(Y \mid X_i)) / V(Y)$ | **Correlation-inclusive** first-order index: what $X_i$ explains through itself *and* through its coupling. Equals VKOGA's `S_TC`. |
| `ST` | $E(V(Y \mid X_{\sim i})) / V(Y)$ | **Correlation-exclusive** total index: what only $X_i$ can explain. Equals VKOGA's `S_TU`. |

Under independent inputs both reduce exactly to the classic Sobol' `S1` and
`ST`. Under correlation `ST >= S1` no longer holds in general.

## Design

`sample(problem, n_samples, *, scramble=True, seed=0)` builds
`base_n * (2D + 1)` rows, where `base_n` is `n_samples` rounded up to the
next power of two: one joint block, then per parameter one block with
$X_i$ kept and the rest redrawn from $p(\mathbf{X}_{\sim i} \mid X_i)$, and
one block with the rest kept and $X_i$ redrawn from
$p(X_i \mid \mathbf{X}_{\sim i})$. Both conditionals are closed-form
Gaussians in the latent copula space.

The dependence structure comes from `problem.correlation`. The sampler is
exempt from the correlated-design error on `sobol` / `morris` / `efast`:
conditioning on the declared copula is the method's purpose. With no declared
correlation the design is exactly the Saltelli column-swap scheme.

Raises `ValueError` for categorical problems (the conditional copula needs
continuous marginals), for fewer than two parameters, and for
`n_samples < 2`.

## Analysis

`analyze(samples, Y)` applies the single-loop estimators of the 2012 paper:
the paired product over the shared-$X_i$ rows for `S1` and the Jansen squared
difference over the shared-$\mathbf{X}_{\sim i}$ rows for `ST`. The exact
formulas are stated in the `jaxgsa.kucherenko._analyze` module docstring.

Base points with any non-finite output are dropped as a group, with a
`UserWarning`. Zero-variance output slices warn and produce NaN indices.

## Result

- `result.S1`, `result.ST` — the two index arrays.
- `result.variance` — output variance under the joint input measure, per
  output slice.
- `result.is_correlated` — whether the problem declared a dependence.
- `result.to_dataset(time_coords=None)` — labeled xarray view (`S1`, `ST`,
  `variance`).

## Persistence

`KucherenkoSamples.save(path)` / `KucherenkoSamples.load(path)` use the same
one-file NPZ format as the other designs; the problem metadata carries the
correlation matrix.

## Reference

- Kucherenko, S., Tarantola, S. & Annoni, P. (2012). Estimation of global sensitivity indices for models with dependent variables. *Computer Physics Communications*, 183(4), 937-946.

See the [Kucherenko example](/examples/kucherenko), [Methods](/guide/methods),
and the [API overview](/api/).
