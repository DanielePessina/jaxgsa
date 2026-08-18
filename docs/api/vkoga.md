# VKOGA (Correlated-Input Indices)

`jaxgsa.vkoga.analyze(problem, X, Y, ...)` fits a VKOGA kernel surrogate to
given `(X, Y)` data and returns `VKOGAResult`. Its variance-based indices stay
meaningful when the inputs are dependent (Hilhorst et al., 2024; Li et al.,
2010). It is the only method in jaxgsa that estimates correlated and
uncorrelated contributions separately from a common surrogate.

Result operations:

- `result.predict(X_new, batch_size=None)` — evaluate the fitted surrogate on
  new inputs.
- `result.to_dataset(time_coords=None)` — labeled xarray view of the indices.
- `result.is_correlated` — whether the indices were computed under a
  non-identity correlation.

`result.shapley()` raises `NotImplementedError`. Shapley effects need a
variance decomposition indexed by parameter subsets. A kernel expansion is a
sum over centres, and every centre involves every parameter, so there is no
membership matrix to allocate from. Use `jaxgsa.hdmr` or `jaxgsa.pce` for
Shapley effects.

## Index reference

Every index has shape `(..., D)`, where `D` is the number of parameters. The
leading axes follow the output contract: `(D,)` for a scalar output `(N,)`,
`(K, D)` for `(N, K)`, and `(T, K, D)` for `(N, T, K)`.

| Index | Definition | Reading |
| --- | --- | --- |
| `S_TC` | $V(E(Y \mid X_i)) / V(Y)$ | Total correlated: what $X_i$ explains through itself, plus what it explains through its correlation with the others. This is the measure for input prioritisation. |
| `S_TU` | $E(V(Y \mid X_{\sim i})) / V(Y)$ | Total uncorrelated: what only $X_i$ can explain, correlated pathways removed. This is the measure for input fixing. |
| `S_U` | independent part of $S_{TC}$ | The contribution of $X_i$ alone. |
| `S_C` | `S_TC - S_U` | The correlation-borne part. It can be negative when a correlation opposes a direct effect. |
| `S_IU` | `S_TU - S_U` | Independent interactions. |

Under independent inputs the five collapse to the familiar picture: `S_TC` is the
first-order Sobol' index $S_1$, `S_TU` is the total index $S_T$, `S_U` equals
`S_TC`, and `S_C` is zero.

## Dependency structure

The `correlation` argument declares the Gaussian copula the indices are computed
under:

| Value | Meaning |
| --- | --- |
| `None` (default) | Read `problem.correlation`. Independent (identity matrix) when the problem declares none. |
| `(D, D)` array | Override the problem's declaration for this call. The matrix must be symmetric with a unit diagonal. If it is not positive definite, `analyze` projects it to the nearest positive-definite matrix and emits a `JaxgsaWarning`. |

To fit a matrix from observed data, use
`jaxgsa.sampling.fit_correlation(problem, X_data)` and attach it with
`problem.with_correlation(...)`. This one workflow makes it explicit which
sample the copula comes from. A string value for `correlation` raises
`ValueError`; pass an array or `None` instead.

The matrix actually used is always returned on `result.correlation`.

`analyze` raises `ValueError` in two more cases:

- The problem has fewer than two parameters. Conditioning on the other
  parameters is meaningless for `D = 1`.
- The problem declares a categorical parameter. The isotropic RBF needs a
  continuous CDF map per coordinate, and a step-CDF coordinate breaks both the
  kernel metric and the copula conditionals. Use `jaxgsa.optimal_transport`,
  `jaxgsa.borgonovo`, `jaxgsa.pawn`, or the Saltelli-based Sobol pipeline
  instead.

## Fit and estimator controls

| Argument | Default | Effect |
| --- | --- | --- |
| `gamma` | `None` | Gaussian RBF shape parameter. `None` cross-validates over ten log-spaced values. |
| `ridge` | `None` | RKHS regularisation. `None` cross-validates over ten log-spaced values. |
| `max_centers` | `300` | Cap on greedily selected kernel centres, itself capped at `N`. |
| `n_folds` | `10` | Folds for the hyperparameter cross-validation. |
| `n_outer` | `512` | Outer (conditioning) sample size per parameter. |
| `n_inner` | `128` | Inner (conditional) sample size per outer point. |
| `n_variance` | `8192` | Sample size for the output variance and the component-function fit. |
| `seed` | `0` | Base seed for the quasi-random draws. |
| `batch_size` | `None` | Rows per batch when evaluating the surrogate. `None` derives one from the memory budget. |

Leaving both `gamma` and `ridge` to cross-validation costs a 10x10 grid of
k-fold refits, which dominates the runtime. Pass both explicitly to skip the
search once you know good values. The fitted values are reported on the
result.

## Diagnostics

- `correlation` — the `(D, D)` copula matrix the indices were computed under.
- `variance` — the output variance under the correlated input measure, per
  output slice.
- `n_centers` — the number of kernel centres in the fitted surrogate.
- `gamma`, `ridge` — the hyperparameters the fit used.
- `rmse` — the training-fit RMSE, per output slice.

## Two things to get right

1. Train on an independent, space-filling design even when the analysis is
   correlated. The correlated measure concentrates on a ridge. But `S_TU`
   conditions on $X_{\sim i}$ and then resamples $X_i$ across its whole
   marginal. A surrogate trained only on correlated data extrapolates for
   exactly those draws.
2. Enable float64. The coefficient solve forms $A^\top A$, which squares the
   condition number of the cross kernel, and float32 cannot carry it for small
   `gamma`. Call `jax.config.update("jax_enable_x64", True)` before fitting.
   `analyze` emits a `JaxgsaWarning` when x64 is off.

## References

- Hilhorst, G., Quicken, S., van de Vosse, F.N. & Huberts, W. (2024). Efficient sensitivity analysis for biomechanical models with correlated inputs. *International Journal for Numerical Methods in Biomedical Engineering*, 40(2), e3797.
- Li, G., Rabitz, H., Yelvington, P.E., Oluwole, O.O., Bacon, F., Kolb, C.E. & Schoendorf, J. (2010). Global sensitivity analysis for systems with independent and/or correlated inputs. *Journal of Physical Chemistry A*, 114(19), 6022-6032.
- Wirtz, D. & Haasdonk, B. (2013). A vectorial kernel orthogonal greedy algorithm. *Dolomites Research Notes on Approximation*, 6, 83-100.

See the [VKOGA example](/examples/vkoga), [Methods](/guide/methods), and the
[API overview](/api/).
