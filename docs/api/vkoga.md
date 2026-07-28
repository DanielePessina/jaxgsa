# VKOGA (Correlated-Input Indices)

`jaxgsa.vkoga.analyze(problem, X, Y, ...)` fits a VKOGA kernel surrogate to given
`(X, Y)` data and returns `VKOGAResult` with variance-based indices that stay
meaningful when the inputs are **dependent** (Hilhorst et al., 2024; Li et al.,
2010). It is the only method in jaxgsa that estimates correlated and
uncorrelated contributions separately from a common surrogate.

Important result operations:

- `result.predict(X_new, batch_size=None)`
- `result.to_dataset(time_coords=None)`
- `result.is_correlated`

`result.shapley()` raises `NotImplementedError`. Shapley effects need a variance
decomposition indexed by parameter subsets; a kernel expansion is a sum over
*centres*, every one of which involves every parameter, so there is no membership
matrix to allocate from. Use `jaxgsa.hdmr` or `jaxgsa.pce` for Shapley effects.

## Index reference

Every index has shape `(..., D)` under the usual output contract: `(D,)` for
`(N,)` outputs, `(K, D)` for `(N, K)`, `(T, K, D)` for `(N, T, K)`.

| Index | Definition | Reading |
| --- | --- | --- |
| `S_TC` | $V(E(Y \mid X_i)) / V(Y)$ | Total **correlated**: what $X_i$ explains through itself *and* through its correlation with the others. The measure for **input prioritisation**. |
| `S_TU` | $E(V(Y \mid X_{\sim i})) / V(Y)$ | Total **uncorrelated**: what only $X_i$ can explain, correlated pathways removed. The measure for **input fixing**. |
| `S_U` | independent part of $S_{TC}$ | The contribution of $X_i$ alone. |
| `S_C` | `S_TC - S_U` | The correlation-borne part. **Can be negative** when a correlation opposes a direct effect. |
| `S_IU` | `S_TU - S_U` | Independent interactions. |

Under independent inputs the five collapse to the familiar picture: `S_TC` is the
first-order Sobol' index $S_1$, `S_TU` is the total index $S_T$, `S_U` equals
`S_TC`, and `S_C` is zero.

## Dependency structure

The `correlation` argument declares the Gaussian copula the indices are computed
under:

| Value | Meaning |
| --- | --- |
| `None` (default) | Treat the inputs as independent (identity matrix). |
| `(D, D)` array | Use this copula correlation matrix directly. Must be symmetric with a unit diagonal; it is projected to the nearest positive-definite matrix if needed. |
| `"empirical"` | Fit one from `X` by Spearman rank correlation, converted to the latent Pearson correlation with $\rho = 2\sin(\pi\rho_s/6)$. |

The matrix actually used is always returned on `result.correlation`. Fewer than
two parameters raises `ValueError`: conditioning on "the other parameters" is
meaningless for `D = 1`.

## Fit and estimator controls

| Argument | Default | Effect |
| --- | --- | --- |
| `gamma` | `None` | Gaussian RBF shape parameter; `None` cross-validates over ten log-spaced values. |
| `ridge` | `None` | RKHS regularisation; `None` cross-validates over ten log-spaced values. |
| `max_centers` | `300` | Cap on greedily selected kernel centres, itself capped at `N`. |
| `n_folds` | `10` | Folds for the hyperparameter cross-validation. |
| `n_outer` | `512` | Outer (conditioning) sample size per parameter. |
| `n_inner` | `128` | Inner (conditional) sample size per outer point. |
| `n_variance` | `8192` | Sample size for the output variance and the component-function fit. |
| `seed` | `0` | Base seed for the quasi-random draws. |
| `batch_size` | `None` | Rows per batch when evaluating the surrogate; `None` derives one from the memory budget. |

Leaving both `gamma` and `ridge` to cross-validation costs a 10x10 grid of
k-fold refits and dominates the runtime; pass both explicitly to skip it once you
know good values (they are reported on the result).

## Diagnostics

- `correlation` — the `(D, D)` copula matrix the indices were computed under.
- `variance` — output variance under the *correlated* input measure, per output slice.
- `n_centers`, `gamma`, `ridge` — the fitted surrogate's size and hyperparameters.
- `rmse` — training-fit RMSE per output slice.

## Two things to get right

1. **Train on an independent, space-filling design even when the analysis is
   correlated.** The correlated measure concentrates on a ridge, but `S_TU`
   conditions on $X_{\sim i}$ and then resamples $X_i$ across its whole marginal —
   a surrogate trained only on correlated data extrapolates for exactly those
   draws.
2. **Enable float64.** The coefficient solve forms $A^\top A$, squaring the
   condition number of the cross kernel, and float32 cannot carry it for small
   `gamma`. Call `jax.config.update("jax_enable_x64", True)` before fitting;
   `analyze` emits a `UserWarning` when x64 is off.

## References

- Hilhorst, G., Quicken, S., van de Vosse, F.N. & Huberts, W. (2024). Efficient sensitivity analysis for biomechanical models with correlated inputs. *International Journal for Numerical Methods in Biomedical Engineering*, 40(2), e3797.
- Li, G., Rabitz, H., Yelvington, P.E., Oluwole, O.O., Bacon, F., Kolb, C.E. & Schoendorf, J. (2010). Global sensitivity analysis for systems with independent and/or correlated inputs. *Journal of Physical Chemistry A*, 114(19), 6022-6032.
- Wirtz, D. & Haasdonk, B. (2013). A vectorial kernel orthogonal greedy algorithm. *Dolomites Research Notes on Approximation*, 6, 83-100.

See the [VKOGA example](/examples/vkoga), [Methods](/guide/methods), and the
[API overview](/api/).
