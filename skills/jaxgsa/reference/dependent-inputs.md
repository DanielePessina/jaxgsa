# Dependent inputs

`sobol`, `efast`, `pce`, `morris`, `dgsm` and PCE-backed `shapley` all assume
independent parameters and raise a `ValueError` when `problem.correlation` is
declared. Two methods are built for dependence instead: `kucherenko`, which
needs its own design, and `vkoga`, which works from data you already have.

## Declaring the dependence

A Gaussian copula. Every marginal is kept exactly as written; only the joint
pairing changes.

```python
import numpy as np
import jaxgsa

R = np.array([[1.0, 0.8, 0.0],
              [0.8, 1.0, 0.0],
              [0.0, 0.0, 1.0]])

problem = jaxgsa.Problem.from_dict(
    {
        "x1": jaxgsa.GaussianSpec(mean=0.0, variance=1.0),
        "x2": jaxgsa.GaussianSpec(mean=0.0, variance=1.0),
        "x3": jaxgsa.GaussianSpec(mean=0.0, variance=1.0),
    },
    correlation=R,                # or correlation_type="spearman"
)
problem.has_correlated_inputs     # True
problem.correlation               # the validated latent matrix
```

`correlation_type="latent"` (the default) reads `R` as the Pearson correlation
of the copula's latent normals. `"spearman"` reads it as a rank correlation and
converts with `2 sin(pi rho_s / 6)` before storing. The distinction is not
cosmetic: a published Spearman 0.8 is a latent 0.8135, and declaring it as
`"latent"` understates the dependence.

To fit the matrix from data instead:

```python
R = jaxgsa.sampling.fit_correlation(problem, X_observed)
problem = problem.with_correlation(R)
```

`jaxgsa.sampling.monte_carlo(problem, n, seed=...)` honors a declared
correlation through the NORTA construction, so it draws correlated rows that
keep every declared marginal. `jaxgsa.sampling.correlate(X, problem)` imposes
the correlation on an existing sample by rank re-pairing, leaving the marginal
values untouched.

A correlation entry that touches a categorical parameter is rejected. Keep that
row and column at identity.

## The running example

`Y = x1 + x2 + x3` with standard normal marginals and `corr(x1, x2) = 0.8`. The
conditional-variance quantities are exact on paper, which is what makes it worth
using here. `Var(Y) = 3 + 2(0.8) = 4.6`, and the variance of a parameter given
the other two is `1 - 0.8^2 = 0.36` for `x1` and `x2`, and 1 for `x3`:

```
true S_TU = [0.36, 0.36, 1.0] / 4.6 = [0.0783, 0.0783, 0.2174]
```

`x1` and `x2` each move the output as much as `x3` does, but fixing either one
alone barely reduces the output variance, because the other still carries most
of the same information. That gap is the whole subject.

## kucherenko

Evaluates the real model on a conditional-copula design. No surrogate.

```python
def model(X):
    return X[:, 0] + X[:, 1] + X[:, 2]

ks = jaxgsa.kucherenko.sample(problem, 8192, seed=0)   # reads problem.correlation
Y = model(jnp.asarray(ks.samples))
result = jaxgsa.kucherenko.analyze(ks, Y)

ks.samples.shape[0]   # 57344 model runs
result.S1             # [0.7041, 0.7043, 0.2170]   correlation-inclusive
result.ST             # [0.0783, 0.0782, 0.2174]   correlation-exclusive
result.variance
```

`ST` reproduces the analytic `[0.0783, 0.0783, 0.2174]` to four decimals. Note
that `ST` is far below `S1` for the coupled parameters: `S1` counts what a
parameter explains through its coupling as well as on its own, and `ST` counts
only what is lost when everything else is held. The usual `ST >= S1` ordering
does not hold under dependence.

Signature:

```python
jaxgsa.kucherenko.sample(problem, n_samples, *, scramble=True, seed=None,
                         verbose=True)
jaxgsa.kucherenko.analyze(samples, Y, *, n_bootstrap=0, conf_level=0.95,
                          ci_method="quantile", key=None, on_invalid="raise",
                          verbose=True, keep_replicates=False)
```

Cost is `N(2D + 1)` model runs. Every parameter needs its own pair of blocks,
because the redraw for parameter `i` comes from a conditional that depends on
`i`, which is why the design cannot reuse one `B` block the way `sobol` does.

Keep in mind:

- **These are not `sobol`'s indices under another name.** Under a declared
  correlation `S1` is correlation-inclusive and `ST` is correlation-exclusive,
  so neither belongs in a table beside `sobol.S1` or `sobol.ST`.
- **An uncorrelated problem is paying for a design it does not need.** With no
  declared correlation the conditionals collapse to independent draws, the
  design reduces to the Saltelli column-swap scheme, and the indices are the
  classic Sobol `S1` and `ST`. On a 3-parameter problem at `base_n=4096` that is
  28,672 model runs against `sobol`'s 20,480, or 32,768 with `S2`. `sample`
  warns and quotes all three counts. It warns rather than raises, because
  cross-checking a correlated analysis against a design-based reference is a
  valid reason to run it that way.
- Even then the two do not agree bit for bit: Kucherenko's `S1` is the
  Homma-Saltelli estimator, not the Sobol-Mauntz form `sobol` uses by default.
- The design is built from the declared copula, so a wrong matrix gives clean
  estimates of the wrong quantity. Conditioning is closed-form only in latent
  normal space, which means tail-dependent or non-monotone dependence is not
  representable here.
- No `S2`, no surrogate, and a categorical parameter raises.

## vkoga

Fits a greedy kernel surrogate, then estimates the correlated variance-based
indices of Li et al. (2010) against it. Given data.

```python
import jax
jax.config.update("jax_enable_x64", True)      # before fitting

# Train on an INDEPENDENT design, even though the analysis is correlated.
X_train = jnp.asarray(
    jaxgsa.sampling.monte_carlo(problem.with_correlation(None), 2000, seed=1)
)
Y_train = model(X_train)

result = jaxgsa.vkoga.analyze(problem, X_train, Y_train, key=jax.random.key(0))

result.S_TC     # [0.6951, 0.6975, 0.2246]   total correlated, prioritisation
result.S_TU     # [0.0787, 0.0803, 0.2272]   total uncorrelated, fixing
result.S_U      # [0.0772, 0.0787, 0.2243]   uncorrelated first order
result.S_C      # [0.6179, 0.6188, 0.0003]   correlated share
result.S_IU     # [0.0015, 0.0015, 0.0029]   independent interaction
result.n_centers, result.gamma, result.ridge, result.rmse   # 300, ..., 0.094

Y_pred = result.predict(X_new, batch_size=2048)
```

`S_TC` matches Kucherenko's `S1` and `S_TU` matches its `ST`, here to about two
decimals against the analytic `[0.0783, 0.0783, 0.2174]`, from 2000 model runs
instead of 57,344. The difference is surrogate error, which is the price of not
being able to run the model again.

`S_C` is the reading the pair of methods is for: 0.62 of `x1`'s influence is
carried by its coupling to `x2`, and only 0.077 is its own. Fixing `x1` alone
buys you almost nothing.

Signature:

```python
jaxgsa.vkoga.analyze(problem, X, Y, *, correlation=None, gamma=None, ridge=None,
                     max_centers=None, n_folds=10, n_outer=512, n_inner=128,
                     n_variance=8192, n_bootstrap=0, conf_level=0.95,
                     ci_method="quantile", key=None, batch_size=None,
                     on_invalid="raise", verbose=True, keep_replicates=False)
```

`key` is required, because the index stage is always a Monte Carlo draw.
`correlation=` overrides `problem.correlation` for one call, and the matrix
actually used comes back on `result.correlation`.

Keep in mind:

- **Train on an independent, space-filling design.** This is the easy way to get
  wrong answers, and it is why the example above drops the correlation before
  drawing `X_train`. A correlated sample concentrates on a ridge through the
  parameter space, but `S_TU` conditions on the other parameters and then
  resamples `X_i` across its whole marginal, which is precisely the off-ridge
  region a correlated training set never visited. The surrogate extrapolates
  exactly where the estimator leans on it hardest. `analyze` warns when it
  detects correlation in the training `X`. If your data is observational you can
  still fit the copula from it, but read `S_TU`, and so `S_U`, `S_C` and `S_IU`,
  as carrying that extrapolation error.
- **Use float64.** The coefficient step forms the normal matrix, which squares
  the condition number of the cross kernel, and for small `gamma` that exceeds
  what single precision carries. `analyze` warns when x64 is off.
- If you can still run the model, run `kucherenko` instead and get the same two
  quantities with no surrogate error in between. VKOGA is the given-data
  fallback, not the better estimator.
- Cost is dominated by a 10 by 10 grid of k-fold refits. Pass `gamma` and
  `ridge` explicitly once you know good values. `n_outer`, `n_inner` and
  `n_variance` only touch the surrogate, so they are cheap to raise.
- `result.shapley()` raises `NotImplementedError`, and there is no `S2`.

## Four indices under dependence

There is no single generalisation of the Sobol indices to dependent inputs.
jaxgsa ships four variance-based routes, they measure different things, and they
disagree on the same data.

| Route | What it estimates | What it needs |
| --- | --- | --- |
| `kucherenko` | `S1 = V(E(Y given X_i))/V(Y)` and `ST = E(V(Y given X_~i))/V(Y)`, exactly, under the declared copula | Its own design, `N(2D+1)` model runs, a declared correlation |
| `vkoga` | The same two, as `S_TC` and `S_TU`, plus `S_U`, `S_C`, `S_IU`, from a fitted kernel surrogate | Any (X, Y) pairs, a declared correlation |
| `hdmr` ANCOVA split | Per term, not per parameter: each component's variance split into a structural share `Sa` and a coupling share `Sb` | Any (X, Y) pairs. The correlation is read implicitly out of X |
| `shapley(backend="hdmr", include_correlative=True)` | One allocation per parameter summing to 1, splitting each term's `Sa + Sb` among its participants | Any (X, Y) pairs |

Three things to hold on to:

- **They are different estimands.** A disagreement is not a bug in one of them.
  `kucherenko` and `vkoga` estimate the same pair and should agree up to
  surrogate and Monte Carlo error, as they do above. The other two estimate
  something else and have no reason to match.
- **Only two of the four are conditional-variance indices.** HDMR's `ST` under
  dependence is a term-membership sum, not a total-effect index. On
  `Y = X1 + X2 + X3` with `corr(X1, X2) = 0.95` and 8192 samples, HDMR reports
  `ST = [0.398, 0.397, 0.207]` where the true conditional-variance totals are
  `[0.020, 0.020, 0.204]`: right about the independent parameter, twenty times
  too high on the two coupled ones. The bias runs toward "cannot be fixed".
  `hdmr.analyze` warns on a correlated problem for this reason. The ANCOVA
  Shapley allocation is likewise not the Song et al. (2016) Shapley effects, and
  its correlative shares can go negative.
- **None of them is comparable to `jaxgsa.sobol`.** Under dependence a
  first-order index that includes coupling is a different number from one that
  does not.

## The distribution-based alternatives

`optimal_transport`, `borgonovo`, `hsic` and `pawn` never assumed independence
in the first place, so they accept a correlated problem without complaint. Their
indices are correlation-inclusive: a parameter the model never reads scores
above zero when it correlates with one the model does read. That is the correct
reading of those indices, not an estimation error, and it is why they cannot
answer the fixing question the way `S_TU` can. See `moment-independent.md`.
