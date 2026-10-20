# Dependent inputs

`sobol`, `efast`, `pce`, `morris`, `dgsm` and PCE-backed `shapley` all assume
independent parameters and raise a `ValueError` when the problem declares
dependence (`problem.correlation` or `problem.sampler`). Two methods are built for dependence instead: `kucherenko`, which
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
problem.dependence                # "gaussian"
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

`jaxgsa.sampling.monte_carlo(problem, n_samples, seed=...)` honors a declared
correlation through the NORTA construction, so it draws correlated rows that
keep every declared marginal. `jaxgsa.sampling.transform(problem, U)` maps your
own unit-cube design (for example a Latin hypercube) through the marginals and
the copula or sampler. `jaxgsa.sampling.correlate(problem, X)` imposes
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
import jax.numpy as jnp

def model(X):
    return X[:, 0] + X[:, 1] + X[:, 2]

ks = jaxgsa.kucherenko.sample(problem, base_n=8192, seed=0)   # reads problem.correlation
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
jaxgsa.kucherenko.sample(problem, n_samples=None, *, base_n=None,
                         subsets="singletons",   # or "all"
                         scramble=True, seed=None, verbose=True)
jaxgsa.kucherenko.analyze(samples, Y, *, n_bootstrap=0, conf_level=0.95,
                          ci_method="quantile", key=None, on_invalid="raise",
                          verbose=True, keep_replicates=False)
```

Cost is `N(2D + 1)` model runs for `N` base points. `n_samples` is the minimum
number of model runs, as in `sobol`. Give exactly one of `n_samples` and
`base_n`. `base_n=N` sets the base points exactly and must be a power of two.
Every parameter needs its own pair of blocks,
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

### Exact Shapley effects and PME

`subsets="all"` builds one block for every proper non-empty subset of inputs,
plus the joint blocks A and B. It costs `base_n * 2^D` runs and needs
`2 <= D <= 12`.

```python
ka = jaxgsa.kucherenko.sample(problem, base_n=1024, subsets="all", seed=0)
ra = jaxgsa.kucherenko.analyze(ka, model(jnp.asarray(ka.samples)),
                               n_bootstrap=20, key=0, keep_replicates=True)
ra.shapley().Sh              # [0.39, 0.39, 0.22]  conditional-variance Shapley effects, sum 1
ra.shapley(cost="closed")    # closed-cost variant (default cost="total")
ra.pme().Sh                  # proportional marginal effects; an unused input gets exactly 0
ra.closed, ra.subset_total, ra.subsets   # per-subset indices and the membership matrix
```

Both methods return a `ShapleyResult` with `backend="exact"`. Intervals need
`n_bootstrap > 0` and `keep_replicates=True`. On a `subsets="singletons"` result,
`shapley()` and `pme()` raise `ValueError`. `D > 12` raises. Use `vkoga` or
`shapley` instead. On this design `S1` and `ST` come from the subset blocks, so
they differ slightly from a singletons run.

## vkoga

Fits a greedy kernel surrogate, then estimates the correlated variance-based
indices of Li et al. (2010) against it. Given data.

```python
import jax
jax.config.update("jax_enable_x64", True)      # before fitting

# An independent design can cover the integration support.
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

`S_TC` matches Kucherenko's `S1` and `S_TU` matches its `ST`, here to about 0.01
against the analytic `[0.0783, 0.0783, 0.2174]`, from 2000 model runs
instead of 57,344. The difference is surrogate error, which is the price of not
being able to run the model again.

`S_C` is the reading the pair of methods is for: 0.62 of `x1`'s influence is
carried by its coupling to `x2`, and only 0.077 is its own. Fixing `x1` alone
buys you almost nothing.

Signature:

```python
jaxgsa.vkoga.analyze(problem, X, Y, *, key,   # key: JAX key or int
                     gamma=None, ridge=None,
                     max_centers=None, n_folds=10, n_outer=512, n_inner=128,
                     n_variance=8192, n_bootstrap=0, conf_level=0.95,
                     ci_method="quantile", batch_size=None,
                     on_invalid="raise", verbose=True, keep_replicates=False)
```

`key` is required and keyword-only, because the index stage uses randomized
integration points. The dependence comes from the problem only. Read the matrix
on `result.problem.correlation`. `result.dependence` is `"independent"`,
`"gaussian"` or `"custom"`. `result.S1` and `result.ST` are aliases of `S_TC`
and `S_TU`. `rmse` is in-sample. `cv_rmse` and `explained_variance` hold one
value per output slice.

Keep in mind:

- **Training data must cover the analysis distribution.** The surrogate must
  be accurate where the distribution in `problem` puts its mass. Both nested
  estimators draw from conditionals of the joint distribution, never from a
  whole marginal with the rest held fixed. So rows drawn from the analysis
  distribution work. An independent design is a safe default when the
  correlation is not fixed yet, or when one fit is reused under several
  correlations. Avoid training data that misses regions the analysis covers,
  for example trained under rho=+0.8 and analysed under rho=-0.8. Check
  `cv_rmse`.
- **Use float64.** The coefficient step forms the normal matrix, which squares
  the condition number of the cross kernel, and for small `gamma` that exceeds
  what single precision carries. `analyze` warns when x64 is off.
- If you can still run the model, run `kucherenko` instead and get the same two
  quantities with no surrogate error in between. VKOGA is the given-data
  fallback, not the better estimator.
- Cost is dominated by a 10 by 10 grid of k-fold refits. Pass `gamma` and
  `ridge` explicitly once you know good values. `n_outer`, `n_inner` and
  `n_variance` only touch the surrogate, so they are cheap to raise.
- `VKOGAResult` has no `shapley()` method, and there is no `S2`.

## Five routes under dependence

There is no single generalisation of the Sobol indices to dependent inputs.
jaxgsa ships five variance-based routes, they measure different things, and they
disagree on the same data.

| Route | What it estimates | What it needs |
| --- | --- | --- |
| `kucherenko` | `S1 = V(E(Y given X_i))/V(Y)` and `ST = E(V(Y given X_~i))/V(Y)` under the target joint distribution | Its own design, `N(2D+1)` model runs for `N` base points, a Gaussian correlation or a custom input sampler |
| `vkoga` | The same two, as `S_TC` and `S_TU`, plus `S_U`, `S_C`, `S_IU`, from a fitted kernel surrogate | Training (X, Y) pairs, a Gaussian correlation or a custom input sampler |
| `kucherenko` with `subsets="all"` | Exact conditional-variance Shapley effects (`.shapley()`) and PME (`.pme()`), one allocation per parameter summing to 1 | Its own design, `base_n * 2^D` runs, `2 <= D <= 12` |
| `hdmr` ANCOVA split | Per term, not per parameter: each component's variance split into a structural share `Sa` and a coupling share `Sb` | Any (X, Y) pairs. The correlation is read implicitly out of X |
| `shapley(backend="hdmr", include_correlative=True)` | One allocation per parameter summing to 1, splitting each term's `Sa + Sb` among its participants | Any (X, Y) pairs |

Three things to hold on to:

- **They are different estimands.** A disagreement is not a bug in one of them.
  `kucherenko` and `vkoga` estimate the same pair and should agree up to
  surrogate and Monte Carlo error, as they do above. The other two estimate
  something else and have no reason to match.
- **Only the `kucherenko` and `vkoga` indices are conditional-variance indices.** HDMR's `ST` under
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

`optimal_transport`, `borgonovo`, `hsic`, `pawn` and `rank` never assumed independence
in the first place, so they accept a correlated problem without complaint. Their
indices are correlation-inclusive: a parameter the model never reads scores
above zero when it correlates with one the model does read. That is the correct
reading of those indices, not an estimation error, and it is why they cannot
answer the fixing question the way `S_TU` can. See `moment-independent.md`.

## Custom input sampler

A problem can carry a custom sampler. It declares a joint distribution that is
not a Gaussian copula. Declare it with the problem:

```python
problem = jaxgsa.Problem(names=..., bounds=..., sampler=f, sampler_dimension=2)
problem = jaxgsa.Problem.from_dict(..., sampler=f, sampler_dimension=2)
problem = problem.with_sampler(f, sampler_dimension=2)   # copy with a sampler
problem = problem.with_sampler(None)             # copy without it
```

A problem has either a Gaussian-copula `correlation` or a `sampler`, never both
(`ValueError`). `problem.dependence` is `"independent"`, `"gaussian"` or
`"custom"`. Kucherenko and VKOGA results carry the same `dependence` field.

The sampler type is `jaxgsa.sampling.InputSampler`:

```python
sampler(uniform_points, *, conditioning_rows=None, fixed_indices=())
```

The package supplies `(n, sampler_dimension)` uniform points. Return `(n, D)`
finite real physical inputs in `problem.names` order. `sampler_dimension`
defaults to `D`. On the default Kucherenko design and in VKOGA, the package makes three kinds
of call. With `subsets="all"`, the sampler also gets every other sorted
subset in `fixed_indices`, so write it to accept any subset.

- `fixed_indices=()`: joint draws, `conditioning_rows=None`.
- `fixed_indices=(i,)`: keep column `i` of `conditioning_rows`, redraw the rest
  given `x_i`.
- `fixed_indices` holding every index except `i`: keep those columns, redraw
  `x_i` given the rest.

Copy the fixed columns exactly. Overwriting fixed columns after a joint draw is
valid only for independent inputs. The callback must be deterministic: identical
arguments give identical outputs. The package passes copies, so changing the
arguments does no harm. All randomness comes from the package: Kucherenko's
`seed` and `scramble`, or VKOGA's required `key`. Everything uses scrambled Sobol' points, and sample sizes round up to
the next power of two. Shapes, finite values, fixed columns and the support of each input (uniform and
truncated bounds, integer codes for categorical inputs) are checked. A value
that overshoots a bound by float32 rounding is clipped. A value further out
raises `ValueError`. The sampler must be a deterministic inverse Rosenblatt
(conditional-quantile) map. Rejection sampling and MCMC are not valid.
Statistical consistency is the user's responsibility.

`kucherenko.sample(problem, n_samples, seed=...)` and
`vkoga.analyze(problem, X_train, Y_train, key=...)` read the sampler from the
problem. Kucherenko and VKOGA share one estimator for both dependence kinds, and
the custom path costs about the same time.

Sobol, Morris and eFAST designs and the independence-only analyses refuse a
custom-sampler problem, as they refuse a correlated one. For given-data
methods, draw `X` with `jaxgsa.sampling.monte_carlo(problem, n_samples, seed=...)`. It
sends pseudo-random uniforms through the joint branch, so rows are independent
draws and row bootstrap and HSIC permutation tests stay valid. This does not
turn an ordinary `X` matrix into a Kucherenko analysis design. Saved designs
retain sampled rows and provenance, not callable code. A loaded problem still
reports `dependence == "custom"` but raises if asked to draw rows. Attach the
sampler again with `problem.with_sampler(f, sampler_dimension=k)`. VKOGA's declared
marginals must match the sampler, because its coordinate transform and
component bases still use them. Training data must cover the analysis
distribution. Dependence fitting remains explicit, never triggered just by
supplying data.

### Higher-order dependence

Independent uniforms `U, V` with `W = (U + V) % 1` give three pairwise
independent inputs whose joint distribution is constrained. An identity
Gaussian copula loses that constraint. For `Y = W`, the true `S1` is
`[0, 0, 1]` and conditional `ST` is `[0, 0, 0]`, because the other two inputs
determine any missing input. Zero conditional total means redundant
information, not causal inactivity; independent uniforms would give
`ST = [0, 0, 1]`. Use this example to test higher-order support rather than
only zero pairwise correlations. Check every fixed subset, including empty
and all columns. The runnable joint/conditional sampler is in
`docs/examples/correlated-inputs.md`. These per-input indices do not identify
groups responsible for dependence; the full joint law is supplied, not
inferred. VKOGA uses the same sampler and still needs matching `Problem`
marginals.
