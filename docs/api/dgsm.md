# DGSM

DGSM ranks inputs by how strongly the output reacts to them on average. The
measure is `nu_i = E[(df/dx_i)^2]`, the mean squared partial derivative over
the input distribution. When the model is JAX-differentiable this is the
cheapest useful method in the package: one autodiff sweep over an ordinary
Monte Carlo sample replaces a whole Saltelli design.

Those moments turn into two numbers that frame the total Sobol index:

- `upper_bound_i = C_i * nu_i / Var(Y)`, where `C_i` is the Poincare constant
  of input `i`'s marginal. The Poincare / Sobol-Kucherenko inequality makes
  this a genuine cap: `ST_i` is never above it, for every marginal this
  package supports.
- `lower_bound_i = Var(x_i) * sigma_i^2 / Var(Y)`, where
  `sigma_i = E[df/dx_i]` is the mean signed derivative. Kucherenko & Song
  (2016), Theorem 4.1, prove `ST_i >=` this **when input `i`'s marginal is an
  untruncated Gaussian**, and only then. On a uniform or truncated marginal it
  is an estimate: exact when the response is linear in that input, and able to
  sit above the true `ST_i` when it is strongly curved. The
  [DGSM example](/examples/dgsm/) works through a case where it reads 1.29 for
  an input whose `ST` is 1 by definition.

An input whose `upper_bound` is near zero is provably negligible. That is a
stronger statement than a small estimated index, and it is what DGSM is for.
Confirm anything that rests on `lower_bound` with `jaxgsa.sobol`.

## How to call it

```python
analyze(
    problem: Problem,
    fn: Callable | None = None,
    X: Array | None = None,
    *,
    Y: Array | None = None,
    dfdx: Array | None = None,
    standardize_outputs: bool = False,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    batch_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> DGSMResult
```

There are two calling conventions. Use exactly one.

- Autodiff path: `analyze(problem, fn, X)`. JAX differentiates the model, and
  one pass returns both the Jacobian and the forward outputs.
- Pre-computed path: `analyze(problem, Y=Y, dfdx=dfdx)`. Use it when the model
  is not JAX-differentiable, or when the Jacobian comes from somewhere else.

Arguments from both groups, or one group only half filled, raise a
`ValueError` that names the argument. Nothing is dropped in silence.

```python
X = jaxgsa.sampling.monte_carlo(problem, 4096, seed=0)

def ishigami_one(x):        # one row (D,) -> scalar ()
    return jnp.sin(x[0]) + 7.0 * jnp.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * jnp.sin(x[0])

result = jaxgsa.dgsm.analyze(problem, ishigami_one, X, verbose=False)

result.nu           # Array([ 7.8056626, 24.486486 , 10.771026 ], dtype=float32)
result.sigma        # Array([-0.06197319, -0.01710795, -0.00092443], dtype=float32)
result.upper_bound  # Array([2.2949116, 7.1991735, 3.1667461], dtype=float32)
result.lower_bound  # Array([9.2871493e-04, 7.0773523e-05, 2.0664183e-07], dtype=float32)
result.var_y        # Array(13.6051655, dtype=float32)
```

The upper bounds are all above 1 here, so on Ishigami they exclude nothing.
That is the honest cost of a bound that must hold for every model with these
marginals. `nu` still ranks the inputs, and it agrees with the Sobol answer
that `x2` dominates. Read the near-zero `sigma` values as a warning too: every
Ishigami derivative averages to about zero over the symmetric domain, which
makes the lower bound useless and is exactly the non-monotonic case `mu` in
Morris also misses.

## fn takes one sample

`fn` is a one-sample function. It maps one row of shape `(D,)` to a scalar
`()`, a vector `(K,)`, or an array `(T, K)`. This is different from every
other method in this package, which calls the model on the whole `(N, D)`
matrix. `analyze()` does the batching itself, because it has to differentiate
per row.

Wrap a batch model like this:

```python
result = jaxgsa.dgsm.analyze(problem, lambda x: model(x[None, :])[0], X)
```

A batch model passed straight in raises a `ValueError` that names the expected
signature. The check traces `fn` on one row with `jax.eval_shape`, so it never
runs the model and costs an expensive model nothing.

The check adds the "wrap a batch model" advice only when the failure looks like
a batch callable: the function indexes or reduces over a sample axis the row
does not have, or it returns more than two axes. Any other trace failure is
reported plainly. You get the original error and the expected signature, and no
guess at the cause. A model that already takes one row and fails for its own
reason is therefore not sent to write a wrapper it does not need.

## The autodiff mode is chosen by shape

There is no keyword for it. `analyze` compares the number of output slices to
the number of inputs and picks `jax.jacfwd` when `T*K > D`, `jax.jacrev`
otherwise. That is the standard rule: reverse mode costs one pass per output,
forward mode one per input, so you sweep whichever axis is shorter. Both modes
compute the same Jacobian; only the order of float arithmetic differs.

The verbose summary reports which one ran, with the numbers behind the choice:

```
    gradients: reverse-mode autodiff (T*K=1, D=3)
```

```
    gradients: forward-mode autodiff (T*K=10, D=3)
```

See `docs/adr/0005-autodiff-mode-selection.md`.

## Arguments

`dfdx` mirrors `Y`'s layout with one extra trailing `(D,)` axis: `(N, D)` for a
`(N,)` `Y`, `(N, K, D)` for `(N, K)`, and `(N, T, K, D)` for `(N, T, K)`.

`standardize_outputs=True` reports `nu`, `sigma` and `var_y` for the
standardized output `(Y - mean) / std`, one mean and one standard deviation per
output slice. DGSM returns dimensional quantities: under `Y -> a*Y + b`,
`sigma` scales by `a` and `nu` by `a^2`. Dividing them out puts every slice in
units of its own standard deviation, so slices of different magnitude compare
directly. `upper_bound` and `lower_bound` are ratios and do not move.

`n_bootstrap` resamples rows with replacement. That is exactly the right unit
here: `nu` and `sigma` are plain i.i.d. means over rows, and both bounds are
plug-in ratios of those means to `Var(Y)`, recomputed on the same resampled
rows. Intervals come back for all four. `var_y` gets none, because it is the
denominator rather than a sensitivity measure, and its uncertainty is already
inside the two bound intervals. The cost is one extra sweep of the Jacobian
over the sample, not one sweep per replicate, because a replicate is a
weighted row sum. `key` is required as soon as `n_bootstrap > 0`.

`batch_size` sets how many sample rows the autodiff path processes at once,
clamped to `N`. The Jacobian accumulates in batches, so this is what bounds
peak memory. `None` derives a width from the memory budget, pricing each row at
a few Jacobian-sized transients (`T*K*D` floats per row times a small live
factor). Set it yourself when a large `T*K*D` blows up device memory.

`on_invalid` checks the derivative as well as the output, on both calling
conventions. A derivative that blows up poisons `nu` even where the output
itself is finite, so the report names `"Y or its derivative"` for a bad
derivative. On the autodiff path `X` is checked too, and rows are masked before
the batch reduction, so `"drop"` gives the same moments as re-running on the
smaller sample.

DGSM raises on a correlated problem, because the Poincare inequality assumes
independent inputs, and on a categorical one, because a derivative along an
unordered level code has no meaning.

## DGSMResult

| Field | Meaning |
| --- | --- |
| `nu` | `E[(df/dx_i)^2]`, shape `(D,)` / `(K, D)` / `(T, K, D)` |
| `sigma` | `E[df/dx_i]`, the mean signed derivative, same shape |
| `upper_bound` | Poincare upper bound on `ST`, valid for every supported marginal |
| `lower_bound` | `Var(x_i) * sigma_i^2 / Var(Y)`; a lower bound on `ST` for a Gaussian marginal, an estimate otherwise |
| `var_y` | output variance per slice, the denominator of both bounds |
| `nu_conf`, `sigma_conf`, `upper_bound_conf`, `lower_bound_conf` | `(2, ...)` for `[lower, upper]`, or `None` |
| `ci`, `problem`, `invalid` | the interval record, the problem, the non-finite report |

A zero-variance output slice makes both bounds NaN for that slice and raises a
`JaxgsaWarning` naming it.

## Poincare constants

```python
poincare_constant(spec: InputSpec) -> float
axis_constants(problem: Problem) -> tuple[np.ndarray, np.ndarray]
```

`poincare_constant` returns `C(p)` for one marginal: `(high - low)^2 / pi^2`
for a uniform, `variance` for an unbounded Gaussian, and a numerical spectral
solve for a truncated one. An open side of a truncated Gaussian is stood in for
at 8 sigma, which carries the whole mass to float precision. A categorical
marginal raises, since the inequality needs a continuous density.

`axis_constants` returns `(C, Var)` for the whole problem, each `(D,)`: the
Poincare constants and the marginal variances, the two factors in the upper and
lower bounds.

```python
jaxgsa.dgsm.axis_constants(problem)
# (array([4., 4., 4.]), array([3.28986813, 3.28986813, 3.28986813]))
```

For a uniform on `[-pi, pi]` that is `(2pi)^2 / pi^2 = 4` and
`(2pi)^2 / 12 = 3.2899`.

## indices

```python
indices(problem, fn=None, X=None, *, Y=None, dfdx=None,
        standardize_outputs=False, batch_size=None) -> tuple[Array, ...]
```

The four measures as plain arrays, with no checks and no result object, so it
composes with `jax.jit`, `jax.vmap`, `jax.grad` and `jax.jacrev`. That last one
matters here: it is what lets you differentiate a bound with respect to
something upstream.

Related docs:

- [DGSM Example](/examples/dgsm)
- [Methods](/guide/methods)
- [API reference](/api/)
