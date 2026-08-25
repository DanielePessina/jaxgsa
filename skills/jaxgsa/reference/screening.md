# Screening methods

Two cheap methods that answer a narrower question than a variance decomposition:
which parameters can I stop worrying about? They are good at finding the ones
that do nothing and less good at ordering the ones that do. Both misrank
Ishigami's top two.

Use them to decide what to **drop**, fix the negligible parameters, then spend
the remaining budget on `sobol` for the survivors.

Both refuse a correlated or categorical problem.

The examples use Ishigami, whose true `ST` is `[0.5576, 0.4424, 0.2437]`:

```python
import jax.numpy as jnp
import numpy as np
import jaxgsa
from jaxgsa.benchmarks import ishigami

PROBLEM = ishigami.PROBLEM
```

## morris

Elementary effects: one finite-difference slope per trajectory and parameter,
reduced to a mean, a mean absolute value, and a standard deviation.

```python
samples = jaxgsa.morris.sample(PROBLEM, n_trajectories=50, seed=0)
samples.samples.shape[0]   # 60 rows, from 200 before deduplication
Y = ishigami.evaluate(jnp.asarray(samples.samples))
result = jaxgsa.morris.analyze(samples, Y)

result.mu        # signed mean effect, cancels on a non-monotone response
result.mu_star   # [5.70, 7.88, 5.50]   the headline measure
result.sigma     # [5.98, 7.14, 8.31]   nonlinearity or interaction
```

Read it as a plot, not a table. `mu_star` on one axis, `sigma` on the other, and
drop what sits near the origin. Here nothing does: all three parameters matter,
which is the correct screening answer even though the ranking is wrong. `x2`
leads on `mu_star` while the true `ST` leads with `x1`.

60 rows, not 200. The default `num_levels=4` grid collides often in low
dimensions, and jaxgsa evaluates each distinct row once. Cost is `r(D + 1)`
model runs before deduplication, against `N(2D + 2)` for Sobol.

Signature:

```python
jaxgsa.morris.sample(problem, n_trajectories, *, num_levels=4,
                     method="trajectory", scramble=True, seed=None,
                     truncation_quantile=1e-4, verbose=True)
jaxgsa.morris.analyze(samples, Y, *, standardize_outputs=False, n_bootstrap=0,
                      conf_level=0.95, ci_method="quantile", key=None,
                      resample_chunk_size=None, on_invalid="raise",
                      verbose=True, keep_replicates=False)
```

Morris spells its batching keyword `resample_chunk_size`, not
`slice_chunk_size`. `MorrisResult.to_physical_units()` rescales to a derivative
scale in the problem's own units, and covers uniform marginals only: for a
Gaussian marginal the inverse-CDF transform is nonlinear, so there is no single
linear rescaling and the call raises rather than return a number on the wrong
scale.

### Free screening from a Saltelli design

A Saltelli design already contains a radial Morris design. Within each base
point it holds a row `A` and `D` rows that differ from `A` in exactly one
parameter, which is what an elementary effect needs.

```python
samples = jaxgsa.sobol.sample(PROBLEM, 8192, seed=0)
Y = ishigami.evaluate(samples.samples)

sobol_result = jaxgsa.sobol.analyze(samples, Y)
morris_result = jaxgsa.morris.analyze(samples.to_morris(), Y)
```

```
jaxgsa.sobol.SobolSamples.to_morris: D=3, mode=second-order, base_n=1024,
  blocks=1024, effects=3072, reusing n_runs=8192 existing evaluations
  (0 new model runs)
ST      [0.5560, 0.4417, 0.2413]
mu_star [8.7048, 15.0253, 6.6204]
sigma   [12.5912, 20.0245, 11.4691]
```

3072 elementary effects out of model runs you had already paid for.

Keep in mind:

- **`mu_star` is a mean absolute slope in unit-cube coordinates, not a variance
  share.** It does not sum to anything. It is a proxy for the `ST` ranking, not
  a substitute, and both designs above swap the top two: `mu_star` puts `x2`
  first while `ST` puts `x1` first. If you screen with Morris and keep only the
  top parameter, you keep the wrong one.
- The magnitude depends on the design, not only on the model. The trajectory
  design above gives `mu_star = [5.70, 7.88, 5.50]` and the radial design
  derived from a Saltelli run gives `[8.70, 15.03, 6.62]` on the same function.
  Compare `mu_star` across parameters, never across designs.
- Large `sigma` relative to `mu_star` means nonlinearity or interactions, and it
  is not attributable to a pair.
- `to_morris()` produces a **radial** design, so compare it against
  `morris.sample(..., method="radial")`, never against the `"trajectory"`
  default. On Ishigami the two differ by a factor of 1.9 on `x2`.
- The derived `mu_star` and the Sobol `ST` come from the same model outputs, so
  their agreement is not an independent check of either.
- With unbounded Gaussian marginals `mu_star` has no fixed magnitude, because
  how far the design reaches into the tail sets it. Only rankings survive a
  change of `truncation_quantile`. Declare `truncate_gaussians=` on the
  `Problem` if magnitudes have to mean anything. `to_morris()` warns when
  unbounded Gaussians are present.
- Keep `scramble=True`. With `scramble=False`, `to_morris()` drops blocks whose
  step is numerically zero, and the survivors are a biased subsequence: 21.9% of
  blocks dropped at `base_n=64`, giving a `mu_star(x3)` 16% low.

## dgsm

Derivative-based measures, and bounds on `ST` built from them. It differentiates
your model with JAX, so it needs no design of its own beyond plain Monte Carlo
points.

```python
X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n=1024, seed=0))

# fn takes ONE sample row, shape (D,), and returns a scalar, (K,) or (T, K).
result = jaxgsa.dgsm.analyze(
    PROBLEM, lambda x: ishigami.evaluate(x[None, :])[0], X
)

result.nu           # [7.7500, 24.4051, 10.2674]   mean squared derivative
result.sigma        # mean derivative, near zero on a symmetric response
result.lower_bound  # [0.0010, 0.0074, 0.0022]
result.upper_bound  # [2.3450, 7.3846, 3.1067]
result.var_y
```

Both bounds hold. Neither is usable. The true `ST` is
`[0.5576, 0.4424, 0.2437]`, the lower bound is near zero on all three, and the
upper bound is above 1 on all three, which excludes nothing since `ST <= 1` by
definition. Worse, the upper bound ranks the parameters `x2 > x3 > x1` while the
true `ST` ranks them `x1 > x2 > x3`. `analyze` warns about both of these.

A batch model must be wrapped, as above. Passing one unwrapped raises a
`ValueError` that spells out the fix. If the Jacobian was computed elsewhere,
pass it directly and skip `fn`:

```python
result = jaxgsa.dgsm.analyze(PROBLEM, Y=Y, dfdx=dfdx)
```

Signature:

```python
jaxgsa.dgsm.analyze(problem, fn=None, X=None, *, Y=None, dfdx=None,
                    standardize_outputs=False, n_bootstrap=0, conf_level=0.95,
                    ci_method="quantile", key=None, batch_size=None,
                    on_invalid="raise", verbose=True, keep_replicates=False)
```

Cost is `N` Jacobians, each about `min(D, T*K)` evaluations. jaxgsa picks
forward or reverse mode from the shapes.

Keep in mind:

- **DGSM finds parameters that do nothing. It does not rank the ones that do.**
  More samples do not help: the Poincare constant sets the slack in the upper
  bound, not the sample size. At `N = 131072` the bounds settle at
  `[2.19, 7.06, 3.17]` and the ranking is unchanged.
- `nu` is a mean **squared** derivative, so a steep slope over a small part of
  the range dominates it. `x3`'s derivative reaches 12 at the ends of its range
  and is near zero over most of it, which gives it a large `nu` and a small
  variance share.
- The Kucherenko-Song lower bound is a proven bound only for an untruncated
  Gaussian marginal. Otherwise it is an estimate that can exceed the true `ST`
  when the response is curved, and `analyze` warns and names the marginals. It
  collapses toward zero for any response that is not monotone in the parameter,
  because the mean derivative cancels.
- At `T*K` much larger than `D` the Jacobian costs `D` forward passes and the
  cost argument for DGSM evaporates. Run Sobol.
- If the model is not JAX-differentiable, use Morris, which is the same idea at
  a finite step size.
- A truncated Gaussian needs its Poincare constant from a finite-element
  spectral solve rather than a closed form.
