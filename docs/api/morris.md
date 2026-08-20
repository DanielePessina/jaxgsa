# Morris

Morris screening ranks parameters by one-at-a-time finite differences, called
elementary effects, taken at points spread across the whole input domain. It
is the cheap first pass: rank many parameters, drop the dead ones, then spend
the model budget on a variance-based method for the rest.

The measures are `mu_star`, the mean absolute elementary effect, and `sigma`,
their standard deviation. `mu_star` is the importance ranking. A `sigma` that
is large next to `mu_star` means the effect changes across the domain, so the
parameter is nonlinear or interacting.

## sample

```python
sample(
    problem: Problem,
    n_trajectories: int,
    *,
    num_levels: int = 4,
    method: Literal["trajectory", "radial"] = "trajectory",
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
    truncation_quantile: float = 1e-4,
    verbose: bool = True,
) -> MorrisSamples
```

`n_trajectories` is the sample size behind every measure: each trajectory
contributes exactly one elementary effect per parameter. It must be at least
2. Typical screening uses 10 to 50. The full design costs at most
`n_trajectories * (D + 1)` model runs, and usually fewer, because exact
duplicate rows are removed.

`num_levels` is the grid resolution `p` of the trajectory design, with step
`delta = p / (2 * (p - 1))`. Keep it even. An odd value makes the grid levels
unequally probable and lands steps off-grid, and jaxgsa warns about it. The
radial design ignores this argument.

`method` picks the design generator. `"trajectory"` walks the Morris (1991)
grid. `"radial"` builds Campolongo (2011) star designs around
scrambled-Sobol' base points, which spreads points quasi-randomly instead of
on a coarse grid and leaves far fewer duplicates to remove. `scramble` applies
only to the radial design.

`seed` sets the design. Pass an `int` or `None` to keep the prefix-nesting
guarantee of `downsample`. A reused `np.random.Generator` advances its state
between calls, which breaks that nesting.

`truncation_quantile` pulls the design away from the unit-cube faces on any
*open* side of a Gaussian marginal, because an unbounded inverse CDF maps 0
and 1 to infinity. A side the problem already bounds with an explicit `low` or
`high` is left alone, so a two-sided truncated Gaussian is sampled as
declared, and uniform and categorical marginals never move.

The default `1e-4` is not arbitrary. At `q = 1e-4` the grid drops 0.29% of the
marginal variance and 5.0% of its fourth moment. The former default of `5e-3`
dropped 7.5% and 24%, and it visibly moved rankings.

On an unbounded marginal `mu_star` has no `q -> 0` limit at all. The design
always includes unit levels 0 and 1 exactly, so a smaller `q` always reaches
further into the tail and the effects grow with it. Compare rankings across
truncation settings, never magnitudes. If you want one bounded input model
that every method shares, declare it once with
`Problem.from_dict(..., truncate_gaussians=q)`.

`sample` raises on a correlated problem, because the one-at-a-time design
assumes independent inputs, and on a categorical problem, because stepping
along a grid has no meaning for an unordered level code.

## MorrisSamples

- `samples` — `(n_runs, D)`, the unique rows to evaluate.
- `n_expanded` — the design size before deduplication,
  `n_trajectories * (D + 1)`.
- `expanded_to_unique` — the map back to `samples`.
- `ee_idx_before`, `ee_idx_after`, `ee_delta` — `(r, D)` each, the
  bookkeeping for `EE = (Y[after] - Y[before]) / delta`.
- `n_trajectories`, `num_levels`, `method`, `n_params`, `problem` — the design
  metadata.
- `n_blocks_dropped` — blocks lost at construction because their step was not
  measurable. Always 0 for a design `sample()` built. It can be positive only
  for a converted design, such as one from `SobolSamples.to_morris()`, and
  `analyze` reports the loss because you did not ask for a smaller design.
- `downsample(n_trajectories, Y=None)`, `save(path)`, `load(path)`.

Deduplication earns its keep here. A 512-trajectory grid design over 3
parameters expands to 2048 rows and collapses to 64 unique ones, because
trajectories on a `num_levels=4` grid keep landing on the same points:

```
jaxgsa.morris.sample: D=3, method=trajectory, n_trajectories=512, num_levels=4,
n_expanded=2048, n_runs=64, duplicates_removed=1984 (96.9%)
```

## analyze

```python
analyze(
    sampling_result: MorrisSamples,
    Y: Array,
    *,
    standardize_outputs: bool = False,
    n_bootstrap: int = 0,
    conf_level: float = 0.95,
    ci_method: Literal["quantile", "gaussian"] = "quantile",
    key: Array | None = None,
    resample_chunk_size: int | None = None,
    on_invalid: OnInvalid = "raise",
    verbose: bool = True,
    keep_replicates: bool = False,
) -> MorrisResult
```

```python
samples = jaxgsa.morris.sample(problem, n_trajectories=512, seed=0, verbose=False)
result = jaxgsa.morris.analyze(samples, ishigami(samples.samples), verbose=False)

result.mu_star   # Array([7.4111, 7.875 , 6.6882], dtype=float32)
result.sigma     # Array([6.248 , 7.8812, 9.1494], dtype=float32)
result.space     # 'unit'
```

`sigma` beats `mu_star` on `x3` here, which is the right answer for Ishigami:
`x3` has almost no effect on its own and acts only through its interaction
with `x1`.

`standardize_outputs=True` standardizes each output slice to mean 0 and unit
standard deviation over the expanded sample axis before the effects are
formed. All three measures are dimensional: under `Y -> a*Y + b` each of `mu`,
`mu_star` and `sigma` scales by `a`. So this keyword changes what the numbers
mean. They come back in units of the output's own standard deviation, which is
what makes a slice in millimetres comparable to a slice in metres. It does not
change the ranking inside one slice.

`resample_chunk_size` caps how many bootstrap replicates one device call may
carry. Each replicate gathers a full `(r, D, T, K)` copy of the elementary
effects, so a time-series run can exhaust device memory at a chunk width that
a scalar run handles easily. `None` derives the width from the memory budget;
an explicit value is honoured as given, capped only at `n_bootstrap`. It
changes no number beyond float summation order.

`on_invalid` counts one trajectory as the unit, so a single non-finite value
condemns the whole trajectory's `D + 1` rows.

## MorrisResult

| Field | Meaning |
| --- | --- |
| `mu` | mean elementary effect, sign kept, shape `(..., D)` |
| `mu_star` | mean absolute elementary effect, the headline ranking |
| `sigma` | standard deviation of the effects, `ddof=1` |
| `mu_conf`, `mu_star_conf`, `sigma_conf` | `(2, ...)` for `[lower, upper]`, or `None` |
| `space` | `"unit"` or `"physical"` |
| `ci`, `problem`, `invalid` | the interval record, the problem, the non-finite report |

Keep `mu` next to `mu_star`. Opposite-sign effects cancel in `mu`, so a
parameter with a large `mu_star` and a near-zero `mu` is non-monotonic.

`space` says which coordinates the measures live in. The default `"unit"`
divides the output change by a step in `[0, 1]` coordinates, which makes
`mu_star` comparable across parameters with different ranges.
`result.to_physical_units()` divides each measure by the parameter range
`high - low` and returns a copy with `space == "physical"`. That is a
derivative scale, directly comparable to DGSM's `sigma`:

```python
result.to_physical_units().mu_star   # Array([1.1795, 1.2533, 1.0645], dtype=float32)
```

It raises for a Gaussian problem, where the transform is nonlinear and no
single range exists, and for a result that is already physical. Any kept
bootstrap draws are rescaled along with the measures.

## indices

```python
indices(sampling_result, Y, *, standardize_outputs=False)
    -> tuple[Array, Array, Array]     # mu, mu_star, sigma
```

Raw arrays, no checks, no result object. Safe inside `jax.jit` and `jax.vmap`,
where `analyze` is not.

## Free screening from a Sobol design

`SobolSamples.to_morris()` reinterprets an already-evaluated Saltelli design as
a radial Morris design, so the screening measures cost no extra model runs:

```python
samples = jaxgsa.sobol.sample(problem, 8192, seed=0)
Y = model(samples.samples)
sobol_result = jaxgsa.sobol.analyze(samples, Y)
morris_result = jaxgsa.morris.analyze(samples.to_morris(), Y)
```

Related docs:

- [Morris Example](/examples/morris)
- [Methods](/guide/methods)
- [API reference](/api/)
