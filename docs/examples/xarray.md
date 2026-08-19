# xarray Labeled Output

By the end of this page you will have two `xarray.Dataset` objects, one from a
Sobol analysis and one from an RS-HDMR analysis of the same model. In both you
can select a number by parameter name, output name, time coordinate, or term
label, instead of counting integer axes.

`to_dataset()` does the conversion. An `xarray.Dataset` is a container of named
arrays that share named axes (dimensions) and axis labels (coordinates), so
`sel(param="amplitude")` replaces an index like `[..., 0]`.

## Self-contained setup

The setup below runs in five steps.

1. Declare the problem, with three named input parameters and two named
   outputs. The names given here become the `param` and `output` coordinate
   labels in every dataset built later.
2. Write the model. It returns displacement and velocity at 30 timepoints,
   stacked last, which is the `(N, T, K)` layout: `N` model runs, `T`
   timepoints, `K` outputs.
3. Run Sobol. `num_resamples=100` and a random `key` turn on bootstrap
   resampling, which is what produces the confidence intervals used further
   down. Without them there would be no `S1_lower` or `S1_upper` to export.
4. Run RS-HDMR on a separate set of 1500 uniform random points. HDMR fits a
   surrogate to arbitrary `(X, Y)` pairs, so it does not need the structured
   Saltelli design that Sobol uses, and it gets its own sample.
5. Export both results. Passing `time_coords=time_values` labels the time axis
   with the real time values rather than the integers 0 to 29.

```python
import jax
import jax.numpy as jnp
import numpy as np
import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {
        "amplitude": (0.5, 2.0),
        "frequency": (1.0, 5.0),
        "damping": (0.01, 0.5),
    },
    output_names=("displacement", "velocity"),
)

time_values = np.linspace(0.0, 5.0, 30)


def model(X):
    amp = X[:, 0, None]
    freq = X[:, 1, None]
    damping = X[:, 2, None]
    tt = jnp.asarray(time_values)[None, :]

    displacement = amp * jnp.sin(2 * jnp.pi * freq * tt) * jnp.exp(-damping * tt)
    velocity = amp * jnp.cos(2 * jnp.pi * freq * tt) * jnp.exp(-damping * tt)
    return jnp.stack([displacement, velocity], axis=-1)  # (N, T, K)


sampling_result = jaxgsa.sobol.sample(problem, n_samples=2048, seed=42)
X_sobol = jnp.asarray(sampling_result.samples)
Y_sobol = model(X_sobol)

sobol = jaxgsa.sobol.analyze(
    sampling_result,
    Y_sobol,
    num_resamples=100,
    key=jax.random.key(0),
)

bounds = jnp.array(problem.bounds)
X_hdmr = jax.random.uniform(
    jax.random.key(1),
    (1500, problem.num_vars),
    minval=bounds[:, 0],
    maxval=bounds[:, 1],
)
Y_hdmr = model(X_hdmr)
hdmr = jaxgsa.hdmr.analyze(problem, X_hdmr, Y_hdmr, maxorder=2)

ds_sobol = sobol.to_dataset(time_coords=time_values)
ds_hdmr = hdmr.to_dataset(time_coords=time_values)
```

## Sobol dataset

```python
print(ds_sobol)
# <xarray.Dataset>
# Dimensions:  (time: 30, output: 2, param: 3, param_i: 3, param_j: 3)

print(ds_sobol.S1.sel(param="amplitude"))
print(ds_sobol.ST.sel(output="velocity"))
print(
    ds_sobol.S2.sel(
        time=time_values[10],
        output="displacement",
        param_i="amplitude",
        param_j="frequency",
        method="nearest",
    )
)
```

The printed dimensions line accounts for every axis in the result. `time: 30`
is the 30 timepoints, `output: 2` the two named outputs, and `param: 3` the
three input parameters. `param_i` and `param_j` are both 3 as well, because the
second-order indices `S2` cover pairs of parameters and so need a parameter
axis twice.

The three selections then return progressively smaller arrays.

- `ds_sobol.S1.sel(param="amplitude")` fixes one of the three parameters and
  leaves `(time, output)`, so 60 first-order values: amplitude's own share of
  the variance at each timepoint, for each output.
- `ds_sobol.ST.sel(output="velocity")` fixes the output and leaves
  `(time, param)`, so 90 total-order values for velocity alone.
- The `S2` selection fixes all four axes and returns a single number: the
  amplitude-frequency interaction in displacement, at the eleventh timepoint.
  `method="nearest"` is there because `time` holds floating-point values, and
  asking for an exact float match is fragile.

## Confidence intervals

The bootstrap resampling requested with `num_resamples=100` gives each index a
lower and an upper bound. `to_dataset()` stores those bounds as their own
dataset variables, named after the index they belong to:

```python
print(ds_sobol.S1_lower.sel(param="amplitude"))
print(ds_sobol.S1_upper.sel(param="amplitude"))
print(ds_sobol.ST_lower.sel(output="velocity"))
```

Each of these has the same dimensions as the index it bounds, so
`S1_lower.sel(param="amplitude")` lines up element by element with
`S1.sel(param="amplitude")` from the previous section. Comparing the two tells
you how much of an index you can trust: an interval that spans zero means the
sample size does not yet separate that parameter from noise.

## Provenance attributes

A dataset also carries the settings the analysis ran with, in `ds.attrs`.
These are the same scalars the result prints, under the same names, so a saved
dataset says what produced it:

```python
print(ds_sobol.attrs)  # {'estimator': 'saltelli-jansen'}
print(ds_hdmr.attrs)   # {'streamed': False}
```

Read them as follows.

- Sobol records its `estimator`. Six estimator pairs are available and they
  disagree at a finite sample size, so the numbers are ambiguous without it.
- HDMR and PCE record `streamed`, and PCE also records the `order` it fitted,
  which can be lower than the one you asked for.
- eFAST records `omega_0` and `M`, Morris records its `space`, optimal
  transport records its `mode`, Shapley records `backend`, `order` and
  `include_correlative`, and Kucherenko and VKOGA record `is_correlated`.
  VKOGA adds the fitted `n_centers`, `gamma`, `ridge` and `cv_rmse`.

Every value is a plain string, number or boolean, so `ds.to_netcdf(...)` writes
them without further work. A setting that does not apply to a run leaves its
key out, because netCDF has no null attribute. VKOGA drops `cv_rmse` when you
fixed both hyperparameters, because no cross-validation ran.

## HDMR dataset

RS-HDMR fits the output as a sum of terms, one per parameter and one per
interacting group of parameters. That is why its dataset is indexed
differently from the Sobol one. `HDMRResult.to_dataset()` uses a `term`
dimension for `Sa`, `Sb`, `S`, and `select`, whose labels join parameter names
with a slash. `ST` stays
indexed by `param`, because a total-order index belongs to a single parameter.

```python
print(ds_hdmr.ST.sel(param="amplitude"))
print(ds_hdmr.Sa.sel(term="amplitude/frequency"))
print(ds_hdmr.rmse.sel(output="displacement"))
```

Read these as follows.

- `ST.sel(param="amplitude")` is amplitude's total contribution, summed over
  every term it appears in.
- `Sa.sel(term="amplitude/frequency")` is the structural contribution of the
  joint amplitude-frequency term: the share of output variance carried by that
  one term of the fitted surrogate. The matching `Sb` variable holds the
  correlative part, which goes to zero when the inputs are independent.
- `rmse.sel(output="displacement")` is the surrogate fit error for
  displacement. Check it first. If the surrogate does not reproduce the output,
  the indices derived from it describe the surrogate and not your model.

## Practical caveats

- If `problem.output_names` is omitted, outputs are labeled `y0`, `y1`, and so
  on.
- Without `time_coords`, `to_dataset()` uses integer time indices.
- Sobol `S2` becomes dataset variables with `param_i` and `param_j`.
- `select` and `rmse` only appear on the HDMR dataset when the result contains
  those fields.

## See also

- [Multi-Output & Time-Series](/examples/multi-output) for the shape rules that
  feed into `to_dataset()`.
- [RS-HDMR Example](/examples/hdmr) for the surrogate workflow before export.
- [Advanced Workflow](/examples/advanced-workflow) for one page that uses both
  Sobol and HDMR datasets on the same custom model.
