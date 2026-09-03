# xarray Labeled Output

A time-resolved Sobol result on a 3-parameter, 2-output, 30-timepoint model is
180 first-order numbers, and you reach them with `S1[17, 0, 1]`. Get one index
wrong and nothing complains. `to_dataset()` turns the same arrays into an
`xarray.Dataset`, where that number is
`ds.S1.isel(time=17).sel(output="displacement", param="frequency")` and a typo
raises instead of returning the wrong parameter.

Every result class in `jaxgsa` has `to_dataset()`. This page runs Sobol and
RS-HDMR on the same model, exports both, and reads the results against each
other. The comparison does not end the way you might expect.

## Setup

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

time_values = np.linspace(0.1, 5.0, 30)


def model(X):
    amp = X[:, 0, None]
    freq = X[:, 1, None]
    damping = X[:, 2, None]
    tt = jnp.asarray(time_values)[None, :]

    displacement = amp * jnp.sin(2 * jnp.pi * freq * tt) * jnp.exp(-damping * tt)
    velocity = amp * jnp.cos(2 * jnp.pi * freq * tt) * jnp.exp(-damping * tt)
    return jnp.stack([displacement, velocity], axis=-1)  # (N, T, K)


design = jaxgsa.sobol.sample(problem, n_samples=2048, seed=42, verbose=False)
sobol = jaxgsa.sobol.analyze(
    design,
    model(jnp.asarray(design.samples)),
    n_bootstrap=100,
    key=jax.random.key(0),
    verbose=False,
)

bounds = jnp.array(problem.bounds)
X_hdmr = jax.random.uniform(
    jax.random.key(1), (1500, problem.num_vars), minval=bounds[:, 0], maxval=bounds[:, 1]
)
Y_hdmr = model(X_hdmr)
hdmr = jaxgsa.hdmr.analyze(problem, X_hdmr, Y_hdmr, maxorder=2, verbose=False)

ds_sobol = sobol.to_dataset(time_coords=time_values)
ds_hdmr = hdmr.to_dataset(time_coords=time_values)
```

Three choices in there are worth stating.

The time grid starts at 0.1, not 0. At `t = 0` the sine is exactly zero for
every sample, so displacement has zero variance, and a variance share of zero
variance is undefined. `jaxgsa` warns and fills those indices with NaN, and the
NaN then poisons any mean you take over the time axis. Starting at 0.1 avoids
the whole thing.

`n_bootstrap=100` with a `key` is what produces the confidence intervals used
two sections down. Without both, `to_dataset()` has no `S1_lower` to write.

HDMR gets its own 1500 uniform random points. It fits a surrogate to any
`(X, Y)` pair, so it does not need the structured Saltelli design, and giving it
one would be a waste of the structure.

## The Sobol dataset

```python
print(ds_sobol)
```

```text
<xarray.Dataset> Size: 11kB
Dimensions:   (time: 30, output: 2, param: 3, param_i: 3, param_j: 3)
Coordinates:
  * time      (time) float64 240B 0.1 0.269 0.4379 0.6069 ... 4.662 4.831 5.0
  * output    (output) <U12 96B 'displacement' 'velocity'
  * param     (param) <U9 108B 'amplitude' 'frequency' 'damping'
  * param_i   (param_i) <U9 108B 'amplitude' 'frequency' 'damping'
  * param_j   (param_j) <U9 108B 'amplitude' 'frequency' 'damping'
Data variables:
    S1        (time, output, param) float32 720B 0.4297 0.5122 ... -0.04315
    ST        (time, output, param) float32 720B 0.4873 0.5834 ... 1.108 0.3602
    S2        (time, output, param_i, param_j) float32 2kB nan 0.0475 ... nan
    S1_lower  (time, output, param) float32 720B 0.3208 0.4023 ... -0.1374
    S1_upper  (time, output, param) float32 720B 0.5293 0.6488 ... 0.07314
    ST_lower  (time, output, param) float32 720B 0.4008 0.4996 ... 0.813 0.2895
    ST_upper  (time, output, param) float32 720B 0.5596 0.7332 ... 1.404 0.4549
    S2_lower  (time, output, param_i, param_j) float32 2kB nan -0.0889 ... nan
    S2_upper  (time, output, param_i, param_j) float32 2kB nan 0.226 ... nan
Attributes:
    estimator:  saltelli-jansen
```

`param_i` and `param_j` are both the parameter list again. `S2` covers pairs, so
it needs the parameter axis twice, and its diagonal is NaN because a parameter
paired with itself is not a second-order term.

Every confidence interval arrives split into `*_lower` and `*_upper` rather than
as a `(2, ...)` array with a bound axis. The split is deliberate. You select a
bound by name, and the halves keep the same dimensions as the estimate they
bound, so they line up element by element in arithmetic.

The `time` coordinate is float64 while the data is float32. Ask for
`sel(time=2.0)` on that grid and you get a `KeyError`, because 2.0 is not one of
the 30 values. Use `method="nearest"`.

## Selecting

```python
print(ds_sobol.S1.sel(param="frequency", output="displacement").isel(time=slice(0, 5)).values)
```

```text
[0.5121903  0.8914394  0.89829767 0.84917337 0.90702844]
```

Frequency owns roughly 90% of displacement variance from the second timepoint
onward, and only 51% at `t = 0.1`. Early on the oscillator has barely moved, so
amplitude still shows through. After that the phase spread across
`frequency` in `[1, 5]` swamps everything.

Chaining a nearest-match lookup needs care. `method="nearest"` applies to every
dimension in the same `.sel()` call, and it cannot compute a distance between
two strings, so mixing a float coordinate and a string coordinate in one call
raises a `TypeError`. Split it:

```python
s2 = (
    ds_sobol.S2.sel(output="displacement", param_i="amplitude", param_j="frequency")
    .sel(time=time_values[10], method="nearest")
)
print(float(s2), "at t =", float(s2.time))
```

```text
0.08388328552246094 at t = 1.7896551724137932
```

Exact string labels first, nearest-match float second.

## Confidence intervals

The interval width is the number to look at, and taking it across all three
parameters at once is one line:

```python
print((ds_sobol.ST_upper - ds_sobol.ST_lower).mean("time").to_pandas())
```

```text
param         amplitude  frequency   damping
output                                      
displacement   0.055160   0.397972  0.056613
velocity       0.048141   0.407634  0.053547
```

Frequency's total-order index carries an interval about 0.40 wide, seven times
wider than the other two. It is also the parameter that matters most. That
combination is normal, and it is the useful thing a bootstrap tells you. The
dominant parameter is dominant, and at `base_n=256` you cannot say by how much
to better than plus or minus 0.2. If a decision turns on frequency's exact
share, raise `n_samples`. If it turns only on which parameter leads, you already
have the answer.

Amplitude and damping have intervals of about 0.05 around indices of about 0.13,
so they are separated from zero and from each other. That is a real conclusion
from a small sample.

## Provenance attributes

`ds.attrs` carries the settings the analysis ran with, under the same names the
result prints:

```python
print(ds_sobol.attrs)
print(ds_hdmr.attrs)
```

```text
{'estimator': 'saltelli-jansen'}
{'streamed': False}
```

Sobol records its `estimator`, because six estimator pairs are available and
they disagree at finite sample size. A dataset without it is ambiguous.

HDMR and PCE record `streamed`, and PCE adds the `order` it actually fitted,
which can be below the one you asked for. eFAST records `omega_0` and `M`.
Morris records its `space`. Optimal transport records its `mode`. Shapley
records `backend`, `order` and `include_correlative`. Kucherenko and VKOGA
record `is_correlated`, and VKOGA adds the fitted `n_centers`, `gamma`, `ridge`
and `cv_rmse`.

Every value is a plain string, number or boolean, so `ds.to_netcdf(...)` writes
them with no further work. A setting that did not apply leaves its key out
rather than writing a null, because netCDF has no null attribute. VKOGA drops
`cv_rmse` when you fixed both hyperparameters, since no cross-validation ran.

## The HDMR dataset, and why you check `rmse` first

RS-HDMR fits the output as a sum of terms, one per parameter and one per
interacting group. Its dataset is indexed by `term` rather than by `param` for
`Sa`, `Sb`, `S` and `select`, with labels joining parameter names with a slash.
`ST` stays on `param`, because a total-order index belongs to one parameter.

```python
print(list(ds_hdmr.coords["term"].values))
print(ds_hdmr.rmse.mean("time").to_pandas())
print(np.asarray(jnp.std(Y_hdmr, axis=0).mean(axis=0)))
```

```text
[np.str_('amplitude'), np.str_('frequency'), np.str_('damping'), np.str_('amplitude/frequency'), np.str_('amplitude/damping'), np.str_('frequency/damping')]
output
displacement    0.494926
velocity        0.505360
Name: rmse, dtype: float32
[0.5756984  0.58360267]
```

The surrogate's RMSE is 0.495 against an output standard deviation of 0.576. It
explains about a quarter of the variance and leaves the rest as residual. This
surrogate is not usable, and here is what that does to the indices:

```python
print(ds_sobol.ST.sel(output="displacement").mean("time").to_pandas())
print(ds_hdmr.ST.sel(output="displacement").mean("time").to_pandas())
```

```text
param
amplitude    0.127266
frequency    0.969359
damping      0.126892
Name: ST, dtype: float32
param
amplitude    0.047862
frequency    0.128679
damping      0.025192
Name: ST, dtype: float32
```

HDMR puts frequency at 0.13 where Sobol puts it at 0.97, and its three indices
sum to 0.20 rather than to something near 1. The ranking survives, but nothing
else does.

Neither number is a bug. A B-spline HDMR expansion at `maxorder=2` cannot
represent $\sin(2\pi f t)$ across $f \in [1, 5]$ at $t = 5$, where the output
turns over 25 times inside the input range. The surrogate fits a nearly flat
function, honestly reports that it did, and returns the sensitivity of the flat
function it fitted. That is the failure mode: HDMR indices describe the
surrogate, and they are only about your model to the extent that the surrogate
is. Read `rmse` before you read anything else on an HDMR dataset.

For a fair comparison against Sobol on this model you would need a far higher
`maxorder`, more `m` basis functions per term, and many more than 1500 points.
On a smooth model HDMR is much cheaper than Sobol for the same accuracy. On an
oscillator sampled this coarsely it is not competitive.

```python
print(ds_hdmr.Sa.sel(term="amplitude/frequency", output="displacement").mean("time").values)
print(ds_hdmr.select.to_pandas())
```

```text
0.019177476
term
amplitude               4.0
frequency              38.0
damping                 5.0
amplitude/frequency     9.0
amplitude/damping       4.0
frequency/damping       1.0
Name: select, dtype: float32
```

`Sa` is the structural share carried by one term of the fitted surrogate, and
the matching `Sb` is the correlative part, which goes to zero when the inputs
are independent. `select` counts how many of the 60 output slices kept each
term after the backfitting selection. Frequency was kept in 38 and
`frequency/damping` in only 1, which is another way of seeing that the fit
struggled.

## Practical caveats

- Without `output_names`, outputs are labeled `y0`, `y1`, and so on. Set them on
  the `Problem`, not on the dataset afterwards, so the analyzer can also use
  them to catch a mislaid output axis. See
  [Multi-Output & Time-Series](/examples/multi-output#the-full-shape-table).
- Without `time_coords`, the `time` dimension gets integer indices.
- A field is exported only when the result carries it. HDMR drops `S2` when the
  fit kept no second-order terms, and `select` and `rmse` appear only when the
  result has them.
- Other result types export their own fields. Optimal transport gives `ot`,
  `advective`, `diffusive`, `S1` and `above_dummy`, plus a per-parameter
  `ot_dummy` when you passed `dummy=True`, and the `*_lower` / `*_upper` pairs when you
  bootstrapped. PAWN gives `pawn` and `n_valid_bins`, the per-parameter count of
  bins that held at least 2 samples.
- `ds.to_netcdf(path)` writes the whole thing, attributes included. NaN slices
  survive the round trip.

## See also

- [Multi-Output & Time-Series](/examples/multi-output) for the shape rules that
  decide which dimensions a dataset gets.
- [Bootstrap Confidence Intervals](/examples/bootstrap) for the resampling
  behind `S1_lower` and `S1_upper`.
- [RS-HDMR Example](/examples/hdmr) for making the surrogate fit before you
  trust its indices.
- [Screen first, then quantify](/examples/advanced-workflow) for cutting 20 inputs to 4 with Morris,
  then spending the Sobol budget on the survivors.
