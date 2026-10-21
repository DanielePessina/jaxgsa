# Sampling

Four namespaces build a design: `jaxgsa.sobol.sample` for Saltelli designs,
`jaxgsa.morris.sample` for elementary-effects trajectories,
`jaxgsa.efast.sample` for Fourier search curves, and
`jaxgsa.kucherenko.sample` for the conditional-copula design that handles
dependent inputs. `jaxgsa.sampling` holds the plain draws and the correlation
helpers that the given-data methods use.

The method pages give the scientific sources for their designs:
[Saltelli designs](/api/sobol#references),
[Morris trajectories and radial designs](/api/morris#references),
[eFAST search curves](/api/efast#references), and
[Kucherenko conditional designs](/api/kucherenko#references).

## Seeds

Every function that draws anything takes the same keyword:

```python
seed: int | np.random.Generator | None = None
```

`None` (the default) pulls fresh OS entropy, so two calls give two different
designs. Pass an int for a reproducible one, or pass an existing
`np.random.Generator` to draw from a stream you already own. It is
keyword-only everywhere.

The four design samplers also take `verbose: bool = True`, which prints the
one-line design summary. `jaxgsa.sampling.monte_carlo` does not, and that is
deliberate: it returns a bare `np.ndarray`, runs nothing on the device, and
deduplicates nothing, so there is no compile step and no row count worth
reporting.

The seed feeds the scrambling, not the sequence. `sobol.sample`,
`morris.sample` and `kucherenko.sample` take `scramble: bool = True`; the
underlying Sobol' sequence is deterministic, and the seed only chooses the
linear matrix scramble and digital shift applied to it, as implemented by
[SciPy's Sobol' engine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)
following [Matoušek (1998)](#ref-matousek-1998). `kucherenko.sample` makes that explicit and
raises rather than accepting a seed that would do nothing:

```python
jaxgsa.kucherenko.sample(problem, base_n=4096, scramble=False, seed=0)
```

```
ValueError: jaxgsa.kucherenko.sample: seed has no effect with scramble=False.
The unscrambled Sobol' sequence is deterministic, so the seed would do
nothing. Use scramble=True, or drop the seed.
```

Use `scramble=False` only to reproduce an unscrambled reference design.
Scrambling is what removes the Sobol' sequence's structural duplicates, and
without it a Saltelli design loses real blocks to deduplication.

The underlying SciPy engine uses the direction numbers of
[Joe & Kuo (2008)](#ref-joe-kuo-2008).

`jaxgsa.efast.sample` has no `scramble`: its randomness is the phase shift of
each search curve, which the seed sets directly.

## jaxgsa.sampling

```python
monte_carlo(problem, n_samples, *, seed=None) -> np.ndarray            # (n, D)
correlate(problem, X, *, seed=None) -> np.ndarray              # (N, D)
transform(problem, U) -> np.ndarray                           # (n, D)
fit_correlation(problem, X) -> np.ndarray                      # (D, D)
correlation_from_covariance(cov) -> np.ndarray                 # (D, D)
InputSampler                                                   # protocol type
```

`monte_carlo` generates an ordinary joint input matrix, rather than a
method-specific analysis design. It follows the problem's dependence. A
problem with a custom sampler (`problem.with_sampler(f, sampler_dimension=k)`) gets
pseudo-random uniform points of width `k`, passed through the joint branch of
the sampler (`fixed_indices=()`). See
[the callback contract](/api/kucherenko#custom-input-dependence).

```python
X = jaxgsa.sampling.monte_carlo(problem, 1024, seed=7)
Y = model(X)
result = jaxgsa.pawn.analyze(problem, X, Y)
```

The same `X, Y` workflow applies to given-data methods such as HSIC,
Borgonovo, optimal transport and HDMR. Their interpretations under dependence
remain method-specific. In particular, PCE orthogonal variance decompositions
retain their independence assumptions; custom samples do not remove them.
`monte_carlo` rows are independent draws, so row-bootstrap intervals and
permutation tests such as HSIC's stay valid. A plain input matrix does not
replace Kucherenko's conditional design or make a Saltelli design valid for
dependent inputs.

`transform` is the last step of every sampler, exposed for your own design.
It maps an `(n, D)` array of unit-cube points (or `(n, sampler_dimension)` for
a custom sampler) through the problem's custom sampler, Gaussian copula, or
marginals, and returns physical rows. `monte_carlo` is `transform` applied to
pseudo-random points.

```python
from scipy.stats import qmc

U = qmc.LatinHypercube(d=problem.n_params, seed=0).random(256)
X = jaxgsa.sampling.transform(problem, U)
```

`seed` takes an `int` or a NumPy `Generator`. A JAX key raises `TypeError`:
JAX keys go to `analyze(key=...)`, which also accepts a plain `int`.

`jaxgsa.sampling.InputSampler` is the protocol type of the sampler callback.
Use it to annotate your own function.

`monte_carlo` draws plain pseudo-random rows and pushes each column through
its declared inverse CDF, so a categorical column comes back as level codes.
There is no low-discrepancy structure and no Saltelli layout. Given-data
methods consume the resulting ordinary input rows and their outputs. When `problem.correlation` is set it
draws correlated latent normals first (a normal-to-anything construction;
see [Gaussian-copula sampling in Hilhorst et al. (2024)](/api/vkoga#references)), so every
column keeps its declared marginal and the joint sample carries the declared
dependence. An independent problem keeps the plain uniform path bit-for-bit,
so old seeds still reproduce old samples.

```python
p = jaxgsa.Problem(("a", "b"), ((0.0, 1.0), (0.0, 1.0))).with_correlation(
    np.array([[1.0, 0.8], [0.8, 1.0]])
)
jaxgsa.sampling.monte_carlo(p, 5, seed=0)
```

```
array([[0.5500, 0.5102],
       [0.7391, 0.7197],
       [0.2961, 0.4108],
       [0.9039, 0.9465],
       [0.2408, 0.0954]])
```

`correlate` goes the other way. It takes a sample you already have and
reorders each column so the ranks follow the declared correlation, using the
[Iman–Conover method (1982)](#ref-iman-conover-1982). Every output column is a permutation of the matching
input column, so the marginal values survive untouched, including whatever
structure a low-discrepancy design put into them. Only the pairing changes.
It raises if `problem` has no correlation, or if `X` holds a non-finite
value: `np.sort` puts `NaN` last, so a bad row would be pinned to the extreme
rank scores and bias the correlation of the good rows.

`fit_correlation` estimates the latent matrix from data. It computes the
Spearman rank correlation and converts it with `2 sin(pi rho_s / 6)`. Ranks
make the estimate invariant to the marginals, so a heavily skewed parameter
does not distort the dependence structure. Feed the result to
`problem.with_correlation(...)`.

`correlation_from_covariance` rescales a published covariance matrix to unit
diagonal. Use it when a paper reports `Cov` and jaxgsa wants a correlation.

## Correlated designs

One design builder reads the problem's dependence rather than refusing it.
`jaxgsa.kucherenko.sample(problem, base_n=..., seed=...)` places its blocks with
the conditionals of the declared copula or custom sampler, so the indices it feeds are valid under
dependence. `sobol.sample`, `morris.sample` and `efast.sample` all raise a
`ValueError` on a correlated or custom-sampler problem instead of quietly producing indices that
assume independence. See the [Kucherenko page](/api/kucherenko). To analyze
correlated data you already have, see [VKOGA](/api/vkoga).

## SobolSamples

`jaxgsa.sobol.sample(problem, n_samples, *, base_n=None, calc_second_order=True,
scramble=True, seed=None, verbose=True)` returns a `SobolSamples`.

Fields:

- `samples` — `(n_runs, D)`, the rows to evaluate, in physical units.
- `n_runs` — unique rows to evaluate, one model run per row.
- `n_expanded` — the design size before deduplication.
- `expanded_to_unique` — the index map from the expanded design back to
  `samples`, which is how `analyze` rebuilds the Saltelli blocks.
- `base_n`, `n_params`, `calc_second_order`, `problem` — the design metadata.
- `unit`, `expanded_to_unit` — the same design in the unit cube, before the
  input distributions are applied.

`n_runs` and `n_expanded` differ because scrambled Sobol' points repeat across
the A, B and AB blocks; jaxgsa evaluates each distinct row once. Watch them in
the `sample()` summary line, where the gap is reported as a percentage.

Methods:

- `save(path)` / `load(path)` — one compressed NPZ file, sample matrix plus a
  JSON metadata blob. `MorrisSamples`, `EFASTSamples` and `KucherenkoSamples`
  use the same format.
- `downsample(base_n, Y=None)` — a prefix-nested smaller design. Pass the
  evaluated `Y` and it returns the matching subset of outputs too, so you can
  plot convergence against sample size from one evaluated design.
- `to_morris(*, verbose=True)` — reinterpret the evaluated design as a radial
  Morris design. Free screening measures, no extra model runs.
- `transform(theta=None)` — the design in physical units, for the distribution
  parameters you pass.

`jaxgsa.morris.MorrisSamples` shares the `n_runs` / `n_expanded` vocabulary and
the same `downsample`, `save` and `load`.

### Reusing one design under different input ranges

`unit` holds the quasi-random points before any distribution is applied, so
they do not depend on the input distributions at all. `transform` applies a
set of distribution parameters to them. Pass `None` to get the design the
problem itself describes.

That lets you test your assumed input ranges without drawing a new design:

```python
design = jaxgsa.sobol.sample(problem, n_samples=8192, seed=0)

narrow = design.transform({"x1": {"low": 0.0, "high": 1.0}, ...})
wide   = design.transform({"x1": {"low": -1.0, "high": 2.0}, ...})
```

`theta` has the type `jaxgsa.sobol.Theta`: a mapping keyed by parameter name, then
by the field names of that parameter's distribution. `low` and `high` for a
uniform, `mean` and `variance` for a Gaussian, and all four for a truncated
Gaussian.

`transform` is written in JAX, so it is differentiable with respect to
`theta`. See [Sobol](/api/sobol) for how to get the derivative of
an index.

`transform` raises for a problem with categorical parameters. A categorical
inverse CDF is a step function, so `unit` and `samples` do not have the same
number of rows, and a derivative through it has no meaning.

See [Save and Reload Samples](/examples/save-load) and the
[API reference](/api/).

## References

- <a id="ref-matousek-1998"></a>Matoušek, J. (1998).
  [On the L2-discrepancy for anchored boxes](https://doi.org/10.1006/jcom.1998.0489).
  *Journal of Complexity*, 14(4), 527–556. Linear scrambling used by SciPy's
  LMS-plus-shift Sobol' engine.
- <a id="ref-joe-kuo-2008"></a>Joe, S., & Kuo, F. Y. (2008).
  [Constructing Sobol' sequences with better two-dimensional projections](https://doi.org/10.1137/070709359).
  *SIAM Journal on Scientific Computing*, 30(5), 2635–2654. Direction numbers
  used by the underlying Sobol' sequence generator.
- <a id="ref-iman-conover-1982"></a>Iman, R. L., & Conover, W. J. (1982).
  [A distribution-free approach to inducing rank correlation among input variables](https://doi.org/10.1080/03610918208812265).
  *Communications in Statistics — Simulation and Computation*, 11(3), 311–334.
  Rank reordering that preserves the observed marginal values in `correlate`.
