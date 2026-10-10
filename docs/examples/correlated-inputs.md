# Correlated Inputs

Use this page when your input parameters are not independent. By the end of it
you will be able to declare the dependence once on the `Problem`, draw samples
that respect it, and pick a method whose indices still mean something when
inputs move together.

Read the next section before you read any index. Correlation does not make
sensitivity indices noisier. It changes what they mean.

## The question splits in two

Take a model that reads one input and ignores the other:

```python
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {
        "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
    },
    correlation=[[1.0, 0.8], [0.8, 1.0]],
)

X = jaxgsa.sampling.monte_carlo(problem, n=8192, seed=42)
Y = X[:, 0]  # the model reads x1 and ignores x2

ot = jaxgsa.optimal_transport.analyze(problem, X, Y)
```

```
jaxgsa.optimal_transport.analyze
  problem: D=2 (x1, x2)
    marginals: gaussian=2
    correlation: correlated (Gaussian copula)
    output: N=8192 runs, T=1 x K=1 output slice
    invalid: none found in 8192 rows (policy 'raise')
  timing:
    estimator (includes compile on the first call): 0.2385 s
    mode: univariate
    epsilon: 0.03
    slice_chunk_size: auto (resolved from the memory budget)
  results: top 2 of 2 parameters by ot
    1. x1  ot=0.9348
    2. x2  ot=0.389
```

`x2` scores 0.389 on an input the model never reads. Borgonovo delta and PAWN
agree, on the same `(X, Y)`:

```python
delta = jaxgsa.borgonovo.analyze(problem, X, Y, verbose=False)
pawn = jaxgsa.pawn.analyze(problem, X, Y, verbose=False)

print("ot   :", np.round(ot.ot, 3))
print("delta:", np.round(delta.delta, 3))
print("pawn :", np.round(pawn.pawn, 3))
```

```
ot   : [0.935 0.389]
delta: [0.928 0.363]
pawn : [0.703 0.316]
```

Nothing is wrong. `x2` carries real information about `Y`, because it is
correlated with `x1` at 0.8. If I show you `x2` I have told you a lot about
`Y`. These three methods answer "how much does watching this input tell me
about the output". Under dependence that is a genuine question with a non-zero
answer for `x2`.

Now put the same model through Kucherenko, which runs it on a
conditional-copula design:

```python
ks = jaxgsa.kucherenko.sample(problem, 8192, seed=0)
result = jaxgsa.kucherenko.analyze(ks, ks.samples[:, 0])

print("S1:", np.round(result.S1, 3))
print("ST:", np.round(result.ST, 3))
```

```
jaxgsa.kucherenko.sample: D=2, base_n=8192, n_blocks=5, n_runs=40960, dependence=copula-conditional, scramble=True
jaxgsa.kucherenko.analyze
  problem: D=2 (x1, x2)
    marginals: gaussian=2
    correlation: correlated (Gaussian copula)
    output: N=40960 runs, T=1 x K=1 output slice
    invalid: none found in 8192 base points (policy 'raise')
  timing:
    compute: 0.01894 s
    design: copula-conditional (2D+1 = 5 blocks of 8192 base points)
  results: top 2 of 2 parameters by ST
    1. x1  ST=0.36
    2. x2  ST=0
S1: [1.   0.64]
ST: [0.36 0.  ]
```

Two indices, two answers, both exact to three decimals. `S1` for `x2` is 0.64,
which is $0.8^2$. `ST` for `x2` is 0, because once you know `x1` the value of
`x2` changes nothing. `ST` for `x1` is 0.36, which is $1 - 0.8^2$: the part of
`x1` that `x2` cannot stand in for.

So under dependence "the effect of `x1` alone" is two different numbers.

- Correlation-inclusive (`S1`, `ot`, `delta`, `pawn`, HSIC): what an input
  explains through itself and through everything it moves with. Use it to
  decide which input to **measure** more accurately.
- Correlation-exclusive (`ST`, VKOGA's `S_TU`): what only that input can
  explain. Use it to decide which input you can **fix** at a nominal value.

Under independence the two coincide and the distinction disappears, which is
why the habit of reading one number per input survives so long before it
breaks.

The rest of this page is the mechanics.

## How jaxgsa models dependence

The built-in path uses a Gaussian copula. Custom joint and conditional samplers
for Kucherenko and VKOGA can represent other dependence structures. A copula separates two questions: what each
parameter's own distribution looks like, and how the parameters move together.
Every parameter keeps its declared marginal (uniform, Gaussian, or truncated
Gaussian) exactly as written. A `(D, D)` correlation matrix couples the
columns. That matrix lives on the copula's latent standard-normal scale, which
is the hidden set of normal variables the copula builds the samples from. The
matrix lives on the `Problem`, so samplers and analyzers see one consistent
declaration.

```python
R = np.array(
    [
        [1.0, 0.8],
        [0.8, 1.0],
    ]
)

problem = jaxgsa.Problem.from_dict(
    {
        "x1": (0.0, 1.0),
        "x2": {"dist": "gaussian", "mean": 1.0, "variance": 4.0},
    },
    correlation=R,
)

print(problem.dependence)
print(problem.correlation)
```

```
gaussian
[[1.  0.8]
 [0.8 1. ]]
```

By default the matrix is read on the latent scale. That is the Pearson
correlation of the copula's underlying normals. A rank (Spearman) correlation
is the natural target for non-Gaussian marginals, and a Gaussian copula
inverts it exactly. Declare a rank correlation with
`correlation_type="spearman"`:

```python
from scipy import stats

p_s = jaxgsa.Problem.from_dict(
    {"x1": (0.0, 1.0), "x2": (0.0, 1.0)},
    correlation=[[1.0, 0.7], [0.7, 1.0]],
    correlation_type="spearman",  # converted via 2 sin(pi * rho_s / 6)
)

print("latent:", round(float(p_s.correlation[0, 1]), 4))
Xs = jaxgsa.sampling.monte_carlo(p_s, n=100_000, seed=7)
print("achieved spearman:", round(float(stats.spearmanr(Xs).statistic), 4))
```

```
latent: 0.7167
achieved spearman: 0.6977
```

You asked for 0.7 in rank terms, jaxgsa stored 0.7167 on the latent scale, and
100,000 draws come back at 0.698. The gap to 0.7 is Monte-Carlo scatter, not
bias. This is what "inverts it exactly" buys you: the number you declared is
the number your samples have.

`correlation_type=` is the one spelling, and it is accepted by `Problem(...)`,
`Problem.from_dict(...)` and `Problem.with_correlation(...)`.

### Validation and repair

The matrix is checked on entry. A wrong shape, an asymmetry, a non-unit
diagonal, or entries outside `[-1, 1]` raise `ValueError`. A matrix that is not
positive definite is repaired by eigenvalue clipping. Such a matrix usually
means the pairwise correlations you declared cannot all hold at once. How
loudly the repair reports itself depends on how far it has to move the matrix.
The measure is the largest change to a single entry, on the scale you declared:

| Largest entry change | What happens |
|---|---|
| below `1e-6` | nothing. This is floating-point noise. |
| `1e-6` to `0.05` | a `JaxgsaWarning` reports the change and the minimum eigenvalue. |
| `0.05` or more | a `ValueError`. The matrix is structurally inconsistent. |

So you never silently sample a different dependence structure than you
declared. If the third case applies, correct the matrix, or fit a valid one
from data with `jaxgsa.sampling.fit_correlation`. Check also that you did not
pass a rank correlation without `correlation_type="spearman"`.

`fit_correlation` itself never raises for this reason. Inconsistent data is
not a user error. A fit that had to move an entry by `0.05` or more only
warns.

Rescale a published covariance matrix before you pass it. The variances on
its diagonal are discarded in favor of the declared marginals:

```python
R = jaxgsa.sampling.correlation_from_covariance(cov)
```

## Which matrix do I pass?

Match your starting point to the right entry path:

| You have | Do this |
|---|---|
| A rank (Spearman) correlation you want the samples to have | Pass that matrix with `correlation_type="spearman"`. The conversion to the latent scale is exact. |
| A published covariance for Gaussian variables | `correlation_from_covariance(cov)`, default `correlation_type`. Latent equals Pearson here, so this is exact. |
| Observed data | `fit_correlation(problem, X_observed)`, then `with_correlation`. The fit uses ranks and converts internally. |
| A rough Pearson target with non-Gaussian marginals | Prefer `correlation_type="spearman"` with the same number. Only the Spearman route carries an exact guarantee. The two scales differ little (Spearman 0.7 maps to latent 0.7167). |

The short version: the latent matrix is what the copula machinery uses, and
the Spearman matrix is what your samples measurably have. When in doubt,
declare Spearman.

## Draw correlated samples

`jaxgsa.sampling.monte_carlo` honors the declared correlation transparently
(NORTA: correlated latent normals pushed through each marginal's inverse
CDF). Problems without a correlation keep the previous pseudo-random path
bit-for-bit.

```python
X = jaxgsa.sampling.monte_carlo(problem, n=4096, seed=42)
```

Each column follows its declared marginal exactly, and the ranks across
columns follow the declared correlation.

## Fit the correlation from observed data

When the dependence structure lives in a historical data set rather than a
paper, estimate it from the data and attach it to the (frozen) problem:

```python
R_fit = jaxgsa.sampling.fit_correlation(problem, X_observed)
problem = problem.with_correlation(R_fit)
```

The fit works on Spearman ranks, so it is invariant to the declared
marginals. Heavy ties (discrete or heavily rounded columns) bias the rank
estimate toward zero. A polychoric estimator is future work.

Fitting is explicit: supplied `X, Y` do not automatically trigger a Gaussian
copula fit. A fitted pairwise rank matrix cannot describe every dependence
structure.

## Retrofit correlation onto an existing sample

`correlate` imposes the declared correlation on a sample you already have by
Iman-Conover rank re-pairing. Each column of the result is an exact
permutation of the input column, so the marginal values are preserved,
including any structure a low-discrepancy design put into them:

```python
p = jaxgsa.Problem.from_dict(
    {"x1": (0.0, 1.0), "x2": (0.0, 1.0)},
    correlation=[[1.0, 0.8], [0.8, 1.0]],
)
X_independent = jaxgsa.sampling.monte_carlo(p.with_correlation(None), n=4096, seed=1)
X_correlated = jaxgsa.sampling.correlate(X_independent, p, seed=2)

print("spearman before:", round(float(stats.spearmanr(X_independent).statistic), 4))
print("spearman after :", round(float(stats.spearmanr(X_correlated).statistic), 4))
print("column values unchanged:", np.array_equal(
    np.sort(X_independent[:, 0]), np.sort(X_correlated[:, 0])
))
```

```
spearman before: 0.0059
spearman after : 0.7869
column values unchanged: True
```

The values in each column are the same 4096 numbers. Only the pairing changed.

The method builds a score matrix from van der Waerden scores. It then removes
the score matrix's own sampling noise before it reads off the ranks. A finite
design therefore lands much closer to the declared correlation. At
N = 50 and a target of 0.8, the achieved rank correlation scatters with a
standard deviation of about 0.024, against about 0.065 for a plain correlated
normal draw.

## Picking a method

### Correlation-inclusive, from data you already have

Optimal transport, Borgonovo delta, HSIC, and PAWN take any `(X, Y)` pairs.
They are rank- and distribution-based, so they need no independence assumption
at all. Go back to the `problem`, `X` and `Y` from the top of this page (the
sections above reused those names for other examples):

```python
problem = jaxgsa.Problem.from_dict(
    {
        "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
    },
    correlation=[[1.0, 0.8], [0.8, 1.0]],
)
X = jaxgsa.sampling.monte_carlo(problem, n=8192, seed=42)
Y = X[:, 0]  # the model reads x1 and ignores x2

ot = jaxgsa.optimal_transport.analyze(problem, X, Y)
delta = jaxgsa.borgonovo.analyze(problem, X, Y)
pawn = jaxgsa.pawn.analyze(problem, X, Y)
hsic = jaxgsa.hsic.analyze(problem, X, Y, key=jax.random.key(0))
```

All four report the correlation-inclusive reading demonstrated at the top of
this page. HSIC is the expensive one here: its permutation test holds `2D+1`
resident `N x N` kernel matrices, and the run above at N=8192 took 84 seconds
against 0.2 for optimal transport. Drop `N` before you reach for it.

### Correlation-exclusive variance fractions

[Kucherenko](/examples/kucherenko) and [VKOGA](/examples/vkoga) both report
`S1`/`S_TC` and `ST`/`S_TU`, the two quantities the opening section separated.
They differ in where the numbers come from. Kucherenko runs your model on a
conditional-copula design of `base_n * (2D + 1)` points and carries no
surrogate error. VKOGA fits a kernel surrogate to `(X, Y)` you already have and
needs no new model runs, at the cost of the surrogate. VKOGA also splits the
totals further into `S_U`, `S_C` and `S_IU`.

VKOGA training data must cover the analysis distribution accurately. Rows
drawn from that distribution work, and so does an independent design. See its
[training-data section](/examples/vkoga#training-data-must-cover-the-analysis-distribution).

### HDMR: the split per term, not per parameter

HDMR's ANCOVA decomposition separates a structural share `Sa` from a
correlative share `Sb`, term by term:

```python
hdmr = jaxgsa.hdmr.analyze(problem, X, Y, verbose=False)
print("terms:", hdmr.terms)
print("Sa:", np.round(hdmr.Sa, 3))
print("Sb:", np.round(hdmr.Sb, 3))
```

On the same `Y = x1` data as above:

```
terms: ('x1', 'x2', 'x1/x2')
Sa: [0.97  0.    0.005]
Sb: [0.01 0.01 0.  ]
```

Read this as a third answer, not a contradiction. HDMR fits component
functions and reports what each one carries. The `x2` component is flat, so
`Sa` for `x2` is 0, and its correlative share `Sb` is 0.010. HDMR is saying the
model has no `x2` structure in it. That is true and useful, and it is not the
same statement as `ot = 0.389` or `S1 = 0.64`.

`analyze` warns on a correlated problem that `HDMRResult.ST` is the SCSA total
from Li et al. (2010), not a Sobol total-order index. It can be negative, it is
not bounded in `[0, 1]`, and it does not measure the variance reduction from
fixing a parameter. Do not use it to decide something can be fixed. Use
`kucherenko.ST` or `vkoga.S_TU`. HDMR's value under dependence is the per-term
`Sa` / `Sb` split, which neither of those provides.

### Allocating the ANCOVA split across parameters

`Sa` and `Sb` are per term. `jaxgsa.shapley.analyze` with the HDMR backend
spreads them over the parameters instead, so each parameter gets one number
again:

```python
sh = jaxgsa.shapley.analyze(
    problem, X, Y, backend="hdmr", include_correlative=True, verbose=False
)
sh_struct = jaxgsa.shapley.analyze(
    problem, X, Y, backend="hdmr", include_correlative=False, verbose=False
)

print("with correlative:", np.round(sh.Sh, 3))
print("structural only :", np.round(sh_struct.Sh, 3))
```

```
with correlative: [0.987 0.013]
structural only : [0.997 0.003]
```

`include_correlative=True` folds each term's `Sb` into the allocation, which
moves `x2` from 0.003 to 0.013 on the `Y = x1` data. That is small here because
HDMR found almost no `x2` structure to attach a correlative share to. Both
numbers are an ANCOVA-based attribution, not conditional-variance Shapley
effects, so do not read them against the Shapley effects a PCE backend returns
on an independent problem.

The PCE backend refuses a correlated problem outright:

```python
jaxgsa.shapley.analyze(problem, X, Y)  # backend="pce" is the default
```

```
ValueError: jaxgsa.shapley.analyze with backend='pce' computes a variance
allocation that assumes independent inputs, but this problem declares dependent
inputs (problem.correlation or problem.sampler). Use backend='hdmr' with include_correlative=True, which
allocates the ANCOVA (structural + correlative) decomposition instead — an
ANCOVA-based attribution, not conditional-variance Shapley effects — or one of
the dependence-tolerant methods: ...
```

## Methods that refuse dependent problems

Some methods have indices that are only defined for independent inputs. They
raise `ValueError` rather than return silently wrong numbers. Two groups do
this: the structured design samplers (`sobol.sample`, `morris.sample`,
`efast.sample`) and the analyzers whose theory needs independence
(`pce.analyze`, `dgsm.analyze`, `shapley.analyze` with the PCE backend).

```python
jaxgsa.pce.analyze(problem, X, Y)
```

```
ValueError: jaxgsa.pce.analyze computes indices that assume independent inputs.
They are invalid, not merely approximate, when the problem declares dependent
inputs (problem.correlation or problem.sampler). Use a dependence-tolerant
given-data method instead: jaxgsa.vkoga (variance-based indices from given data, through a kernel
surrogate), jaxgsa.kucherenko (variance-based indices from its own
conditional-copula design), jaxgsa.optimal_transport, jaxgsa.borgonovo,
jaxgsa.hdmr (whose ANCOVA Sb term quantifies the correlation-induced
contribution, and whose result supports shapley(include_correlative=True)),
jaxgsa.hsic, jaxgsa.pawn, or jaxgsa.shapley with backend="hdmr". To analyze the
independent problem instead, drop the dependence with problem.with_correlation(None).
```

To run one of these on the independent version of the problem, drop the
dependence explicitly with `problem.with_correlation(None)`, or
`problem.with_sampler(None)` for a custom sampler. A custom-sampler problem is
refused in exactly the same way.

## Practical notes

- A Gaussian copula fixes a dependence family, not only a correlation matrix.
  For correlations strictly below one in magnitude it has no asymptotic tail
  dependence. Real data may carry asymmetric or tail dependence it cannot model.
- `problem.correlation` selects the built-in Gaussian copula. A custom
  `problem.sampler` (joint and conditional transformations) lets Kucherenko and
  VKOGA integrate under other copulas or other joint distributions. The package does not infer such a model from
  observed rows; supply consistent samplers for your target distribution.
- For non-Gaussian marginals, the Pearson correlation of the physical samples
  usually differs from the latent matrix. This is the NORTA
  correlation-matching problem. The Spearman rank correlation is the
  exactly invertible route, so declare it with `correlation_type="spearman"`.
  When every marginal is Gaussian, latent and Pearson coincide and the
  sample covariance reproduces the declared one.
- The correlation round-trips through problem serialization. NPZ design
  files and the JSON problem metadata both carry it.

## See also

- [Non-Uniform Inputs](/examples/non-uniform-inputs) for the marginal specs
  the copula couples.
- [Kucherenko](/examples/kucherenko) and [VKOGA](/examples/vkoga) for the two
  variance fractions under dependence.
- [RS-HDMR](/examples/hdmr) for the ANCOVA decomposition in depth.
- [Optimal Transport](/examples/optimal-transport) and
  [Borgonovo Delta](/examples/borgonovo) for the distribution-based indices.
- [Methods guide](/guide/methods) for which method to pick under correlation.

## Custom dependence without a copula

A custom sampler lets Kucherenko and VKOGA work under dependence beyond a
Gaussian copula. You attach it to the `Problem`. It must implement joint and
conditional sampling. Representative correlated rows alone do not determine
conditional redraws. PAWN, Borgonovo, optimal transport and HSIC still accept
`X, Y` directly.

Consider independent uniform variables `u` and `v`, and set
`w = (u + v) % 1`. All three marginals are uniform on `[0, 1]`, and every pair
is independent, yet knowing any two determines the third. This is higher-order
dependence, stronger evidence than merely having zero pairwise correlations.
A Gaussian copula with an identity correlation matrix would instead generate
three independent variables and lose this constraint.

The sampler receives uniform points. It makes exactly three kinds of call:

- `fixed_indices=()`: return joint draws.
- `fixed_indices=(i,)`: keep column `i` of `conditioning_rows` and redraw the
  other columns given it.
- `fixed_indices` holding every index except `i`: keep those columns and redraw
  column `i` given the rest.

It must return `(n, D)` finite rows, copy the fixed columns exactly, and be
deterministic. The package passes copies, so changing the arguments does no
harm.

```python
import jax
import jax.numpy as jnp
import numpy as np

import jaxgsa

jax.config.update("jax_enable_x64", True)


def sample_modular(uniform_points, *, conditioning_rows=None, fixed_indices=()):
    fixed = frozenset(fixed_indices)
    if not fixed:
        u, v = uniform_points[:, 0], uniform_points[:, 1]
        return np.column_stack((u, v, (u + v) % 1.0))
    if conditioning_rows is None:
        raise ValueError("conditioning_rows is required")
    rows = conditioning_rows.copy()
    if len(fixed) == 1:
        free = 1 if 0 in fixed else 0
        rows[:, free] = uniform_points[:, 0]
    if 2 not in fixed:
        rows[:, 2] = (rows[:, 0] + rows[:, 1]) % 1.0
    elif 1 not in fixed:
        rows[:, 1] = (rows[:, 2] - rows[:, 0]) % 1.0
    elif 0 not in fixed:
        rows[:, 0] = (rows[:, 2] - rows[:, 1]) % 1.0
    return rows


problem = jaxgsa.Problem(
    names=("u", "v", "w"),
    bounds=((0.0, 1.0),) * 3,
    sampler=sample_modular,
    sampler_dimension=2,
)
print(problem.dependence)  # custom

design = jaxgsa.kucherenko.sample(problem, 8192, seed=7)
Y = jnp.asarray(design.samples[:, 2])  # the model is Y = w
result = jaxgsa.kucherenko.analyze(design, Y)
print(result.S1)  # [-0.0018 -0.0044  1.], so [0, 0, 1] within noise
print(result.ST)  # [0. 0. 0.], up to floating-point noise
```

`sampler_dimension=2` says the sampler uses two uniform coordinates per row,
because `u` and `v` are the only free inputs. You can also attach a sampler
to an existing problem with `problem.with_sampler(sample_modular, dimension=2)`.
`problem.with_sampler(None)` drops it. A problem has either a `correlation` or
a `sampler`, never both.

`u` and `v` each tell us nothing about `Y` individually, while `w` tells us
everything. However, the conditional total index of `w` is zero: once `u` and
`v` are known, `w` has no remaining uncertainty. This means redundant
information under the joint distribution, not that `w` is causally inactive.
With three independent uniforms instead, `Y = w` has `ST = [0, 0, 1]`.
These per-input summaries respect the supplied higher-order dependence; they
do not identify which groups of inputs create it. The sampler must supply
the full joint law and its conditionals; fitting a pairwise matrix cannot
recover this constraint.

The package generates scrambled Sobol' points and assembles the analysis
blocks. Sample sizes round up to the next power of two. The sampler returns
physical values in `problem.names` order. Its public callable type is
`jaxgsa.sampling.InputSampler`.

Methods whose indices assume independent inputs refuse this problem, as they
refuse a correlated one. This covers the Sobol', Morris and eFAST designs, PCE,
DGSM and the PCE-based Shapley backend. Given-data methods such as PAWN, HSIC,
Borgonovo, optimal transport and HDMR accept an input matrix drawn with
`monte_carlo`, which sends pseudo-random uniforms through the joint branch of
the sampler:

```python
X = jaxgsa.sampling.monte_carlo(problem, 1024, seed=7)
Y = jnp.asarray(X[:, 2])
pawn_result = jaxgsa.pawn.analyze(problem, jnp.asarray(X), Y)
```

These calls generate joint observations. Kucherenko still needs its own block
design, and VKOGA also calls the conditional branches during index
integration.

Use the same problem to integrate a VKOGA surrogate. The training data and the
analysis distribution are separate inputs. This independent training design
covers the cube. The integration distribution obeys the modular constraint:

```python
X_train = jnp.asarray(
    jaxgsa.sampling.monte_carlo(problem.with_sampler(None), n=512, seed=1)
)
Y_train = X_train[:, 2]
vk = jaxgsa.vkoga.analyze(
    problem,
    X_train,
    Y_train,
    gamma=1.0,
    ridge=1e-8,
    max_centers=100,
    n_outer=512,
    n_inner=128,
    key=jax.random.key(7),
)
assert isinstance(vk, jaxgsa.vkoga.VKOGAResult)
print(vk.dependence)  # custom
print(vk.S_TC)  # [1.4e-04 1.1e-04 1.0e+00], subject to surrogate error
print(vk.S_TU)  # about 2e-31 for every input, subject to surrogate error
```

Note that `monte_carlo(problem, ...)` would draw `w = (u + v) % 1` for the
training rows. Here the training rows come from the problem without its
sampler, so `u`, `v` and `w` are independent and cover the cube. Training on
the modular draws themselves also works, because the analysis uses conditionals
of the joint distribution. See
[VKOGA training data](/examples/vkoga#training-data-must-cover-the-analysis-distribution).

VKOGA's declared marginals must match the sampler's marginals, because its
coordinate transform and component basis still use `Problem`. All three
uniforms match here. See [VKOGA's API](/api/vkoga#custom-input-dependence) for
estimator and decomposition limitations.

Save a Kucherenko design and its evaluated outputs to analyze it later. The
sampled rows and sampling provenance are saved, but Python callbacks are not.
Retain callback code separately to generate another design.
