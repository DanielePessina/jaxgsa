# Correlated Inputs

Use this page when your input parameters are not independent. By the end of it
you will be able to declare the dependence once on the `Problem`, draw samples
that respect it, and pick a method whose indices still mean something when
inputs move together.

jaxgsa models dependence with a Gaussian copula. A copula separates two
questions: what each parameter's own distribution looks like, and how the
parameters move together. Every parameter keeps its declared marginal (uniform,
Gaussian, or truncated Gaussian) exactly as written. A `(D, D)` correlation
matrix couples the columns. That matrix lives on the copula's latent
standard-normal scale, which is the hidden set of normal variables the copula
builds the samples from. The matrix lives on the `Problem`, so samplers and
analyzers see one consistent declaration.

## Declare correlation on a Problem

```python
import numpy as np

import jaxgsa

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

print(problem.has_correlated_inputs)  # True
print(problem.correlation)            # the validated latent matrix
```

By default the matrix is interpreted on the latent scale. That is the Pearson
correlation of the copula's underlying normals. A rank (Spearman) correlation
is the natural target for non-Gaussian marginals, and a Gaussian copula
inverts it exactly. Declare a rank correlation with
`correlation_type="spearman"`:

```python
problem = jaxgsa.Problem.from_dict(
    {"x1": (0.0, 1.0), "x2": (0.0, 1.0)},
    correlation=[[1.0, 0.7], [0.7, 1.0]],
    correlation_type="spearman",  # converted via 2 sin(pi * rho_s / 6)
)
```

The matrix is validated on entry. A wrong shape, an asymmetry, a non-unit
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
| A rough Pearson target with non-Gaussian marginals | Prefer `correlation_type="spearman"` with the same number. Only the Spearman route carries an exact guarantee. The two scales differ little (Spearman 0.6 maps to latent 0.618). |

The short version: the latent matrix is what the copula machinery uses; the
Spearman matrix is what your samples measurably have. When in doubt, declare
Spearman.

## Draw correlated samples

`jaxgsa.sampling.monte_carlo` honors the declared correlation transparently
(NORTA: correlated latent normals pushed through each marginal's inverse
CDF). Problems without a correlation keep the previous pseudo-random path
bit-for-bit.

```python
X = jaxgsa.sampling.monte_carlo(problem, n=4096, seed=42)
# Each column follows its declared marginal exactly;
# the ranks across columns follow the declared correlation.
```

## Fit the correlation from observed data

When the dependence structure lives in a historical data set rather than a
paper, estimate it from the data and attach it to the (frozen) problem:

```python
R_fit = jaxgsa.sampling.fit_correlation(problem, X_observed)
problem = problem.with_correlation(R_fit)
```

The fit works on Spearman ranks, so it is invariant to the declared
marginals. Heavy ties (discrete or heavily rounded columns) bias the rank
estimate toward zero — a polychoric estimator is future work.

## Retrofit correlation onto an existing sample

`correlate` imposes the declared correlation on a sample you already have by
Iman–Conover rank re-pairing. Each column of the result is an exact
permutation of the input column, so the marginal values — including any
structure a low-discrepancy design put into them — are preserved:

```python
X_independent = jaxgsa.sampling.monte_carlo(problem.with_correlation(None), n=4096, seed=1)
X_correlated = jaxgsa.sampling.correlate(X_independent, problem, seed=2)
```

The method builds a score matrix from van der Waerden scores. It then removes
the score matrix's own sampling noise before it reads off the ranks. A finite
design therefore lands much closer to the declared correlation. At
N = 50 and a target of 0.8, the achieved rank correlation scatters with a
standard deviation of about 0.024, against about 0.065 for a plain correlated
normal draw.

## Analyze with a correlation-tolerant method

Several methods accept a declared correlation. They divide into two groups
by what they report.

The first group returns a total, correlation-inclusive score per input:
optimal transport, Borgonovo delta, HSIC, and PAWN. These are rank- and
distribution-based, so they need no independence assumption at all:

```python
Y = model(X)

ot = jaxgsa.optimal_transport.analyze(problem, X, Y)
delta = jaxgsa.borgonovo.analyze(problem, X, Y)
hsic = jaxgsa.hsic.analyze(problem, X, Y)
pawn = jaxgsa.pawn.analyze(problem, X, Y)
```

All four share the same caveat. Under correlation these indices deliberately
credit an input for influence it carries through its correlated partners.
Suppose `Y` depends only on `x1` and `corr(x1, x2) = 0.8`. Then `x2` also gets
a clearly non-zero index. That is the correct correlation-inclusive reading,
not an estimation error. It answers "what does watching `x2` tell me about
`Y`", not "what happens if I change `x2` on its own".

The second group answers the other question. It separates an input's direct
effect from the part it borrows through its correlated partners, and reports
both as variance fractions. Two of the three read the declared correlation
matrix directly:

```python
vk = jaxgsa.vkoga.analyze(problem, X, Y)
print(vk.S_TC)  # total correlated — what to measure more accurately
print(vk.S_TU)  # total uncorrelated — what is safe to freeze
```

[VKOGA](/examples/vkoga) fits a kernel surrogate to the given `(X, Y)` data
and splits each input's effect into correlated and uncorrelated parts.
[Kucherenko](/examples/kucherenko) answers the same two questions without a
surrogate: it generates a conditional-copula design and you evaluate your
actual model on it. Both return conditional-variance quantities with their
exact meaning under dependence.

HDMR is the third. It separates the two contributions per term rather than
per parameter, via its ANCOVA decomposition — its `Sb` term is precisely the
correlation-induced share, so it doubles as a diagnostic:

```python
hdmr = jaxgsa.hdmr.analyze(problem, X, Y)
print(hdmr.Sa)  # structural variance fraction per term
print(hdmr.Sb)  # correlative variance fraction — the correlation diagnostic
```

Read HDMR's `ST` with care here. Under dependence it is a sum over the terms
each parameter appears in, not a total-effect index: it has no
variance-reduction reading, and it can be far from the true total effect. Use
`vkoga.S_TU` or `kucherenko.ST` when you need a total index. HDMR's value
under dependence is the per-term `Sa` / `Sb` split, which the other two do
not provide.

`jaxgsa.shapley.analyze(..., backend="hdmr", include_correlative=True)`
allocates the full ANCOVA decomposition across the parameters (an
ANCOVA-based attribution, not conditional-variance Shapley effects).

## Methods that refuse correlated problems

Some methods have indices that are only defined for independent inputs. They
raise a `ValueError` rather than return silently wrong numbers. Two groups do
this: the structured design samplers (`sobol.sample`, `morris.sample`,
`efast.sample`) and the analyzers whose theory needs independence
(`pce.analyze`, `dgsm.analyze`, `shapley.analyze` with the PCE backend).

```python
jaxgsa.pce.analyze(problem, X, Y)
# ValueError: jaxgsa.pce.analyze computes indices that assume independent
# inputs, and they are invalid — not merely approximate — when
# problem.correlation declares a dependence structure. ...
```

To run one of these on the independent version of the problem, drop the
matrix explicitly with `problem.with_correlation(None)`.

## Practical notes

- A Gaussian copula fixes the whole dependence family, not only the
  correlation. It has no tail dependence. On the latent scale, the variables
  are conditionally independent given the rest. Real data may carry
  asymmetric or tail dependence that this smooths away.
- For non-Gaussian marginals, the Pearson correlation of the physical samples
  usually differs from the latent matrix. This is the NORTA
  correlation-matching problem. The Spearman rank correlation is the
  exactly invertible route: declare it with `correlation_type="spearman"`.
  When every marginal is Gaussian, latent and Pearson coincide and the
  sample covariance reproduces the declared one.
- The correlation round-trips through problem serialization: NPZ design
  files and the JSON problem metadata both carry it.

## See also

- [Non-Uniform Inputs](/examples/non-uniform-inputs) for the marginal specs
  the copula couples.
- [RS-HDMR](/examples/hdmr) for the ANCOVA decomposition in depth.
- [Optimal Transport](/examples/optimal-transport) and
  [Borgonovo Delta](/examples/borgonovo) for the distribution-based indices.
- [VKOGA](/examples/vkoga) and [Kucherenko](/examples/kucherenko) for
  variance fractions under dependence.
- [Methods guide](/guide/methods) for which method to pick under correlation.
