# Correlated Inputs

Use this page when your input parameters are not independent. jaxgsa models
dependence with a **Gaussian copula**: every parameter keeps its declared
marginal (uniform, Gaussian, or truncated Gaussian) exactly as written, and a
`(D, D)` correlation matrix on the copula's latent standard-normal scale
couples the columns. The matrix lives on the `Problem`, so samplers and
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

By default the matrix is interpreted on the **latent** scale — the Pearson
correlation of the copula's underlying normals. If you have a **rank**
(Spearman) correlation instead — the natural target for non-Gaussian
marginals, and exactly invertible under a Gaussian copula — declare it with
`correlation_kind="spearman"`:

```python
problem = jaxgsa.Problem.from_dict(
    {"x1": (0.0, 1.0), "x2": (0.0, 1.0)},
    correlation=[[1.0, 0.7], [0.7, 1.0]],
    correlation_kind="spearman",  # converted via 2 sin(pi * rho_s / 6)
)
```

The matrix is validated on entry: wrong shape, asymmetry, non-unit diagonal,
or entries outside `[-1, 1]` raise `ValueError`. A matrix that is not
positive definite — usually a sign of inconsistent pairwise correlations — is
repaired by eigenvalue clipping with a `UserWarning` reporting how much it
changed, so you never silently sample a different dependence structure than
you declared.

If you have a published **covariance** matrix rather than a correlation
matrix, rescale it first — note the variances on its diagonal are discarded
in favor of the declared marginals:

```python
R = jaxgsa.sampling.correlation_from_covariance(cov)
```

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
Iman–Conover-style rank re-pairing. Each column of the result is an exact
permutation of the input column, so the marginal values — including any
structure a low-discrepancy design put into them — are preserved:

```python
X_independent = jaxgsa.sampling.monte_carlo(problem.with_correlation(None), n=4096, seed=1)
X_correlated = jaxgsa.sampling.correlate(X_independent, problem, seed=2)
```

## Analyze with a correlation-tolerant method

Rank- and distribution-based given-data methods remain valid under
correlated inputs. Optimal transport and Borgonovo delta measure the total,
correlation-inclusive influence of each input:

```python
Y = model(X)

ot = jaxgsa.optimal_transport.analyze(problem, X, Y)
delta = jaxgsa.borgonovo.analyze(problem, X, Y)
```

Under correlation these indices deliberately credit an input for influence
it carries *through* its correlated partners: if `Y` depends only on `x1`
but `corr(x1, x2) = 0.8`, then `x2` also gets a clearly non-zero index. That
is the correct correlation-inclusive reading, not an estimation error.

HDMR goes further and **separates** the two contributions via its ANCOVA
decomposition — its `Sb` term is precisely the correlation-induced share, so
it doubles as a diagnostic:

```python
hdmr = jaxgsa.hdmr.analyze(problem, X, Y)
print(hdmr.Sa)  # structural variance fraction per term
print(hdmr.Sb)  # correlative variance fraction — the correlation diagnostic
```

`hsic` and `pawn` also accept correlated problems, and
`jaxgsa.shapley.analyze(..., backend="hdmr", include_correlative=True)`
allocates the full ANCOVA decomposition (an ANCOVA-based attribution, not
conditional-variance Shapley effects).

## Methods that refuse correlated problems

Methods whose indices are only defined for independent inputs raise a
`ValueError` instead of returning silently wrong numbers — the structured
design samplers (`sobol.sample`, `morris.sample`, `efast.sample`) and the
analyzers whose theory needs independence (`pce.analyze`, `dgsm.analyze`,
`shapley.analyze` with the PCE backend):

```python
jaxgsa.pce.analyze(problem, X, Y)
# ValueError: jaxgsa.pce.analyze computes indices that assume independent
# inputs, and they are invalid — not merely approximate — when
# problem.correlation declares a dependence structure. ...
```

To run one of these on the independent version of the problem, drop the
matrix explicitly with `problem.with_correlation(None)`.

## Practical notes

- A Gaussian copula fixes the whole **dependence family**, not just the
  correlation: it has no tail dependence, and variables are conditionally
  independent given the rest on the latent scale. Real data may carry
  asymmetric or tail dependence this smooths away.
- The Pearson correlation of the physical samples will generally **not**
  equal the latent matrix for non-Gaussian marginals (the NORTA
  correlation-matching problem). The Spearman rank correlation is the
  exactly invertible route: declare it with `correlation_kind="spearman"`.
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
- [Methods guide](/guide/methods) for which method to pick under correlation.
