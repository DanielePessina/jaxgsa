# HSIC (Kernel-Based Sensitivity Analysis)

This page turns a plain set of input-output samples into kernel-based
sensitivity indices. You end with four arrays that each hold one number per
input: a normalized dependence score, a total-order score, a significance
p-value, and the raw statistic the first two are built from.

HSIC stands for Hilbert-Schmidt Independence Criterion. It scores how strongly
each input and the output depend on each other. It does not split the output
variance into parts. Instead it maps inputs and outputs into reproducing kernel
Hilbert spaces. A reproducing kernel Hilbert space is a space of functions in
which a kernel — a smooth similarity function between two points — plays the
role of an inner product. Dependence that plain correlation misses shows up in
that space. HSIC therefore detects any form of statistical dependence:
nonlinear, non-monotone, and heteroscedastic. Heteroscedastic here means that
the spread of the output changes with the input, not only its average.

The method produces two index types:

- **R2-HSIC** (first-order): normalized kernel dependence between each input
  and the output, analogous to a kernel correlation coefficient.
- **Total HSIC**: captures dependence through interactions, computed via
  complement product kernels. An interaction is an effect that appears only
  when two or more inputs move together.

When to use HSIC:

- You want a sensitivity measure that captures all forms of dependence, not
  just variance-based effects.
- Your inputs may be correlated. HSIC works without independence assumptions.
- You have existing (X, Y) sample pairs and want sensitivity indices without
  additional model evaluations. HSIC is a given-data method, meaning it accepts
  whatever samples you already have instead of asking you to run the model on a
  sampling design of its own.
- You want a third lens alongside Sobol indices, which split the output
  variance among the inputs, and distribution-based methods such as optimal
  transport (OT) and PAWN, which compare whole output distributions.

## Import style

The HSIC module lives at `jaxgsa.hsic`:

```python
from jaxgsa import hsic
# hsic.analyze(...)
```

`monte_carlo` is in `jaxgsa.sampling`, not in `jaxgsa.hsic`. Call it as
`jaxgsa.sampling.monte_carlo()`.

## Key difference from other methods

HSIC is a given-data method like RS-HDMR. It takes any (X, Y) sample pairs, so
no special sampling design is required. Sobol indices decompose output
variance. HSIC instead measures statistical dependence in a reproducing kernel
Hilbert space. Three consequences follow:

- HSIC detects nonlinear, non-monotone, and heteroscedastic effects that
  Sobol indices may underweight.
- R2-HSIC indices do not sum to 1 (they are individual dependence
  measures, not variance fractions).
- The indices depend on the kernel bandwidth, which is set automatically
  via the median heuristic. The bandwidth sets how far apart two points can
  be and still count as similar; the median heuristic picks it from the median
  distance between sample points.

## Scalar example (Ishigami)

Ishigami is a standard three-input test function with a known answer, which
makes it a good place to read the output of a new method. The steps are:

1. Draw samples. HSIC accepts any sampling scheme, so a plain Monte Carlo draw
   is enough and there is nothing to match up later.
2. Run the model on those samples. HSIC needs the paired (X, Y) values and
   nothing else.
3. Call `analyze`, which builds the kernel matrices and returns the indices.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM

# Generate Monte Carlo samples (any sampling works — no special design needed)
X = jaxgsa.sampling.monte_carlo(PROBLEM, n=2048, seed=42)
print("X shape:", X.shape)  # (2048, 3)

# Evaluate the model
Y = jaxgsa.benchmarks.ishigami.evaluate(jnp.asarray(X))
print("Y shape:", Y.shape)  # (2048,)

# Compute HSIC indices
result = jaxgsa.hsic.analyze(PROBLEM, jnp.asarray(X), Y)

print("R2_HSIC:", result.R2_HSIC)   # (D,) = (3,) — normalized first-order
print("T_HSIC:", result.T_HSIC)     # (D,) = (3,) — total-order
print("p_values:", result.p_values) # (D,) = (3,) — permutation p-values
print("hsic_raw:", result.hsic_raw) # (D,) = (3,) — unnormalized HSIC(Xi, Y)
```

Each of the four printed arrays has length 3, one entry per input, in the
order the parameters appear in `PROBLEM`. Read them together rather than one at
a time. `R2_HSIC[i]` is the normalized dependence of the output on input i, so
entries can be compared across inputs but do not add up to 1. `T_HSIC[i]` adds
the dependence that input i carries through interactions, so a gap between
`T_HSIC[i]` and `R2_HSIC[i]` points to an input that matters mostly in company
with others. `p_values[i]` says whether the measured dependence is larger than
what the permutation test produces by chance. `hsic_raw[i]` is the
unnormalized statistic; it is on the scale of the kernels, so use it for
diagnostics rather than for ranking.

## Multi-output example

When your model returns K outputs, the resulting index arrays have shape
`(K, D)`. K is the number of outputs and D is the number of inputs. The steps
match the scalar case, with one addition: the problem carries `output_names`,
which fixes the row order of the index arrays.

1. Build a `Problem` with three inputs and two named outputs.
2. Draw Monte Carlo samples and evaluate both outputs on the same X.
3. Call `analyze` once. It handles both outputs in a single pass.

```python
import jax.numpy as jnp
import jaxgsa

problem = jaxgsa.Problem.from_dict(
    {
        "x1": (0.0, 1.0),
        "x2": (0.0, 1.0),
        "x3": (0.0, 1.0),
    },
    output_names=("linear", "quadratic"),
)

X = jaxgsa.sampling.monte_carlo(problem, n=2048, seed=42)
Xj = jnp.asarray(X)

# Two outputs: linear combination and sum of squares
Y = jnp.column_stack([
    Xj @ jnp.array([1.0, 2.0, 3.0]),
    jnp.sum(Xj**2, axis=1),
])

result = jaxgsa.hsic.analyze(problem, Xj, Y)

print("R2_HSIC shape:", result.R2_HSIC.shape)  # (K, D) = (2, 3)
print("T_HSIC shape:", result.T_HSIC.shape)    # (K, D) = (2, 3)
```

Both printed shapes are `(2, 3)`: two rows for the two outputs, three columns
for the three inputs. Row 0 belongs to `"linear"` and row 1 to `"quadratic"`,
following the order given in `output_names`. So `result.R2_HSIC[1, 2]` is the
dependence of the `"quadratic"` output on `x3`.

## xarray export

`HSICResult.to_dataset()` converts results to a labeled `xarray.Dataset`. That
lets you select by parameter name or output name instead of by integer index.

```python
ds = result.to_dataset()
print(ds)
# <xarray.Dataset>
# Dimensions:    (output: 2, param: 3)

print(ds.R2_HSIC.sel(param="x1"))
print(ds.p_values.sel(output="linear"))
```

The printed dataset reports `(output: 2, param: 3)`, the same two axes as the
raw arrays above but now named. The first `sel` call returns the R2-HSIC value
of `x1` for both outputs. The second returns the p-values of all three inputs
for the `"linear"` output.

For scalar output, the dataset has dimension `(param,)` only.

## Bandwidth control

By default, HSIC uses the median heuristic to set the Gaussian kernel
bandwidth. No tuning is required. You can override with a fixed bandwidth for
convergence studies, where the bandwidth must stay the same while the sample
size changes:

```python
# Default: median heuristic (recommended)
result = jaxgsa.hsic.analyze(problem, X, Y)

# Fixed bandwidth for sweep studies
result = jaxgsa.hsic.analyze(problem, X, Y, bandwidth=0.3)
```

The two calls differ only in how the kernel width is chosen. The first lets the
data set it and so it moves with the sample. The second pins it at 0.3 for
every run, which makes results from different runs directly comparable.

## Permutation p-values

A permutation test shuffles one input against the output many times to see how
large the HSIC statistic gets when the two are independent by construction. The
p-value is the fraction of shuffles that beat the real value, adjusted by the
Phipson-Smyth correction so that a p-value is never reported as exactly zero.
The number of permutations controls precision:

```python
# Default: 200 permutations (p-value resolution ≈ 0.005)
result = jaxgsa.hsic.analyze(problem, X, Y, n_perms=200, seed=42)

# Faster with fewer permutations
result = jaxgsa.hsic.analyze(problem, X, Y, n_perms=50, seed=42)
```

The first call resolves p-values to about 0.005. The second does a quarter of
the shuffling work, so it runs faster and its p-values are coarser. A small
p-value
(< 0.05) indicates that the input has a statistically significant dependence
with the output. The `seed` parameter ensures reproducibility.

## Shape rules

N is the number of samples, T the number of time steps, K the number of
outputs, and D the number of inputs.

| `Y` shape | R2_HSIC / T_HSIC / p_values / hsic_raw |
|---|---|
| `(N,)` | `(D,)` |
| `(N, K)` | `(K, D)` |
| `(N, T, K)` | `(T, K, D)` |

D is always the last axis of the index arrays.

## Practical caveats

- HSIC is O(N²) in computation and memory (kernel matrices). For N > 8000,
  use `batch_size` to limit peak memory. `batch_size` splits the kernel
  computation into chunks so the whole matrix never sits in memory at once.
- R2-HSIC indices do not sum to 1. They are individual dependence measures,
  not variance fractions.
- The total HSIC index uses product kernels across all D inputs. For very
  high D (> 15), the product kernel can underflow in float32.
- For outputs with large magnitude, set `prenormalize=True` to standardize
  Y before kernel computation.
- Inputs are automatically transformed to [0, 1] via their marginal CDF,
  ensuring comparable bandwidths across dimensions. The marginal CDF of an
  input maps its values to the probability of drawing something smaller.

## See also

- [Basic Example](/examples/basic) for the Sobol workflow with structured
  Saltelli sampling.
- [DGSM](/examples/dgsm) for derivative-based sensitivity measures.
- [eFAST](/examples/efast) for frequency-based variance decomposition.
- [PCE](/examples/pce) for analytical Sobol indices from polynomial expansion
  coefficients.
- [Methods](/guide/methods) for the theory behind HSIC and when to choose it
  over other methods.
- [API Reference](/api/#given-data-methods) for full
  parameter documentation.
