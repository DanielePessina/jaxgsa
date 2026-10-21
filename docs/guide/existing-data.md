# Sensitivity indices from existing data

You have input rows `X` and model outputs `Y`. You did not build them with a
sensitivity design. This page shows which jaxgsa methods accept such data and
what each one gives you.

All four methods below are called the same way: `method.analyze(problem, X, Y)`.
`X` has shape `(N, D)`. `Y` has shape `(N,)`, `(N, T)` or `(N, T, K)`.

| Method | What it gives | Main limit |
| --- | --- | --- |
| `rank` | First-order index `S1` | No total-effect or interaction index. |
| `pce` | `S1`, `S2` and `ST` from a polynomial surrogate | Needs a smooth model and independent inputs. |
| `hdmr` | Per-term variance shares from a fitted surrogate | Order 2 at most. Check the fit before you trust the shares. |
| `borgonovo` | Moment-independent index `delta` | Measures influence on the whole output distribution, not variance. |
| `pawn` | KS distance between conditional and unconditional output distributions | Same kind of measure as `delta`. It is not a variance share. |

## Rank: first-order effects

`rank.analyze` estimates `S1 = Var(E[Y | X_i]) / Var(Y)`. It sorts the rows by
one input at a time and compares each output with the next one in that order.
It needs one sort per input and no tuning choices. It gives no `ST`.

```python
import jax.numpy as jnp
import jaxgsa
from jaxgsa.benchmarks.ishigami import PROBLEM, evaluate

X = jnp.asarray(jaxgsa.sampling.monte_carlo(PROBLEM, n_samples=2048, seed=0))
Y = evaluate(X)

result = jaxgsa.rank.analyze(PROBLEM, X, Y, verbose=False)
print("S1:", result.S1)
```

## PCE and HDMR: total effects and interactions

`pce.analyze` fits a polynomial surrogate and reads `S1`, `S2` and `ST` from its
coefficients. `hdmr.analyze` fits a component-wise surrogate. Both report fit
diagnostics. Read them before you read the indices. The Ishigami model needs
a high PCE `order`; the default of 3 fits it poorly and `analyze` warns.

```python
pce = jaxgsa.pce.analyze(PROBLEM, X, Y, order=9, verbose=False)
print("ST (PCE):", pce.ST)

hdmr = jaxgsa.hdmr.analyze(PROBLEM, X, Y, verbose=False)
print("HDMR terms:", hdmr.terms)
```

## Borgonovo and PAWN: distribution-based measures

`borgonovo.analyze` and `pawn.analyze` ask how much an input changes the shape
of the output distribution. They do not split variance. Use them when the
variance is a poor summary, for example for skewed outputs.

```python
delta = jaxgsa.borgonovo.analyze(PROBLEM, X, Y, verbose=False)
print("delta:", delta.delta)

pawn = jaxgsa.pawn.analyze(PROBLEM, X, Y, verbose=False)
print("PAWN:", pawn.pawn)
```

## Limits that apply to all of them

- Sample size matters. Use `n_bootstrap` to get confidence intervals.
- Check the intervals before you rank close inputs.
- If `X` has correlated columns, the values include correlation effects.
  See [dependent inputs](/guide/dependent-inputs).
- Failed runs are covered in [failed model runs](/guide/failed-runs).

For the full method comparison, see [choosing a method](/guide/methods).
