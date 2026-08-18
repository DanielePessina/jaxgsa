# Roadmap to jaxgsa 1.0

**Status: draft for discussion. Not committed. Nothing here is final.**

This document plans the work between version 0.8.0 and version 1.0.0.

Every method below has two things attached to it:

1. The paper it comes from.
2. The external reference we will check our numbers against.

Section 2 explains that second requirement, because it shapes everything else.

Every claim in this document was checked against the primary source by a
verification pass on 2026-08-18. Where that pass corrected an earlier draft,
the correction is marked **[corrected]**. Where a claim could not be confirmed,
it is marked **[unverified]** and must not be repeated as fact.

---

## Contents

1. What this release is for
2. How we prove the numbers are right
3. Fix first: three defects that block new work
4. Release 0.9 — methods that work on data you already have
5. Release 0.10 — methods that use gradients
6. Release 1.0 — coverage and stability
7. Papers
8. Out of scope
9. Undecided
10. Summary table

---

## 1. What this release is for

### 1.1 The problem with "more methods"

jaxgsa has thirteen methods. That is more than SALib, UQpy, Chaospy or
GlobalSensitivity.jl. Method count is no longer a reason to choose jaxgsa. It
is the minimum a user expects.

### 1.2 The position we will take instead

No other global sensitivity analysis (GSA) package is written in JAX. Code
searches for `jax sobol sensitivity` and `shapley effects jax.numpy` return
only our own files. Neither of the two awesome-jax lists has a
sensitivity-analysis entry. No PyTorch-native GSA package exists either.

So the position for 1.0 is:

> **jaxgsa is the GSA library where the model evaluation loop is one `vmap` on
> the device, and gradients are cheap.**

### 1.3 The cost argument, stated correctly

Two terms first.

- **Automatic differentiation (AD)** computes exact derivatives of a program.
- **Reverse mode** is the AD mode that computes the derivative of one output
  with respect to all inputs in a single backward pass.

The Saltelli column-swap design needs `N * (d + 2)` model runs to get
first-order and total-order indices, where `d` is the number of inputs. A
derivative-based method needs `N` gradients.

One reverse-mode gradient costs about `c` model runs. The value of `c` is:

| Source | Value |
|---|---|
| Griewank and Walther, cheap-gradient result | `c <= 4`, provable |
| Baydin et al., JMLR 2018 | `c < 6` guaranteed, `c` about 2 to 3 in practice |
| JAX autodiff cookbook | about 3 |

The derivative route is cheaper when `N * c < N * (d + 2)`, that is when
**`d > c - 2`**.

**[corrected]** An earlier draft said the crossover is near `d = 4`. That
assumed `c = 6`, the worst case no source calls typical. At `c = 4` the
crossover is `d > 2`. In practice it is `d >= 2`.

The saving grows linearly with `d`. The reason is worth stating: reverse-mode
cost does not depend on the number of inputs.

### 1.4 The limit of that argument. Read this before quoting any speed claim.

**[corrected]** One reverse-mode pass returns **one row** of the Jacobian
matrix. That is the gradient of **one** output.

So for a model with `T` outputs, the full Jacobian costs **`T` reverse-mode
passes**, not one. The derivative route then costs `N * c * T`, and the
crossover becomes:

```
d > c * T - 2
```

At `c = 3` and `T = 10`, that is `d > 28`, not `d > 2`.

jaxgsa supports outputs of shape `(N, T, K)`. So **every cost claim must
either carry the `T` factor or say it applies to scalar output only.**

Forward mode is the opposite: it costs `d` passes regardless of `T`. So the
correct rule is:

- Use reverse mode when `T * K < d`.
- Use forward mode when `T * K > d`.

`src/jaxgsa/dgsm/_analyze.py:118` currently hard-codes `jax.jacrev`, which is
reverse mode. Its own comment at line 112 already notes the limitation. **Task
for 0.10: choose `jacfwd` or `jacrev` by comparing `T * K` against `d`.** This
is a real speed gain on time-series outputs, not a cosmetic change.

### 1.5 Correct the README before release

The README says jaxgsa is "up to 668x faster than SALib". That figure compares
against a single-process NumPy baseline.

**Do one of two things. Do not leave it as it is.**

1. State the baseline in the same sentence.
2. Or re-measure against a parallel CPU baseline.

Published speedups for large Monte Carlo work on a GPU, against a fully
parallel CPU, are near 13x, not 100x. A reviewer will check this.

### 1.6 Timing

Two projects are moving into the same space.

- **jaxonomy** was created on 2026-07-06. It is a JAX simulation engine with a
  sensitivity submodule: Saltelli sampling with bootstrap, Morris, quasi-random
  sampling, categorical inputs, correlated marginals. It already has more
  downloads per month than jaxgsa.
- **equadratures** has an unmerged documentation pull request, dated
  2026-07-19, adding an `equadratures.jax` namespace with differentiable
  quadrature and Sobol indices.

The first-mover position is real. It is not permanent.

---

## 2. How we prove the numbers are right

**Every method in this roadmap must have a paper and an external numerical
check.** A method that only agrees with itself is not verified.

### 2.1 The five tiers

Use the strongest tier available. Tier 0 is strongest.

| Tier | What it means | Examples |
|---|---|---|
| **T0** | A closed-form answer, derived independently | Analytic Sobol indices of the Ishigami function |
| **T1** | Reference numbers published in a paper, typed into the test | Tables of indices in a benchmark paper |
| **T2** | A permissive-licence library, installed as a development extra | SALib, UQpy, POT, ATHENA (MIT); OpenTURNS (LGPL-3+) |
| **T3** | A copyleft library, run in a separate process | R `sensitivity`, `sensobol`, `gsaot`; SAFEpython |
| **T4** | Internal consistency only | Finite differences, coverage simulation, invariants |

### 2.2 Rules

1. A method **must not** ship at T4 alone, unless this document states why no
   external oracle exists. Three items below are in that position. Each says so.
2. **A test that retypes the source's own formula is not an oracle.** It is a
   mirror. It proves only that two copies of one expression agree. Mirrors count
   as no tier at all.
3. **A published table is not automatically T1.** Check it against an
   independent derivation first. Section 2.4 explains why.
4. Record the tier in the test's docstring, so a reader knows what the test
   proves.

### 2.3 Mirror tests already in the codebase

The audit found tests that look like external checks but are mirrors. These
**must** be reclassified, and the ones that matter must be replaced:

| Test | Problem |
|---|---|
| `test_borgonovo.py:381` | Named `test_matches_salib_formula`, but it never imports SALib. It retypes a constant that is character-identical to `borgonovo/_analyze.py:136`. |
| `test_pce.py:308-325` | Mirror. |
| `test_hdmr_streaming.py:225-237` | Mirror. |
| `test_vkoga.py:144-151` | Mirror. |
| `test_dgsm.py:61` | Retypes `(2*pi)**2/pi**2` from `_poincare.py:52`. Line 62's `== 4.0` is the real check. |

### 2.4 A published table that is wrong

Azzini and Rosati (2022), *Data in Brief* 42:108071, publishes indices for 17
test functions. An earlier draft named it as a T1 oracle. **Do not use it
without checking each row.**

Confirmed problems:

- Its Ishigami row prints `S2 = 0.4413` and `ST2 = 0.4424111`. Input `x2` has no
  interactions, so these two numbers must be equal. One is wrong.
- Its `S1 = 0.3138` is a mis-rounding of `0.3139`.
- Its Hartmann 6-D total column is labelled `S1...S6`.
- Row F3 prints `S3` inside the ST column.
- Several rows are numerical, not analytic, and nothing marks which.

The verification pass derived the Ishigami values independently and confirmed
them by Monte Carlo with 4 million samples. Use these:

For `a = 7`, `b = 0.1`, inputs uniform on `[-pi, pi]`:

```
V1 = b*pi^4/5 + b^2*pi^8/50 + 1/2 = 4.3458880
V2 = a^2/8                        = 6.1250000
V3 = 0
V13 = 8*b^2*pi^8/225              = 3.3736999
V   = 13.8445879

S1  = 0.3139   S2  = 0.4424   S3  = 0.0000
ST1 = 0.5576   ST2 = 0.4424   ST3 = 0.2437
```

These match the table's ST row to all seven printed digits, which confirms the
`S2` entry is the error.

### 2.5 Pin the Ishigami convention

Three parameter conventions are in use. They give very different answers, so a
cross-package comparison breaks silently if the convention is not stated.

| Convention | S1, S2, S3 |
|---|---|
| `a = 7, b = 0.1` (dominant; SALib default) | 0.3139, 0.4424, 0 |
| `a = 7, b = 0.05` | 0.2185, 0.6869, 0 |
| `a = 2, b = 1` (sensobol, hard-coded) | 0.3830, 0.0009, 0 |

**[unverified]** The `b = 0.05` variant is often called "the Sobol-Levitan
convention". The Sobol and Levitan paper is real, but it is closed access and
the attribution could not be confirmed. **Do not use that name.** Call it
"`a = 7, b = 0.05`".

### 2.6 Licences

Using a copyleft library as a test oracle in a development-only extra is
acceptable. Three independent reasons:

1. Copyleft duties trigger on **distribution**. Running a tool in CI is private
   use.
2. The published wheel contains no copyleft code. A dev extra is a declaration,
   not a bundle.
3. Running in a separate process is the safe case in the FSF's own guidance.

**This is not legal advice.**

Practical rules:

- Copyleft oracles **must not** appear in `[project.dependencies]`. Put them in
  an extra named `oracles`.
- Prefer out-of-process invocation. The four R oracles satisfy this by
  construction.
- **Do not read copyleft source while implementing.** Work from the paper. This
  is the practice already used for the optimal-transport module.
- OpenTURNS is LGPL-3+, which is weaker. It is safe to import directly.

| Permissive (MIT / BSD / LGPL) | Copyleft (dev extra only) |
|---|---|
| SALib, UQpy, POT, ATHENA, GlobalSensitivity.jl, OpenTURNS | SAFEpython (GPL-3), R `sensitivity` (GPL-2), `sensobol` (GPL-3), `gsaot` (GPL >= 3) |

---

## 3. Fix first: three defects that block new work

Do these before any new method. Two are silent-failure risks. One invalidates a
verification claim we already make.

### 3.1 eFAST computes its frequency plan twice

**The defect.** `efast/_sampling.py:230` computes `omega_0`, builds the design
from it, then discards it. `efast/_analyze.py:188` recomputes the same
expression. Worse, the analyzer picks its complementary frequency band as
`jnp.arange(omega_0 // 2)` at line 67, while the sampler assigns complementary
frequencies in the range `[1, omega_0 // (2*M)]`.

**Why it is dangerous.** Those two bounds were written separately. They agree
today only because `omega_0 // (2*M) <= omega_0 // 2`. If either changes, the
analyzer reads a different frequency bin and returns plausible but wrong S1 and
ST values, with no error raised. The tests recompute the same formula, so they
would not catch it.

**The fix.** Write one `_frequency_plan(D, n_per_curve, M)` that returns both
`omega_0` and the assigned complementary frequencies. Call it from both sides.

### 3.2 eFAST claims a check it does not perform

**The defect.** `tests/test_efast.py:442` guards with `if analytical[i] > 0.01:`
and has no `else` branch. Every near-zero entry of the Sobol g-function
total-order vector is therefore asserted by nothing.

**Why it matters.** eFAST advertises a T0 check against an analytic function
while not performing it on part of the vector. Under section 2, that claim is
currently false. The sibling `test_s1` at line 430 does have an `else` branch
asserting `abs(S1[i]) < 0.02`.

**The fix.** Add the `else` branch. Check `test_pce.py:138-162` for the same
pattern.

### 3.3 A failed model evaluation poisons the results

**The current state.** Only two modules handle a non-finite model output:

- `sobol` drops affected Saltelli groups in `_drop_nonfinite` and **reports the
  damage** through `SobolResult.nan_counts`.
- `morris` cleans non-finite trajectories, warns, and enforces a minimum
  trajectory count.

Everywhere else, a NaN flows straight into the indices. A search for `isfinite`
or `isnan` returns **zero hits in `pce`, `hdmr` and `optimal_transport`.**

**Why it matters.** This is an old, unsolved complaint across the whole field.
SALib issue #273 has been open for seven years. It is the most-discussed issue
in GlobalSensitivity.jl.

**The fix.** Add one shared policy, controlled by an `on_invalid=` argument:

| Value | Behaviour |
|---|---|
| `"raise"` (default) | Fail. Report how many evaluations failed, which rows, and where those rows sit in the design. |
| `"propagate"` | Return NaN only for the affected indices, not the whole array. |
| `"drop"` | Drop and renormalise. Warn with the count and the resulting effective `N`. |

Silently dropping rows changes what the estimator computes, so it **must not**
be the default. Generalise `sobol`'s `nan_counts` as the reporting pattern.

**A feature hiding inside this.** A failed evaluation is data. Build the
indicator `1{y is not finite}` and run a regional sensitivity analysis on it
(section 4.5). The result says which inputs make the solver fail. This reuses
the same kernel and no package does it.

**Oracle.** T4, correctly. This is a behavioural contract, not a number. Test it
by injecting NaNs into a model with known analytic indices, then checking each
of the three settings behaves as specified.

---

## 4. Release 0.9 — methods that work on data you already have

Theme: get more out of a sample that has already been evaluated, and say how
confident we are in the result.

### 4.1 Rank-based estimators

**What it does.** Sort the sample by input `x_i`. Look at which output values
end up next to each other. If `x_i` matters, neighbouring outputs are similar.
If it does not, they are not.

**Chatterjee's coefficient.** Sort by `X`. Let `r_i` be the rank of the matching
`Y`. Then:

```
xi_n = 1 - 3 * sum_{i=1}^{n-1} |r_{i+1} - r_i| / (n^2 - 1)
```

This form assumes no ties in `X` and none in `Y`. With ties in `Y`, using
`l_i = #{j : Y_(j) >= Y_(i)}`:

```
xi_n = 1 - n * sum_{i=1}^{n-1} |r_{i+1} - r_i| / (2 * sum_{i=1}^{n} l_i * (n - l_i))
```

Break ties in `X` uniformly at random. The coefficient is **not symmetric** in
`X` and `Y`. That is deliberate.

**The Sobol index version.** Gamboa et al. give, with `N(j)` the rank-successor
permutation:

```
xi_n = [ (1/n) sum Y_j * Y_{N(j)} - ((1/n) sum Y_j)^2 ]
       / [ (1/n) sum Y_j^2 - ((1/n) sum Y_j)^2 ]
```

The verification pass ran this on Ishigami with `N = 2e5` and got
`0.317 / 0.444 / -0.003` against analytic `0.3139 / 0.4424 / 0`.

**Which orders you get. [corrected]** First-order indices, and **closed group
indices** `S^u = Var(E[Y|X^u]) / Var(Y)` through multivariate nearest
neighbours.

**Total-order indices are not direct.** You must build `ST_i = 1 - S^{~i}`,
which needs a nearest-neighbour search in `d - 1` dimensions. The convergence
rate degrades there. OpenTURNS agrees: its `getTotalOrderIndices()` raises
"Method not yet implemented".

**Cost.** `O(N log N)` in one dimension. The multivariate case is `O(N log N)`
expected time from a k-d tree at fixed dimension, with constants that grow
exponentially in dimension.

**Confidence intervals. [corrected]** Lin and Han give a central limit theorem
and a consistent variance estimator, so intervals need no bootstrap. **This
holds for `d = 1` only.** For `d > 1` a non-ignorable bias appears and the
interval covers `E[xi_n]`, not `xi`. State this where group indices are offered.

**Papers.**

- Chatterjee, S. (2021). A new coefficient of correlation. *JASA*
  116(536):2009-2022. doi:10.1080/01621459.2020.1758115
- Gamboa, Gremaud, Klein, Lagnoux (2022). Global sensitivity analysis: a novel
  generation of mighty estimators based on rank statistics. *Bernoulli*
  28(4):2345-2374. doi:10.3150/21-BEJ1421
- Lin, Z. and Han, F. (2023). On boosting the power of Chatterjee's rank
  correlation. *Biometrika* 110(2):283-299. doi:10.1093/biomet/asac048
- Lin, Z. and Han, F. (2022). Limit theorems of Chatterjee's rank correlation.
  arXiv:2204.08031

**Two citation traps.**

1. **arXiv:2003.01772 is withdrawn.** Cite the *Bernoulli* DOI, or the
   superseding preprint arXiv:2605.23760. Do not cite the withdrawn version
   anywhere.
2. **An erratum exists.** Gamboa, Klein, Lagnoux and Rochet correct two
   computational errors in the **asymptotic variance** of the rank-based Sobol
   estimator (HAL hal-04125285v2). The estimator and its consistency are
   unchanged. **Cite the erratum wherever we use the CLT variance**, because the
   published variance is wrong. [unverified] whether a journal erratum exists.

**Oracles.**

- **T2**: `scipy.stats.chatterjeexi`. Confirmed present in SciPy 1.15.0 and
  absent in 1.14.1. Signature
  `(x, y, *, axis=0, y_continuous=False, method='asymptotic', nan_policy, keepdims)`.
  It implements the original 1-nearest-neighbour version. **No new dependency
  needed.**
- **T2**: OpenTURNS `RankSobolSensitivityAlgorithm`. It is in the main
  namespace as of version 1.26; before that it was in `openturns.experimental`.
- **T0**: the Ishigami and Sobol g-function values in section 2.4.
- **T3**: R `sensitivity::sobolrank`.

**Effort.** Very low. This remains the best value per line in the plan.

### 4.2 Dummy-parameter significance test

**What it does.** Estimate the index of an input that provably cannot affect the
output. The answer is not zero, because estimators are noisy. That value is the
noise floor. A real input matters only if its index clears the floor.

**Why it matters.** It turns a ranking into a decision. Most GSA output is a
sorted list with no cut-off. This supplies the cut-off.

**How to implement it. [corrected]** Do **not** append a column to the design and
re-run the model. SAFEpython and sensobol compute the dummy index **analytically
from the existing sample, at zero extra model cost**. SAFEpython's form:

```python
Sdummy  = (mean(YA[~idx] * YB[~idx]) - f0**2) / VARy
STdummy = 1 - (mean(YB[~idx]**2) - f0**2) / VARy
```

For PAWN, the dummy is the Kolmogorov-Smirnov distance between two independent
sub-samples of the same unconditional distribution.

Only GlobalSensitivity.jl appends real columns, and it does so in its regional
sensitivity analysis, not in a Sobol estimator.

**Paper. [corrected]** The primary citation is **Khorashadi Zadeh et al. (2017),
*Environmental Modelling and Software* 91:210-222,
doi:10.1016/j.envsoft.2017.02.001**. Equations 3, 4, 12 and 13 give the
estimators. Both SAFEpython and sensobol cite this.

Do **not** cite Pianosi, Sarrazin and Wagener (2015) for the dummy parameter.
That paper is the SAFE toolbox description and predates the feature.

Do **not** cite Andres (1997) as the origin. [unverified] The paper is real but
closed access, its abstract concerns a different screening method, and it does
not appear in Khorashadi Zadeh's reference list.

**Oracles.**

- **T4 by construction**, and this is correct here. The dummy test is a
  statistical protocol, not a quantity with a true value. Verify two things: on
  a model that truly ignores an input, the estimated index centres near zero;
  and the test holds its nominal false-positive rate over repeated runs.
- **T3**: cross-check the floor against `sensobol::sobol_dummy`.

**Effort.** Very low.

### 4.3 Confidence intervals in every module

**Current state.** jaxgsa computes intervals in five of thirteen modules:
`sobol`, `morris`, `pawn`, `borgonovo`, `optimal_transport`. All five share one
convention through `_core/bootstrap.py`, which is unusually consistent.

Intervals are absent from `hsic`, `dgsm`, `hdmr`, `pce`, `shapley`, `efast`,
`kucherenko`, `vkoga`.

**What to add.**

1. Bootstrap intervals in the remaining eight modules.
2. Three interval types: percentile, basic, and bias-corrected and accelerated
   (BCa).
3. Asymptotic intervals where the estimator has a known limit distribution.

**Why BCa.** An index bounded in `[0, 1]` has a skewed sampling distribution near
either end. The percentile method is only first-order accurate and corrects
neither median bias nor a variance that changes with the parameter. BCa is
second-order accurate and transformation-respecting, so it covers correctly in
exactly that regime.

**Two known defects to fix at the same time.**

1. `sobol`'s bootstrap path recomputes point estimates through a different XLA
   graph than the non-bootstrap path (`_analyze.py:338` versus `:257`). Nothing
   tests that the two agree. Add that test.
2. `sobol`'s `slice_chunk_size` reaches only the bootstrap path, and the test
   that nominally covers it passes `num_resamples=0`. So it is validated and
   discarded, untested. Morris and Borgonovo both have real invariance tests;
   Sobol does not.

Separately, `pawn`'s `slice_chunk_size` is declared, documented as "accepted for
signature parity", and appears nowhere else in the file. Decide whether to
implement it or remove it.

**Papers.**

- Efron, B. (1987). Better bootstrap confidence intervals. *JASA*
  82(397):171-185. doi:10.1080/01621459.1987.10478410 — the primary source for
  BCa.
- Efron, B. and Tibshirani, R. (1993). *An Introduction to the Bootstrap*.
  Chapman and Hall. Chapter 14.
- Janon, Klein, Lagnoux, Nodet, Prieur (2014). Asymptotic normality and
  efficiency of two Sobol index estimators. *ESAIM: Probability and Statistics*
  18:342-364. doi:10.1051/ps/2013040

**Oracles.**

- **T4 by simulation, and this is the right test**: coverage. Generate many
  independent datasets from a model with known analytic indices. A nominal 95%
  interval must contain the true value about 95% of the time. This tests the
  interval, not just the point estimate.
- **T2**: SALib bootstrap intervals for `sobol`.
- **T3**: `sensobol` for BCa. Note its default is `type = "norm"`; BCa must be
  requested with `type = "bca"`.

**Effort.** Low, but it touches eight modules.

### 4.4 A choice of Sobol estimator

**Current state. [corrected]** An earlier draft said jaxgsa uses the Saltelli
estimator. That is not accurate. `sobol/_indices.py` uses:

| Order | Estimator |
|---|---|
| First | Saltelli 2002 form, `E[B * (AB_j - A)] / Var(Y)` |
| Total | **Jansen (1999)**, `0.5 * E[(A - AB_j)^2] / Var(Y)` |
| Second | Saltelli 2002 |

Jansen was chosen because it is non-negative by construction. Section 4.4.2
shows that choice is better founded than the docstring says.

**The gap** is that the estimator is fixed with no option, not that we only have
Saltelli.

#### 4.4.1 Estimators to add

| Estimator | Source |
|---|---|
| Jansen for **first** order | Jansen, M.J.W. (1999). *Computer Physics Communications* 117(1-2):35-43. doi:10.1016/S0010-4655(98)00154-4 |
| Janon-Monod | Janon et al. (2014), above. **The estimator is not original to them**: Monod, Naud and Makowski (2006) introduced it, which is why the name is hyphenated. |
| Martinez | **[unverified as a journal paper]** The provenance is a 2011 GdR MASCOT-NUM presentation. It uses an empirical correlation, which is what gives it Fisher-transform intervals. |
| Mauntz-Kucherenko | Sobol', Tarantola, Gatelli, Kucherenko, Mauntz (2007). *RESS* 92(7):957-960. |
| Azzini-Rosati | **Azzini, Mara and Rosati (2021)** — three authors. "Comparison of two sets of Monte Carlo estimators of Sobol' indices". *Environmental Modelling and Software* 144:105167. Cost `2N(d+1)`. Guarantees `S_i <= T_i`, which Saltelli and Jansen do not. |

**Do not attribute the choice guidance to Puy et al.** Their paper is "A
comprehensive comparison of total-order estimators **for global sensitivity
analysis**", *IJUQ* 12(2):1-18. It frames results by **goal and by
dimensionality**, never by index magnitude or output variance:

- For ranking inputs: Razavi-Gupta, Jansen, or Janon-Monod.
- For approaching true total-order values: Jansen, Janon-Monod, or
  Azzini-Rosati.
- Saltelli, Homma-Saltelli, Glen-Isaacs and pseudo-Owen degrade badly for
  `d > 10`.

**[corrected]** The magnitude-based guidance in an earlier draft came from the
individual papers, and two of the three statements were imprecise:

- Janon-Monod's asymptotic variance is **always at most** the classical one,
  with equality only when `S = 0` or `S = 1`. It is not worse for small indices.
- Azzini-Rosati's robustness is to a **large mean relative to the variance**, and
  it applies to the **first-order** estimator. Its total-order advantage is in
  the high-index regime, `ST_i > 0.55`.
- Mauntz-Kucherenko's small-index accuracy is [unverified] from the 2007 paper,
  which is closed access.

#### 4.4.2 Negative index estimates, explained correctly

**[corrected]** An earlier draft said a negative estimate means the estimator is
a poor fit for that index magnitude. That misdescribes the mechanism.

The real cause: these estimators form the index as a **difference of two
correlated Monte Carlo estimates** — a cross-moment minus a squared mean. That
is unbiased but noisy. When the true index is near zero, the sampling error
exceeds it, so estimates land on both sides of zero.

Owen (2013, *ACM TOMACS* 23(2):11) states it directly: the cross-moment form
"has very large variance when `tau^2_u << mu^2`", whereas the squared-difference
form "is a sum of squares, hence nonnegative. If the true `tau^2_u = 0`, then the
estimate is 0 with probability one."

Puy et al. add that Saltelli's mixing of a B matrix in the numerator with an A
matrix in the denominator increases the volatility, and that negatives also
appear at large sample sizes and low dimension. So it is **not** simply an
under-sampling artefact.

Negative values are expected, not a bug, for estimators that admit them.
Investigate only if they are large, appear for demonstrably large indices, or
persist as `N` grows.

Remedies, in order:

1. Use a sum-of-squares numerator (Jansen, or Azzini-Rosati). **jaxgsa already
   does this for total order.**
2. Use consistent mean and variance across numerator and denominator
   (Janon-Monod).
3. Report bootstrap intervals, so a negative value reads as "the interval covers
   zero".
4. Clip to zero **for display only**. **Never clip before ranking** — clipping
   biases upward in exactly the near-zero regime where ranking decisions are
   made.

**Oracles.**

- **T0**: the values in section 2.4. Every estimator must converge to the same
  answer.
- **T2**: OpenTURNS ships exactly four `SobolIndicesAlgorithm` implementations —
  `SaltelliSensitivityAlgorithm`, `JansenSensitivityAlgorithm`,
  `MartinezSensitivityAlgorithm`, `MauntzKucherenkoSensitivityAlgorithm`. This
  is a direct four-way check under a licence we can import.
- **T3**: `sensobol` for Azzini-Rosati.

**Design cost.** The Saltelli design costs `N(d+2)` for first plus total order,
and `N(2d+2)` when second order is added. Total order alone is `N(d+1)`.

**Effort.** Low.

### 4.5 Regional sensitivity analysis

**What it does.** Split the sample into two groups by an output condition:
"behavioural" and "non-behavioural". For each input, compare the two conditional
input distributions. A large distance means that input controls whether the
output lands in the region of interest.

**What question it answers.** "Which inputs drive the output into this region?",
rather than "which inputs move the output".

**Relation to PAWN.** The two are dual, not mirror images:

| | Condition on | Compare |
|---|---|---|
| PAWN | an input | output distributions |
| RSA | the output | input distributions |

**[corrected]** They are not exact reflections. PAWN compares a conditional
distribution against the **unconditional** one. RSA compares **two complementary
conditional** distributions against each other. PAWN also aggregates its
statistic over many conditioning intervals; RSA produces one number per input
from a binary split.

**[corrected] Two implementations exist and they are not interchangeable.**

| | SAFEpython `RSA_indices_thres` | SALib `SALib.analyze.rsa` |
|---|---|---|
| Conditioning | one behavioural threshold | `bins=20` percentile bins |
| Statistic | max vertical distance between CDFs (Kolmogorov-Smirnov) | **Cramer-von Mises**, range `[0, inf)` |
| Output | one number per input | one value per bin, so sensitivity across output space |
| Direction | output space | `target="Y"` or `target="X"` |

SALib's signature is
`analyze(problem, X, Y, bins=20, target="Y", print_to_console=False, seed=None)`.

**A cross-library agreement test between SALib and SAFEpython would fail by
construction.** Decide which formulation jaxgsa implements, then pick the
matching oracle. If we implement the classical threshold-and-KS form,
SAFEpython is the oracle, not SALib.

**Papers.**

- Spear, R.C. and Hornberger, G.M. (1980). Eutrophication in peel inlet — II.
  Identification of critical uncertainties via generalized sensitivity analysis.
  *Water Research* 14(1):43-49. doi:10.1016/0043-1354(80)90040-8
- Roux, Loisel, Buis (2025). Maximizing regional sensitivity analysis indices to
  find sensitive model behaviors. *IJUQ* 15(1):47-60.
  doi:10.1615/Int.J.UncertaintyQuantification.2024051424 — finds the region best
  explained by each input, instead of fixing a threshold.

**Two citation traps.**

1. SAFEpython's own docstring miscites Spear and Hornberger as "Water Resour.
   Res." The journal is **Water Research**. Do not copy the error.
2. **[corrected]** Pianosi and Wagener (2018) does **not** cover RSA. A full-text
   check found RSA mentioned twice, both times as dataset provenance. Neither
   Spear and Hornberger paper is in its reference list. It is a PAWN-only paper,
   titled "Distribution-based sensitivity analysis from a generic input-output
   sample", *EMS* 108:197-207.

**Oracles.**

- **T3**: SAFEpython `RSA_indices_thres` and `RSA_indices_groups` for the
  threshold form. GPL-3, so implement from the papers.
- **T2**: SALib `SALib.analyze.rsa` for the binned Cramer-von Mises form.

**Effort.** Very low. It shares the PAWN kernel.

### 4.6 HSIC significance tests, and distance correlation as a kernel

First, a definition. **HSIC** is the Hilbert-Schmidt Independence Criterion. It
measures dependence between an input and the output using kernels, and it
detects any kind of dependence, not only monotone or linear.

#### 4.6.1 Add p-values

jaxgsa returns an HSIC score. A score has no cut-off. Add three routes:

1. A **permutation test**. Shuffle one input against the output many times. The
   shuffled scores give the null distribution.
2. An **asymptotic test**. Under independence, `m * HSIC_b` is approximately
   Gamma-distributed, by moment matching.
3. A **spectral test**.

De Lozzo and Marrel offer all three, with the third route being **bootstrap
resampling with replacement**, not permutation. [corrected] An earlier draft
listed only two routes.

**Implementation detail that is a common bug.** In the Gamma approximation,
`alpha = E[HSIC_b]^2 / var(HSIC_b)` and `beta = m * var(HSIC_b) / E[HSIC_b]`.
The factor `m` appears **in beta only**.

**Bandwidth. [corrected]** An earlier draft said the median heuristic is the
standard choice. It is the standard in **kernel testing**, but GSA practice —
OpenTURNS, De Lozzo and Marrel, Marrel and Chabridon — uses the **empirical
standard deviation** of the variable. Use that default and document it.

Known limits of the median heuristic: it maximises power for location-type
differences but degrades for scale-type differences and in high dimension. Cite
Reddi et al. (2015, AAAI) — note the author is Reddi, not Ramdas. Do not cite
Garreau, Jitkrittum and Kanagawa (2017) as a criticism; it largely supports the
heuristic.

**Biased and unbiased forms.** The V-statistic `HSIC_b = (1/m^2) tr(KHLH)` has
bias `O(1/m)`. The U-statistic is unbiased and zeroes the kernel diagonals. Both
cost `O(m^2)` in time and memory. The spectral null needs `O(m^3)`.
**[unverified which normalisation to prefer]**: two conventions circulate,
`1/m^2` and `(m-1)^-2`. Pick one deliberately and document it.

#### 4.6.2 Ship distance correlation as a kernel, not a module

**The claim, verified.** Sejdinovic, Sriperumbudur, Gretton and Fukumizu (2013),
*Annals of Statistics* 41(5):2263-2291, Theorem 24, prove:

```
dCov^2(X, Y) = 4 * HSIC with distance-induced kernels
```

and, in their Appendix A, that distance **correlation** equals **normalised
HSIC** with those kernels, with the constants cancelling exactly.

So shipping distance correlation as a separate method would present a kernel
choice as a new method. Ship it as a kernel option and say so in the
documentation.

**Four constraints this places on the kernel abstraction.**

1. It is a **pair** of kernels, one per argument, combined by tensor product.
   The abstraction must support that, not a single kernel.
2. The distance kernel is **unbounded, not translation-invariant, and
   parameterised by a base point**. Section 5.3 of that paper proves you cannot
   reach it by tuning a Gaussian bandwidth. It is a different point in kernel
   space, not a bandwidth setting.
3. Its moment conditions are **stronger** than for a bounded kernel.
4. **The asymptotic Gamma p-value is invalid for distance kernels.** The moment
   formulas assume normalised kernels with `k_ii = 1`. The distance kernel has
   `k(z,z) = rho(z, z0)`, which is not constant. **Route distance kernels to the
   spectral or bootstrap null instead.**

**Papers.**

- Gretton, Bousquet, Smola, Scholkopf (2005). Measuring statistical dependence
  with Hilbert-Schmidt norms. *ALT 2005*, LNCS 3734:63-77.
  doi:10.1007/11564089_7
- Gretton et al. (2007). A kernel statistical test of independence. *NIPS 2007*.
- De Lozzo, M. and Marrel, A. (2016). New improvements in the use of dependence
  measures for sensitivity analysis and screening. *JSCS* 86(15):3038-3058.
- Da Veiga, S. (2015). Global sensitivity analysis with dependence measures.
  *JSCS* 85(7):1283-1305. doi:10.1080/00949655.2014.945932
- Szekely, Rizzo, Bakirov (2007). Measuring and testing dependence by
  correlation of distances. *Annals of Statistics* 35(6):2769-2794. Note the
  title says "dependence", not "independence"; the latter is a common misquote.
- Sejdinovic et al. (2013), as above.

**Oracles.**

- **T2**: OpenTURNS `HSICEstimatorGlobalSensitivity`, with
  `getPValuesAsymptotic()` and `getPValuesPermutation()`. It also exposes the
  biased and unbiased forms as `HSICVStat` and `HSICUStat`, and offers
  `getR2HSICIndices()`. This checks both p-value routes and both estimator forms
  directly.
- **T2**: the `dcor` package for distance correlation.
- **T3**: R `sensitivity::testHSIC`, which has asymptotic, Gamma, permutation and
  sequential p-values.
- **T4**: false-positive rate. A test at level 0.05 must reject about 5% of the
  time when the input truly does not matter.

**Deferred: target and conditional HSIC.** [corrected] An earlier draft said
target HSIC puts a weight `w(Y)` on the output **kernel**. It does not. It
transforms the output **values**: `T-HSIC = HSIC(X_i, w(Y))`. The kernel changes
only because the variable type changes. Conditional HSIC is a different
construction again — a change of measure using two asymmetric centering
matrices, which is exactly why it has no unbiased form and why OpenTURNS's
`getPValuesAsymptotic()` raises for it. Design the output path so a value
transform can be inserted later. Do not build it yet. Reference: Marrel, A. and
Chabridon, V. (2021), *RESS* 214:107711.

**Effort.** Low.

---

## 5. Release 0.10 — methods that use gradients

Theme: the selling point, made real. Ship this as one story, with one benchmark.

### 5.1 Correction: Poincare-certified DGSM already ships

**[corrected] This was planned as new work. It is not. It is already
implemented.**

`src/jaxgsa/dgsm/_poincare.py` implements per-marginal Poincare constants and
wires them through to the result:

- `poincare_constant(spec, *, grid=512)` returns the constant. Uniform gives
  `(b-a)^2 / pi^2` exactly. Unbounded Gaussian gives `sigma^2`. A truncated or
  one-sided-truncated Gaussian goes to a P1 finite-element spectral solve.
- `axis_constants(problem)` returns the per-axis constants.
- `DGSMResult.upper_bound` is the Poincare bound `C_i * nu_i / Var(Y)`.
- `DGSMResult.lower_bound` is a Kucherenko-Song lower bound.
- `analyze()` warns when an upper bound falls below its lower bound, and raises
  when the problem declares correlated inputs.
- Both functions are public and documented.

**What is actually left to do:**

1. **Get an external oracle for the truncated-Gaussian branch.** The uniform and
   unbounded-Gaussian constants are closed form and already T0. The spectral
   solve has no closed form, and current coverage calls the private function
   directly. Use R `sensitivity::PoincareOptimal` as a T3 oracle, and the tables
   in Roustant, Barthe and Iooss (2017) as T1.
2. **Extend the constants** to the marginals added in section 6.4.
3. **Add the PCE route.** Sudret and Mai (2015) compute DGSM analytically from
   polynomial chaos coefficients, with no extra model runs. jaxgsa already has
   the PCE module. This also gives a T4 cross-check: the PCE-derived DGSM and
   the autodiff DGSM must agree.
4. **Update the signature** when `_NormalizedInputSpec` becomes a public frozen
   dataclass. `poincare_constant` is public but currently typed on a private
   6-tuple alias.

**Two citation errors to fix in the existing docstring.**

| Currently says | Should say |
|---|---|
| Roustant et al. (2017). *Stat. Comp.* 27:879-894 | Roustant, Barthe, Iooss (2017). *Electronic Journal of Statistics* 11(2):3081-3119. doi:10.1214/17-EJS1310 |
| Lamboni et al. (2013). *Math. Comp. Sim.* 87:44-54 | *Math. Comp. Sim.* **87:45-54**. doi:10.1016/j.matcom.2013.02.002 |

**Three facts about the bound worth recording in the docs.**

1. The bound is `S_i^T <= C(mu_i) * nu_i / D`. The constant multiplies `nu_i` in
   the numerator. `D` is the output variance. The index is the **total** Sobol
   index. This form is Roustant, Barthe and Iooss equation (1.2).
2. **It requires independent inputs.** The derivation assumes a product measure.
   The code already raises when correlation is declared, which is correct.
3. **[corrected]** Do not attribute the normalised form to Lamboni et al.
   Theorem 3.1. That theorem is unnormalised: `D_j^tot <= C(mu_j) * nu_j`.
   Lamboni's theorems also assume a Boltzmann or log-concave measure, and the
   paper states explicitly that a uniform density on a finite interval is
   neither. The uniform case comes from Sobol' and Kucherenko's direct argument
   instead.

### 5.2 Active subspaces

**What it does.** Ordinary GSA asks which of the `d` inputs matter. Active
subspaces asks which **directions** in input space matter. A direction can be a
mixture, such as `0.7*x1 - 0.3*x4`.

**How it works.**

1. Build `C = E[grad f(x) grad f(x)^T]`, a `d` by `d` symmetric positive
   semi-definite matrix. Estimate it as `G^T G / N`, where `G` holds the stacked
   gradients.
2. Eigendecompose: `C = W L W^T`.
3. Read the eigenvalues.

**What question it answers.** "Is my 50-parameter model really a 50-parameter
model?" Often it is not.

**Why Sobol indices cannot do this.** If the model depends only on `x1 - x2`,
every individual Sobol index is moderate, and nothing in the output says the
model is one-dimensional. The active subspace says so, and gives the direction.

**[corrected] What justifies the approximation.** An earlier draft said a large
gap between the first and second eigenvalues justifies a one-dimensional
approximation. That is wrong. Two different guarantees are involved:

| Quantity | What it controls |
|---|---|
| **Trailing eigenvalues** `lambda_{n+1} + ... + lambda_m` | The **approximation error**. Theorem 3.1: `E[(f - F)^2] <= C1 * (lambda_{n+1} + ... + lambda_m)`. |
| **The eigenvalue gap** | How **stably the subspace is estimated** from a finite sample. |

So a large `lambda_1 / lambda_2` gap alone does **not** certify a
one-dimensional approximation. You need the trailing eigenvalues to be small.
Constantine's own paper notes the Poincare inequality in that bound is "a
notoriously loose bound".

**[corrected] Activity scores.** Constantine and Diaz define, in their equation
21:

```
alpha_i(n) = sum_{j=1}^{n} lambda_j * w_{i,j}^2
```

The sum runs to `n`, the **truncation**, not to `d`. Their Theorem 4.1 gives
`alpha_i(n) <= nu_i`, with equality only when `n = m`. Their Theorem 4.2 links
it to the total Sobol index:

```
tau_i <= (1 / (4 * pi^2 * V)) * (alpha_i(n) + lambda_{n+1})
```

The `+ lambda_{n+1}` term is required whenever `n < m`. The `1/(4*pi^2)` factor
is specific to the `[-1,1]^m` scaling.

**One practical caveat.** `C` is **not invariant to input scaling**. Inputs must
be standardised to a common scale, or the eigenvectors just report units.

**When it fails.** It needs a differentiable model with square-integrable
gradients. It finds **linear** combinations only, so a model with curved level
sets has no active subspace even if it is genuinely low-dimensional.

**Papers.**

- Constantine, Dow, Wang (2014). Active subspace methods in theory and practice.
  *SIAM J. Sci. Comput.* 36(4):A1500-A1524. doi:10.1137/130916138
- Constantine, Dow, Wang (2014). Erratum. *SIAM J. Sci. Comput.*
  36(6):A3030-A3031. doi:10.1137/140983598
- Constantine, P. (2015). *Active Subspaces*. SIAM Spotlights, volume 2.
- Constantine, P. and Diaz, P. (2017). Global sensitivity metrics from active
  subspaces. *RESS* 162:1-13. doi:10.1016/j.ress.2017.01.013
- Zahm, Constantine, Prieur, Marzouk (2020). Gradient-based dimension reduction
  of multivariate vector-valued functions. *SIAM J. Sci. Comput.*
  42(1):A534-A558. doi:10.1137/18M1221837

**Three citation traps.**

1. **[corrected]** The erratum corrects Theorems 3.2, 3.3, 3.6 and 3.7 — a
   misapplied triangle inequality — plus a sign in equation 5.3. It does **not**
   touch Theorem 3.1 or any Poincare constant.
2. **Do not cite arXiv:1304.2070** for those four theorems. It was last updated
   in December 2013 and carries the uncorrected statements.
3. In Zahm et al., the matrix is `H = integral (grad f)^T R_V (grad f) dmu`,
   where `R_V` is the **output-space** metric. The input covariance enters
   separately: the eigenvalues come from the pencil `(H, Sigma^-1)`. It is easy
   to conflate the two.

**Competitive position.** The reference package `active_subspaces` has no commit
on any branch since December 2016 and supports Python 2.7 only. **[corrected]**
It is dormant and unmaintained, but it is **not archived** — do not say it is.
Its successor ATHENA has 54 stars. OpenTURNS does not implement this.
GlobalSensitivity.jl has had an open issue for it since 2021.

**Oracles.**

- **T0, and unusually strong.** For a linear model `f = a^T x`, the gradient is
  constant, so `C = a a^T` **exactly, for any input distribution**. Rank one,
  single eigenvalue `|a|^2`, eigenvector `a/|a|`. Verified numerically to machine
  zero under uniform, Gaussian and exponential inputs.
- **T0**: for a quadratic `f = 0.5 x^T A x` with `x ~ N(mu, Sigma)` and `A`
  symmetric, `C = A (Sigma + mu mu^T) A`. This reduces to `A^2` for standard
  normal inputs. If `A` is not symmetric, use `0.5(A + A^T)`.
- **T4 invariant**: `trace(C) = sum_i nu_i`, because the trace commutes with the
  expectation. Verified to 13 digits. This ties active subspaces to the existing
  DGSM module and catches axis errors immediately.
- **T2**: ATHENA (`athena-mathlab` on PyPI, MIT, active as of June 2026).

**Effort.** Very low. About twenty lines plus tests.

### 5.3 Crossed DGSM

**What it does.** Take the same idea one derivative further:
`nu_ij = E[(d2f / dx_i dx_j)^2]`. A large value means inputs `i` and `j`
interact.

**What it bounds. [corrected]** An earlier draft said it bounds the second-order
Sobol index. It does not. Theorem 1 of Roustant, Fruth, Iooss and Kuhnt gives:

```
D_{i,j}  <=  D_{i,j}^super  <=  C(mu_i) * C(mu_j) * nu_{i,j}
```

The tight quantity is the **superset importance**
`D_{i,j}^super = sum over all sets I containing both i and j of D_I`. Bounding
the pure second-order index `S_ij` goes through the first, loose inequality.
**Advertise it as a bound on superset importance.**

The product `C_opt(mu_i) * C_opt(mu_j)` is the best constant. Independence is
required.

**Why it fits JAX.** `jax.hessian` makes this nearly free. Second-order Sobol
designs are expensive; this is a cheap screen for which pairs deserve one.

**When it fails.** The model must be twice differentiable. Memory is `O(d^2)`
per sample, so chunk over inputs for large `d`.

**Paper.** Roustant, Fruth, Iooss, Kuhnt (2014). Crossed-derivative based
sensitivity measures for interaction screening. *Mathematics and Computers in
Simulation* 105:105-118. doi:10.1016/j.matcom.2014.05.005. They coin the term
"crossed DGSM". The superset importance itself is Fruth, Roustant and Kuhnt
(2014), *JSPI* 147:212-223.

**Oracles.**

- **T0**: analytic on polynomials. For `f = x1 * x2`, the cross derivative is
  exactly 1 and every other second derivative is 0.
- **T4**: ranking agreement with second-order Sobol indices on Ishigami, where
  the `x1`-`x3` interaction is the only non-zero one.
- **T2 — but do not trust it.** GlobalSensitivity.jl `DGSM(crossed=true)` exists
  and returns `crossedsq`, which is `nu_ij`. Two defects make it a poor oracle:
  its `tao` weight is coded as `(1 - 3x + x^2)/6` where the literature has
  `(1 - 3x + 3x^2)/6`, and the coded version goes negative at `x = 1`, which is
  impossible for a weight on a squared derivative; and `tao` and `sigma` assume
  uniform `[0,1]` inputs yet are applied to samples from any distribution. It
  also computes only the strict upper triangle, leaving the diagonal at zero.
  Use it for `crossedsq` only, on uniform `[0,1]` problems.

**Effort.** Very low.

### 5.4 Differentiating through a sensitivity index

**What it does.** Computes `d S_i / d theta`: the derivative of a sensitivity
index with respect to a model parameter, or a parameter of an input
distribution.

**What question it answers.** Not "what is the sensitivity structure" but "how
do I **change** it". For example: which design change most reduces the model's
sensitivity to a parameter I cannot control?

**Why this is the flagship.** A dedicated search found no prior work.
`"Sobol index" AND "automatic differentiation"` returns zero arXiv hits. No GSA
library exposes it. The reason is mundane: in a NumPy library an index is the
output of a long procedural pipeline that nobody can differentiate. Here it is
already a differentiable function of the samples.

**How to phrase the claim. [corrected]** The broad phrasing does not survive
scrutiny. Use one of these instead:

> The first to compute exact gradients of **variance-based and
> moment-independent global** sensitivity indices with respect to model and
> distribution parameters.

or

> The first GSA library in which sensitivity indices are **end-to-end
> differentiable**, enabling gradient-based calibration against a target
> sensitivity profile.

**Four neighbours the paper must distinguish itself from.** State these first;
a reviewer will otherwise raise them.

| Neighbour | What it differentiates |
|---|---|
| Derivative-based measures (DGSM, active subspaces) | the **model** |
| Robust design and reliability-based optimisation | output **moments and failure probabilities** |
| Perturbed-law indices (Lemaitre et al., arXiv:1210.1074) | a **finite perturbation** of the input law, measuring an **output statistic**. Definition 3.1 is a ratio at perturbation size delta, not a derivative. |
| Explanation regularisation in machine learning | **local, per-instance attributions** with respect to **model weights** |

The last is the real near-miss. **Wang, Wang and Inouye, "Shapley Explanation
Networks", ICLR 2021 (arXiv:2104.02297)** embeds a differentiable Shapley
transform in a network and backpropagates through exact Shapley values. Those
are local attributions and model weights, not global indices and distribution
parameters. The distinction holds, but only if we draw it first.

**Which estimators support it. [corrected]** An earlier draft said rank-based and
sorting-based estimators have zero gradients. That conflates two different
operations.

| Operation | Behaviour | Consequence |
|---|---|---|
| **Sorting** | piecewise **linear** | differentiable almost everywhere; gradients are generally non-zero. Only ties are a problem. |
| **Ranking** | piecewise **constant** | derivative zero or undefined |
| **Empirical CDF** (PAWN) | piecewise constant | zero almost everywhere. The genuinely dead case. |
| **Histogram binning** (Borgonovo delta) | bin membership constant, within-bin statistics smooth | gradient is **non-zero but biased**, with jumps at bin edges. Fix with a kernel density estimate, not a relaxation. |

So a rank-based Sobol estimator is **piecewise constant in `X`** but **smooth in
`Y`**. Gradients flowing through the model output survive; gradients routed
through `X` do not. That is a far more useful statement than "gradients are
zero". For the constant cases, soft relaxations exist: Cuturi, Teboul and Vert
(2019) and Blondel et al. (2020) for soft ranks; smooth kernel CDFs for PAWN.

**How the distribution path works.** Samples are generated as
`X = F^{-1}(u; theta)` from quasi-random points `u`. That is already a
reparameterisation, so it is differentiable in `theta`. Failure cases, corrected:

1. **Discrete and categorical inputs** — a real failure. The inverse CDF is a
   step function. Use a score-function estimator or a Gumbel-Softmax relaxation.
2. **A non-differentiable inverse CDF** — real but usually fixable. For Gamma,
   Beta, Dirichlet and von Mises, use implicit reparameterisation gradients
   (Figurnov, Mohamed, Mnih, NeurIPS 2018), which differentiate the CDF
   implicitly and need no explicit inverse.
3. **[corrected] "The support depends on theta" is not itself a failure.** For
   `Uniform(0, theta)` the map `x = theta * u` is perfectly smooth. The real
   failure is a **discontinuous integrand at a theta-dependent boundary**, where
   a boundary term is lost.
4. **Correlation parameters** are fine. Differentiate through a Cholesky factor,
   so the matrix stays positive definite.

**Quasi-random sampling adds one subtlety.** The points `u` do not depend on
`theta`, so the chain rule is untouched. But:

- A deterministic quasi-Monte Carlo estimator is a quadrature rule, not an
  unbiased average, so "unbiased gradient" does not apply. Convergence needs the
  **derivative** integrand to have bounded variation, which is a stronger
  requirement than for the index itself.
- **Randomised quasi-Monte Carlo (scrambling) restores unbiasedness** for both
  the estimate and the gradient. Make it the default.
- **Hold the scramble fixed across an optimisation.** Re-scrambling each step
  makes the objective a moving target and destroys the variance reduction.

**Bootstrap. [corrected]** "Bootstrap resampling is non-differentiable" is not
quite right — the resample indices do not depend on `theta`, so each replicate
is smooth given a fixed index set. The real reasons to differentiate only the
point estimate are that interval endpoints are order statistics, giving a
high-variance discontinuous gradient, and that optimising a confidence interval
is conceptually wrong anyway. Keep the rule; change the justification.

**Papers.** None for the sensitivity-index case; that is the point. Supporting:

- Kingma, D.P. and Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR
  2014*. arXiv:1312.6114
- Rezende, Mohamed, Wierstra (2014). *ICML 2014*. Co-discoverers of the pathwise
  estimator. Omitting them is a noticed oversight.
- Mohamed, Rosca, Figurnov, Mnih (2020). Monte Carlo gradient estimation in
  machine learning. *JMLR* 21(132):1-62.
- Figurnov, Mohamed, Mnih (2018). Implicit reparameterization gradients.
  *NeurIPS 2018*. arXiv:1805.08498

**Oracles.**

- **T0, and worth building deliberately.** For the linear-Gaussian model
  `Y = a^T X` with `X ~ N(0, R)` and `R` a correlation matrix, the index and its
  derivatives are all closed form. With `v = Ra` and `s = a^T R a`:

  ```
  S_i      = v_i^2 / (R_ii * s)
  dS_i/da  = (2 v_i / (R_ii * s)) * [ R e_i - (v_i / s) * v ]
  dS_i/dR  = 2 v_i (e_i a^T)/(R_ii s)
             - v_i^2 (e_i e_i^T)/(R_ii^2 s)
             - v_i^2 (a a^T)/(R_ii s^2),   symmetrised as (G + G^T)/2
  ```

  These were verified against finite differences to 5e-11. Differentiate through
  a Cholesky parameterisation so `R` stays positive definite.
- **T4**: central finite differences of the index at several step sizes, on any
  model where the index is analytic.

**Effort.** Medium. Most of the work is the API and documenting the limits
honestly.

---

## 6. Release 1.0 — coverage and stability

### 6.1 Nearest-neighbour Shapley effects from given data

**What it does.** Shapley effects need conditional expectations `E[Y | X_S]` for
every subset `S`. The usual estimator samples them, which needs a designed
experiment and a known input distribution. Broto, Bachoc and Depecker replace
the sampling with nearest neighbours: to approximate conditioning on `X_S`, find
the points whose `X_S` values are closest and average their outputs.

**What question it answers.** "What are the Shapley effects of the dataset I
already have?" No design. No known input distribution. Correlated inputs are
handled naturally, because the data carries the dependence.

**Why it matters.** jaxgsa computes Shapley effects only from a PCE or HDMR
surrogate. R `sensitivity` has three given-data variants. No permissive-licence
package in any language has this.

**[corrected] Only one variant is truly given-data.** The paper offers a `knn`
variant and a `mix` variant. The `mix` variant still calls the model at new
inputs. Implement `knn`.

**[corrected] There is a hard dimension ceiling. This is the main risk.**
Corollary 1 of Broto et al. gives the rate:

```
error = o_p( N^(-1 / (2 * (p - |u|))) )
```

The governing quantity is **how many variables you condition on**, not the
ambient dimension. Conditioning on `p-1` variables gives about `N^(-1/2)`, which
is good. But Shapley effects need **all** subsets, including `|u| = 1`, which
gives `N^(-1/(2(p-1)))`. At `p = 10` that is `N^(-1/18)`: halving the error needs
about 260,000 times the samples.

**A live contradiction in the literature.** Huang and Joseph (2025), section 3.2,
summarise Broto's rate as "almost `N^(-1/2)`, independent of the problem
dimension". That contradicts Broto's own Corollary 1, since their estimator is
the `|u| = 1` case. Azadkia and Chatterjee independently obtain
`n^(-1/(p+q))` and believe it is the true rate. **Treat dimension-independence
as unsupported. Benchmark before relying on it.** Practical envelope: `p` up to
about 5 to 8.

**Consistency conditions.** A bounded, almost-everywhere-continuous density with
respect to a product measure, and bounded `f`. The **rate** result needs more:
`f` continuously differentiable, compact support, and a density **bounded
below** — which excludes untruncated Gaussians and even triangular inputs.

**The architectural decision.** Subset aggregation is vectorised algebra and maps
well. The cost is the nearest-neighbour searches. Brute-force vmapped distance
matrices are fine to about `N = 10^4`. Beyond that you need an approximate
nearest-neighbour index, and JAX has none natively. Decide this explicitly.

**Papers.**

- Broto, B., Bachoc, F., Depecker, M. (2020). Variance reduction for estimation
  of Shapley effects and adaptation to unknown input distribution. *SIAM/ASA
  JUQ* 8(2):693-716. doi:10.1137/18M1234631
- Song, E., Nelson, B.L., Staum, J. (2016). Shapley effects for global
  sensitivity analysis: theory and computation. *SIAM/ASA JUQ* 4(1):1060-1083.
  The baseline, needing conditional sampling.
- Owen, A.B. and Prieur, C. (2017). On Shapley value for measuring importance of
  dependent inputs. *SIAM/ASA JUQ* 5(1):986-1002.

**Oracles.**

- **T0, but exponential.** Owen and Prieur Theorem 2 gives a closed form for
  Shapley effects of the linear-Gaussian model:

  ```
  phi_j = (1/d) * sum over u subset of -j
          of  C(d-1, |u|)^-1 * cov(x_j, x_{-u}' beta_{-u} | x_u)^2 / var(x_j | x_u)
  ```

  This is exact, but it is a sum over `2^(d-1)` subsets. **It is a small-`d`
  oracle**, usable to about `d = 15`, not a general reference implementation.
  Useful cross-check: this equals the population LMG (Lindeman, Merenda, Gold)
  R-squared decomposition. Fully explicit non-exponential forms exist only for
  `d = 2` and a block-structured `d = 3` (Iooss and Prieur 2019).
- **T3**: R `sensitivity::shapleysobol_knn` and `sobolshap_knn`.
- **T4**: agreement with the existing surrogate-based Shapley module where both
  apply.

**Effort.** Medium.

**Two follow-ons that share the machinery, with corrections.**

1. **Azadkia and Chatterjee (2021)**, *Annals of Statistics* 49(6):3070-3102.
   **[corrected] This is not a total-order index for deterministic models.**
   Their Theorem 2.1 says `T = 1` if and only if `Y` is a measurable function of
   `Z` given `X`. For a deterministic `Y = f(X)` that is always true, so
   `T` is identically 1 and the measure degenerates to a 0/1 screening flag. The
   correct GSA use is the **unconditional** version: `1 - T(Y, X_{-i})` is a
   **Cramer-von Mises** total index, not a Sobol one. Also, "distribution-free"
   is imprecise: what is proven is assumption-free **consistency**, not a
   distribution-free null. Their Remark 8 states that a consistent
   conditional-independence test with bounded level is provably impossible.
2. **Proportional marginal effects**, Herin, Il Idrissi, Chabridon, Iooss (2024),
   *SIAM/ASA JUQ* 12(2):667-692. **[corrected] The Shapley behaviour it fixes is
   not an axiom violation.** In their Example 1, "the Shapley's joke", `Y = X1`
   with correlated `(X1, X2)` gives `Sh2 = rho^2 / 2 > 0` for an input the model
   never uses. But `X2` is a genuinely non-null player in the Sobol game, since
   `v({2}) = rho^2`. Shapley behaves exactly as axiomatised. The mismatch is that
   equal-split redistribution is wrong for **factor fixing** while being
   defensible for **factor prioritisation**. PME swaps balanced contributions for
   equal proportional gains, and loses additivity. It still costs `2^d - 1`
   evaluations.
3. **Huang and Joseph (2025)**, *Technometrics* 67(4):573-589. **[corrected] This
   targets the total Sobol index, not Shapley effects.** Its noise correction is
   real: it estimates the noise-only total effect using all `p` coordinates as
   the conditioning set and subtracts it. Practical detail: `N_I = 2` is more
   robust than Broto's `N_I = 3` under noise.

### 6.2 Optimal-transport parity with gsaot

**Context.** `gsaot` is the direct competitor: an R package by Leonardo Chiani,
reviewed by Borgonovo, Plischke and Tavoni. jaxgsa already has the
Wasserstein-Bures decomposition.

**What to add.**

1. **An exact solver.** POT provides `ot.emd`, a network-simplex solver, and POT
   is already a development extra.
2. **An irrelevance threshold.** The expected index value for a genuinely
   non-influential input. **[corrected]** It is computed by a **dummy-variable
   Monte Carlo**: draw a fresh input column independent of `y`, then run the
   normal estimator on it. The output is never permuted. This is the
   optimal-transport form of section 4.2, so the two should share an interface.
3. **Sensitivity maps.** **[corrected]** These are a per-output-component
   one-dimensional sweep returning a matrix of indices. They are **not** a
   transport map, despite the name.
4. **A user-supplied ground cost.** The argument in gsaot is `cost`, accepting
   `"L2"` or a function. Note it is available on `ot_indices` but **not** on
   `ot_indices_wb`.

**[corrected] Two items to remove from the plan.**

1. **The entropic bound does not bound the Sinkhorn bias.** An earlier draft said
   `entropic_bound()` bounds the regularisation bias. It does not. It returns
   `K_eps(P_Y, P_Y) / M^K[Y]` — the **lower bound of the entropic index itself**,
   the value the index takes under independence. Its signature never sees `x`. It
   says nothing about the gap between the regularised and exact costs. It also
   **cannot** serve as an independence test: the gsaot paper states that
   dependent inputs can attain the bound.
2. **`higher_order_terms` is not released.** It exists only on gsaot's GitHub
   master, in an unnumbered section of NEWS.md, and master's DESCRIPTION carries
   the same version string as the CRAN release. Do not plan against it. It is a
   two-line difference of two existing results in any case.

**Three technical facts to get right.**

1. **The Sobol relation carries a factor of one half.** For squared-Euclidean
   cost the index splits as `iota^K = iota^V + iota^Sigma + iota^Gamma`, and the
   advective term is `iota^V = S_i / 2`, because `M^K[Y] = 2 * Var(Y)` in the
   scalar case. Writing "recovers the Sobol index" without the half reproduces
   the factor-two normaliser trap already recorded in the project notes.
2. **Wasserstein-Bures is a lower bound, not an exact split.** The
   advective-plus-diffusive decomposition is exact **only for Gaussians**. In
   general it is Gelbrich's bound:
   `W2^2 >= ||m1 - m2||^2 + B^2(S1, S2)`, with a non-negative remainder that
   vanishes for elliptical laws sharing a generator. Describe it as a
   semi-metric and a lower bound.
   The closed form is
   `W2^2 = ||m1-m2||^2 + tr(S1) + tr(S2) - 2 tr[(S1^0.5 S2 S1^0.5)^0.5]`.
   Prefer that symmetrised nesting numerically over `tr[(S1 S2)^0.5]`.
3. **Log-domain Sinkhorn is not optional in float32.** `exp(-C/eps)` underflows
   to zero once `C/eps` exceeds about 745 in float64 and about **88 in
   float32**, after which Sinkhorn divides by zero. float32 is the JAX default.

**Do not write "Sinkhorn converges to the exact solution".** Costs converge
unconditionally, but **plans** converge only along subsequences when the optimal
plan is not unique, and the entropic solution selects the maximum-entropy plan.
Also **do not quote the `O(eps log(1/eps))` rate**: that needs absolutely
continuous, compactly supported marginals. A Monte Carlo pipeline transports
**empirical** measures, where the rate is exponential in `1/eps` instead
(Cominetti and San Martin, 1994).

**Papers.**

- Borgonovo, Figalli, Plischke, Savare (2025). Global sensitivity analysis via
  optimal transport. *Management Science* 71(5):3809-3828.
  doi:10.1287/mnsc.2023.01796. Online first in August 2024, which is why "2024"
  circulates. Note the accent in Savare.
- Chiani, Borgonovo, Plischke, Tavoni (2025). gsaot: an R package for optimal
  transport-based sensitivity analysis. arXiv:2507.18588
- Gelbrich, M. (1990). *Mathematische Nachrichten* 147:185-203.

**Oracles.**

- **T2**: POT `ot.emd`. Sinkhorn must converge to the exact cost as the
  regularisation goes to zero. POT is MIT, with a vendored LEMON-licensed
  network-simplex component — record it that way in a dependency audit.
- **T0**: the Wasserstein-Bures distance between two Gaussians, closed form
  above.
- **T3**: R `gsaot`. Note a live bug: `entropic_bound(solver = "sinkhorn_log")`
  errors in both the CRAN release and master, because the dispatch switches on a
  string that `match.arg` never produces.

**Effort.** Low to medium.

### 6.3 Generalised Sobol indices for functional output

**What it does.** jaxgsa already accepts `(N, T, K)` outputs and returns an index
per output point. That is a **curve** of indices. A generalised index is a
**single number per input for the whole output**.

**How it works.** The Hoeffding decomposition gives a decomposition of the output
**covariance matrix**: `Sigma = C_u + C_{~u} + C_{u,~u}`. Scalarising with a
matrix `M` gives `S^u(M) = Tr(M C_u) / Tr(M Sigma)`. The canonical index takes
`M = Id`, so:

```
S^u = Tr(Cov(E[Y | X_u])) / Tr(Cov(Y))
```

**What question it answers.** "Which input matters for the trajectory as a
whole?" A per-timestep curve cannot answer that, because indices at different
times are not comparable when the output variance changes with time.

**Three corrections to an earlier draft.**

1. **[corrected] There is no total effect defined.** Gamboa et al. define
   `S^u`, `S^{~u}` and `S^{u,~u}`. A total index must be built as `1 - S^{~u}`.
2. **[corrected] Isometry invariance is not just a property, it is a converse.**
   Proposition 3.1 gives `S^u(Of) = S^u(f)` for orthogonal `O`. Proposition 3.2
   proves `M = lambda * Id` is the **only** choice with that invariance. So the
   trace form is **forced, not a convention**. That is good news: the index is
   canonical.
3. **[corrected] Rescaling invariance holds only for a single common scalar.**
   `S^u(lambda f) = S^u(f)`. **Per-component rescaling is not invariant.** So
   `Tr(C_u) / Tr(Sigma)` is **not unit-invariant** across outputs with different
   units. Ship a documented standardisation policy for mixed-unit outputs.

**Scalar output.** At `K = 1` the generalised index reduces **exactly** to the
ordinary Sobol index, for **every** non-zero `M`, not only the identity. This is
a sharp consistency test.

**Papers.**

- Gamboa, Janon, Klein, Lagnoux (2014). Sensitivity analysis for
  multidimensional and functional outputs. *Electronic Journal of Statistics*
  8(1):575-603. doi:10.1214/14-EJS895. Open access; arXiv:1311.1797.
- Alexanderian, Gremaud, Smith (2020). *RESS* 196:106722.
  doi:10.1016/j.ress.2019.106722
- Lamboni, M. (2018). Multivariate sensitivity analysis: minimum variance
  unbiased estimators of the first-order and total-effect covariance matrices.
  *RESS*.

**Oracles.**

- **T2, and close to ideal**: UQpy `GeneralisedSobolSensitivity` — MIT licensed,
  pure Python, importable as
  `from UQpy.sensitivity import GeneralisedSobolSensitivity`. Three traps: the
  class name uses British spelling but its result attributes are Americanised
  (`generalized_first_order_indices`); the class was called `GeneralisedSobol` in
  earlier 4.x releases; and it builds `C_u` as the cross-covariance block
  `Cov(Y^u, Y)`, which equals `Cov(E[Y|X_u])` only **in expectation**, so
  finite-sample estimates can go slightly negative and UQpy does not clamp. **Do
  not treat it as a bit-exact oracle.**
- **T0**: for a multi-output linear model the covariance matrices are closed
  form.
- **T4**: at `K = 1`, the generalised index must equal the ordinary Sobol index
  exactly.

**Effort.** Low.

### 6.4 More input distributions

**Current state.** `Problem` supports uniform, Gaussian and categorical only.

**The finding.** No distribution library supplies what GSA needs. Per marginal,
jaxgsa needs: the inverse CDF, the density, moments, support bounds, **the
orthogonal polynomial family and its three-term recurrence** for PCE, **the
Poincare constant** from section 5.1, and **the conditional-copula transform**
for Kucherenko and VKOGA. The last three exist in no distribution package in any
language, so a side table is required whichever dependency we choose.

**The decision.** Keep a small internal `Marginal` protocol carrying the
GSA-specific extras. Supply **adapters** for external distributions.

| Source | Role |
|---|---|
| `scipy.stats` frozen distributions | Already a dependency. About 100 marginals. Host-side and not traceable, which is acceptable: sampling happens once, outside `jit`. |
| `numpyro.distributions` | Optional extra. JAX-native and actively maintained. |
| `tensorflow_probability.substrates.jax` | Reject. No release in 21 months. |
| `distrax` | Reject. Machine-learning focused, missing most marginals. |

Supply first-class native specs only where the extras are needed: uniform,
Gaussian, lognormal, triangular, truncated normal, beta, exponential, and
discrete or empirical.

**Result.** No new required dependency. A user who wants a lognormal input writes
`scipy.stats.lognorm(s=0.5)`.

**Oracles.**

- **T2**: `scipy.stats` for the inverse CDF, density and moments.
- **T0**: orthogonality of the PCE basis. For each new marginal, the Gram matrix
  of the polynomial basis under that measure must equal the identity to numerical
  precision. This catches a wrong recurrence immediately.
- **T1 and T3**: the Poincare constants, per section 5.1.
- **T4**: round-trip the copula conditional transform, and check the sampled
  marginals against the requested ones with a Kolmogorov-Smirnov test.

**Effort.** Medium, spread over many small pieces.

### 6.5 Fix the linear-Gaussian test fixture

`tests/_linear_gaussian.py` is the project's strongest analytic oracle. Two
things must be stated or fixed before it carries more weight.

**1. The formulas assume a correlation matrix.** The general forms are:

```
S1_i = (Ra)_i^2 / (R_ii * a'Ra)
ST_i = a_i^2 / ((R^-1)_ii * a'Ra)
```

The fixture uses `R` with unit diagonal, so the current code is correct. **State
the assumption explicitly**, or use the general form. Verified numerically:
without the `R_ii` terms, a general covariance gives `S1 = 1.208`, which is
impossible.

Implementation note: `ST_i = a_i^2 / ((R^-1)_ii * a'Ra)` gives all `d` total
indices from one Cholesky factorisation in `O(d^3)`, with no per-input inverse.

**2. The pair mixes two conventions.** `S1` is the **full** first-order index
`S_i^full`; `ST` is the **independent** total index `ST_i^ind`. So the usual
invariant `S_i <= ST_i` **does not hold under dependence**. Any test that assumes
it will fail for the right reason and look like a bug. Document the convention in
the module docstring.

**3. Do not assert the Shapley sandwich unconditionally.** The ordering
`ST_i^ind <= Sh_i <= S_i^full` held in every random draw the verification pass
tried, but Iooss and Prieur show it **can** invert, and give the `d = 2`
condition. Use it as a diagnostic, not a test invariant.

### 6.6 Release engineering

A 1.0 is a promise about stability. Required before the tag:

- **Versioning.** Semantic versioning, with a documented public API surface.
- **Deprecations.** Follow NEP 23. Each one names the version, gives an
  alternative, raises `DeprecationWarning` with the correct `stacklevel`, appears
  in the release notes, and survives at least two releases, about one year.
  Deprecations **must not** land in a bugfix release.
- **Support windows.** Adopt SPEC 0. Drop Python three years after release and
  dependencies after two. So Python 3.12 and above, with 3.13 and 3.14 in the
  matrix.
- **Random numbers.** Adopt SPEC 7. Take an `rng` argument. Do not use a global
  seed.
- **Typing.** Ship the `py.typed` marker.
- **CI.** Keep `ruff check` and `ruff format` as separate gates. Add a
  reproducible benchmark script to the repository, because performance is the
  selling point and a reviewer must be able to confirm it. State plainly what is
  and is not tested on a GPU; no free GPU runner exists.
- **Packaging.** PyPI Trusted Publishing, configured before the next release. A
  conda-forge feedstock through staged-recipes.
- **Citation.** `CITATION.cff` with a `preferred-citation` block, and a Zenodo
  concept DOI.
- **Licence.** Keep MIT. Do not move to GPL; it would deter the industrial JAX
  users this package targets.

---

## 7. Papers

Two, with different jobs.

**A software paper, in JOSS.** The bar has risen since SALib passed in 2017. The
current checklist requires at least six months of public development history,
evidence of sustained rather than sudden development, and engagement from users
outside the author group. The paper must contain five named sections: statement
of need; state of the field, answering "why did you not contribute this to
SALib?"; software design with trade-offs weighed; a research impact statement;
and an AI usage disclosure. **Do not submit** until the six-month and
external-user conditions are met.

**A methodology paper, in RESS.** On differentiating through a sensitivity index,
section 5.4. That is the part with no prior art, and RESS is the field's main
venue. A software paper makes jaxgsa a citable tool; a methodology paper makes it
a citable method.

**Before either:** an arXiv preprint, a Zenodo DOI, and `CITATION.cff`.

**Before publishing the novelty claim**, re-run the search against Google Scholar
and ScienceDirect for the robust-design and reliability-based-optimisation
literature. The verification pass covered arXiv, GitHub and direct fetches
thoroughly, but its search budget ran out before that specific area. A hit there
would at worst force the sentence "moments and failure probabilities were already
differentiated; indices were not". It does not threaten the narrowed claim.

**An adoption note.** `sensobol` was published in the *Journal of Statistical
Software* and has 16 GitHub stars. SALib invented no method, was published in
JOSS, and has 1,003 stars and 677,000 downloads per month. Venue prestige did not
drive adoption. A short copy-pasteable API, method breadth under one interface,
and a domain beachhead did.

---

## 8. Out of scope for 1.0

| Item | Reason |
|---|---|
| Plotting module | Decided against. |
| Command-line interface | Decided against. |
| Convergence analysis API | Deferred. See the note below. |
| PoinCE, gradient-enhanced Poincare chaos | Deferred. A strong fit for JAX, since the method was designed for cheap gradients its authors did not have. Luthen, Roustant, Gamboa, Iooss, Marelli, Sudret (2023), *IJUQ* 13(6):57-82, arXiv:2107.00394. |
| Distance correlation as a separate module | Rejected as misleading. It is HSIC with a distance kernel, so it ships as a kernel option. See 4.6.2. |
| Target and conditional HSIC | Deferred, but design the output path so a value transform can be added. See 4.6.2. |

**A note on the convergence API, for whenever it is revisited.** The idea is to
recompute indices on nested prefixes of an already-evaluated sample, at no extra
model cost. Two facts must be respected:

1. **Prefixes are valid only at powers of two.** SciPy's documentation is
   explicit: Sobol sequences "lose their balance properties if one uses a sample
   size that is not a power of 2, or skips the first point, or thins the
   sequence."
2. **A Saltelli design cannot be halved by slicing the flat output array.** The
   design is a base sample split into A and B, plus `k` cross matrices, stacked
   into an expanded run matrix. Slicing the expanded matrix mixes complete A rows
   with partial cross-matrix blocks and produces meaningless indices. To halve it
   correctly, take rows `0 : N/2` of A, of B, and of **every** cross matrix
   independently, then reassemble. Expose nesting at the **sample** level, never
   by slicing `Y`.

Reference for what to report: Sarrazin, Pianosi, Wagener (2016), *EMS*
79:135-152. Their finding is that "convergence of screening and ranking can be
reached before sensitivity estimates stabilize", so the three must be reported
separately. They also warn that convergence is case-dependent, so fixed sample-
size rules should not be given.

---

## 9. Undecided

**Non-Gaussian copulas.** GlobalSensitivity.jl, with Copulas.jl, computes Shapley
effects for Clayton, Frank, Gumbel and t copulas, with exact per-family
conditional sampling. That is strictly more general than jaxgsa's Gaussian-only
support, and it is the capability gap a reviewer notices first. **This needs a
decision.**

**Terminology: "pick-freeze".** Project policy is to avoid the term and write
"Saltelli column-swap scheme". The verification pass notes that Janon et al. use
"Sobol Pick-Freeze" as the formal name, and that the two are **not** synonyms:
pick-freeze names the sampling principle, while Saltelli 2002 and 2010 name the
specific `N(d+2)` bookkeeping. The substitution is therefore not always
semantically neutral. **Decide whether to keep the blanket substitution or allow
the term where it is technically correct.**

---

## 10. Summary table

| Item | Release | Paper | Strongest oracle | Effort |
|---|---|---|---|---|
| eFAST frequency-plan duplication | 0.9 | - | T4 shared-plan test | low |
| eFAST missing `else` branch | 0.9 | - | T0, restores an existing claim | very low |
| Failed-evaluation policy | 0.9 | - | T4 behavioural | low |
| Rank-based estimators | 0.9 | Chatterjee 2021; Gamboa+ 2022 | T2 `scipy.stats.chatterjeexi` | very low |
| Dummy-parameter test | 0.9 | Khorashadi Zadeh+ 2017 | T4 by construction | very low |
| Intervals in all modules | 0.9 | Efron 1987; Janon+ 2014 | T4 coverage | low |
| More Sobol estimators | 0.9 | Jansen 1999; Azzini+ 2021 | T2 OpenTURNS, four-way | low |
| Regional sensitivity analysis | 0.9 | Spear and Hornberger 1980 | T3 SAFEpython or T2 SALib | very low |
| HSIC p-values; dCor kernel | 0.9 | De Lozzo 2016; Sejdinovic 2013 | T2 OpenTURNS | low |
| Poincare: oracle plus PCE route | 0.10 | Roustant+ 2017; Sudret 2015 | T3 R `PoincareOptimal` | low |
| Forward vs reverse mode selection | 0.10 | - | T4 equality of the two paths | low |
| Active subspaces | 0.10 | Constantine+ 2014; Diaz 2017 | T0 linear and quadratic | very low |
| Crossed DGSM | 0.10 | Roustant+ 2014 | T0 polynomial | very low |
| d(index)/d(theta) | 0.10 | none; that is the point | T0 linear-Gaussian derivatives | medium |
| kNN Shapley from given data | 1.0 | Broto+ 2020; Owen and Prieur 2017 | T0 linear-Gaussian, small `d` | medium |
| Optimal-transport parity | 1.0 | Borgonovo+ 2025 | T2 POT | low-medium |
| Generalised Sobol | 1.0 | Gamboa+ 2014 | T2 UQpy (MIT) | low |
| More distributions | 1.0 | - | T2 scipy; T0 orthogonality | medium |
| Linear-Gaussian fixture fixes | 1.0 | Owen and Prieur 2017 | T0 | very low |
| Release engineering | 1.0 | - | - | low |
