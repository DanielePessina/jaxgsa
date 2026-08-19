# ADR 0020: Constraints on methods not yet built

Status: accepted (2026-08-18). Binding when each method is implemented.

## Context

A verification pass read the primary source for every planned method and found
that the obvious reading of several of them is wrong. These findings cost real
time to reproduce, and each one would otherwise be rediscovered by whoever
builds the method — or, worse, not rediscovered. They are recorded as
constraints rather than as a reading list.

Each entry says what to do and why. Where a candidate oracle is unusable, that
is recorded too, because a bad oracle is more expensive than none.

---

## Active subspaces

- **The eigenvalue *gap* does not license truncation.** The **trailing**
  eigenvalues bound the approximation error: Theorem 3.1 gives
  `E[(f - F)^2] <= C1 * (lambda_{n+1} + ... + lambda_m)`. The gap controls
  something else — how *stably* the subspace is estimated from a finite
  sample. A large `lambda_1 / lambda_2` alone certifies nothing.
  Constantine's own paper calls the Poincare inequality in that bound
  "a notoriously loose bound".
- **Activity scores sum to the truncation, not to `d`:**
  `alpha_i(n) = sum_{j=1..n} lambda_j * w_{i,j}^2`. The total-index link
  (Theorem 4.2) is
  `tau_i <= (1/(4*pi^2*V)) * (alpha_i(n) + lambda_{n+1})`; the
  `+ lambda_{n+1}` is **required** whenever `n < m`, and the `1/(4*pi^2)`
  factor is specific to `[-1,1]^m` scaling.
- **`C` is not invariant to input scaling.** Standardise inputs, or the
  eigenvectors report units.
- **Citation trap.** The erratum corrects Theorems 3.2, 3.3, 3.6 and 3.7 (a
  misapplied triangle inequality) plus a sign in eq. 5.3. It does **not**
  touch Theorem 3.1. **Do not cite arXiv:1304.2070** for those four — it
  carries the uncorrected statements. In Zahm et al. the matrix is
  `H = integral (grad f)^T R_V (grad f) dmu` with `R_V` the *output-space*
  metric; the input covariance enters separately, through the pencil
  `(H, Sigma^-1)`.
- **Oracle:** T0. The linear and quadratic cases are exact. ATHENA is a dead
  end (ADR 0004).

## Crossed DGSM

- **It bounds superset importance, not the second-order index:**
  `D_ij <= D_ij^super <= C(mu_i) C(mu_j) nu_ij`. Advertising it as a bound on
  `S_ij` goes through the loose first inequality. Say "superset importance".
- **Do not use GlobalSensitivity.jl's `DGSM(crossed=true)` as an oracle**
  except for `crossedsq` on uniform `[0,1]`. Its `tao` weight is coded
  `(1-3x+x^2)/6` where the literature has `(1-3x+3x^2)/6`, which goes negative
  at `x = 1`; and `tao`/`sigma` assume uniform `[0,1]` but are applied to any
  distribution.
- `jax.hessian` costs about `D` times a gradient, per output, so this stays
  off by default.

## Optimal transport

- **The Sobol relation carries a factor of one half.** For squared-Euclidean
  cost, `iota^K = iota^V + iota^Sigma + iota^Gamma` with `iota^V = S_i / 2`,
  because `M^K[Y] = 2*Var(Y)` in the scalar case. Writing "recovers the Sobol
  index" without the half reproduces the factor-two normaliser bug already
  hit once in this project.
- **Wasserstein-Bures is Gelbrich's lower bound**, exact only for Gaussians
  (and elliptical laws sharing a generator). Describe it as a semi-metric and
  a lower bound. Numerically, prefer the symmetrised nesting
  `tr[(S1^0.5 S2 S1^0.5)^0.5]` over `tr[(S1 S2)^0.5]`.
- **Log-domain Sinkhorn is not optional.** `exp(-C/eps)` underflows to zero
  once `C/eps` exceeds about **88 in float32**, which is the JAX default (745
  in float64). After that Sinkhorn divides by zero.
- **Do not write "Sinkhorn converges to the exact solution."** Costs converge
  unconditionally; *plans* converge only along subsequences when the optimal
  plan is not unique. **Do not quote the `O(eps log(1/eps))` rate** — it needs
  absolutely continuous, compactly supported marginals, and a Monte Carlo
  pipeline transports *empirical* measures, where the rate is exponential in
  `1/eps` (Cominetti and San Martin, 1994).

## Generalised Sobol

- **The trace form is forced, not conventional.** Proposition 3.2 proves
  `M = lambda * Id` is the *only* isometry-invariant choice, so the index is
  canonical. Say so; it answers the first question a reader has.
- **It is not invariant to per-component rescaling**, so mixed-unit outputs
  need a documented standardisation policy.
- **No total effect is defined.** Build it as `1 - S^{~u}`.
- **UQpy is not a bit-exact oracle**: it builds `C_u` as a cross-covariance
  block that equals `Cov(E[Y|X_u])` only in expectation, and it does not
  clamp.

## kNN Shapley from given data

- **There is a hard dimension ceiling, and the literature disagrees about
  it.** Broto's Corollary 1 rate is governed by how many variables are
  conditioned on. Shapley needs `|u| = 1`, giving `N^(-1/(2(p-1)))`: at
  `p = 10`, halving the error needs about 260,000 times the samples. Huang and
  Joseph (2025) §3.2 assert dimension-independence, which contradicts Broto's
  own corollary. Practical envelope: `p` about 5 to 8, and say so in the
  docstring.
- The rate result needs a density **bounded below**, which excludes
  untruncated Gaussians and even triangular inputs.
- Only the `knn` variant is given-data. `mix` still calls the model.

## Differentiability of an index pipeline

- **Sorting is piecewise *linear*** — differentiable almost everywhere, with
  generally non-zero gradients. **Ranking is piecewise constant.** So a
  rank-based estimator is constant in `X` but smooth in `Y`.
- **PAWN's empirical CDF is the genuinely dead case.**
- **Borgonovo's histogram binning gives non-zero but *biased* gradients**,
  with jumps at bin edges. Fix it with a KDE, not with a relaxation of the
  binning.
- **Two QMC rules.** Randomised QMC restores unbiasedness of the gradient, so
  make it the default. The scramble must be **held fixed** across an
  optimisation, or the objective is a moving target.

## HSIC

- **A distance kernel invalidates the Gamma null.** The moment formulas assume
  normalised kernels with `k_ii = 1`; a distance kernel has
  `k(z,z) = rho(z,z0)`. It is a *pair* of kernels, unbounded and base-point
  parameterised, and it cannot be reached by tuning a Gaussian bandwidth.
  Route distance kernels to the spectral or bootstrap null.
- **The median-heuristic quantile position is
  `q = (N^2 + N - 1) / (2*(N^2 - 1))`.** The obvious
  `q = (N + (N^2 - N)/2) / N^2` is wrong: it lands half a slot short and
  biases toward the lower order statistic at even counts — 15% relative error
  at `N = 4`. The correct position reproduces the strict-upper-triangle median
  to zero error for every `N` from 4 to 1024. Using it in place of explicit
  upper-triangle index machinery took working memory from `3.57*N^2` to
  `1.00*N^2`, saving 164 MiB at `N = 4096`.
- **Normalisation is settled**: Gretton eq. (4), the V-statistic `1/m^2`.
  `(m-1)^-2` is *not* an alternative convention for the same quantity.
- **The permutation test's citation is Gretton et al. §3**, not De Lozzo and
  Marrel, whose three routes are asymptotic, spectral and non-asymptotic
  bootstrap.
- **[unverified]** The claim that the HSIC bandwidth heuristic uses a standard
  deviation rather than the median could not be confirmed in the body text of
  Marrel and Chabridon (2021). Do not state it as established.

## Dummy parameter

- **Compute it analytically from the existing sample. Do not append a column.**
  SAFEpython and sensobol both do it at zero extra model cost.
- Cite Khorashadi Zadeh et al. (2017), eqs 3, 4, 12, 13. **Do not cite Pianosi
  et al. (2015)** — it predates the feature. **Do not cite Andres (1997)** as
  the origin: the attribution is unverified, it is a different method, and it
  is absent from Khorashadi Zadeh's own references.

## Poincare

- **Do not attribute the *normalised* bound to Lamboni et al. Theorem 3.1.**
  That theorem is unnormalised, and Lamboni assumes a Boltzmann/log-concave
  measure, which explicitly excludes the uniform density on a finite interval.
  The uniform case comes from Sobol' and Kucherenko's direct argument.
- The bound requires **independent** inputs (a product measure).
- Bibliography: Roustant et al. 2017 is *EJS* 11(2):3081-3119, not
  *Stat. Comp.*; Lamboni 2013 is pages 45-54, not 44-54; there is no
  standalone "Diaz (2017)"; the author order of Reddi et al. (2015) differs
  between venues (arXiv 1406.2083 lists Reddi first, the AAAI proceedings list
  Ramdas first) — pick one venue and be consistent.

## Rank-based estimators (cut)

Planned, then **cut after reading the primary source.** The plan claimed
closed group indices via multivariate nearest neighbours; Gamboa et al. do not
support it. Their eq. (17) assumes `V = X1` real-valued, Remark 4.4 defers the
extension, multivariate nearest neighbours appear only in a post-submission
citation to Broto, the CLT (Thm 4.1) is per-index and not joint, total order
is never discussed, and ties are excluded by assumption with no tie-breaking
rule given. **If this returns, Broto is the citation for anything beyond first
order.**
