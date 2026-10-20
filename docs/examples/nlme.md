# Population PK models (NLME)

This page shows how to run a sensitivity analysis on a nonlinear mixed-effects
(NLME) model, the model type of population pharmacokinetics. It needs no
special API: four short recipes with `kucherenko`, `rank` and plain NumPy
answer the usual questions.

In an NLME model each patient has:

- **covariates** `X`, such as age, weight and creatinine clearance (CrCL).
  They are often correlated.
- **random effects** `eta`, one per kinetic parameter. They describe the
  between-patient variability that the covariates do not explain (IIV, for
  example "30% IIV on CL"). Each `eta_k` is Gaussian with variance
  `omega_k^2`, independent of `X` and of the other random effects.
- **kinetic parameters** `theta = g(X, eta)`, for example
  `CL = CL_pop * (WT/75)^0.75 * exp(eta_CL)`.

The output `Y` is a function of `theta`, for example a concentration or an
AUC.

| Question | Recipe | Method |
|---|---|---|
| How much does each random effect matter? | 1 | `kucherenko` on `(X, eta)` |
| How much do the covariates explain, and how much is IIV? | 2 | NumPy |
| Which covariate matters? | 3 | `kucherenko`, `subsets="all"` |
| Which kinetic parameter matters? | 4 | `rank` |

## The example model

Three standardized covariates with a correlation matrix, three random effects,
and a one-compartment oral model. The output is the log concentration 2 h after
a 100 mg dose.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import jaxgsa

COVS = ["z_age", "z_wt", "z_crcl"]
ETAS = ["eta_CL", "eta_V", "eta_ka"]
OMEGA = {"eta_CL": 0.3, "eta_V": 0.2, "eta_ka": 0.4}
R_COV = np.array([
    [1.0, 0.2, -0.6],
    [0.2, 1.0, 0.3],
    [-0.6, 0.3, 1.0],
])


def kinetics(z):
    """Individual CL, V and ka from covariate z-scores and random effects."""
    z_age, z_wt, z_crcl, eta_cl, eta_v, eta_ka = (z[..., i] for i in range(6))
    log_wt = 0.2 * z_wt  # log(weight / 75 kg)
    log_crcl = 0.3 * z_crcl  # log(CrCL / 90 mL/min)
    cl = 5.0 * jnp.exp(0.75 * log_wt + 0.5 * log_crcl + eta_cl)
    v = 50.0 * jnp.exp(log_wt + eta_v)
    ka = 1.5 * jnp.exp(eta_ka)
    return cl, v, ka


def model(z, dose=100.0, t=2.0):
    """log concentration at time t after an oral dose (one compartment)."""
    cl, v, ka = kinetics(z)
    k = cl / v
    c = dose * ka / (v * (ka - k)) * (jnp.exp(-k * t) - jnp.exp(-ka * t))
    return jnp.log(c)


def gaussian(variance):
    return {"dist": "gaussian", "mean": 0.0, "variance": variance}


corr = np.eye(6)
corr[:3, :3] = R_COV  # covariates correlated, random effects independent
problem = jaxgsa.Problem.from_dict(
    {**{c: gaussian(1.0) for c in COVS}, **{e: gaussian(OMEGA[e] ** 2) for e in ETAS}},
    correlation=corr,
)
```

Age has no direct effect in this model. It acts only through its correlation
with CrCL.

## Recipe 1: how much each random effect matters

Put the covariates and the random effects into one problem and run
[Kucherenko](/examples/kucherenko). The total index `ST` of `eta_k` is the
share of `Var(Y)` that is lost if the IIV on that parameter is removed. Because
each `eta_k` is independent of everything else, `ST` here has its usual
meaning, and it works at any number of inputs.

```python
ks = jaxgsa.kucherenko.sample(problem, base_n=2**13, seed=0, verbose=False)
res = jaxgsa.kucherenko.analyze(ks, model(jnp.asarray(ks.samples)), verbose=False)
for name, st in zip(problem.names[3:], np.asarray(res.ST)[3:], strict=True):
    print(f"{name:7s} ST = {st:.3f}")
```

```text
eta_CL  ST = 0.031
eta_V   ST = 0.380
eta_ka  ST = 0.062
```

At 2 h the concentration depends mostly on V, so the IIV on V matters most.

## Recipe 2: covariates against IIV

The covariate share is `S_X = Var(E[Y | X]) / Var(Y)`. The rest, `1 - S_X`, is
the share of the random effects. Estimate it from `N` patients with `M`
random-effect draws each:

```python
rng = np.random.default_rng(1)
N, M = 20_000, 8  # N patients, M random-effect draws per patient
X = rng.multivariate_normal(np.zeros(3), R_COV, size=N)  # or your real cohort
eta = rng.normal(size=(N, M, 3)) * np.array([OMEGA[e] for e in ETAS])
Z = np.concatenate([np.repeat(X[:, None, :], M, axis=1), eta], axis=-1)
Y = np.asarray(model(jnp.asarray(Z)))  # shape (N, M)

within = Y.var(axis=1, ddof=1).mean()  # E[Var(Y | X)]
between = Y.mean(axis=1).var(ddof=1) - within / M  # Var(E[Y | X])
S_X = between / (between + within)
print(f"covariates explain {S_X:.3f} of Var(Y); random effects {1 - S_X:.3f}")
```

```text
covariates explain 0.523 of Var(Y); random effects 0.477
```

The `- within / M` term removes the noise in each patient's mean, so the
estimate is unbiased for any `M >= 2`.

Do not use `1 - sum(ST of the random effects)` for this. It is biased when
the random effects interact. Here the sum is 0.473 against 0.477.

The covariates can come from anywhere: a registry, a trial population, or a
cohort that passed eligibility rules. This recipe uses only the rows you give
it.

## Recipe 3: which covariate matters

Correlated covariates share their effect, so ask for Shapley effects. Run
Kucherenko on the covariates alone with `subsets="all"`, and give each row
one fresh random-effect draw:

```python
cov_problem = jaxgsa.Problem.from_dict({c: gaussian(1.0) for c in COVS}, correlation=R_COV)
ks_x = jaxgsa.kucherenko.sample(cov_problem, base_n=2**12, subsets="all", seed=0, verbose=False)
Xs = jnp.asarray(ks_x.samples)
eta1 = rng.normal(size=(Xs.shape[0], 3)) * np.array([OMEGA[e] for e in ETAS])
Y_x = model(jnp.concatenate([Xs, jnp.asarray(eta1)], axis=-1))
res_x = jaxgsa.kucherenko.analyze(ks_x, Y_x, verbose=False)

Sh = np.asarray(res_x.shapley().Sh) - (1.0 - S_X) / len(COVS)
for name, sh in zip(COVS, Sh, strict=True):
    print(f"{name:7s} Shapley = {sh:.3f}")
print(f"sum = {Sh.sum():.3f} (S_X = {S_X:.3f})")
```

```text
z_age   Shapley = 0.016
z_wt    Shapley = 0.448
z_crcl  Shapley = 0.059
sum = 0.523 (S_X = 0.523)
```

The random effects act as noise in this design. Shapley effects split that
noise equally over the covariates, so the last line subtracts
`(1 - S_X) / P` from each one, with `S_X` from recipe 2. The corrected
effects then sum to `S_X`. Age gets a small share through its correlation
with CrCL, although the model never reads it.

`subsets="all"` costs `base_n * 2^P` model runs for `P` covariates and
accepts up to 12 covariates.

## Recipe 4: which kinetic parameter matters

`theta` depends on both `X` and `eta`, so the kinetic parameters are
correlated. Their full index `Var(E[Y | theta_j]) / Var(Y)` needs only a
sample of `(theta, Y)`, and [rank](/api/rank) estimates it from given data:

```python
Zf = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 16_384, seed=2))
log_theta = jnp.log(jnp.stack(kinetics(Zf), axis=-1))  # log CL, log V, log ka
theta_problem = jaxgsa.Problem(
    names=["log_CL", "log_V", "log_ka"],
    bounds=[(float(c.min()), float(c.max())) for c in log_theta.T],
)
res_full = jaxgsa.rank.analyze(theta_problem, log_theta, model(Zf), verbose=False)
for name, s1 in zip(theta_problem.names, np.asarray(res_full.S1), strict=True):
    print(f"{name:7s} full index = {s1:.3f}")
```

```text
log_CL  full index = 0.266
log_V   full index = 0.895
log_ka  full index = 0.067
```

The full indices do not sum to 1, because each one includes what the
parameter shares with the others. Read them as "how well does knowing this
parameter predict the output", not as a split of the variance.

## Checked against exact values

On a linear NLME test case with 8 correlated covariates and 8 random effects,
where every index has a closed form, the four recipes gave:

| Recipe | Largest error against the exact value |
|---|---|
| 1, random-effect `ST` | 1e-4 |
| 2, `S_X` (N = 40 000, M = 8) | 2e-4 |
| 3, covariate Shapley | 4e-3 |
| 4, full index (`rank`, N = 8000) | 6e-3 |

## Limits

- **Eligibility rules are not truncation.** If your cohort is selected by
  rules such as "age 18 to 65 and eGFR above 60", the joint law of the
  covariates changes, not only their ranges. Truncating each marginal in a
  `Problem` does not reproduce it. Recipes 2 and 4 use your cohort rows
  directly, so they are correct for it. Recipes 1 and 3 need a joint law that
  `kucherenko` can sample conditionally (a Gaussian copula), so use them for
  the population the copula describes.
- **Random effects must be independent of the covariates.** This is the usual
  NLME assumption.
- **Categorical covariates** such as sex need a custom sampler in recipes 1
  and 3, declared with `problem.with_sampler(f, sampler_dimension=k)`. The
  sampler draws the covariates jointly and given any fixed subset, for example
  sex first and then age and weight from a Gaussian per sex. The Gaussian
  copula alone cannot condition on a categorical input, so `kucherenko`
  refuses a categorical input without a custom sampler.
