# Validation

This page shows how the tests check each method. The aim is that you can see what a number rests on before you trust it.

Most checks compare jaxgsa with a closed-form answer. A few compare it with another library. The rest check that the code agrees with itself. The page lists all three kinds, and it says plainly where the outside evidence is thin.

## Tiers

Each test names its tier in its docstring. A lower number means stronger evidence.

- **T0, closed form.** The expected value comes from a formula worked out on paper, such as the analytic Ishigami indices.
- **T1, reference implementation.** The expected value comes from independent code, such as plain NumPy or POT, run inside the test.
- **T2, cross-estimator.** The expected value comes from another jaxgsa estimator or another library that estimates the same quantity.
- **T3, R package.** The test runs an R package as a black box and compares the numbers. It skips when R is missing.
- **T4, internal consistency.** The test checks invariants, shapes, batching, `jit` and `vmap` agreement, and error handling. No outside number is involved.

Every method also has many T4 tests. The table below lists only the checks that use an outside value. Every method in the table has at least one such check.

## What the table shows

- **Tolerance** is the value in the test code. `atol` is an absolute error. `rtol` is a relative error. A bare number such as `0.05` is an absolute bound on the largest error.
- **Slow** tests are marked `slow`. The default `pytest` run skips them. Run them with `uv run pytest -m slow`.
- Sampling methods carry Monte Carlo error. Their tolerances are wide on purpose. They check that the error is sampling error, not that the code is exact.

## Results by method

| Method | Reference | Tier | Tolerance | Test file |
| --- | --- | --- | --- | --- |
| sobol | Ishigami closed-form S1, ST, S2 (small run) | T0 | `atol` 0.02 (S2: 0.03) | [test_sobol_estimators.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_sobol_estimators.py) |
| sobol | Sobol G-function closed-form S1, ST | T0 | `atol` 0.03 | [test_sobol_estimators.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_sobol_estimators.py) |
| sobol | Ishigami, Sobol G and Oakley-O'Hagan closed forms, large runs; tolerance shrinks with N | T0 (slow) | 0.15 / 0.05 / 0.02 at N = 1024 / 4096 / 16384; 5% to 10% relative for the others | [test_analytical.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_analytical.py) |
| sobol | Linear additive model, exact S1 = ST | T0 | default `assert_allclose` (`rtol` 1e-7) | [test_analytical.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_analytical.py) |
| efast | Ishigami closed-form S1, ST | T0 | `atol` 0.08 | [test_efast.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_efast.py) |
| efast | Linear model: S1 equals the closed form, and ST equals S1 | T0 | `atol` 0.08 (S1), 0.05 (ST vs S1) | [test_efast.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_efast.py) |
| kucherenko | Linear Gaussian with correlated inputs, closed-form S1, ST and variance | T0 | `atol` 5e-3; variance `rtol` 2e-2 | [test_kucherenko.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_kucherenko.py) |
| kucherenko | Closed and total indices of all 15 subsets of 4 correlated inputs; Shapley effects by exact permutation | T0 | `atol` 0.015 | [test_kucherenko_subsets.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_kucherenko_subsets.py) |
| kucherenko | R `sensitivity::shapleyLinearGaussian`; R `pme_knn` (loose) | T3 (slow) | `atol` 0.015; `atol` 0.08 | [test_kucherenko_subsets.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_kucherenko_subsets.py) |
| morris | Linear model: mu* equals the coefficients, mu equals mu*, sigma is zero | T0 | `atol` 1e-5 | [test_morris.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_morris.py) |
| morris | Ishigami: mu* ranking follows the analytic ST ranking; x3 reads as nonlinear | T0 (ranking only) | exact rank order | [test_morris.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_morris.py) |
| morris | Scaling law under `Y -> a*Y + b` | T0 | `rtol` 1e-4 to 1e-3 | [test_morris.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_morris.py) |
| dgsm | Linear model: exact nu, mu, and the lower bound against analytic ST | T0 | `atol` 1e-5 (nu, mu); `rtol` 0.08 (bound) | [test_dgsm.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_dgsm.py) |
| dgsm | Truncated-normal variance used for the Poincare constant, against `scipy.stats.truncnorm` | T2 | `rtol` 1e-11 | [test_dgsm_followups.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_dgsm_followups.py) |
| borgonovo | Linear Gaussian closed-form delta and S1 | T0 | `atol` 0.04 (delta), 0.05 (S1) | [test_borgonovo.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_borgonovo.py) |
| pawn | `Y = x1`: closed-form KS distance per bin, for the median and max statistics | T0 | `abs` 0.03 on the active input; below 0.15 on the others | [test_pawn.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_pawn.py) |
| hsic | NumPy V-statistic oracle built from scratch in float64 (R2, T and raw HSIC) | T1 | `rtol` 1e-8 | [test_hsic.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_hsic.py) |
| hsic | Population HSIC of a Gaussian pair (closed form) against the V-statistic | T0 | relative 0.15 (default), 0.05 (slow) | [test_hsic.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_hsic.py) |
| hsic | Sobol G ranking of R2 and T | T0 (ranking only) | rank order | [test_hsic.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_hsic.py) |
| optimal_transport | Gaussian additive and Gaussian linear models, closed-form index, advective and diffusive parts | T0 | `atol` 0.05 (N = 4096); 0.015 to 0.02 (N = 32000, slow) | [test_optimal_transport.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_optimal_transport.py) |
| optimal_transport | Correlated Gaussian inputs, conditional-normal formula | T0 (slow) | `atol` 0.02 | [test_optimal_transport.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_optimal_transport.py) |
| optimal_transport | Multivariate index rebuilt from POT's log-domain Sinkhorn solver | T1 (slow) | `rtol` 1e-4 (OT cost), 1e-6 (advective part) | [test_optimal_transport.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_optimal_transport.py) |
| optimal_transport | S1 against the Borgonovo S1 estimate | T2 | `atol` 0.03 | [test_optimal_transport.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_optimal_transport.py) |
| rank | Ishigami S1; linear Gaussian S1 with a copula and with a custom sampler | T0 | `atol` 0.03 | [test_rank.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_rank.py) |
| rank | A written-out loop of the paper's formula (eq. 20) | T1 | `atol` 2e-5 | [test_rank.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_rank.py) |
| rank | R `sensitivity::sobolrank` on Ishigami data, after the exact `n/(n-1)` rescale | T3 (slow) | `atol` 2e-4 | [test_rank.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_rank.py) |
| pce | Linear additive model: S1, ST, S2 and the fit | T0 | `atol` 0.02 with `rtol` 0.05 | [test_pce.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_pce.py) |
| pce | Ishigami closed-form S1, ST and S2 (order 8) | T0 | `atol` 0.01 | [test_pce.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_pce.py) |
| pce | Hermite and Legendre bases are orthonormal under Gauss quadrature | T0 | `atol` 1e-6 | [test_pce.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_pce.py) |
| hdmr | Ishigami closed-form S1 and ST (order 2) | T0 | 30% relative; an index that should be zero stays below 0.1 | [test_hdmr.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_hdmr.py) |
| hdmr | Ishigami with `n_intervals=4` and N = 4096, S1, ST and S2[x1, x3] | T0 | `atol` 0.05 (S1), 0.06 (ST, S2) | [test_hdmr_order2.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_hdmr_order2.py) |
| hdmr | Indices do not change under `Y -> 3Y + 17` | T0 | `rtol` and `atol` 2e-4 | [test_hdmr.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_hdmr.py) |
| shapley | Linear model closed-form Shapley effects, PCE and HDMR backends | T0 | `atol` 0.05; effects sum to 1 within 1e-5 | [test_shapley.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_shapley.py) |
| shapley | Ishigami closed-form Shapley effects, PCE backend (order 8) | T0 | `atol` 0.04 | [test_shapley.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_shapley.py) |
| shapley | Correlated Gaussian pair, closed form (2/7, 5/7), HDMR backend with the correlative term | T0 | `atol` 0.06 | [test_shapley.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_shapley.py) |
| vkoga | Linear Gaussian with correlated inputs, closed-form S_TC, S_TU and variance | T0 | `atol` 1e-2 (S_TC), 2e-2 (S_TU); variance `rtol` 5e-2 | [test_vkoga.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_vkoga.py) |
| vkoga | Additive model has no interaction part | T0 | `atol` 3e-2 | [test_vkoga.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_vkoga.py) |
| vkoga | Custom Gaussian sampler recovers the closed form; Ishigami under a custom copula agrees with the built-in copula | T0, T2 | `atol` 0.04; 0.06 (default tier) | [test_vkoga_custom_sampling.py](https://github.com/DanielePessina/jaxgsa/blob/master/tests/test_vkoga_custom_sampling.py) |

## Where the evidence is thinner

- **Most evidence is T0.** A closed form needs a model with a known answer. It checks the maths on a few test functions. It does not prove the code is right on your model.
- **dgsm, pawn, borgonovo and morris** have a small number of outside checks. Each rests on one or two closed-form cases. For morris, the Ishigami check tests the rank order, not the values. For pawn, the closed form covers one active input.
- **hdmr** has wide tolerances (30% relative in the basic check, 0.05 to 0.06 with `n_intervals=4`). The fit is a polynomial approximation. A script compares it with a direct-form estimator by hand (see below). No test runs that comparison.
- **hsic** has no comparison with an outside library. Its checks are a NumPy re-derivation and a population closed form.
- **Only rank, kucherenko and optimal_transport** compare with an outside package (R or POT) inside the test suite, and those tests are slow and skip when the package is missing. Live comparisons with SALib, POT and SciPy for sobol, morris, borgonovo, pawn and optimal_transport are in a separate script (see below).
- **Nothing here is a proof.** The tests run on a few functions, at fixed seeds, in float32 and float64. They cannot cover every model.

## External oracles

An oracle is a source of truth outside jaxgsa. The oracles never ship in the package. CI does not run them.

| Oracle | Used for | How to run |
| --- | --- | --- |
| POT (`ot`), a development extra | `optimal_transport` multivariate index (T1, slow) | `uv run --extra dev pytest -m slow tests/test_optimal_transport.py` |
| R `sensitivity` (`sobolrank`, `shapleyLinearGaussian`, `pme_knn`) | `rank`, `kucherenko` Shapley and PME (T3, slow) | Install R and the packages. Then run `uv run pytest -m slow tests/test_rank.py tests/test_kucherenko_subsets.py`. |
| SciPy (`truncnorm`) | `dgsm` Poincare constant (T2) | Runs in the default suite. |

R code runs in a separate process through [`tests/_r_oracle.py`](https://github.com/DanielePessina/jaxgsa/blob/master/tests/_r_oracle.py). jaxgsa never reads or copies the R code. If `Rscript` or a package is missing, the test skips.

Four scripts under [`scripts/oracles/`](https://github.com/DanielePessina/jaxgsa/tree/master/scripts/oracles) run by hand. Three of them print a table and exit non-zero on disagreement.

| Script | Oracle | Checks |
| --- | --- | --- |
| `openturns_sobol_estimators.py` | OpenTURNS 1.27 | The Jansen, Martinez and Mauntz-Kucherenko Sobol estimators on Ishigami, against recorded literals |
| `salib_delta_class_counts.py` | SALib 1.5.2 | The class counts of the Plischke heuristic that `borgonovo` uses |
| `hdmr_direct_form.py` | Re-derived NumPy direct-form HDMR, `scipy.stats.f.ppf` | `hdmr.analyze` against an independent estimator. This is T4: it has no external reference. |
| `external_checks.py` | SALib, POT, SciPy (live) | A pytest file. See below. |

The first two scripts produce saved values. Those values are back in the default test suite as T2 checks, so the suite needs neither OpenTURNS nor SALib:

- `TestOpenTurnsOracle` in `tests/test_sobol_estimators.py` compares the estimators with the `OPENTURNS_ISHIGAMI` literals.
- `TestPlischkeHeuristic` in `tests/test_borgonovo.py` compares the class counts with the SALib values.

To run one script:

```bash
uv run --extra dev scripts/oracles/hdmr_direct_form.py
```

The OpenTURNS script is not in the dev extra. Run it with `uv run --with openturns scripts/oracles/openturns_sobol_estimators.py`.

### Live checks

`external_checks.py` imports the outside libraries at run time. It is not in the default test run, and CI does not run it. It covers:

- sobol: the default estimator against `SALib.analyze.sobol`.
- morris: `mu`, `mu_star` and `sigma` against `SALib.analyze.morris` on the same design.
- borgonovo: delta and S1 against `SALib.analyze.delta`.
- pawn: the median index against `SALib.analyze.pawn`, and the tie-aware KS statistic against `scipy.stats.ks_2samp`.
- optimal_transport: per-bin W2, the Sinkhorn solver and the multivariate index against POT.

Run it with:

```bash
uv run --extra dev pytest scripts/oracles/external_checks.py
```

It takes about 20 seconds.

## Run the tests yourself

```bash
uv run pytest -n auto            # default tier
uv run pytest -m slow -n auto    # tight tolerances and the R and POT checks
```

The default tier takes about three minutes on a cold cache. The slow tier takes longer.
