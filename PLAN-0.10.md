# Execution plan: jaxgsa 0.10

Status: planned 2026-08-18, after the design interview. Release 0.9 is
committed on `chore/tranche-0` and not yet released.

This plan supersedes section 6 of `PLAN-V1.0.md`. That section listed seven
items. Four are cut, two were already shipped, and the largest was moved to
1.0. Section 4 gives every reason.

**0.10 adds capability and breaks one thing.** The break is
`SobolResult.nan_counts`, which the new non-finite policy replaces.

---

## 1. What ships

| # | Item | Batch |
| --- | --- | --- |
| 1 | `on_invalid`: one non-finite policy for all thirteen entry points | 1 |
| 2 | Crossed DGSM, `dgsm.analyze(crossed=True)` | 2 |
| 3 | `jaxgsa.active_subspace`, a new module | 2 |
| 4 | Poincare: two citation fixes, an external oracle, and `PCEResult.dgsm()` | 2 |
| 5 | `sobol.analyze(estimator=...)`, a choice of published estimator pair | 3 |
| 6 | The dummy-parameter threshold on `sobol` and `pawn` | 3 |
| 7 | Optimal-transport parity: an exact solver and a user-supplied cost | 3 |
| 8 | HSIC: the Gamma p-value, the distance kernel, a bandwidth choice | 4 |

## 2. What was cut, and why

| Item | Reason |
| --- | --- |
| **Rank-based estimators** (`jaxgsa.rank`) | Cut after reading the paper. See correction C1: the group indices the roadmap promised are not in it. What remains is first-order only, and `scipy.stats.chatterjeexi`, UQpy and OpenTURNS all already have it. It was the most-covered item on the list. |
| **Regional sensitivity analysis** | Both formulations already exist in Python: SAFEpython uses a threshold and a Kolmogorov-Smirnov distance, SALib uses percentile bins and a Cramer-von Mises statistic. They disagree, so shipping one settles an argument that is not ours. Open question Q3 in `PLAN-V1.0.md` is therefore withdrawn, not answered. |
| **Generalised Sobol indices** | UQpy ships `GeneralisedSobolSensitivity`, correctly attributed to Gamboa et al. OpenTURNS has an equivalent aggregated form. Our `(N, T, K)` support already covers most of the use. |
| **Confidence intervals in eight modules** | Deferred. One scoped exception ships: `active_subspace` gets the bootstrap that is part of its published method. The percentile / basic / BCa sweep with a coverage simulation stays a 1.0 item. |
| **`d(index)/d(theta)`** | Moved to 1.0. See section 3. |

## 3. Why the flagship moved to 1.0

The flagship differentiates a sensitivity index with respect to a model
parameter or a parameter of an input distribution. The distribution half needs
samples built as `X = F_inverse(u; theta)`, so that `theta` is a traced value
and `jax.grad` has something to follow.

jaxgsa cannot do this today, and the obstacle is deliberate.
`problem.py:113` coerces every specification value with `float(...)`, with the
comment "to prevent JAX tracers or numpy scalars from leaking into metadata".
The specification is a hashable jit-cache key. Nothing in it can be a tracer.

Making jaxgsa supply the inverse map means separating the *structure* of a
marginal, which stays static, from its *numbers*, which must become a traced
array. That is the `Marginal` protocol already planned for 1.0.

A second obstacle is independent of the first. `jax.grad(analyze)` fails today
even for a model parameter, because several host-side checks read values out of
a tracer: the zero-variance warning in `_core/validation.py`, the non-finite
row drops in `sobol` and `morris`, and the range check at
`borgonovo/_analyze.py:884`. A traceable path has to be built through each
module we want to support.

Doing the flagship before the distributions work means building a throwaway
version of the distributions work first. It waits.

---

## 4. Corrections to the source documents

Every correction below was checked against the paper or against the working
tree. Three of them change what gets built.

### C1. The rank-based estimators do not give group indices

`ROADMAP-1.0.md` section 4.1 says the method yields "first-order indices, and
**closed group indices** `S^u` through multivariate nearest neighbours". The
paper does not support this.

Gamboa, Gremaud, Klein and Lagnoux (2022), read in full:

- The new estimator, equation (17) in section 4.1, is defined "with respect to
  `V = X1` **assumed to be real-valued**". Remark 4.4 defers any extension to
  more variables to "the forthcoming paper".
- Multivariate nearest neighbours appear nowhere in the paper's own method.
  The single mention is a citation added after submission, to Broto et al., for
  Shapley effects. That is the separate 1.0 item.
- The central limit theorem, Theorem 4.1, is **per index, not joint**.
  Remark 4.5 states that a joint theorem for the vector of all `p` indices "is
  not a direct generalization" and is future work. Theorem 4.1 also assumes
  `f` is twice differentiable in its first coordinate, with `f` and both
  derivatives bounded.
- Total-order indices are not discussed. The term does not occur.
- Ties are excluded by assumption, not handled. Section 3.1: "we assume that
  the laws of `V` and `Y` are both diffuse (ties are excluded)." No
  tie-breaking rule is given.

The item is cut. If it ever returns, it returns with Broto et al. as the
citation for anything beyond first order.

### C2. HSIC p-values already ship

`ROADMAP-1.0.md` section 4.6.1 says "jaxgsa returns an HSIC score. A score has
no cut-off." It has one. `HSICResult.p_values` exists, and
`hsic/_analyze.py:313` runs a permutation test to fill it.

What is actually missing is the asymptotic route, the distance kernel, and a
bandwidth choice.

### C3. De Lozzo and Marrel's three routes do not include a permutation test

The roadmap lists "a permutation test", "an asymptotic test" and "a spectral
test". The paper's own three routes are **asymptotic, spectral, and a
non-asymptotic bootstrap**. Our shipped permutation test is real and correct,
but its citation is Gretton et al. section 3, not De Lozzo and Marrel.

### C4. The HSIC normalisation is settled

The roadmap records "[unverified which normalisation to prefer]: two
conventions circulate, `1/m^2` and `(m-1)^-2`." Gretton et al. equation (4)
defines the biased V-statistic with `1/m^2`, equal to
`(1/m^2) * trace(KHLH)`. Equation (3) gives the unbiased U-statistic with
falling factorials, not `(m-1)^-2`. We use the V-statistic, so `1/m^2` is
correct and matches what we already ship.

### C5. The optimal-transport irrelevance threshold already ships

`ROADMAP-1.0.md` section 6.2 lists it as work to do.
`optimal_transport/_analyze.py:449` already takes `dummy: bool`, and
`OTResult.ot_dummy` carries the result. It builds the baseline by permuting a
synthetic column, at `:777`.

### C6. Three citations were wrong or incomplete

| Was | Is |
| --- | --- |
| Roustant et al. (2017), *Stat. Comp.* 27:879-894, in `dgsm/_poincare.py:21` | Roustant, Barthe, Iooss (2017), *Electronic Journal of Statistics* 11(2):3081-3119, doi:10.1214/17-EJS1310 |
| Lamboni et al. (2013), *Math. Comp. Sim.* 87:**44**-54, in `_poincare.py:20` and `dgsm/_analyze.py:10` | *Math. Comp. Sim.* 87:**45**-54, doi:10.1016/j.matcom.2013.02.002 |
| "Roustant+ 2014" for crossed DGSM, no volume | Roustant, Fruth, Iooss, Kuhnt (2014), *Math. Comp. Sim.* **105:105-118**, doi:10.1016/j.matcom.2014.05.005 |

There is also no standalone "Diaz (2017)". The bootstrap for active-subspace
eigenvalues is in Constantine and Diaz (2017), *Reliability Engineering and
System Safety* 162:1-13, section 2.1.

Azzini and Rosati (2021) is *Reliability Engineering and System Safety*
213:107647, "Sobol' main effect index: an Innovative Algorithm (IA) using
Dynamic Adaptive Variances". Two other Azzini papers, from 2021 and 2023, are
easy to confuse with it.

### C7. Two claims to record as unverified

1. That GSA practice defaults the HSIC bandwidth to the empirical standard
   deviation rather than the median heuristic. The claim could not be
   confirmed in the body text of Marrel and Chabridon (2021). Do not state it
   as established.
2. The author order of Reddi et al. (2015) differs between venues. The arXiv
   preprint (1406.2083) lists Reddi first; the AAAI proceedings list Ramdas
   first. Cite one venue and be consistent.

---

## 5. Decisions

### 5.1 The non-finite policy

`on_invalid` takes `"raise"`, `"propagate"` or `"drop"`, and defaults to
`"raise"`. It reaches all thirteen `analyze()` entry points, including the
three named `analyze_pce`, `analyze_hdmr` and `analyze_vkoga`, which a search
for `def analyze(` misses.

It covers non-finite values in **both** `Y` and `X`, under the one keyword,
with one restriction: `"drop"` is refused for `X` on a design-based method.
Removing a row from a Saltelli, Morris or eFAST design breaks the block
structure that the estimator depends on, so a dropped `X` row there would
produce a silently wrong index. `"raise"` and `"propagate"` apply everywhere.

Every result carries a report that names how many rows were affected, which
row indices, and where they sat in the design. Under `"raise"` the same
information goes into the message. This is what tells a user which model
evaluation to investigate.

`SobolResult.nan_counts` is **removed**. It counted NaNs in the output indices
and threw row identity away, which is the defect the report fixes. This is the
release's one breaking change.

Today's behaviour, for the record: `sobol/_analyze.py:67` and
`morris/_analyze.py:119` drop rows silently, `efast/_analyze.py:180` raises,
`kucherenko/_analyze.py:102` filters, and the rest let NaN reach the indices.

### 5.2 Gradient items

**Crossed DGSM** goes on `dgsm.analyze(crossed=True)`, off by default, adding a
`(D, D)` field. It reuses the argument resolution, validation and batching that
batch 3 of release 0.9 fixed. `jax.hessian` costs about `D` times the gradient,
per output, so the default matters. Advertise it as a bound on superset
importance, not on the second-order index.

**`jaxgsa.active_subspace`** is a new module. It mirrors `dgsm.analyze`, taking
either a model or a precomputed `dfdx`, so one gradient sample serves both. It
returns the eigenvalues and eigenvectors of `C = E[grad f grad f^T]`, the
per-input activity scores, and Constantine and Diaz's bootstrap intervals on
the eigenvalues and the subspace distance.

It refuses categorical and correlated problems, exactly as `dgsm.analyze` does
at `_analyze.py:383-384`. A derivative with respect to a level code has no
meaning, and the theory assumes a product measure.

**Poincare** already ships end to end. Three things are left: fix the two
citations in C6, get an external oracle for the truncated-Gaussian spectral
solve, which is currently only covered by calling the private function
directly, and add `PCEResult.dgsm()` for the Sudret and Mai route. That last
one follows the `result.shapley()` pattern and gives a free cross-check,
because the PCE-derived DGSM and the autodiff DGSM must agree.

### 5.3 Given-data items

**`sobol.analyze(estimator=...)`** takes an author-pair name:
`"saltelli-jansen"` (the default, so no existing number moves), `"jansen"`,
`"janon-monod"`, `"martinez"`, `"mauntz-kucherenko"`, `"azzini-rosati"`. The
current code pairs a Saltelli first-order estimator with a Jansen total, which
is why the default is named for both.

`"azzini-rosati"` raises unless the design carries `BA` columns.
`sobol/_sampling.py:342` produces them only when `calc_second_order=True`,
which makes the design step `2D+2` instead of `D+2`. The estimator therefore
costs about twice as much, and the error must say so.

**The dummy-parameter threshold** goes on `sobol` and `pawn`. It estimates the
index of an input that provably cannot matter, which gives the noise floor for
that sample and that estimator. An index below the floor is not distinguishable
from nothing.

The construction differs by design, under one `dummy=True` keyword:

- `sobol` uses the analytic form, which costs no extra model runs. It is
  Khorashadi Zadeh et al. (2017) equations 3, 4, 12 and 13.
- `pawn` uses the Kolmogorov-Smirnov distance between two independent
  sub-samples of the same unconditional distribution.
- `optimal_transport` already has one, built from a permuted synthetic column.
  It stays as it is.

Document per method which construction ran. A floor is not a p-value: it says
"not distinguishable from nothing at this sample size", and nothing more.

**Optimal transport** gains `solver="exact"` for multivariate and trajectory
mode, through POT, promoted from a development extra to an optional one. The
call leaves the device and runs on CPU, and the docstring says so. Sinkhorn
stays the default, and univariate mode is already exact. A missing POT raises
with the install command.

It also gains a user-supplied ground cost, mirroring gsaot, which offers it on
`ot_indices` but not on `ot_indices_wb`. Read the gsaot paper's definition of
"sensitivity maps" before building anything: our univariate mode already
returns `(T, K, D)`, and it may be the same object under another name.

### 5.4 HSIC

Add the Gamma asymptotic p-value, from Gretton et al. equation (9), with the
null mean and variance from Theorems 3 and 4. The factor `m` appears in `beta`
only. Record which method produced `p_values`, because the field currently
documents itself as permutation-based.

Add `bandwidth="median" | "std" | float`. The default stays `"median"`, so no
existing number moves. The `"std"` option exists because OpenTURNS uses it, and
without it a comparison against the only other Python package with HSIC cannot
agree by construction.

Add the distance-induced kernel. Sejdinovic et al. Theorem 24, section 5.2,
proves `dCov^2 = 4 * HSIC` with those kernels, and their Appendix A proves that
distance correlation equals normalised HSIC. So it ships as a kernel option and
not as a separate method. Route distance kernels away from the Gamma null.

Keep the V-statistic. The Gamma null is derived for it.

The spectral route is not in scope. It needs `O(m^3)`.

---

## 6. Batches

One branch, off `chore/tranche-0`, in a new worktree. One draft PR. Four
batches, each through the six gates that release 0.9 used: local gate,
correctness review, verification review, blast-radius review, numerical
baseline, documentation. The three review agents must not be the agents that
wrote the code. That protocol found a defect in every batch of 0.9.

| Batch | Content |
| --- | --- |
| 1 | `on_invalid` alone. It touches all thirteen entry points, and everything else lands on top of it. |
| 2 | Crossed DGSM, `active_subspace`, Poincare. |
| 3 | Sobol estimators, the dummy threshold, optimal-transport parity. All three touch the Saltelli design or the transport kernel. |
| 4 | HSIC Gamma p-value, distance kernel, bandwidth option. |

The fixed-seed baseline extends to the new surface: `active_subspace`, crossed
DGSM, each new `estimator=` pair, both dummy thresholds, and the Gamma
p-values. **Every pre-existing entry must stay bit-identical.** No decision in
this plan changes a default. Any movement is a defect until it is proved
otherwise. That rule caught the PAWN and eFAST problems in 0.9.

---

## 7. Verification

The oracle policy of `PLAN-V1.0.md` section 3 stands. Oracles run locally, in
any language. What lands in the repository is the number, typed into a test as
a literal, with the oracle, its version and the script that produced it
recorded next to it.

**No formula is typed from memory.** Every estimator is transcribed from the
paper, and the section or equation number goes in the docstring.

| Item | Oracle |
| --- | --- |
| `on_invalid` | T4 behavioural, one test per policy per entry point |
| Crossed DGSM | T0 on a polynomial with known cross derivatives |
| Active subspaces | T0: `C = a a^T` exactly for a linear model, `A(Sigma + mu mu^T)A` for a Gaussian quadratic, and `trace(C) = sum nu_i` against DGSM |
| Poincare truncated branch | T3 R `sensitivity::PoincareOptimal`, installed; T1 from the tables in Roustant, Barthe and Iooss section 4 |
| `PCEResult.dgsm()` | T4: it must agree with the autodiff DGSM |
| Sobol estimators | T2 OpenTURNS for Saltelli, Jansen, Mauntz-Kucherenko and Martinez, which it ships as separate classes. Janon-Monod and Azzini-Rosati have no Python implementation anywhere, so T1 from the papers and T0 on Ishigami. |
| Dummy threshold | T4 by construction, which is right here: it is a protocol, not a quantity with a true value. Check that it centres near zero on an input the model ignores, and that it holds its nominal false-positive rate. T3 `sensobol::sobol_dummy`, installed. |
| Optimal transport exact solver | T2 POT `ot.emd`. Sinkhorn must converge to the exact cost as the regularisation falls. |
| HSIC Gamma p-value | T2 OpenTURNS. T4: the Gamma p-value and the existing permutation p-value must agree as the permutation count rises. |
| Distance kernel | T2 `dcor`, through Sejdinovic Theorem 24's factor of four |

---

## 8. Papers

Nine of fifteen were retrieved in full and are on hand: Chatterjee, Gamboa,
Jansen, Janon, Constantine and Dow and Wang, Constantine and Diaz, Roustant and
Barthe and Iooss, Sudret and Mai, Borgonovo, and the gsaot paper. All five HSIC
papers were retrieved: De Lozzo and Marrel, Gretton, Sejdinovic, Marrel and
Chabridon, Reddi.

**Three are still needed, and each one blocks an item:**

| Paper | Blocks |
| --- | --- |
| Khorashadi Zadeh et al. (2017), *Environmental Modelling and Software* 91:210-222, doi:10.1016/j.envsoft.2017.02.001, equations 3, 4, 12, 13 | The dummy threshold on `sobol` |
| Azzini and Rosati (2021), *RESS* 213:107647 | The `"azzini-rosati"` estimator |
| Roustant, Fruth, Iooss, Kuhnt (2014), *Math. Comp. Sim.* 105:105-118 | Crossed DGSM |

A green open-access copy of the first exists at the University of Bristol
repository. The other two look genuinely closed.

One further paper is wanted but blocks nothing: Monod, Naud and Makowski
(2006), "Uncertainty and sensitivity analysis for crop models", chapter 3 of
*Working with Dynamic Crop Models*, Elsevier, pages 55-99. It is needed only to
attribute the Janon-Monod estimator correctly. The estimator itself is in Janon
et al. (2014) equation (6), which is on hand.

---

## 9. What 1.0 now holds

| Item | Note |
| --- | --- |
| **Input distributions**, carrying D1 and D2 | The `Marginal` protocol. The release's breaking change, and the prerequisite for the flagship. |
| **`d(index)/d(theta)`** | The flagship, built on the protocol above. |
| **kNN Shapley effects from given data** | Broto et al. The real capability gap: no Python package has it. Group rank indices, if they are ever wanted, come with this k-d tree. |
| **Confidence intervals in eight modules**, percentile, basic, BCa | With a coverage simulation. |
| **Linear-Gaussian fixture fixes** | The strongest oracle in the repository, currently correct by accident. |
| **Release engineering** | SPEC 0, SPEC 7 `rng`, NEP 23, Trusted Publishing, conda-forge, Zenodo, `CITATION.cff`. |

Open question Q5 of `PLAN-V1.0.md`, on non-Gaussian copulas, is still open and
is still needed by 1.0. Q6, on the term "pick-freeze", is still open. Q3, on
the regional sensitivity formulation, is withdrawn with the item.
