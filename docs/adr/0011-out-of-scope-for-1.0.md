# ADR 0011: Out of scope for 1.0

Status: accepted (2026-08-18)

## Context

Several capabilities are proposed often enough that the reasons for leaving
them out need to be written down, or they get re-proposed every review.

## Decision

| Item | Decision and reason |
|---|---|
| **Plotting module** | **Rejected.** Not deferred. |
| **Command-line interface** | **Rejected.** Not deferred. |
| **Distance correlation as its own module** | **Rejected as misleading.** Distance correlation *is* HSIC with a distance kernel. Shipping it as a separate module implies two methods where there is one. The capability ships, as a **kernel option on `hsic`**. |
| **Convergence analysis API** | **Deferred.** See ADR 0006, which records the two traps in the obvious implementation. |
| **Target and conditional HSIC** | **Deferred**, but design the HSIC output path so a value transform can be added without breaking it. |
| **PoinCE, gradient-enhanced Poincare chaos** | **Deferred.** A strong fit for JAX — the method was designed for cheap gradients its authors did not have. Luthen, Roustant, Gamboa, Iooss, Marelli, Sudret (2023), *IJUQ* 13(6):57-82, arXiv:2107.00394. |
| **Regional sensitivity analysis** | **Withdrawn, not answered.** Two incompatible formulations already exist in Python: SAFEpython uses a threshold and a Kolmogorov-Smirnov distance and returns one number per input; SALib uses percentile bins and a Cramer-von Mises statistic and returns one value per bin. They disagree by construction, so a cross-library agreement test would fail no matter what we shipped, and shipping one settles an argument that is not ours. If it ever returns, picking the formulation also picks the oracle. Two citation traps: SAFEpython's docstring miscites Spear and Hornberger to *Water Resour. Res.* (it is *Water Research*), and Pianosi and Wagener (2018) does not cover RSA at all. |

## Consequences

- The distance-kernel HSIC option has its own constraint: the Gamma null is
  invalid for it. See ADR 0018.

## Rejected alternatives

- **A plotting module for convenience.** It is a dependency, an API surface
  and a taste argument, in a library whose output is arrays a user already
  knows how to plot.
