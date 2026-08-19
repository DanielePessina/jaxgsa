# ADR 0004: Oracle inventory

Status: accepted (2026-08-18)

## Context

ADR 0001 requires an external check per method. Finding out whether a
candidate oracle installs at all is slow, and two of them do not. This is the
record so the next person does not repeat the search.

## Decision

Verified available, 2026-08-18:

| Oracle | Tier | Status |
| --- | --- | --- |
| `scipy.stats.chatterjeexi` | T2 | scipy 1.18.0. The project pin is `scipy>=1.15`, so the function is always available. |
| SALib `analyze.rsa` | T2 | 1.5.2, already a dev extra. |
| OpenTURNS | T2 | 1.27, wheel installs clean. All four Sobol estimators, `RankSobolSensitivityAlgorithm` and `HSICEstimatorGlobalSensitivity` are in the main namespace. LGPL, so safe to import directly. |
| POT `ot.emd` | T2 | 0.9.7, already a dev extra. |
| dcor | T2 | 0.7. |
| UQpy `GeneralisedSobolSensitivity` | T2 | 4.2.1, but needs Python 3.12 and `setuptools<81`. Use its two pure-array methods and feed them our own A/B/C blocks rather than installing it into the project environment. |
| R `sensitivity` | T3 | 1.31.0. `sobolrank`, `sobolshap_knn`, `shapleyPermEx`, `PoincareOptimal`. |
| R `gsaot` | T3 | 1.1.1. |
| R `sensobol` | T3 | 1.2.0. Needed for BCa intervals and the dummy-parameter floor. |
| SAFEpython | T3 | 0.2.0rc1 resolves. GPL-3: subprocess only, and do not read its source while implementing. |
| **ATHENA** | T2 | **Dead end.** Its dependency chain cannot be resolved on any supported Python. Do not try again without evidence that upstream fixed it. |

Active subspaces, which ATHENA would have checked, uses T0 oracles instead —
the linear and quadratic cases have exact closed-form answers, which is
strong evidence, not a fallback.

## Consequences

- Versions here are the ones recorded in test provenance blocks. When an
  oracle is re-run at a new version, update the provenance block, not this
  table alone.

## Rejected alternatives

- **Waiting for ATHENA.** Its T0 replacement is stronger than what ATHENA
  would have provided.
