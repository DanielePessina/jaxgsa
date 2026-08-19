# ADR 0007: `on_invalid` covers both `X` and `Y`, and refuses `"drop"` for `X` on a design

Status: accepted (2026-08-18), shipped in 0.10

## Context

Before this policy, non-finite handling was per-module and inconsistent:
`sobol/_analyze.py` and `morris/_analyze.py` dropped rows silently,
`efast/_analyze.py` raised, `kucherenko/_analyze.py` filtered, and the rest
let NaN reach the indices. Silently dropping rows changes what the estimator
computes, and the caller is never told which model evaluation failed.

## Decision

One keyword, `on_invalid`, taking `"raise"` (default), `"propagate"` or
`"drop"`, on all thirteen `analyze()` entry points.

It covers non-finite values in **both `X` and `Y`**, under the one keyword,
with one restriction:

> **`"drop"` is refused for `X` on a design-based method.**

Removing a row from a Saltelli, Morris or eFAST design breaks the block
structure the estimator depends on. The estimator would still return a
number, and the number would be silently wrong. `"raise"` and `"propagate"`
apply everywhere.

Every result carries a report naming how many rows were affected, which row
indices, and where they sat in the design. Under `"raise"` the same
information goes into the message. That is what tells a caller which model
evaluation to investigate.

`SobolResult.nan_counts` is removed. It counted NaNs in the output indices
and threw row identity away, which is exactly the defect the report fixes.

## Consequences

- `"raise"` as the default is a behaviour change for the four modules that
  previously dropped or filtered. It is a correctness fix, and it was the one
  deliberate behaviour change in its release.
- A caller who genuinely wants rows dropped from a design has to drop them at
  the sampling level and rerun, which is the honest answer.

## Rejected alternatives

- **Separate keywords for `X` and `Y`.** Two keywords, one question. The
  asymmetry is a per-method restriction, not a second concept.
- **Allowing `"drop"` for `X` everywhere with a warning.** A warning does not
  make a structurally invalid index valid, and the estimator gives no signal
  that anything is wrong.
- **Keeping silent dropping as the default for compatibility.** Silence is
  the defect.
