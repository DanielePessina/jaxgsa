# ADR 0021: The default Sobol estimator is `saltelli-jansen`

Status: accepted (2026-08-20)

## Context

`jaxgsa.sobol.analyze` offers six estimator pairings. All of them converge to
the same indices. They differ in small-sample noise and in whether an
estimate can leave `[0, 1]`. One of them has to be the default.

The docstring used to justify the default with "changing it would move every
stored number". That is a cost of changing, not a reason the choice is right.
The v1.0 grilling asked for the real reason to be recorded.

## Decision

The default stays `"saltelli-jansen"`: Sobol'-Mauntz for the first order,
Jansen (1999) for the total order. Two reasons:

- **Jansen's total-order estimator cannot go negative.** It is a mean of
  squares, `E[(f(A) - f(AB))^2] / (2 Var)`. Users screen on `ST`, and a
  negative `ST` invites the clipping that ADR 0017 refuses.
- **SALib parity.** SALib's `analyze.sobol` computes the same pairing by
  default, so a user who moves between the two libraries gets the same
  numbers with no keyword. This parity is also what the cross-library tests
  and benchmarks lean on.

## Consequences

- The default first-order estimate can still be negative near zero. That is
  intended: ADR 0017 reports it rather than clipping it.
- A user who wants `S1 <= ST` on every sample must ask for
  `"azzini-rosati"` and draw the design with `calc_second_order=True`.

## Rejected alternatives

- **`"jansen"` for both orders.** Neither index can go negative, but both are
  then biased upward at a true zero, and it breaks the out-of-the-box match
  with SALib.
- **`"janon-monod"` for its variance guarantee.** The asymptotic variance is
  never worse, but the guarantee is asymptotic, the improvement is small in
  the measured cases, and it also breaks SALib parity.
- **Keeping the "historical continuity" wording.** Stability of stored
  numbers is enforced by the baseline check, not by the docstring; a
  docstring rationale must say why the choice is good, not why it is old.
