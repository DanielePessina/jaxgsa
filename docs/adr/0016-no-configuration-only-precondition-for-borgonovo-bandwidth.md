# ADR 0016: No up-front check on Borgonovo's `degenerate_bandwidth`

Status: accepted (2026-08-18). **Reverses an earlier audit decision.**

## Context

An audit decision (D9) said `borgonovo.analyze` should raise at the top when
`degenerate_bandwidth * h_full < grid_step`, on the premise that a kernel
narrower than one grid step "guarantees" a failed computation.

**The premise is false**, and this ADR exists because anyone reading the audit
text alone will rebuild the wrong thing.

Two conditions must *both* hold before the integral breaks: a conditioning
class must actually be **degenerate**, so the floor is applied at all, and the
resulting spike must land **on a grid point**, so the trapezoid rule sees it.
Measured on a fixture built to have a genuinely degenerate class, where one
grid step is `0.108 * h_full`:

| `degenerate_bandwidth` | spike on a grid point | spike off the boundary |
| --- | --- | --- |
| 0.100 | delta 0.6721 | 0.7336 |
| 0.010 | delta 0.9433 | 0.5982 |
| 0.001 | **fails**, delta 4.01 | 0.5982 |
| 1e-05 | — | 0.5982 |

A kernel one tenth of a grid step returns a valid answer, and off the boundary
the estimate is stable five orders of magnitude below the step.

A first attempt shipped the audit's rule and refused four configurations that
return bit-identical, correct results — including `degenerate_bandwidth=0.1`,
the very fraction `"auto"` uses internally. It was caught in review before it
left the branch.

There is also a mechanical objection, which stands on its own: the test cannot
run at the top of `analyze`. `h_full` is computed per output column *inside
the jitted kernel*, from `jnp.std(y_r)`, and `grid_step` is data-dependent
too. Neither value exists where the check was to go.

## Decision

**No up-front raise.** Whether the computation breaks is a property of where a
class sits relative to the grid, which is **data, not configuration**, so no
configuration-only check can be a true precondition.

Instead, the existing out-of-range error builds its advice from what the
kernel actually did: whether a class was floored, which column, the floor it
used against the real grid step, and the fraction that would fix it.

## Consequences

- The failure is reported after the fact, with the data that caused it, which
  is more useful than a refusal that guesses.
- Cost was never the obstacle and should not be raised as one: a host-side
  degeneracy scan measured 282 ms against `analyze`'s 5037 ms. It simply
  would not have been correct.

## Rejected alternatives

- **The audit's up-front raise (D9).** Built, measured, rejected. It refuses
  correct configurations, including the library's own default fraction, and
  cannot be evaluated where it was specified.
- **A warning instead of a raise.** Same false premise; it would fire on
  configurations that are fine.
