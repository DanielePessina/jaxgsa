# ADR 0017: Negative Sobol estimates are reported, not clipped

Status: accepted (2026-08-18)

## Context

Sobol estimators sometimes return a small negative index, which is impossible
for the quantity being estimated. The tempting fix is to clamp to zero.

The mechanism matters. A negative value comes from a **difference of two
correlated Monte Carlo estimates** — a cross-moment minus a squared mean.
It is not evidence that the wrong estimator was used for that magnitude, and
it is **not** an under-sampling artefact: negatives also appear at large `N`
and low `d`.

## Decision

**Do not clip negative estimates in the returned arrays.** Clip only for
display, if at all, and say so where it is done.

Clipping to zero biases the estimate **upward**, and it does so in exactly the
near-zero regime where a user is making a ranking or screening decision. A
truly-zero index estimated as a symmetric cloud around zero becomes a
one-sided positive cloud, so an irrelevant input looks slightly relevant, and
it looks more relevant than another irrelevant input purely by noise.

Two related attribution rules:

- **Do not present estimator choice as a function of index magnitude, citing
  Puy et al.** Their framing is by *goal* and *dimensionality*, not by how
  large the index is.
- The Janon-Monod estimator's asymptotic variance is **always at most** the
  classical one. State it that way; it is a uniform result, not a
  regime-dependent one.

## Consequences

- A user will see negative numbers and may ask about them. The docstring
  should say what they mean: the magnitude is an indication of the Monte Carlo
  error at that index, which is useful information a clamp would destroy.
- A confidence interval that straddles zero is the honest report for a
  near-zero index, and it needs the unclipped values to be computed correctly.

## Rejected alternatives

- **`jnp.maximum(s, 0.0)` on the returned indices.** Biases upward where it
  matters most, and hides the error estimate.
- **Raising or warning on a negative index.** It is a normal outcome of a
  correct estimator, not a fault.
