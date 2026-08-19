# ADR 0006: Convergence analysis is deferred, and the obvious implementation is wrong

Status: deferred (2026-08-18)

## Context

The idea is attractive: recompute indices on nested prefixes of a sample that
has already been evaluated, and show the caller how the estimates settle, at
no extra model cost. It is deferred out of 1.0. This ADR exists because the
implementation a reader will reach for first produces plausible, wrong
numbers with no error raised.

## Decision

Defer the API. When it is revisited, two facts constrain it.

1. **Sobol prefixes are valid only at powers of two.** SciPy's documentation
   is explicit that Sobol sequences "lose their balance properties if one uses
   a sample size that is not a power of 2, or skips the first point, or thins
   the sequence." Nesting must step by halving, not by arbitrary fractions.

2. **A Saltelli design cannot be halved by slicing the flat output array.**
   The design is a base sample split into A and B, plus `k` cross matrices,
   stacked into one expanded run matrix. Slicing the expanded matrix mixes
   complete A rows with partial cross-matrix blocks, and the indices that come
   out are meaningless. To halve it correctly, take rows `0 : N/2` of A, of B,
   and of **every** cross matrix independently, then reassemble.
   **Expose nesting at the sample level, never by slicing `Y`.**

What to report, per Sarrazin, Pianosi and Wagener (2016), *EMS* 79:135-152:
screening, ranking and estimate convergence separately, because "convergence
of screening and ranking can be reached before sensitivity estimates
stabilize". They also warn that convergence is case-dependent, so the API
must not offer a fixed sample-size rule.

## Consequences

- Any caller who slices `Y` themselves to "check convergence" is getting
  wrong numbers. If the shape of the expanded matrix is ever exposed, this
  needs to be said in the docstring.

## Rejected alternatives

- **Slicing the flat `Y`.** The whole reason for this ADR.
- **Shipping a single scalar convergence criterion.** The cited work is that
  the three quantities converge at different rates, so one number hides the
  answer the caller wanted.
