# ADR 0019: PCE leave-one-out leverage comes from a Cholesky factor

Status: accepted (2026-08-18)

## Context

The PCE leave-one-out error was computed twice, two different ways, with a
comment asserting the two agree. The dense path formed an intermediate that
grows with `N`; the streaming path could not afford it, so it carried the
small Gram inverse and re-implemented the formula inline.

## Decision

Compute the leverage **once**, from a **Cholesky factor of the Gram matrix**,
and use it on both paths.

**Not an explicit inverse.** The default ridge is `1e-8`, deliberately small.
PCE Gram matrices become badly conditioned as the polynomial order rises, and
forming an explicit inverse worsens that. A degraded leave-one-out value does
not just report a slightly wrong error — it **feeds back into automatic order
selection**, so the conditioning problem changes which model is chosen.

## Consequences

- The memory estimate charged three `N x n_terms` arrays, one of them for the
  intermediate that is now removed. Correcting it to two moves the point at
  which the streaming fit engages, so some fits that streamed before stay
  single-pass. **This is a deliberate, reviewed movement of a number**, and it
  is the accurate estimate rather than the conservative one.
- A `streamed` flag on the fit result records which path ran. Without it, a
  fit that silently stops streaming passes every test.

## Rejected alternatives

- **Keep the two implementations and test that they agree.** They did agree,
  by assertion in a comment rather than in a test, and the duplication is
  what made the memory estimate wrong on one side only.
- **An explicit `inv(Gram)`.** Cheaper to write, and it degrades the quantity
  that selects the polynomial order.
