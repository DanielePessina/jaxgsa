# ADR 0013: `Problem` stays a plain value object, not a JAX pytree

Status: accepted (2026-08-19)

## Context

A wanted capability for 1.0 is `d(index)/d(marginal parameter)` — the
derivative of a sensitivity index with respect to, say, the bounds of a
uniform input. The obvious route is to register `Problem` as a JAX pytree so
that `jax.grad` can differentiate through it.

PR #47 did exactly that. It was closed unmerged.

## Decision

**`Problem` is a plain, hashable value object. It is not registered as a JAX
pytree.**

Differentiation with respect to marginal parameters goes through **`Theta`**
instead: a mapping pytree of marginal parameters, consumed by
`SobolSamples.transform(theta)`.

## Consequences

- Registering `Problem` would make *every* `Problem` traceable. A caller who
  passes one into a jitted function and never wanted a gradient would find
  their marginal parameters turned into tracers. That is a cost paid by
  everyone for a feature wanted by few.
- `Problem` stays hashable and usable as jit-cache metadata, which several
  code paths rely on (see ADR 0019).
- The differentiation surface is **opt-in and explicit**: it exists only
  where a caller builds a `Theta`. That is a smaller and more auditable
  surface than "the whole problem specification is differentiable".
- The cost is one extra concept in the vocabulary, and a caller who wants
  gradients must route through `transform`.

## Rejected alternatives

- **Register `Problem` as a pytree** (PR #47). Closed unmerged for the
  tracer-leak reason above.
- **Two `Problem` classes, one traceable.** Doubles the type surface and
  every function that accepts a problem then accepts two things.
