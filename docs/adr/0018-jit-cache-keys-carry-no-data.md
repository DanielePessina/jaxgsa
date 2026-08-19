# ADR 0018: A jit cache key carries metadata, never data

Status: accepted (2026-08-18)

## Context

Several modules pass structured objects as jit arguments or memoise a compiled
kernel on a tuple of parameters. JAX pytree rules make this a repeatable trap:
**a `NamedTuple` is a pytree**, so giving one a field holding Python data turns
that data into traced values inside the jit cache key. The symptom is a cache
that never hits, or in the worst case a recompilation per call with values
baked in.

## Decision

Anything used as, or reachable from, a jit cache key holds only hashable
metadata: Python scalars, strings, and tuples of them. Never a list, never a
NumPy array, never a container that a pytree flattener will walk into.

Concrete cases already load-bearing in this codebase:

- **Categorical partition groups must not become a `NamedTuple` carrying their
  levels.** The groups are passed as jit arguments in both optimal transport
  and Borgonovo. A list-valued field would become traced entries in the cache
  key.
- **`CategoricalSpec.probs` and `.labels` are tuples, not lists.** The spec
  tuple is jit-cache metadata on a hashable `Problem` (see ADR 0013), and the
  categorical payload relies on tuple-of-tuples.
- **A frequency-plan object holding a NumPy array must stay out of the eFAST
  kernel memoisation key**, which is keyed on `(N, M, omega_0, batched)`.
- **HDMR's kernel closure captures concrete Python integers that must not
  become traced.** This is why the HDMR kernel's long positional argument
  lists were deliberately left alone rather than bundled into an object: HDMR
  is the most numerically delicate module here, and the naming fix that was
  actually needed did not require threading a new object through it.

## Consequences

- Structs that would be pleasant as `NamedTuple`s stay as plain tuples or
  frozen dataclasses on the jit boundary. That is the price.
- A refactor that "tidies" an argument list into a container is exactly the
  change ADR 0009 says must earn its place, and this is one of the ways it
  can go wrong silently.

## Rejected alternatives

- **Bundle jit arguments into `NamedTuple`s for readability.** Readability at
  the cost of a silently poisoned cache.
- **Mark fields static with `jax.tree_util` registration per struct.** More
  machinery to get wrong, for containers that are two or three values long.
