# ADR 0014: float32 stays the default, and jaxgsa ships no `enable_x64` wrapper

Status: accepted (2026-08-19)

## Context

Batching an estimator over more output slices at once changes the last bits of
a float32 reduction, because XLA schedules the reduction differently at a
different width. That shows up in the numerical baseline as a few units in the
last place. The proposal was to switch the default to float64, or at least
ship `jaxgsa.enable_x64()`, on the theory that more precision would make batch
width irrelevant.

## Decision

**Keep float32 as the default. Ship no `jaxgsa.enable_x64()` wrapper.**

jaxgsa computes in whatever precision JAX is configured for and infers dtype
from the caller's arrays. It does not set `jax_enable_x64` itself.

Three reasons, in order of weight:

1. **float64 does not buy what was wanted.** Measured: the batch-width
   discrepancy goes from about `2e-7` to about `2e-16`. It does **not** become
   bit-exact. The cause is *reassociation* — the reduction is summed in a
   different order — which is arithmetic, not precision. More bits make the
   difference smaller and never zero.
2. **The cost is real.** Up to 2.1x memory, and up to 1/64 throughput on
   consumer NVIDIA GPUs. TPUs have no float64 support at all.
3. **The wrapper would add nothing.** `jax.enable_x64()` is already the
   primitive. It is thread-local, it is a context manager, and it works with
   jaxgsa as-is. A wrapper would be a re-export with a jaxgsa name on it,
   implying jaxgsa-specific behaviour that does not exist.

What the library does owe the caller: **never silently destroy precision.**
Passing a float64 array while x64 is off truncates it to float32, so say so
once.

## Consequences

- The numerical baseline stays float32 and machine-specific, and the one
  reviewed batch-width exception recorded in `scripts/baseline/README.md`
  stays a reviewed exception rather than something a flag would have removed.
- A caller who needs float64 writes `jax.config.update("jax_enable_x64", True)`
  or uses `jax.experimental.enable_x64()`. Document that; do not wrap it.

## Rejected alternatives

- **float64 by default.** Pays 2.1x memory and a large throughput loss on
  every user, for a difference that is still not exact.
- **`jaxgsa.enable_x64()`.** A rename of an existing primitive.
