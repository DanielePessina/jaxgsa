# ADR 0005: Choose forward or reverse mode from the output shape

Status: accepted (2026-08-18). Implemented and closed (2026-08-20): the
v1.0 behavior-flips change made `jaxgsa.dgsm._core.jacobian_of` select
`jax.jacfwd` when `T*K > D` and `jax.jacrev` otherwise, with no user-facing
keyword. The numerical baseline was regenerated once for it; the reviewed
exception is recorded in `scripts/baseline/README.md`.

## Context

jaxgsa's stated position is that it is the GSA library where gradients are
cheap. The argument for a derivative-based method is that the Saltelli
column-swap design needs `N * (d + 2)` model runs, while a derivative method
needs `N` gradients, and one reverse-mode gradient costs about `c` model runs
(`c <= 4` provable, Griewank and Walther; `c < 6` guaranteed and about 2 to 3
in practice, Baydin et al. JMLR 2018; about 3 per the JAX cookbook). That
gives a crossover at `d > c - 2`, so in practice `d >= 2`.

**That argument is only true for scalar output.** One reverse-mode pass
returns one *row* of the Jacobian — the gradient of one output. A model with
`T` outputs costs `T` reverse passes, so the derivative route costs
`N * c * T` and the crossover becomes `d > c * T - 2`. At `c = 3`, `T = 10`
that is `d > 28`, not `d > 2`.

jaxgsa supports `Y` of shape `(N, T, K)`, so `T*K` output slices, not one.

## Decision

- Forward mode costs `d` passes regardless of `T`. Reverse mode costs `T*K`
  passes regardless of `d`. So: **use reverse when `T*K < d`, forward when
  `T*K > d`.** Select at call time by comparing the two. A tie (`T*K == d`)
  costs the same either way; the implementation keeps reverse there, so
  every case that was reverse-only before the change stays bit-identical.
- **Every cost or speed claim must either carry the `T` factor or state that
  it applies to scalar output only.** This includes README and docs prose.

Before this change, `dgsm/_analyze.py` hard-coded `jax.jacrev` (reverse
mode); its own comment noted the limitation. The comparison is a real speed
gain on time-series outputs, not a cosmetic change, in proportion to
`T*K / d`. The oracle for the change is T4: the two modes must return equal
Jacobians, which `tests/test_dgsm.py::TestJacobianModeSelection` pins.

## Consequences

- Any benchmark table that does not say what `T` was is not comparable.
- **An open obligation on the README.** It claims jaxgsa is "up to 668x faster
  than SALib". That figure compares against a single-process NumPy baseline.
  Published speedups for large Monte Carlo work on a GPU against a *fully
  parallel* CPU are near 13x, not 100x, and a reviewer will check. Either
  state the baseline in the same sentence, or re-measure against a parallel
  CPU baseline. Do not leave it as it is.
- **The measured numbers themselves stand.** The suspicion that the benchmark
  harness stops its timer before the device finishes was investigated and is
  false: every Sobol array is blocked, and `_count_nans` calls `int()` on
  device arrays inside `analyze`, which forces synchronisation before the
  function returns; the HDMR fields that are not blocked come from the same
  compiled kernels as the ones that are. Recorded so the finding is not
  re-raised.

## Rejected alternatives

- **Always reverse.** What the code does today; wrong whenever `T*K > d`,
  which is the normal case for time-series output.
- **A user-facing `mode=` keyword as the only route.** The right mode is
  determined by two numbers the library already has. Making the caller supply
  it exports an implementation detail and they will get it wrong.
