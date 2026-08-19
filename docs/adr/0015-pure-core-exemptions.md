# ADR 0015: `kucherenko` and `vkoga` are exempt from the pure-core rule

Status: accepted (2026-08-19)

## Context

Every method is expected to have a **pure core**: an `indices()` that takes
the design object or `(problem, X, Y)`, returns a bare tuple of arrays, and
survives `jit`, `vmap` and `jacrev`. It is what makes the library's stated
position — one `vmap` on the device, gradients cheap — true method by method.

Two methods do not have one and are not going to.

## Decision

**`kucherenko` and `vkoga` are exempt from the pure-core rule.** The exemption
is **declared in the method registry**, not left as an absence, so
`tests/test_vocabulary.py` and any future conformance test can read it rather
than special-casing two names.

Both are host NumPy/SciPy end to end.

## Consequences

- Neither method is jit-able, vmap-able or differentiable, and the
  documentation must not imply otherwise.
- **This is not a defect to be fixed.** `kucherenko` is the fastest method in
  the library *because* it never touches the device: its work is many small
  operations on modest arrays, where dispatch and transfer dominate, and the
  host path avoids both.
- A conformance test that asserts "every method has a pure core" reads the
  registry flag and skips these two. A new method must not add itself to the
  exempt list without a measured reason of the same kind.

## Rejected alternatives

- **Port both to JAX for uniformity.** Measured to be slower for
  `kucherenko`, which is the whole reason it is written the way it is. This
  is a change with no behavioural target and a negative one at that; see
  ADR 0009.
- **Leave the exemption undeclared.** Then it is indistinguishable from an
  oversight, and the next conformance sweep "fixes" it.
