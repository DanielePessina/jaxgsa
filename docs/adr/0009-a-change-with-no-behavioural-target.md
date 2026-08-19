# ADR 0009: A change with no behavioural target must earn its place

Status: accepted (2026-08-18)

## Context

An audit of the codebase produced twenty-one decisions. Seven of them improved
only the shape of the code: new spec dataclasses, a layout enum, frozen
result classes, a changed `.npz` payload. The original plan shipped all
twenty-one in one release. Two batches shipped under that plan, and their
outcome is the evidence for this ADR.

**What the completed work showed.** The items that paid were the ones with a
behavioural target: the single warning class, a 3.4x faster row deduplication,
a real Morris `downsample` bug, and a test sweep that found unasserted
near-zero entries, a Sobol coverage hole, and a test comparing SciPy against
SciPy.

**The two changes that produced *new* defects were both pure refactors.** The
`np.unique` rewrite was a performance *regression* until it was re-measured.
The rewritten Sobol test covered the wrong code path. Three independent review
agents were needed to catch them.

## Decision

Restructuring is not free. It costs a full review cycle, and it has a defect
rate of its own. **Where a change improves only the shape of the code, it
waits until a release needs that shape for something else.**

Three supporting reasons:

1. **The breaking-change budget is better spent elsewhere.** A user pays a
   migration cost either way. Spending it on internal ergonomics buys them
   nothing.
2. **A deferred representation change pays for itself once, later.** Adding
   input distributions needs an internal marginal protocol, and that is when
   the specification representation genuinely has to change. Doing it early
   means doing it twice.
3. **Timing.** No other GSA package is written in JAX, but `jaxonomy` was
   created on 2026-07-06 and already has more downloads per month. The
   first-mover position is real and not permanent. Internal refactoring is the
   most expensive way to spend that window.

## Consequences

- A pull request whose description cannot name a behaviour that changes, a
  number that moves, or a defect class it forecloses is a candidate for
  rejection, not for review.
- Refactors that do ship are held to a bit-identical numerical baseline. A
  changed number in a "plumbing only" change is a wiring error, not a
  tolerance issue. See `scripts/baseline/README.md`.

## Rejected alternatives

- **Clean up first, then build.** The measured defect rate of the cleanup was
  higher than that of the feature work it was meant to make safe.
- **Never refactor.** Not the rule. The rule is that the shape change rides
  along with the feature that needs it.
