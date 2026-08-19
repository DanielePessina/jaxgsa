# ADR 0003: Copyleft libraries as test oracles

Status: accepted (2026-08-18)

## Context

The strongest external checks for several methods live in GPL software: R
`sensitivity`, `sensobol`, `gsaot`, and SAFEpython. jaxgsa ships under a
permissive licence. The question is whether a copyleft oracle can be used at
all, and under what handling.

## Decision

Using a copyleft library as a **development-only** test oracle is acceptable.
Three independent reasons:

1. Copyleft duties trigger on **distribution**. Running a tool during
   development is private use.
2. The published wheel contains no copyleft code. A dev extra is a
   declaration, not a bundle.
3. Running in a separate process is the safe case in the FSF's own guidance.

**This is not legal advice.**

Practical rules, which are the operative part of this ADR:

- Copyleft oracles **must not** appear in `[project.dependencies]`.
- Prefer **out-of-process invocation**. The R oracles satisfy this by
  construction; SAFEpython is subprocess-only for this reason.
- **Do not read copyleft source while implementing.** Work from the paper.
  This is the practice already used for the optimal-transport module, which
  was written clean-room against the papers rather than against gsaot.
- **OpenTURNS is LGPL-3+**, which is weaker. It is safe to import directly.

| Permissive (MIT / BSD / LGPL) | Copyleft (out-of-process, local only) |
|---|---|
| SALib, UQpy, POT, GlobalSensitivity.jl, OpenTURNS | SAFEpython (GPL-3), R `sensitivity` (GPL-2), `sensobol` (GPL-3), `gsaot` (GPL >= 3) |

Because oracles stay out of the package entirely (ADR 0001), `pyproject.toml`
gains no `oracles` extra at all. Local environment recipes live in
`scripts/oracles/README.md`.

## Consequences

- A T3 check can never run in CI, so its output must be committed as a
  literal with a provenance block.
- An implementer who has read a GPL implementation of a method cannot
  cleanly write our version of it. Assign the reading and the writing to
  different people, or read only the paper.

## Rejected alternatives

- **Avoiding copyleft oracles entirely.** For several methods the only
  independent implementation is GPL. Dropping them would push those methods
  to T4, which ADR 0001 forbids without a recorded reason.
- **An `oracles` extra in `pyproject.toml`.** Since oracles never run in CI
  or in the package, the extra would declare a dependency nothing installs.
