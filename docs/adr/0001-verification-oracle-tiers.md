# ADR 0001: Verification oracle tiers T0-T4

Status: accepted (2026-08-18)

## Context

A sensitivity index is a number. A test that computes the number the same way
the source does proves nothing. Before this policy the suite mixed real
external checks with tests that retyped a formula out of the module they were
testing, and nothing in the test told a reader which kind it was looking at.

## Decision

Every method carries a paper and an external numerical check. Rank the check
by tier, and use the strongest tier available. Tier 0 is strongest.

| Tier | What it means | Examples |
|---|---|---|
| **T0** | A closed-form answer, derived independently | Analytic Sobol indices of the Ishigami function |
| **T1** | Reference numbers published in a paper, typed into the test | Tables of indices in a benchmark paper |
| **T2** | A permissive-licence library, installed as a development extra | SALib, UQpy, POT (MIT); OpenTURNS (LGPL-3+) |
| **T3** | A copyleft library, run in a separate process | R `sensitivity`, `sensobol`, `gsaot`; SAFEpython |
| **T4** | Internal consistency only | Finite differences, coverage simulation, invariants |

Four rules:

1. A method must not ship at T4 alone unless there is a recorded reason no
   external oracle exists.
2. **A test that retypes the source's own formula is not an oracle. It is a
   mirror.** It proves only that two copies of one expression agree. Mirrors
   count as no tier at all.
3. **A published table is not automatically T1.** Check it against an
   independent derivation first. ADR 0002 is the case that forced this rule.
4. Record the tier in the test docstring, so a reader knows what the test
   proves.

Oracles run locally, never in CI and never in the package. What is committed
is the number, typed into the test as a literal with a provenance block:
tier, oracle, exact version, date run, and the path to the regenerating script
under `scripts/oracles/`. See `scripts/oracles/README.md`.

## Consequences

- A T2 or T3 oracle becomes a T1 literal for CI purposes. A live comparison
  would catch a regression on either side; a recorded literal only catches
  ours. That is the intended trade, but the docstring must say which it is.
- A number with no provenance block is rejected in review.
- Named mirrors already found in the suite: a test called
  `test_matches_salib_formula` that never imports SALib, and DGSM's retyping
  of `(2*pi)**2/pi**2` from `_poincare.py`.
- A related failure, from the same habit of not asking what a test proves:
  **`slice_chunk_size` means two different things inside `sobol`.** In
  `_analyze_no_bootstrap` it chunks output columns. In `_analyze_bootstrap` it
  chunks *resamples*, and the point estimates there come from `jit_ft`/`jit_so`
  per slice and do not depend on it at all. So a chunk-invariance test written
  only against the bootstrap path asserts nothing about `S1`, `ST` or `S2` —
  only about the `_conf` fields. A real one needs both a `num_resamples=0`
  pair and a `num_resamples>0` pair.

## Rejected alternatives

- **Running oracles in CI.** It couples our build to another project's
  releases, needs R and GPL packages on the runner, and turns their change
  into our red build.
- **Shipping oracle libraries as runtime dependencies.** See ADR 0003.
