# ADR 0002: The Ishigami convention, and a published table that is wrong

Status: accepted (2026-08-18)

## Context

The Ishigami function is the default T0 oracle for almost every method here.
Two things make it dangerous to use casually: its parameters have three
conventions in circulation that give very different indices, and the most
convenient published table of reference values is corrupted.

## Decision

**Pin the convention.** jaxgsa uses `a = 7`, `b = 0.1`, inputs uniform on
`[-pi, pi]` — the dominant convention and the SALib default. Any test or doc
that quotes an Ishigami index states the convention beside it.

| Convention | S1, S2, S3 |
|---|---|
| `a = 7, b = 0.1` (jaxgsa, SALib) | 0.3139, 0.4424, 0 |
| `a = 7, b = 0.05` | 0.2185, 0.6869, 0 |
| `a = 2, b = 1` (sensobol, hard-coded) | 0.3830, 0.0009, 0 |

**Do not call the `b = 0.05` variant "the Sobol-Levitan convention."** The
Sobol and Levitan paper is real, but it is closed access and the attribution
could not be confirmed. Write "`a = 7, b = 0.05`".

**Use these reference values.** Derived independently and confirmed by Monte
Carlo at 4 million samples:

```
V1 = b*pi^4/5 + b^2*pi^8/50 + 1/2 = 4.3458880
V2 = a^2/8                        = 6.1250000
V3 = 0
V13 = 8*b^2*pi^8/225              = 3.3736999
V   = 13.8445879

S1  = 0.3139   S2  = 0.4424   S3  = 0.0000
ST1 = 0.5576   ST2 = 0.4424   ST3 = 0.2437
```

**Do not use Azzini and Rosati (2022), *Data in Brief* 42:108071 as a T1
oracle without checking each row.** An earlier draft named it as one.
Confirmed problems:

- Its Ishigami row prints `S2 = 0.4413` and `ST2 = 0.4424111`. Input `x2` has
  no interactions, so those two must be equal. `S2` is the wrong one — the ST
  row matches the independent derivation to all seven printed digits.
- Its `S1 = 0.3138` is a mis-rounding of `0.3139`.
- The Hartmann 6-D total column is labelled `S1...S6`.
- Row F3 prints `S3` inside the ST column.
- Several rows are numerical rather than analytic, and nothing marks which.

## Consequences

- A cross-package comparison that does not state the convention is not
  evidence; the three conventions differ by more than any plausible tolerance.
- This is the case that produced rule 3 of ADR 0001: a published table is not
  automatically T1.

## Rejected alternatives

- **Quoting the published table directly.** Faster, and wrong.
- **Supporting all three conventions in fixtures.** Multiplies the reference
  numbers by three for no gain; a reader comparing against another package
  can change `b` themselves.
