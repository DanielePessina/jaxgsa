# ADR 0008: One warning class, `JaxgsaWarning`

Status: accepted (2026-08-18), shipped

## Context

No warning in the package passed a category, so all of them defaulted to
`UserWarning`. The only way to tell a jaxgsa warning from a NumPy or JAX one
was the message text, and that text was inconsistent: six different prefixes
were in use and two sites had no prefix at all. Filtering on the most common
prefix silenced 28 of 34 sites and missed eFAST, PAWN and DGSM entirely.
Filtering on `UserWarning` also silenced NumPy, SciPy and JAX. No filter
selected exactly these warnings. This matters because several of them fire
once per call inside a loop.

## Decision

Add `JaxgsaWarning(UserWarning)`, export it from the package root, and pass
`category=JaxgsaWarning` at every `warnings.warn` site.

**Do not subdivide it.** One class, no hierarchy of
`JaxgsaConvergenceWarning`, `JaxgsaPrecisionWarning` and so on.

Add a guard test that walks the source with an AST walk and asserts every
`warnings.warn` call passes a category, so a new module cannot drift back.
Use an AST walk rather than a grep: a plain grep counts `warnings.warn`
mentioned inside a comment.

## Consequences

- `warnings.simplefilter("ignore", jaxgsa.JaxgsaWarning)` selects exactly the
  library's warnings and nothing else.
- Subclassing `UserWarning` keeps existing `pytest.warns(UserWarning, ...)`
  assertions passing and keeps the documented behaviour true.

## Rejected alternatives

- **A warning subclass per category of problem.** The filtering problem is
  "jaxgsa versus everything else", and one class solves it. Sub-classes are
  a taxonomy nobody asked for and a compatibility surface to maintain; add
  one only when a caller states a case for silencing one kind and not
  another.
- **Standardising the message prefix instead.** Filtering on text is not an
  API.
