# Domain documentation

jaxgsa keeps its domain knowledge in **one context**, at the repository root.
There is no per-package or per-module context file, and adding one is a
mistake: knowledge split across directories drifts, and no reader knows which
copy is current.

Two files, with a clean split.

## `CONTEXT.md` — what is true now

The shared vocabulary and the current rules. One term, one meaning, one
spelling. It covers:

- Domain terms: `Problem`, `Theta`, input spec, design-based versus given-data
  method, output slice, sample row, invalid unit, pure core, preamble.
- The parameter vocabulary: which batching keyword a method takes and why,
  the bootstrap and confidence-interval keywords, entry-point naming,
  precision.
- Current policies: warn versus error, the T0-T4 tier summary, atomic kernel
  then vmap, ragged chunks, numerical baselines, tests justify their presence.

It is **normative for the public interface**. `tests/test_vocabulary.py` reads
these rules off the method registry and fails when a signature drifts from
them, so `CONTEXT.md` is enforced, not aspirational.

Write in `CONTEXT.md` when the answer to "what does this word mean here?" or
"what does the library do?" changes.

## `docs/adr/` — why it is true

One file per decision: **context, decision, consequences, rejected
alternatives**, with a status line. An ADR exists so that a future reviewer who
re-proposes a rejected option finds the answer instead of repeating the work.

Write an ADR when there was a real choice, when something was measured and
rejected, or when the obvious implementation is wrong and needs a recorded
reason. See [`docs/adr/README.md`](../adr/README.md) for the index and the
template.

## Which file

| You want to record | Where |
|---|---|
| What a keyword means, or which methods take it | `CONTEXT.md` |
| Why the keyword is spelled that way and not the obvious alternative | ADR |
| That negative Sobol indices are returned unclipped | `CONTEXT.md`, if a caller needs to know |
| Why clipping was rejected, with the bias argument | ADR |
| A measurement that settled an argument | ADR |
| A plan, a task list, a status | `.scratch/<slug>/`, and delete it when it lands. See [issue-tracker.md](issue-tracker.md) |

The two files are allowed to overlap in *subject* but not in *content*. Where
`CONTEXT.md` states a policy in one paragraph and the full reasoning lives in
an ADR, `CONTEXT.md` links the ADR rather than restating it.

## Rules

- **Do not duplicate.** A fact stated in both files will be updated in one of
  them. Link instead.
- **Never renumber an ADR.** A superseded one keeps its number and gains a
  `Status: superseded by ADR NNNN` line. Deleting it destroys the record of
  why the earlier answer was wrong, which is the part with value.
- **Keep them short.** A document nobody reads is as useless as a deleted one.
- **No new top-level planning file.** That is how the repository accumulated
  three thousand lines of stale schedule. Work in progress goes in
  `.scratch/`; conclusions go here.
