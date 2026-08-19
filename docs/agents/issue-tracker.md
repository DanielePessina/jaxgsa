# Issue tracker

jaxgsa has no external issue tracker for in-progress work. **Issues and specs
live as markdown files in this repository**, under `.scratch/<feature-slug>/`.

They are committed, so a pull request, a commit message or another document can
cite a revision of them.

## Layout

```
.scratch/
  <feature-slug>/
    <whatever the work needs>.md
```

One directory per unit of work. The slug is short, lower case and hyphenated,
and it names the *work*, not the release: `architecture-v1`, not `v0.9`.

There is no required file naming inside the directory. Write the files the work
actually needs — a spec, a measurement dump, an adjudicated task list — and name
them for what they hold (`test-sweep.md`, `perf-before.json`).

## What belongs here

- The specification for a piece of work: what changes, why, and how it will be
  verified.
- Findings a later batch depends on: measurements, sweep results, adjudicated
  lists of things to do.
- Anything a second agent must read to do its half of the job.

## What does not belong here

- **A durable decision with a rationale.** That is an ADR. Write it in
  `docs/adr/` and delete it from the scratch directory. See
  [domain.md](domain.md).
- **A description of how the library currently behaves.** That is `CONTEXT.md`
  or the user documentation.
- **A schedule, a status log, or a progress tracker.** They are stale within a
  week and nobody deletes them. Keep status in the pull request.

## Lifecycle

A scratch directory is **temporary by intent**. When the work lands:

1. Move anything durable into `docs/adr/` or `CONTEXT.md`.
2. Delete the rest.

A scratch directory that survives its feature is a maintenance liability: the
next reader cannot tell which parts still describe reality. Four planning files
totalling three thousand lines were deleted for exactly this reason, and the
roughly two hundred load-bearing lines inside them became ADRs 0001 to 0020.

## Working practice

- One file, one agent. Two agents must never hold the same file — including
  files here. See [ADR 0010](../adr/0010-multi-agent-file-ownership.md).
- Implementation happens in its own git worktree, never the main checkout.
- Pull requests are opened as drafts (`gh pr create --draft`) for review before
  they are marked ready.
