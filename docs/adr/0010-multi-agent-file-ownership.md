# ADR 0010: One file, one agent

Status: accepted (2026-08-18)

## Context

Work on this repository is done by several coding agents at a time, often in
one worktree. The failure mode is two agents editing the same file, which
produces a merge no reviewer can read and, worse, a half-applied change that
still passes tests.

## Decision

**Two agents must never hold the same file.**

- Give each agent an explicit list of the files it owns.
- Tell it to **report** rather than edit anything outside that list.
- Batches that touch disjoint files may run in parallel. Batches that do not,
  do not.
- Implementation runs in its own git worktree, never the main checkout.
- A review agent must be a **fresh** agent, not the implementer and not a
  fork of it. An agent that wrote the code cannot audit its own blast radius.
  Where several review passes are wanted, they run in parallel and
  independently.

## Consequences

- This rule is what let two batches run concurrently in one worktree without
  a single collision.
- A change that genuinely needs two agents in one file is a signal that the
  batching is wrong, not that the rule should be relaxed.
- Because agents report rather than edit outside their list, the coordinator
  applies those reported edits. Reported-but-unapplied edits are the residual
  risk; collect them at the end of each batch.

## Rejected alternatives

- **Locking by convention and merging afterwards.** Tried implicitly; the
  cost is a review cycle per collision.
- **One agent for everything.** Serialises work that is genuinely
  independent, and a single long-running agent loses the fresh-reviewer
  property.
