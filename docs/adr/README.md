# Architecture decision records

An ADR records **one decision, why it was made, and what was rejected**. It is
written so that a future reviewer who re-proposes the rejected option finds the
answer instead of repeating the work.

`CONTEXT.md` at the repository root is the other half of this pair. It says
what the world is like *now* — the vocabulary, the current policies, the
current shape of the API. An ADR says why. If the two disagree, `CONTEXT.md`
is right about the present and the ADR is right about the reasoning; fix the
ADR's status line rather than deleting it.

## Writing one

Copy the shape of any file here: **Context / Decision / Consequences /
Rejected alternatives**, with a status line at the top. Number it with the
next free number and never renumber. Keep it short — an ADR nobody reads is
as useless as a deleted one. A superseded ADR keeps its number and gains a
`Status: superseded by ADR NNNN` line.

## The records

### Verification

| # | Decision |
|---|---|
| [0001](0001-verification-oracle-tiers.md) | Verification oracle tiers T0-T4; a test that retypes the source is a mirror, not an oracle |
| [0002](0002-ishigami-reference-values.md) | The Ishigami convention, the derived reference values, and a published table that is wrong |
| [0003](0003-copyleft-oracles-and-licences.md) | Copyleft libraries as test oracles: out-of-process, never a dependency, do not read the source |
| [0004](0004-oracle-inventory.md) | Which oracles exist, at which versions, and that ATHENA is a dead end |

### Numerics and API

| # | Decision |
|---|---|
| [0005](0005-autodiff-mode-selection.md) | Choose forward or reverse mode from the output shape; every speed claim carries the `T` factor |
| [0006](0006-convergence-api-deferred.md) | Convergence analysis is deferred, and the obvious implementation is wrong |
| [0007](0007-on-invalid-policy.md) | `on_invalid` covers `X` and `Y`; `"drop"` is refused for `X` on a design |
| [0008](0008-single-warning-class.md) | One warning class, `JaxgsaWarning`, not subdivided |
| [0013](0013-problem-is-not-a-pytree.md) | `Problem` stays a plain value object; gradients go through `Theta` |
| [0014](0014-float32-default-no-x64-wrapper.md) | float32 stays the default, and there is no `jaxgsa.enable_x64()` |
| [0015](0015-pure-core-exemptions.md) | `kucherenko` and `vkoga` are exempt from the pure-core rule, by declaration |
| [0016](0016-no-configuration-only-precondition-for-borgonovo-bandwidth.md) | No up-front check on Borgonovo's `degenerate_bandwidth` |
| [0017](0017-negative-sobol-estimates-are-not-clipped.md) | Negative Sobol estimates are reported, not clipped |
| [0018](0018-jit-cache-keys-carry-no-data.md) | A jit cache key carries metadata, never data |
| [0019](0019-pce-leave-one-out-uses-a-cholesky-factor.md) | PCE leave-one-out leverage comes from a Cholesky factor |

### Scope and process

| # | Decision |
|---|---|
| [0009](0009-a-change-with-no-behavioural-target.md) | A change with no behavioural target must earn its place |
| [0010](0010-multi-agent-file-ownership.md) | One file, one agent |
| [0011](0011-out-of-scope-for-1.0.md) | Out of scope for 1.0: plotting, CLI, distance correlation as a module, RSA |
| [0012](0012-open-questions.md) | **Open:** non-Gaussian copulas; the "pick-freeze" terminology rule |
| [0020](0020-constraints-on-methods-not-yet-built.md) | Constraints on methods not yet built — read before implementing one |
