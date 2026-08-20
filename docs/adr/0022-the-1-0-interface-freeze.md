# ADR 0022: The 1.0 interface freeze

Status: accepted (2026-08-20)

1.0 fixes the public interface. After this release a rename, a changed
default, or a keyword that disappears costs a major version. This ADR records
what was frozen and why, so a later reader can tell a decision from an
accident.

Everything here was settled in two review sessions on 2026-08-19 and
2026-08-20, working from a code review of the whole package and from running
every example in the documentation.

## Why a single release, 0.8.0 straight to 1.0.0

The work between them was one arc, not three. There were never 0.9 or 0.10
releases: those numbers were development milestones. Shipping them as
separate releases would have published two intermediate interfaces that
nobody used and that we would then be bound by.

One consequence is worth stating, because it caused real errors. The
changelog was written across that arc, so its entries described whatever the
code looked like at the time, including states nobody outside the repository
ever saw. Before release, every entry was re-derived against released 0.8.0.
Twenty-four were wrong. Two examples: a rename of a class that did not exist
in 0.8, and a repositioned keyword that was never there. Release notes must
compare against the last release, not against the previous commit.

## One word for one concept

The same idea now has the same name, type and position everywhere it appears.
Every rename is a clean break with no alias, because an alias is a second
spelling and the point was to have one.

- `standardize_outputs` for output standardization. `standardize` and
  `prenormalize` are gone.
- `correlation_type` for the correlation scale, on `Problem`, `from_dict` and
  `with_correlation`. `correlation_kind` and the bare `kind` are gone.
- `n_bootstrap` for the resample count. `num_resamples` is gone.
- `seed: int | Generator | None = None` on every sampler.
- `key` for a JAX PRNG key, always keyword-only.

`tests/test_vocabulary.py` enforces this by walking the registry, so a
fourteenth method cannot quietly reintroduce a retired spelling.

## The batching contract

Four rules, in every method:

1. `batch_size` sizes row blocks, clamped to N. It never selects a different
   algorithm.
2. `None` means "derive the width from the memory budget".
3. An explicit value always wins over the budget.
4. A degenerate chunk takes the unchunked path.

Rule 4 arrived last and is the one worth remembering. PCE follows it when
`batch_size >= N`, and the Sobol bootstrap follows it for a chunk of one
slice. Both cases were bugs before: PCE streamed when it should not, and the
bootstrap paid for an outer `vmap` over a length-one axis, which cost 2.6x on
scalar output because it makes every gather batched on two axes.

`hsic` has no batching keyword at all. Its kernel matrices are about
`(2D + 1)` resident `N x N` arrays and no keyword bounds that, so offering one
would have been a keyword that could not do what its name promised.

## Observability is on by default

`verbose: bool = True` on all thirteen `analyze()` functions and all four
samplers, printing a problem summary, timings and a ranked result to stdout
through one seam.

Default on was the deliberate part. A sensitivity analysis is nearly always
run interactively at least once, and the cost of a user not knowing their
design collapsed to 64 unique rows is higher than the cost of a printed
block. Anyone who wants silence passes `verbose=False`, and the pure
`indices()` cores never print at all.

Note for anyone reading old notes: `sobol.sample` and `morris.sample` already
printed in 0.8.0. What 1.0 adds is `analyze()` everywhere, plus the other two
samplers.

## What the numbers promise

Bit-for-bit reproducibility is promised at a fixed seed on one machine, and
nothing more. It is not promised across tile widths, batch widths, output
layouts, or CPU targets.

This was learned the hard way. Two tests asserted bit-identity that the
algorithm justified and the compiler did not: XLA picks its vectorization from
an array's whole shape, so a tile width or a batch width reaches the last bit.
They passed on ARM and failed on x86. Where the library reorders nothing, say
so as an algorithmic guarantee and compare with a tolerance.

`scripts/baseline_check.py` holds the line at zero moved values, and every
exception is written down in `scripts/baseline/README.md` with its cause.
1.0 spent two: the DGSM autodiff-mode flip, and the PCE `explained_variance`
correction.

## Claims must be conditional when the mathematics is

Two API-level claims were wrong, and both were found by testing a case the
docs never tried.

`dgsm.lower_bound` implements Kucherenko and Song (2016) Theorem 4.1, which
that paper states for normally distributed inputs. Five docstrings called a
large value a certificate of importance. On `U(0.1, 0.4)` with `1/p` it
reports 1.286 where the true total index is 1. The value is kept, the
condition is now stated, and `analyze` warns when a marginal does not meet it.

`pce.explained_variance` divided a Parseval variance under the input measure
by a sample variance, so it could exceed 1. It reported an almost perfect fit
as pathological. It is now a coefficient of determination.

The rule this sets: a bound is a bound only under its hypotheses, and the code
should say which. If a diagnostic can leave its own range, that is a defect in
the diagnostic, not a curiosity to document.

## Deliberately not in 1.0

Recorded so they are not mistaken for oversights: non-Gaussian copulas (see
ADR 0012, closed as won't-do), an eFAST phase-shift replicate helper, dummy
baselines for PAWN and Borgonovo, machine-readable clip diagnostics on result
objects, Morris trajectory vectorization, and buffer reuse in the Kucherenko
and VKOGA hot loops.

Two known numerical quirks are documented rather than fixed, because both sit
below the noise floor of any design anyone would run: `janon-monod` and
`azzini-rosati` disagree between the scalar and 3-D point-estimate paths by
about 1e-6 in float32, and HSIC's V-statistic is ill-conditioned for small
indices in float32, where a weak parameter can cancel away 2000 parts in 2001.

## References

- ADR 0001, verification oracle tiers. Every method carries its tier.
- ADR 0005, autodiff mode by output shape. Closed by this release.
- ADR 0012, open questions. Closed by this release.
- ADR 0014, float32 default with no x64 wrapper.
- ADR 0021, the Sobol default estimator.
