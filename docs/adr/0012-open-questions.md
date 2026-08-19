# ADR 0012: Open questions

Status: **open** (2026-08-18)

Two questions are genuinely undecided. They are recorded here so they are not
mistaken for settled in either direction.

## Q1. Non-Gaussian copulas

jaxgsa's `Problem.correlation` supports a Gaussian copula only.
GlobalSensitivity.jl, with Copulas.jl, computes Shapley effects for Clayton,
Frank, Gumbel and t copulas, with exact per-family conditional sampling. That
is strictly more general, and it is the capability gap a reviewer notices
first.

The work is not just sampling: every method that reads `problem.correlation`
assumes a Gaussian conditional, so the conditional-sampling routine has to be
per family.

**Undecided:** whether to add families before 1.0, and if so which.

## Q2. Whether to keep the blanket "pick-freeze" substitution

Project style avoids the term "pick-freeze" and writes "Saltelli column-swap
scheme".

The problem: **the two are not synonyms.** "Pick-freeze" names the *sampling
principle* — hold one input, resample the rest — and Janon et al. use "Sobol
Pick-Freeze" as the formal name of an estimator. Saltelli (2002) and Saltelli
et al. (2010) name the specific `N(d+2)` bookkeeping that jaxgsa implements.
So the substitution is correct where we mean our design, and wrong where a
source means the principle or the Janon estimator.

**Undecided:** keep the blanket substitution for consistency, or allow
"pick-freeze" where it is technically the right word. The second needs a rule
a writer can apply without thinking about it, or it will be applied
inconsistently, which is worse than either.
