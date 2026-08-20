# ADR 0012: Open questions

Status: **closed** (2026-08-20). Both questions were answered for the 1.0
freeze. The answers are at the bottom of this file, under "Resolution". The
question text is kept as written so the reasoning that led to each answer
stays readable.

Two questions were genuinely undecided. They are recorded here so they are not
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

## Resolution (2026-08-20)

### Q1: the Gaussian copula is the scope of the library

Decided as won't-do, not as deferred. `Problem.correlation` describes
dependence with a Gaussian copula, and that is the dependence model jaxgsa
implements. The question is closed rather than carried forward.

The reasoning is that the cost is not in the sampling. Every method that reads
`problem.correlation` assumes a Gaussian conditional, so adding a family means
writing a conditional routine per family and re-deriving the conditional
variance argument in each method that uses one. That is a second dependence
system living beside the first, and it would need its own oracles at every
tier before any of its numbers could be trusted.

What this costs a user: a model whose inputs have tail dependence, or any
dependence a Gaussian copula cannot express, is outside what jaxgsa measures.
The declared rank correlation still holds, so the marginals and the pairwise
ranks are right, but the joint behaviour in the tails is not. Say so where the
docs describe `correlation`, rather than leaving a reader to infer it.
GlobalSensitivity.jl with Copulas.jl covers Clayton, Frank, Gumbel and t, and
is the honest recommendation for that case.

Reopening this needs a new ADR, not an edit to this one.

### Q2: keep the blanket substitution, with one cited exception

Write "Saltelli column-swap scheme" everywhere by default. Never write
"pick-freeze" for jaxgsa's own design, in code, comments, docs or commit
messages.

The one exception, which is mechanical enough to apply without judgement:
**when naming a published estimator or quoting a source, use the source's own
term and cite it.** So "the Sobol Pick-Freeze estimator of Janon et al.
(2014)" is correct, because that is the estimator's name. "jaxgsa uses a
pick-freeze design" is not, because our design is Saltelli's `N(d+2)`
bookkeeping.

The rule is a citation test: if the sentence carries a citation and names
somebody else's estimator, their term wins. Otherwise ours does. That answers
the objection raised in Q2, which was that any exception without a mechanical
rule gets applied inconsistently.
