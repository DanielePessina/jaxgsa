"""Kucherenko Sobol' indices for dependent inputs (design-based).

Design-based method: generate a conditional-copula design, evaluate your
actual model on it, and estimate ``S1 = V(E(Y|X_i))/V(Y)`` and
``ST = E(V(Y|X_{~i}))/V(Y)`` under the problem's declared dependence
structure (Kucherenko, Tarantola & Annoni 2012). No surrogate is fitted. This
is the design-based counterpart to :mod:`jaxgsa.vkoga`, which estimates the
same two quantities against a kernel surrogate. Under independent inputs both
indices reduce exactly to the Saltelli column-swap estimators of the classic
Sobol' ``S1`` and ``ST``.

**Under a declared correlation these numbers are not comparable to
:mod:`jaxgsa.sobol`'s.** They are a different estimand, not the same estimand
reached by a different route. ``S1`` here is correlation-inclusive: it counts
what ``X_i`` explains through its coupling with the other parameters as well
as what it explains alone. ``ST`` is correlation-exclusive. ``ST >= S1``, which
always holds for the classic indices, does not hold here. Do not put a
``kucherenko`` index and a ``sobol`` index in the same table or the same plot
unless the problem is independent.

On the name: Kucherenko calls these Sobol' sensitivity indices, generalized to
dependent inputs, and the 2012 paper is titled for the general case rather than
for its author. The module is named ``kucherenko`` because that is the handle
UQLab and UQpy use for this specific estimator pair, and because a name that
reads as ``sobol`` would invite exactly the false comparison the paragraph
above warns about. It is one of several competing generalizations of Sobol' to
dependent inputs; :mod:`jaxgsa.vkoga`, :mod:`jaxgsa.hdmr`'s ANCOVA split, and
the ANCOVA-based Shapley allocation are three others, and they do not agree
with each other. See the Methods guide for what each one answers.

This module has **no pure core**. It is host NumPy end to end, which is why
it is the fastest method here, so there is no ``indices()`` that survives
``jit``, ``vmap`` or ``jacrev``. That is a declared exemption (see
``jaxgsa._core.registry.MethodSpec.pure_core``), not a gap: no traceable
core exists to export.

Gating: ``sample`` conditions on the problem's declared dependence, either a
``problem.correlation`` or a custom sampler declared with
``problem.with_sampler(f, dimension=k)``, so it is exempt from the
correlated-design refusal that ``sobol``, ``morris``, and ``efast`` apply. It
refuses categorical parameters. The copula conditionals need a continuous
marginal CDF on every coordinate. Use :mod:`jaxgsa.optimal_transport` or
:mod:`jaxgsa.borgonovo` for categorical inputs.

A custom sampler maps uniform points to physical input rows. It receives exactly
three call shapes: joint draws (``fixed_indices=()``), draws that keep ``x_i``
(``fixed_indices=(i,)``), and draws that keep every column except ``i``.

Example::

    import jaxgsa

    problem = jaxgsa.Problem(names=("x1", "x2"), bounds=[(0, 1)] * 2)
    problem = problem.with_correlation(R)      # optional dependence

    ks = jaxgsa.kucherenko.sample(problem, 4096, seed=0)
    Y = model(ks.samples)                      # (n_runs,) evaluations
    result = jaxgsa.kucherenko.analyze(ks, Y)
    result.S1   # correlation-inclusive first-order indices
    result.ST   # correlation-exclusive total indices

Exact Shapley effects and proportional marginal effects: draw the design with
``subsets="all"`` (``N * 2^D`` runs, ``D <= 12``). It adds one block for every
proper subset of the inputs. The result then holds the closed index ``closed``
and the total index ``subset_total`` of every subset, ordered by bitmask::

    ks = jaxgsa.kucherenko.sample(problem, 4096, subsets="all", seed=0)
    result = jaxgsa.kucherenko.analyze(ks, model(ks.samples))
    result.shapley().Sh   # Owen (2014); sums to 1
    result.pme().Sh       # Herin et al. (2024); 0 for an exogenous input

References:
    Kucherenko, Tarantola & Annoni (2012). Comput. Phys. Commun. 183:937-946.
    Owen (2014). SIAM/ASA J. Uncertain. Quantif. 2:245-251.
    Song, Nelson & Staum (2016). SIAM/ASA J. Uncertain. Quantif. 4:1060-1083.
    Herin, Il Idrissi, Chabridon & Iooss (2024). SIAM/ASA J. Uncertain.
    Quantif. 12(2):667-692.
"""

from jaxgsa._core.invalid import InvalidUnit
from jaxgsa._core.registry import MethodSpec, register
from jaxgsa.kucherenko._analyze import analyze
from jaxgsa.kucherenko._result import KucherenkoResult
from jaxgsa.kucherenko._sampling import KucherenkoSamples, sample

__all__ = ["KucherenkoResult", "KucherenkoSamples", "analyze", "sample"]

SPEC = register(
    MethodSpec(
        name="kucherenko",
        analyze=analyze,
        sample=sample,
        result=KucherenkoResult,
        # Deliberately exempt from the correlated-design guard on sobol,
        # morris and efast: this sampler conditions on the dependence
        # structure rather than assuming it away.
        correlation="accepts",
        categorical="refuses",
        bootstrap="n_bootstrap",
        # Host NumPy end to end, so there is no traceable indices(). The
        # exemption is measured, not an oversight. This method is the fastest
        # in the library because it never touches the device: its work is many
        # small operations on modest arrays, where dispatch and transfer
        # dominate. The flag is declared here so a conformance sweep reads it
        # instead of treating the missing core as a defect.
        pure_core=False,
        # One base point carries the 2D+1 conditional rows drawn around it.
        invalid_unit=InvalidUnit.BASE_POINT,
    )
)
