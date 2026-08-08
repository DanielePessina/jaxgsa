"""Kucherenko conditional-copula design for dependent-input Sobol' indices.

The design follows the single-loop scheme of Kucherenko, Tarantola & Annoni
(2012). For each parameter ``i``, write ``y = X_i`` and ``z = X_{~i}``. One
base point ``k`` contributes three input rows:

1. ``(y_k, z_k)`` drawn from the joint distribution,
2. ``(y_k, z'_k)`` with ``z'_k`` drawn from the conditional ``p(z | y = y_k)``,
3. ``(y'_k, z_k)`` with ``y'_k`` drawn from the conditional ``p(y | z = z_k)``.

Row 1 is shared by every parameter, so the full design has ``n * (2D + 1)``
rows. The dependence structure is the problem's Gaussian copula. Both
conditionals are closed-form Gaussians in the latent standard-normal space
(:func:`jaxgsa._core.copula.build_conditional_plan`). The marginal inverse CDFs
map the latent draws to physical units at the very end. Under an identity
correlation the conditionals collapse to fresh independent draws, and the
design is exactly the Saltelli column-swap scheme with ``A``,
``AB_i``-complement, and ``AB_i`` blocks.

The base draws come from one scrambled Sobol' sequence of dimension ``2D``. The
first ``D`` columns drive the joint block and the last ``D`` drive the
conditional redraws. That is the same pairing the Saltelli ``(A, B)``
construction uses.

References:
    Kucherenko, Tarantola & Annoni (2012). Comput. Phys. Commun. 183:937-946.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from jaxgsa._core.copula import (
    assemble_latent,
    build_conditional_plan,
    draw_rest_given_self,
    draw_self_given_rest,
    independent_correlation,
    latent_normal_sample,
    latent_to_physical,
)
from jaxgsa._core.samples import UniqueDesignSamples
from jaxgsa._core.sampling import _next_power_of_2
from jaxgsa._core.validation import _raise_categorical_design
from jaxgsa.problem import Problem


@dataclass(frozen=True)
class KucherenkoSamples(UniqueDesignSamples):
    """Kucherenko conditional-copula design, returned by :func:`sample`.

    Evaluate your model at every row of ``samples`` (shape
    ``(n_runs, D)``), then pass this object and the outputs to
    :func:`jaxgsa.kucherenko.analyze`.

    The design is stored as ``2D + 1`` stacked blocks of ``base_n`` rows each.
    The joint block comes first. Then comes one conditional block per parameter
    for the first-order index, and one per parameter for the total index. Every
    conditional draw is a distinct continuous point, so the design has no
    duplicate rows and the expansion map is the identity. The
    :class:`~jaxgsa._core.samples.UniqueDesignSamples` base is used for its NPZ
    persistence and metadata schema, not for deduplication.

    Attributes:
        samples: Rows to evaluate in the problem's physical units, shape
            ``(base_n * (2D + 1), D)``.
        n_expanded: Equal to ``n_runs``, because the design has no duplicate
            rows.
        expanded_to_unique: Identity index map, shape ``(n_runs,)``.
        base_n: Base points ``n`` per block, a power of two.
        n_params: Number of parameters ``D``.
        problem: Problem definition used for the analysis, including its
            ``correlation``.
    """

    samples: np.ndarray
    n_expanded: int
    expanded_to_unique: np.ndarray
    base_n: int
    n_params: int
    problem: Problem

    def __repr__(self) -> str:
        """Return a concise summary of the design dimensions."""
        return (
            f"KucherenkoSamples(n_runs={self.n_runs}, base_n={self.base_n}, "
            f"n_params={self.n_params}, correlated={self.problem.has_correlated_inputs})"
        )

    def _extra_metadata(self) -> dict[str, Any]:
        """Persist the design parameters in the metadata blob."""
        return {"base_n": self.base_n}

    @classmethod
    def _from_payload(
        cls,
        *,
        samples: np.ndarray,
        n_expanded: int,
        expanded_to_unique: np.ndarray,
        problem: Problem,
        arrays: Mapping[str, np.ndarray],
        meta: Mapping[str, Any],
    ) -> KucherenkoSamples:
        """Rebuild a ``KucherenkoSamples`` from a loaded NPZ payload."""
        return cls(
            samples=samples,
            n_expanded=n_expanded,
            expanded_to_unique=expanded_to_unique,
            base_n=int(meta["base_n"]),
            n_params=problem.num_vars,
            problem=problem,
        )


def sample(
    problem: Problem,
    n_samples: int,
    *,
    scramble: bool = True,
    seed: int = 0,
) -> KucherenkoSamples:
    """Generate the conditional-copula design for :func:`jaxgsa.kucherenko.analyze`.

    The design costs ``base_n * (2D + 1)`` model evaluations, where ``base_n``
    is ``n_samples`` rounded up to the next power of two (the Sobol' sequence
    keeps its balance guarantees only at powers of two).

    The dependence structure comes from ``problem.correlation``. An
    uncorrelated problem gives the classic Saltelli column-swap design, and
    the analysis then reproduces the classic Sobol' indices. This sampler is
    deliberately exempt from the correlated-design guard on ``sobol``,
    ``morris``, and ``efast``. Conditioning on the declared copula is exactly
    what the Kucherenko estimators are for.

    Args:
        problem: Problem definition with parameter names, marginals, and an
            optional ``correlation``.
        n_samples: Requested base points per block, at least 2. Rounded up to
            the next power of two.
        scramble: Whether to Owen-scramble the Sobol' sequence. Keep it on:
            different seeds then give statistically independent designs.
        seed: Seed for the scrambling.

    Returns:
        A :class:`KucherenkoSamples` carrying the stacked design.

    Raises:
        ValueError: If the problem has fewer than two parameters or any
            categorical parameter, or if ``n_samples < 2``.
    """
    _raise_categorical_design(problem, "jaxgsa.kucherenko.sample")
    D = problem.num_vars
    if D < 2:
        raise ValueError(f"Kucherenko indices need at least 2 parameters, got {D}")
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    n = _next_power_of_2(n_samples)

    R = problem.correlation
    if R is None:
        R = independent_correlation(D)
    plan = build_conditional_plan(R)
    chol_full = np.linalg.cholesky(R)

    # One scrambled Sobol' block of dimension 2D: the first D columns drive
    # the joint sample, the last D the conditional redraws. Pairing them
    # inside one QMC point set is the Saltelli (A, B) construction.
    draws = latent_normal_sample(n, 2 * D, seed=seed, scramble=scramble)
    base = draws[:, :D]
    redraw = draws[:, D:]
    Z_joint = base @ chol_full.T

    blocks = [Z_joint]
    # First-order blocks: keep Z_i, redraw the rest from p(z_rest | z_i).
    for i in range(D):
        others = plan.others[i]
        z_rest = draw_rest_given_self(plan, i, Z_joint[:, i], redraw[:, others])
        blocks.append(assemble_latent(i, others, Z_joint[:, i], z_rest))
    # Total blocks: keep Z_rest, redraw Z_i from p(z_i | z_rest).
    for i in range(D):
        others = plan.others[i]
        z_self = draw_self_given_rest(plan, i, Z_joint[:, others], redraw[:, i])
        blocks.append(assemble_latent(i, others, z_self, Z_joint[:, others]))

    X = latent_to_physical(problem, np.concatenate(blocks, axis=0))
    n_total = n * (2 * D + 1)
    return KucherenkoSamples(
        samples=X,
        n_expanded=n_total,
        expanded_to_unique=np.arange(n_total, dtype=np.int64),
        base_n=n,
        n_params=D,
        problem=problem,
    )
