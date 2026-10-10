"""Kucherenko conditional-copula design for dependent-input Sobol' indices.

The design follows the single-loop scheme of Kucherenko, Tarantola & Annoni
(2012). For each parameter ``i``, write ``y = X_i`` and ``z = X_{~i}``. One
base point ``k`` contributes three input rows:

1. ``(y_k, z_k)`` drawn from the joint distribution,
2. ``(y_k, z'_k)`` with ``z'_k`` drawn from the conditional ``p(z | y = y_k)``,
3. ``(y'_k, z_k)`` with ``y'_k`` drawn from the conditional ``p(y | z = z_k)``.

Row 1 is shared by every parameter, so the full design has ``n * (2D + 1)``
rows. One :class:`~jaxgsa._core.dependence.Draws` strategy builds every block,
chosen by ``problem.dependence``:

* ``"independent"`` and ``"gaussian"`` use the problem's Gaussian copula. Both
  conditionals are closed-form Gaussians in the latent standard-normal space.
  The marginal inverse CDFs map the latent draws to physical units at the very
  end. Under an identity correlation the conditionals collapse to fresh
  independent draws, and the design is the Saltelli column-swap scheme.
* ``"custom"`` uses the sampler declared on the problem. It maps uniform points
  to physical rows and receives exactly three call shapes: joint draws
  (``fixed_indices=()``), draws that keep ``x_i`` (``fixed_indices=(i,)``), and
  draws that keep every column except ``i``.

The base draws come from one scrambled Sobol' sequence. Its first
``draws.dimension`` columns drive the joint block, and the remaining columns
drive the conditional redraws. That is the same pairing the Saltelli
``(A, B)`` construction uses.

``subsets="all"`` builds a larger design for exact Shapley effects and
proportional marginal effects. For each proper non-empty subset ``w`` of the
inputs it adds one block that keeps ``X_w`` from the joint block and redraws
``X_{~w}`` from ``p(X_{~w} | X_w)``, with
:meth:`~jaxgsa._core.dependence.Draws.free_given_fixed`. The joint block ``A``
and an independent joint block ``B`` complete the design, so it has
``n * 2^D`` rows. The singleton blocks are the same two families of blocks
that ``subsets="singletons"`` builds: keeping ``{i}`` is the first-order block,
and keeping every column except ``i`` is the total block.

References:
    Kucherenko, Tarantola & Annoni (2012). Comput. Phys. Commun. 183:937-946.
    Owen (2014). SIAM/ASA J. Uncertain. Quantif. 2:245-251.
    Song, Nelson & Staum (2016). SIAM/ASA J. Uncertain. Quantif. 4:1060-1083.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.dependence import draws_for, sobol_uniforms
from jaxgsa._core.samples import UniqueDesignSamples
from jaxgsa._core.sampling import _next_power_of_2
from jaxgsa._core.validation import _raise_categorical_design
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.problem import Dependence, Problem


@dataclass(frozen=True)
class KucherenkoSamples(UniqueDesignSamples):
    """Kucherenko conditional-copula design, returned by :func:`sample`.

    Evaluate your model at every row of ``samples`` (shape
    ``(n_runs, D)``), then pass this object and the outputs to
    :func:`jaxgsa.kucherenko.analyze`.

    The default design is stored as ``2D + 1`` stacked blocks of ``base_n``
    rows each. The joint block comes first. Then comes one conditional block
    per parameter for the first-order index, and one per parameter for the
    total index. With ``subsets="all"`` it is ``2^D`` blocks instead: the
    joint block ``A``, an independent joint block ``B``, and one conditional
    block for each subset ``w`` with ``1 <= |w| <= D - 1``, in the order of the
    bitmask of ``w`` (bit ``j`` is parameter ``j``). Every
    row is retained, including repeats under deterministic dependence, and the
    expansion map is the identity. The
    :class:`~jaxgsa._core.samples.UniqueDesignSamples` base is used for its NPZ
    persistence and metadata schema, not for deduplication.

    Attributes:
        samples: Rows to evaluate in the problem's physical units, shape
            ``(base_n * (2D + 1), D)``.
        n_expanded: Equal to ``n_runs``; every design row is retained.
        expanded_to_unique: Identity index map, shape ``(n_runs,)``.
        base_n: Base points per block; a power of two for Sobol sampling.
        dependence: How the inputs depend on each other, as the design was
            drawn: ``"independent"``, ``"gaussian"`` or ``"custom"``. A saved
            file keeps this label and the sampler dimension, but not the
            sampler function. A loaded custom problem still reports
            ``"custom"``, and it raises if asked to draw new rows.
        n_params: Number of parameters ``D``.
        problem: Problem definition used for the analysis, including its
            ``correlation``.
        subsets: ``"singletons"`` for the ``2D + 1`` block design, or
            ``"all"`` for the ``2^D`` block design that supports exact
            Shapley effects and proportional marginal effects.
    """

    samples: np.ndarray
    n_expanded: int
    expanded_to_unique: np.ndarray
    base_n: int
    n_params: int
    problem: Problem
    dependence: Dependence = "independent"
    subsets: Literal["singletons", "all"] = "singletons"

    @property
    def n_blocks(self) -> int:
        """Number of stacked blocks: ``2D + 1``, or ``2^D`` for ``subsets="all"``."""
        return n_blocks_for(self.n_params, self.subsets)

    def __repr__(self) -> str:
        """Return a concise summary of the design dimensions."""
        return (
            f"KucherenkoSamples(n_runs={self.n_runs}, base_n={self.base_n}, "
            f"n_params={self.n_params}, dependence={self.dependence!r}, "
            f"subsets={self.subsets!r})"
        )

    def _extra_metadata(self) -> dict[str, Any]:
        """Persist the design parameters in the metadata blob."""
        return {"base_n": self.base_n, "dependence": self.dependence, "subsets": self.subsets}

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
        # The signature is the UniqueDesignSamples._from_payload hook. This
        # design stores no extra arrays, so the slot goes unread here.
        del arrays
        return cls(
            samples=samples,
            n_expanded=n_expanded,
            expanded_to_unique=expanded_to_unique,
            base_n=int(meta["base_n"]),
            n_params=problem.num_vars,
            problem=problem,
            dependence=_parse_dependence(meta.get("dependence"), problem),
            subsets=_parse_subsets(meta.get("subsets")),
        )


MAX_SUBSET_PARAMS = 12
"""Largest ``D`` that ``subsets="all"`` accepts. The design has ``n * 2^D`` rows."""


def n_blocks_for(n_params: int, subsets: str) -> int:
    """Return the number of design blocks for ``n_params`` and a ``subsets`` mode."""
    return 2**n_params if subsets == "all" else 2 * n_params + 1


def _parse_subsets(value: Any) -> Literal["singletons", "all"]:
    """Read the stored ``subsets`` label. Old files have none, so they are singletons."""
    if value is None:
        return "singletons"
    if value == "singletons" or value == "all":
        return value
    raise ValueError(f"Unknown subsets {value!r} in saved Kucherenko design")


def _parse_dependence(value: Any, problem: Problem) -> Dependence:
    """Read the stored dependence label, falling back to the problem's."""
    if value is None:
        return problem.dependence
    if value == "independent" or value == "gaussian" or value == "custom":
        return value
    raise ValueError(f"Unknown dependence {value!r} in saved Kucherenko design")


def _warn_independent_problem(problem: Problem, n: int) -> None:
    """Say that an uncorrelated problem is paying for a design it does not need.

    With no declared dependence the conditional redraws are plain independent
    draws, so the design degenerates to a Saltelli column-swap scheme that
    carries one block per parameter where ``sobol`` reuses a single ``B`` block
    for all of them. The indices are then the classic Sobol' ``S1`` and ``ST``:
    the same estimand, at ``(2D + 1) / (D + 2)`` times the model cost, and with
    no ``S2``. Running it that way on purpose is legitimate — it is how you
    cross-check a correlated analysis against a design-based reference — so
    this warns rather than raises.

    The message quotes all three run counts. The choice is a budget choice, and
    the ratio between the three block formulas is not obvious at a glance.

    Args:
        problem: Problem the design was requested for.
        n: Base points per block, already rounded up to a power of two.

    Warns:
        JaxgsaWarning: If ``problem.dependence`` is ``"independent"``.
    """
    if problem.dependence != "independent":
        return
    D = problem.num_vars
    warnings.warn(
        "jaxgsa.kucherenko.sample: the problem declares no dependence. The "
        "conditional redraws are then plain independent draws, so this design "
        "reduces to the Saltelli column-swap scheme and the indices are the "
        f"classic Sobol' S1 and ST. It costs {n * (2 * D + 1)} runs "
        f"(base_n {n} x (2D+1)). jaxgsa.sobol.sample gives the same two "
        f"indices from {n * (D + 2)} runs (base_n x (D+2)), or those plus S2 "
        f"from {n * (2 * D + 2)} runs (base_n x (2D+2)). Run kucherenko on an "
        "independent problem only to cross-check a correlated analysis, or "
        "declare the dependence with problem.with_correlation(R) or "
        "problem.with_sampler(f, dimension=k).",
        category=JaxgsaWarning,
        stacklevel=3,
    )


def sample(
    problem: Problem,
    n_samples: int,
    *,
    subsets: Literal["singletons", "all"] = "singletons",
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
    verbose: bool = True,
) -> KucherenkoSamples:
    """Generate the conditional design for :func:`jaxgsa.kucherenko.analyze`.

    The design costs ``base_n * (2D + 1)`` model evaluations, where ``base_n``
    is ``n_samples`` rounded up to the next power of two (the Sobol' sequence
    keeps its balance guarantees only at powers of two).

    The dependence structure comes from the problem. ``problem.correlation``
    declares a Gaussian copula. A sampler declared with
    ``problem.with_sampler(f, dimension=k)`` supplies custom joint and
    conditional draws instead. The sampler receives exactly three call shapes:
    joint draws (``fixed_indices=()``), draws that keep ``x_i``
    (``fixed_indices=(i,)``), and draws that keep every column except ``i``.
    Problem marginals then give metadata, not a fit of the custom distribution.
    Saved designs can be analyzed without the original sampler. The file does
    not store the sampler function, so attach it again with
    ``problem.with_sampler`` before you draw a new design.

    An independent problem gives the classic Saltelli column-swap design, and
    the analysis then reproduces the classic Sobol' indices. It warns in that
    case: the same indices cost ``base_n * (D + 2)`` runs through
    :func:`jaxgsa.sobol.sample`, or ``base_n * (2D + 2)`` runs with ``S2``
    included, against ``base_n * (2D + 1)`` here. This sampler is
    deliberately exempt from the correlated-design guard on ``sobol``,
    ``morris``, and ``efast``. Conditioning on the declared dependence is
    exactly what the Kucherenko estimators are for.

    ``subsets="all"`` builds the design for exact Shapley effects and
    proportional marginal effects. It costs ``base_n * 2^D`` runs and is
    limited to ``D <= 12``. For each proper non-empty subset ``w`` it keeps
    ``X_w`` from the joint block and redraws the other inputs from their
    conditional given ``X_w``. A custom sampler must then accept any sorted
    ``fixed_indices`` and act as a conditional-quantile (inverse Rosenblatt)
    map; see :class:`~jaxgsa._core.dependence.InputSampler`.

    Args:
        problem: Problem definition with parameter names, marginals, and an
            optional ``correlation`` or custom ``sampler``.
        n_samples: Requested base points per block, at least 2. Rounded up to
            the next power of two.
        subsets: ``"singletons"`` (default) builds the ``2D + 1`` block design
            for ``S1`` and ``ST``. ``"all"`` builds the ``2^D`` block design
            that adds closed and total indices of every subset, and with them
            ``result.shapley()`` and ``result.pme()``.
        scramble: Whether to Owen-scramble the Sobol' sequence. Keep it on:
            different seeds then give statistically independent designs.
        seed: Random seed or generator for the Sobol' scrambling. Requires
            ``scramble=True``.
        verbose: If ``True`` (default), print a one-line summary of the
            design: base points, blocks, total runs, and the dependence kind.
            Pass ``False`` for a silent run.

    Returns:
        A :class:`KucherenkoSamples` carrying the stacked design.

    Warns:
        JaxgsaWarning: If ``problem.dependence`` is ``"independent"``. The
            design then reduces to a Saltelli column-swap scheme and
            ``jaxgsa.sobol`` reaches the same two indices for fewer model runs.

    Raises:
        ValueError: If ``subsets`` is not ``"singletons"`` or ``"all"``, if
            ``subsets="all"`` and the problem has more than 12 parameters, if
            the problem has fewer than two parameters or any
            categorical parameter, if ``n_samples < 2``, if a ``seed`` is
            given with ``scramble=False`` (the unscrambled sequence is
            deterministic, so the seed would do nothing), or if a custom
            sampler returns invalid rows.
    """
    if not scramble and seed is not None:
        raise ValueError(
            "jaxgsa.kucherenko.sample: seed has no effect with scramble=False. "
            "The unscrambled Sobol' sequence is deterministic, so the seed "
            "would do nothing. Use scramble=True, or drop the seed."
        )
    _raise_categorical_design(problem, "jaxgsa.kucherenko.sample")
    D = problem.num_vars
    if D < 2:
        raise ValueError(f"Kucherenko indices need at least 2 parameters, got {D}")
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    if subsets not in ("singletons", "all"):
        raise ValueError(f"subsets must be 'singletons' or 'all', got {subsets!r}")
    n = _next_power_of_2(n_samples)
    if subsets == "all":
        if D > MAX_SUBSET_PARAMS:
            raise ValueError(
                f"subsets='all' needs {n} * 2^{D} = {n * 2**D} model runs for {D} "
                f"parameters, and it supports at most {MAX_SUBSET_PARAMS} parameters. "
                "The cost doubles with every parameter. For more parameters, use a "
                "method that works on given data, such as jaxgsa.shapley.analyze "
                "with a fitted surrogate, or jaxgsa.vkoga."
            )
        return _sample_all_subsets(problem, n, scramble=scramble, seed=seed, verbose=verbose)
    _warn_independent_problem(problem, n)

    draws = draws_for(problem)
    # One Sobol' block: the first k columns drive the joint block, the rest
    # the conditional redraws (the Saltelli (A, B) pairing). Each redraw reads
    # the leading columns it needs.
    k = draws.dimension
    width = max(draws.rest_dimension, draws.self_dimension)
    points = sobol_uniforms(n, k + width, seed=seed, scramble=scramble)
    redraw = points[:, k:]
    joint = draws.joint(points[:, :k])
    blocks = [joint]
    # First-order blocks: keep x_i, redraw the rest given x_i.
    rest_noise = redraw[:, : draws.rest_dimension]
    blocks.extend(draws.rest_given_self(rest_noise, joint, i) for i in range(D))
    # Total blocks: keep x_-i, redraw x_i given the rest.
    self_noise = redraw[:, : draws.self_dimension]
    blocks.extend(draws.self_given_rest(self_noise, joint, i) for i in range(D))
    X = draws.to_physical(np.concatenate(blocks, axis=0))

    n_total = n * (2 * D + 1)
    if verbose:
        _verbose.sample_summary(
            method="jaxgsa.kucherenko.sample",
            problem=problem,
            entries=[
                ("base_n", n),
                ("n_blocks", 2 * D + 1),
                ("n_runs", n_total),
                ("dependence", problem.dependence),
                ("scramble", scramble),
            ],
        )
    return KucherenkoSamples(
        samples=X,
        n_expanded=n_total,
        expanded_to_unique=np.arange(n_total, dtype=np.int64),
        base_n=n,
        n_params=D,
        problem=problem,
        dependence=problem.dependence,
        subsets="singletons",
    )


def subset_fixed(mask: int, n_params: int) -> tuple[int, ...]:
    """Return the sorted parameter indices in a subset bitmask (bit ``j`` is parameter ``j``)."""
    return tuple(j for j in range(n_params) if (mask >> j) & 1)


def _sample_all_subsets(
    problem: Problem,
    n: int,
    *,
    scramble: bool,
    seed: int | np.random.Generator | None,
    verbose: bool,
) -> KucherenkoSamples:
    """Build the ``2^D`` block design: ``A``, ``B``, then one block per proper subset.

    Block ``m`` (bitmask ``m`` in ``1 .. 2^D - 2``) keeps the columns of ``m``
    from ``A`` and redraws the others from their conditional. All blocks read
    the leading columns of one shared noise block, as the singleton design does.
    """
    D = problem.num_vars
    draws = draws_for(problem)
    k = draws.dimension
    masks = range(1, 2**D - 1)
    fixed_sets = {m: subset_fixed(m, D) for m in masks}
    width = max(draws.free_dimension(f) for f in fixed_sets.values())
    points = sobol_uniforms(n, 2 * k + width, seed=seed, scramble=scramble)
    block_a = draws.joint(points[:, :k])
    block_b = draws.joint(points[:, k : 2 * k])
    noise = points[:, 2 * k :]
    blocks = [block_a, block_b]
    for m in masks:
        fixed = fixed_sets[m]
        blocks.append(
            draws.free_given_fixed(noise[:, : draws.free_dimension(fixed)], block_a, fixed)
        )
    X = draws.to_physical(np.concatenate(blocks, axis=0))

    n_blocks = 2**D
    n_total = n * n_blocks
    if verbose:
        _verbose.sample_summary(
            method="jaxgsa.kucherenko.sample",
            problem=problem,
            entries=[
                ("base_n", n),
                ("subsets", "all"),
                ("n_blocks", n_blocks),
                ("n_runs", n_total),
                ("dependence", problem.dependence),
                ("scramble", scramble),
            ],
        )
    return KucherenkoSamples(
        samples=X,
        n_expanded=n_total,
        expanded_to_unique=np.arange(n_total, dtype=np.int64),
        base_n=n,
        n_params=D,
        problem=problem,
        dependence=problem.dependence,
        subsets="all",
    )
