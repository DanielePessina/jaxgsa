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
from typing import Any, Callable, Literal, Mapping

import numpy as np

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.dependence import draws_for, sobol_uniforms
from jaxgsa._core.samples import UniqueDesignSamples
from jaxgsa._core.sampling import (
    _is_power_of_2,
    _next_power_of_2,
    _power_of_2_error,
    check_budget_choice,
    check_seed,
    check_seed_scramble,
)
from jaxgsa._core.validation import _raise_categorical_design
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.kucherenko._subsets import subset_matrix
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
    subsets: Literal["singletons", "all"] = "singletons"

    @property
    def dependence(self) -> Dependence:
        """How the inputs depend on each other, as the problem declares it."""
        return self.problem.dependence

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
        return {"base_n": self.base_n, "subsets": self.subsets}

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
            n_params=problem.n_params,
            problem=problem,
            subsets=_parse_subsets(meta.get("subsets")),
        )


MAX_SUBSET_PARAMS = 12
"""Largest ``D`` that ``subsets="all"`` accepts. The design has ``n * 2^D`` rows."""


def n_blocks_for(n_params: int, subsets: str) -> int:
    """Return the number of design blocks for ``n_params`` and a ``subsets`` mode."""
    return 2**n_params if subsets == "all" else 2 * n_params + 1


def _parse_subsets(value: Any) -> Literal["singletons", "all"]:
    """Read the stored ``subsets`` label."""
    if value == "singletons" or value == "all":
        return value
    raise ValueError(f"Unknown subsets {value!r} in saved Kucherenko design")


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
    D = problem.n_params
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
        "problem.with_sampler(f, sampler_dimension=k).",
        category=JaxgsaWarning,
        stacklevel=3,
    )


def sample(
    problem: Problem,
    n_samples: int | None = None,
    *,
    base_n: int | None = None,
    subsets: Literal["singletons", "all"] = "singletons",
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
    verbose: bool = True,
) -> KucherenkoSamples:
    """Generate the conditional design for :func:`jaxgsa.kucherenko.analyze`.

    ``n_samples`` is the minimum number of model runs, as in
    :func:`jaxgsa.sobol.sample`. The design has ``base_n * (2D + 1)`` rows, and
    ``base_n`` is the smallest power of two whose design reaches ``n_samples``
    (the Sobol' sequence keeps its balance guarantees only at powers of two).
    Pass ``base_n`` instead to set the base points per block exactly.

    The dependence structure comes from the problem. ``problem.correlation``
    declares a Gaussian copula. A sampler declared with
    ``problem.with_sampler(f, sampler_dimension=k)`` supplies custom joint and
    conditional draws instead. On the default design the sampler receives
    three kinds of call: joint draws (``fixed_indices=()``), draws that keep
    ``x_i`` (``fixed_indices=(i,)``), and draws that keep every column except
    ``i``. With ``subsets="all"`` it also receives every other subset of fixed
    columns.
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
        n_samples: Minimum number of model runs. The design may contain more
            rows than this, never fewer. Give exactly one of ``n_samples`` and
            ``base_n``.
        base_n: Exact number of base points per block, a power of two of at
            least 2. The design has ``base_n * (2D + 1)`` rows (``base_n *
            2^D`` with ``subsets="all"``). Give exactly one of ``n_samples``
            and ``base_n``.
        subsets: ``"singletons"`` (default) builds the ``2D + 1`` block design
            for ``S1`` and ``ST``. ``"all"`` builds the ``2^D`` block design
            that adds closed and total indices of every subset, and with them
            ``result.shapley()`` and ``result.pme()``.
        scramble: Whether to scramble the Sobol' sequence. SciPy applies a
            linear matrix scramble and a digital random shift, not Owen
            nested scrambling. Keep it on:
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
            Also if the resolved ``base_n`` is below 16. The design is
            then degenerate and the indices are unreliable; this is a
            warning, not an error, because a tiny design may be a deliberate
            probe.

    Raises:
        ValueError: If both or neither of ``n_samples`` and ``base_n`` are
            given, if ``base_n`` is not a power of two, if ``subsets`` is not
            ``"singletons"`` or ``"all"``, if
            ``subsets="all"`` and the problem has more than 12 parameters, if
            the problem has fewer than two parameters, if it has a
            categorical parameter and no custom sampler, if
            ``n_samples < 2`` or ``base_n < 2``, if a ``seed`` is
            given with ``scramble=False`` (the unscrambled sequence is
            deterministic, so the seed would do nothing), or if a custom
            sampler returns invalid rows.
    """
    check_seed(seed, "jaxgsa.kucherenko.sample")
    check_budget_choice(n_samples, base_n, "jaxgsa.kucherenko.sample")
    check_seed_scramble(scramble, seed, "jaxgsa.kucherenko.sample")
    # The Gaussian copula cannot condition on a categorical input. A custom
    # sampler draws every conditional itself, so it can.
    if problem.dependence != "custom":
        _raise_categorical_design(problem, "jaxgsa.kucherenko.sample")
    D = problem.n_params
    if D < 2:
        raise ValueError(f"Kucherenko indices need at least 2 parameters, got {D}")
    if subsets not in ("singletons", "all"):
        raise ValueError(f"subsets must be 'singletons' or 'all', got {subsets!r}")
    if subsets == "all" and D > MAX_SUBSET_PARAMS:
        raise ValueError(
            f"subsets='all' needs base_n * 2^{D}"
            + (f" = {base_n * 2**D}" if base_n is not None else "")
            + f" model runs for {D} parameters, and "
            f"it supports at most {MAX_SUBSET_PARAMS} parameters. The cost doubles "
            "with every parameter. For more parameters, use a method that works on "
            "given data, such as jaxgsa.shapley.analyze with a fitted surrogate, or "
            "jaxgsa.vkoga."
        )
    n_blocks = n_blocks_for(D, subsets)
    if base_n is not None:
        if base_n < 2:
            raise ValueError(f"base_n must be >= 2, got {base_n}")
        if not _is_power_of_2(base_n):
            raise ValueError(_power_of_2_error("base_n", base_n, reason="Sobol' sequence balance"))
        n = base_n
        requested = None
    else:
        assert n_samples is not None
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")
        requested = n_samples
        n = max(2, _next_power_of_2(-(-n_samples // n_blocks)))
    if n < 16:
        warnings.warn(
            f"jaxgsa.kucherenko.sample: base_n={n} is below the floor of 16 "
            "base points. The design is degenerate: the indices will be "
            "unreliable. Raise n_samples (or base_n) for a usable design.",
            stacklevel=2,
            category=JaxgsaWarning,
        )
    if subsets == "all":
        return _sample_all_subsets(
            problem, n, requested=requested, scramble=scramble, seed=seed, verbose=verbose
        )
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
    rest_noise = redraw[:, : draws.rest_dimension]
    self_noise = redraw[:, : draws.self_dimension]
    blocks = _stack_blocks(
        n_blocks,
        joint,
        # First-order blocks: keep x_i, redraw the rest given x_i.
        *(lambda i=i: draws.rest_given_self(rest_noise, joint, i) for i in range(D)),
        # Total blocks: keep x_-i, redraw x_i given the rest.
        *(lambda i=i: draws.self_given_rest(self_noise, joint, i) for i in range(D)),
    )
    X = draws.to_physical(blocks)

    return _finish_design(
        X,
        problem,
        n,
        subsets="singletons",
        requested=requested,
        scramble=scramble,
        verbose=verbose,
    )


def _stack_blocks(n_blocks: int, first: np.ndarray, *rest: Callable[[], np.ndarray]) -> np.ndarray:
    """Stack equal-shaped design blocks into one preallocated array.

    Gives what ``np.concatenate([first, *(f() for f in rest)])`` gives, but
    each block is written into the output as soon as it is drawn and then
    dropped. Holding every block in a list and concatenating keeps two full
    copies of the design alive at once.

    Args:
        n_blocks: Total number of blocks, ``1 + len(rest)``.
        first: The first block, ``(n, D)``. Its shape is every block's shape.
        *rest: One callable per further block, in order, each returning it.

    Returns:
        ``(n_blocks * n, D)`` array, block-major, in the dtype
        ``np.concatenate`` would choose.

    Raises:
        ValueError: If a block's shape differs from ``first``'s.
    """
    assert n_blocks == 1 + len(rest)
    n = first.shape[0]
    out = np.empty((n_blocks * n, *first.shape[1:]), dtype=first.dtype)
    out[:n] = first
    for b, draw in enumerate(rest, start=1):
        block = draw()
        if block.shape != first.shape:
            raise ValueError(f"design block {b} has shape {block.shape}, expected {first.shape}")
        dtype = np.result_type(out, block)
        if dtype != out.dtype:
            # A later block of a wider dtype: widen, as concatenate would.
            out = out.astype(dtype)
        out[b * n : (b + 1) * n] = block
    return out


def _finish_design(
    X: np.ndarray,
    problem: Problem,
    n: int,
    *,
    subsets: Literal["singletons", "all"],
    requested: int | None,
    scramble: bool,
    verbose: bool,
) -> KucherenkoSamples:
    """Print the design summary and wrap the stacked rows in a ``KucherenkoSamples``."""
    n_blocks = n_blocks_for(problem.n_params, subsets)
    n_total = n * n_blocks
    if verbose:
        _verbose.sample_summary(
            method="jaxgsa.kucherenko.sample",
            problem=problem,
            entries=[
                *_requested_entry(requested),
                ("base_n", n),
                *([("subsets", "all")] if subsets == "all" else []),
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
        n_params=problem.n_params,
        problem=problem,
        subsets=subsets,
    )


def _requested_entry(requested: int | None) -> list[tuple[str, object]]:
    """Return the ``requested_runs>=N`` verbose entry, or nothing when ``base_n`` was set."""
    return [] if requested is None else [("requested_runs>", requested)]


def _sample_all_subsets(
    problem: Problem,
    n: int,
    *,
    requested: int | None,
    scramble: bool,
    seed: int | np.random.Generator | None,
    verbose: bool,
) -> KucherenkoSamples:
    """Build the ``2^D`` block design: ``A``, ``B``, then one block per proper subset.

    Block ``m`` (bitmask ``m`` in ``1 .. 2^D - 2``) keeps the columns of ``m``
    from ``A`` and redraws the others from their conditional. All blocks read
    the leading columns of one shared noise block, as the singleton design does.
    """
    D = problem.n_params
    draws = draws_for(problem)
    k = draws.dimension
    masks = range(1, 2**D - 1)
    members = subset_matrix(D)  # row m - 1 marks the members of mask m
    fixed_sets = {m: tuple(np.flatnonzero(members[m - 1]).tolist()) for m in masks}
    width = max(draws.free_dimension(f) for f in fixed_sets.values())
    points = sobol_uniforms(n, 2 * k + width, seed=seed, scramble=scramble)
    block_a = draws.joint(points[:, :k])
    block_b = draws.joint(points[:, k : 2 * k])
    noise = points[:, 2 * k :]
    blocks = _stack_blocks(
        2**D,
        block_a,
        lambda: block_b,
        *(
            lambda f=fixed_sets[m]: draws.free_given_fixed(
                noise[:, : draws.free_dimension(f)], block_a, f
            )
            for m in masks
        ),
    )
    X = draws.to_physical(blocks)

    return _finish_design(
        X, problem, n, subsets="all", requested=requested, scramble=scramble, verbose=verbose
    )
