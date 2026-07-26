"""Morris elementary-effects sampling (trajectory and radial designs).

Follows the same two-layer contract as :mod:`jaxgsa.sampling`:

1. ``sample()`` returns only the unique rows that a user should evaluate.
2. ``MorrisSamples`` carries enough metadata to reconstruct the full
   expanded design and to locate each elementary effect inside it.

Both designs cost ``n_trajectories * (D + 1)`` expanded rows. Trajectory
points live on a coarse ``num_levels`` grid, so exact duplicate rows across
trajectories are common in low dimensions and deduplication saves real model
evaluations. Every elementary effect is described by a pair of expanded-row
indices plus a signed unit-cube step, so the analysis reduces to one
gather-subtract-divide regardless of the design.

References:
    Morris (1991). Technometrics 33(2):161-174.
    Campolongo, Cariboni & Saltelli (2007). Environ. Model. Softw. 22:1509-1518.
    Campolongo, Cariboni & Saltelli (2011). Comput. Phys. Commun. 182:978-988.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Mapping, overload

import jax.numpy as jnp
import numpy as np
from scipy.stats.qmc import Sobol

from jaxgsa._core.samples import UniqueDesignSamples
from jaxgsa._core.sampling import (
    _inverse_transform_samples,
    _next_power_of_2,
    _stable_unique_rows,
    _transform_samples,
)
from jaxgsa.problem import Problem

# Offset between the Sobol' draws used for radial base points (a) and
# auxiliary points (b). Reusing draw i for both would give delta = 0;
# Campolongo et al. (2011) recommend a shift of 4 positions.
_RADIAL_SHIFT = 4


def _min_radial_delta() -> float:
    """Smallest |delta| a radial step may have before it is unmeasurable.

    A radial elementary effect is ``(f(a with b_i) - f(a)) / (b_i - a_i)``. If
    ``|b_i - a_i|`` is near the floating-point resolution the model actually
    runs at, the two evaluation rows round to the same value and the effect
    degenerates to ``0`` or amplified rounding noise. JAX defaults to float32
    (float64 only when x64 is enabled), so the guard tracks the JAX default
    dtype rather than the float64 design array.

    Returns:
        ``10 * eps`` of the JAX default floating dtype.
    """
    return 10.0 * float(np.finfo(jnp.zeros(1).dtype).eps)


@dataclass(frozen=True)
class MorrisSamples(UniqueDesignSamples):
    """Unique Morris samples plus metadata to locate elementary effects.

    Note the two row counts: ``n_runs`` is the number of *unique* rows you
    must evaluate (one model run per row), while ``n_expanded`` is the
    (larger or equal) pre-deduplication size of the full Morris design that
    the analysis reconstructs internally.

    Attributes:
        samples: Unique rows to evaluate with the user's model, shape
            ``(n_runs, D)`` in the problem's physical units.
        n_expanded: Row count of the full expanded design before
            deduplication, always ``n_trajectories * (D + 1)``.
        expanded_to_unique: Integer index map of shape ``(n_expanded,)``
            from each expanded row to its row index in ``samples``.
        n_trajectories: Number of trajectories (r), the Morris repetition unit.
        num_levels: Grid levels ``p`` used by the trajectory design
            (unused by the radial design).
        method: Design generator, ``"trajectory"`` or ``"radial"``.
        ee_idx_after: ``(r, D)`` expanded-row index of the perturbed point of
            the elementary effect for each trajectory and parameter.
        ee_idx_before: ``(r, D)`` expanded-row index of the reference point of
            each elementary effect.
        ee_delta: ``(r, D)`` signed unit-cube step of each elementary effect,
            so that ``EE = (Y[after] - Y[before]) / delta``.
        n_params: Number of problem dimensions ``D``.
        problem: Problem definition used to transform the samples.
    """

    samples: np.ndarray  # shape (n_unique, D), scaled to bounds
    n_expanded: int
    expanded_to_unique: np.ndarray
    n_trajectories: int
    num_levels: int
    method: Literal["trajectory", "radial"]
    ee_idx_after: np.ndarray
    ee_idx_before: np.ndarray
    ee_delta: np.ndarray
    n_params: int
    problem: Problem

    @overload
    def downsample(self, n_trajectories: int) -> MorrisSamples: ...

    @overload
    def downsample(
        self, n_trajectories: int, Y: np.ndarray
    ) -> tuple[MorrisSamples, np.ndarray]: ...

    def downsample(
        self, n_trajectories: int, Y: np.ndarray | None = None
    ) -> MorrisSamples | tuple[MorrisSamples, np.ndarray]:
        """Return a smaller result by prefix-slicing to fewer trajectories.

        Trajectories are generated sequentially from independent draws
        (trajectory design) or from prefix-nested Sobol' points (radial
        design), so the first *m* trajectories of an *r*-trajectory run are
        identical to drawing *m* trajectories directly with the same integer
        seed. Simulate once at the largest ``n_trajectories`` and slice down —
        no re-simulation needed. (This holds for an ``int`` or ``None`` seed; a
        reused ``np.random.Generator`` advances its state between calls and so
        is not prefix-nested.)

        Optionally pass ``Y`` (model outputs aligned with ``samples``) to get
        the corresponding output slice back.

        Args:
            n_trajectories: Target trajectory count (``2 <= m <= r``).
            Y: Model outputs with shape ``(n_runs, ...)``. When provided,
                the matching prefix is returned alongside the new result.

        Returns:
            ``MorrisSamples`` when called without ``Y``, or
            ``(MorrisSamples, Y_small)`` when ``Y`` is provided.

        Raises:
            ValueError: If ``n_trajectories`` is out of range or ``Y`` has
                too few rows.
        """
        if n_trajectories < 2:
            raise ValueError(f"n_trajectories must be >= 2, got {n_trajectories}")
        if n_trajectories > self.n_trajectories:
            raise ValueError(
                f"Cannot upsample: requested n_trajectories={n_trajectories} > "
                f"current n_trajectories={self.n_trajectories}"
            )
        self._validate_downsample_Y(Y)
        if n_trajectories == self.n_trajectories:
            return (self, Y) if Y is not None else self

        new_expanded_n = n_trajectories * (self.n_params + 1)
        samples_small, new_exp2uniq, _, Y_small = self._prefix_slice(new_expanded_n, Y)

        sr_small = MorrisSamples(
            samples=samples_small,
            n_expanded=new_expanded_n,
            expanded_to_unique=new_exp2uniq,
            n_trajectories=n_trajectories,
            num_levels=self.num_levels,
            method=self.method,
            ee_idx_after=self.ee_idx_after[:n_trajectories].copy(),
            ee_idx_before=self.ee_idx_before[:n_trajectories].copy(),
            ee_delta=self.ee_delta[:n_trajectories].copy(),
            n_params=self.n_params,
            problem=self.problem,
        )

        if Y_small is not None:
            return sr_small, Y_small
        return sr_small

    def _extra_arrays(self) -> dict[str, np.ndarray]:
        """Persist the elementary-effect bookkeeping alongside the base arrays."""
        return {
            "ee_idx_after": self.ee_idx_after,
            "ee_idx_before": self.ee_idx_before,
            "ee_delta": self.ee_delta,
        }

    def _extra_metadata(self) -> dict[str, Any]:
        """Persist the Morris design parameters in the metadata blob."""
        return {
            "n_trajectories": self.n_trajectories,
            "num_levels": self.num_levels,
            "method": self.method,
        }

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
    ) -> MorrisSamples:
        """Rebuild a ``MorrisSamples`` from a loaded NPZ payload."""
        method = meta["method"]
        if method not in ("trajectory", "radial"):
            raise ValueError(f"method must be 'trajectory' or 'radial', got {method!r}")
        return cls(
            samples=samples,
            n_expanded=n_expanded,
            expanded_to_unique=expanded_to_unique,
            n_trajectories=int(meta["n_trajectories"]),
            num_levels=int(meta["num_levels"]),
            method=method,
            ee_idx_after=arrays["ee_idx_after"],
            ee_idx_before=arrays["ee_idx_before"],
            ee_delta=arrays["ee_delta"],
            n_params=problem.num_vars,
            problem=problem,
        )


def _radial_samples_from_blocks(
    *,
    samples: np.ndarray,
    block_rows: np.ndarray,
    problem: Problem,
) -> MorrisSamples:
    """Assemble a radial ``MorrisSamples`` from an already-evaluated design.

    Generic over where the points came from: the caller supplies the unique
    sample matrix plus, for each radial block, the row index of the block's
    base point followed by the ``D`` points perturbed in one parameter each.
    This lets a design built for another method be reinterpreted as a Morris
    design at zero extra model cost — see
    :meth:`jaxgsa.sobol.SobolSamples.to_morris`.

    ``samples`` is passed through unchanged, so outputs already computed for it
    stay aligned and no re-evaluation is needed.

    Args:
        samples: Unique rows already evaluated, shape ``(n_runs, D)``, in the
            problem's physical units.
        block_rows: ``(n_blocks, D + 1)`` integer indices into ``samples``.
            Column 0 is the block's base point; column ``1 + j`` is the point
            differing from it only in parameter ``j``.
        problem: Problem definition the samples were drawn for.

    Returns:
        A radial ``MorrisSamples`` ready for :func:`jaxgsa.morris.analyze`.

    Raises:
        ValueError: If fewer than two blocks are left with a measurable step.

    Warns:
        UserWarning: If any block is dropped because its base and perturbed
            points coincide at the model's floating-point resolution.
    """
    D = problem.num_vars
    # The elementary-effect denominator must be the unit-cube step, matching
    # what ``sample()`` records, so recover unit coordinates in float64.
    samples_unit = _inverse_transform_samples(problem, samples)

    param_idx = np.arange(D)
    # Read each step off the coordinate that actually differs between the base
    # row and the row perturbed in that parameter. This needs no knowledge of
    # the source layout, and it collapses to exactly 0 when deduplication has
    # merged the two rows.
    base_unit = samples_unit[block_rows[:, 0][:, None], param_idx]
    after_unit = samples_unit[block_rows[:, 1:], param_idx]
    ee_delta = after_unit - base_unit

    tol = _min_radial_delta()
    keep = ~(np.abs(ee_delta) < tol).any(axis=1)
    n_total = int(block_rows.shape[0])
    n_dropped = n_total - int(keep.sum())
    if n_dropped:
        block_rows = block_rows[keep]
        ee_delta = ee_delta[keep]
        warnings.warn(
            f"jaxgsa: dropped {n_dropped} of {n_total} radial blocks whose step is below "
            f"{tol:.1e} in at least one parameter (base and perturbed points coincide at "
            f"the model's floating-point resolution); {block_rows.shape[0]} blocks remain",
            # Reached through a caller's conversion method, so the user's frame
            # is two levels up rather than one.
            stacklevel=3,
        )

    n_blocks = int(block_rows.shape[0])
    if n_blocks < 2:
        raise ValueError(
            f"Only {n_blocks} radial block(s) have a measurable step; "
            "Morris measures need at least 2"
        )

    # Block b occupies expanded rows [b*(D+1), (b+1)*(D+1)): base point first,
    # then the D perturbed points in parameter order. That is exactly the row
    # order of block_rows, so flattening it gives the expansion map directly.
    offsets = (np.arange(n_blocks) * (D + 1)).astype(np.int64)
    return MorrisSamples(
        samples=samples,
        n_expanded=n_blocks * (D + 1),
        expanded_to_unique=np.ascontiguousarray(block_rows, dtype=np.int64).reshape(-1),
        n_trajectories=n_blocks,
        # Unused by the radial design; held at the sample() default.
        num_levels=4,
        method="radial",
        ee_idx_after=offsets[:, None] + 1 + param_idx.astype(np.int64),
        ee_idx_before=np.broadcast_to(offsets[:, None], (n_blocks, D)).copy(),
        ee_delta=ee_delta,
        n_params=D,
        problem=problem,
    )


def _build_trajectories(
    n_trajectories: int,
    n_params: int,
    num_levels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate Morris trajectories on the ``num_levels`` grid.

    Points are constructed on an integer half-level grid (units of
    ``1 / (2 * (p - 1))``, where the Morris step ``delta = p / (2 * (p - 1))``
    is exactly ``p`` half-levels). Converting to float once at the end makes
    equal grid points bitwise identical across trajectories, which is what
    lets exact deduplication collapse them.

    Randomness is consumed strictly per trajectory so that the first *m*
    trajectories of an *r*-trajectory draw match a direct *m*-trajectory draw.

    Returns:
        ``(expanded_unit, ee_idx_after, ee_idx_before, ee_delta)`` where
        ``expanded_unit`` has shape ``(r * (D + 1), D)`` in the unit cube and
        the bookkeeping arrays have shape ``(r, D)``.
    """
    D = n_params
    p = num_levels
    # delta = p / (2(p-1)) expressed in half-level units of 1/(2(p-1)).
    delta_int = p
    delta = p / (2.0 * (p - 1))
    # Base levels l/(p-1) must leave room for a +delta step: l <= p/2 - 1.
    n_start_levels = p // 2

    levels_int = np.empty((n_trajectories * (D + 1), D), dtype=np.int64)
    ee_idx_after = np.empty((n_trajectories, D), dtype=np.int64)
    ee_idx_before = np.empty((n_trajectories, D), dtype=np.int64)
    ee_delta = np.empty((n_trajectories, D), dtype=np.float64)

    for j in range(n_trajectories):
        # Per-trajectory draws, fixed order: base levels, step signs, order.
        base = 2 * rng.integers(0, n_start_levels, size=D)  # even = on-grid
        signs = 2 * rng.integers(0, 2, size=D) - 1  # each +/-1
        perm = rng.permutation(D)

        # A -delta step needs headroom below, so shift its start up by delta.
        x = base + np.where(signs < 0, delta_int, 0)
        offset = j * (D + 1)
        levels_int[offset] = x
        for s, i in enumerate(perm):
            x = x.copy()
            x[i] += signs[i] * delta_int
            levels_int[offset + s + 1] = x
            ee_idx_before[j, i] = offset + s
            ee_idx_after[j, i] = offset + s + 1
            ee_delta[j, i] = signs[i] * delta

    expanded_unit = levels_int / (2.0 * (p - 1))
    return expanded_unit, ee_idx_after, ee_idx_before, ee_delta


def _build_radial(
    n_trajectories: int,
    n_params: int,
    *,
    scramble: bool,
    seed: int | np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate radial star designs from scrambled-Sobol' point pairs.

    Each trajectory uses a base point ``a`` and an auxiliary point ``b`` drawn
    from a ``2D``-dimensional Sobol' sequence (``b`` shifted ``_RADIAL_SHIFT``
    draws ahead). Star row *i* swaps coordinate *i* of ``a`` with ``b_i``, so
    every elementary effect compares a star row against the shared base row
    with per-step delta ``b_i - a_i``.

    Returns:
        Same tuple layout as :func:`_build_trajectories`.

    Raises:
        ValueError: If any ``|delta|`` falls below the working float-precision
            floor (see :func:`_min_radial_delta`).
    """
    D = n_params
    r = n_trajectories
    sampler = Sobol(d=2 * D, scramble=scramble, seed=seed)
    # Draw a power-of-2 count (scipy warns otherwise) and slice; Sobol'
    # prefixes are bit-identical, so the extra rows change nothing.
    draws = sampler.random(_next_power_of_2(r + _RADIAL_SHIFT))
    a = draws[:r, :D]
    b = draws[_RADIAL_SHIFT : r + _RADIAL_SHIFT, D:]

    ee_delta = b - a
    min_delta = _min_radial_delta()
    tiny = np.abs(ee_delta) < min_delta
    if np.any(tiny):
        j, i = np.argwhere(tiny)[0]
        raise ValueError(
            f"Radial design produced a near-zero step |delta|={abs(ee_delta[j, i]):.2e} "
            f"(below the float-precision floor {min_delta:.1e}) for trajectory {j}, "
            f"parameter {i}; the elementary effect would be numerically meaningless. "
            "Use scramble=True or a different seed."
        )

    # Star block j: row 0 is the base point a[j]; row 1+i is a[j] with only
    # coordinate i swapped to b[j, i]. No randomness is consumed here, so the
    # whole design is three vectorized assignments.
    block = np.repeat(a[:, None, :], D + 1, axis=1)  # (r, D+1, D)
    diag = np.arange(D)
    block[:, 1 + diag, diag] = b  # swap coordinate i of star row 1+i
    expanded_unit = block.reshape(r * (D + 1), D)

    offsets = (np.arange(r) * (D + 1)).astype(np.int64)
    ee_idx_before = np.broadcast_to(offsets[:, None], (r, D)).copy()
    ee_idx_after = offsets[:, None] + 1 + diag.astype(np.int64)

    return expanded_unit, ee_idx_after, ee_idx_before, ee_delta


def _print_morris_summary(
    *,
    n_params: int,
    method: str,
    n_trajectories: int,
    num_levels: int,
    n_runs: int,
    n_expanded: int,
) -> None:
    """Print a compact summary of the generated unique Morris design."""
    duplicates_removed = n_expanded - n_runs
    duplicate_fraction = duplicates_removed / n_expanded if n_expanded else 0.0
    levels_label = f", num_levels={num_levels}" if method == "trajectory" else ""
    print(
        "jaxgsa.morris.sample: "
        f"D={n_params}, method={method}, n_trajectories={n_trajectories}{levels_label}, "
        f"n_expanded={n_expanded}, n_runs={n_runs}, "
        f"duplicates_removed={duplicates_removed} ({duplicate_fraction:.1%})"
    )


def sample(
    problem: Problem,
    n_trajectories: int,
    *,
    num_levels: int = 4,
    method: Literal["trajectory", "radial"] = "trajectory",
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
    truncation_quantile: float = 0.005,
    verbose: bool = True,
) -> MorrisSamples:
    """Generate unique Morris elementary-effects samples for model evaluation.

    Morris screening ranks inputs by one-at-a-time finite differences
    (elementary effects) spread across the whole input domain — the usual
    first step for weeding out unimportant parameters before a more
    expensive variance-based analysis. Each trajectory perturbs every
    parameter exactly once, so the full design costs
    ``n_trajectories * (D + 1)`` model evaluations at most.

    This function builds those ``n_trajectories`` paths of ``D + 1`` points
    each, removes exact duplicate rows while preserving first-occurrence
    order, and returns only the unique rows for the user to evaluate.
    :func:`jaxgsa.morris.analyze` reconstructs the expanded layout internally.

    Gaussian marginals are supported through a truncated-quantile grid: the
    Morris design includes the unit-cube boundaries, which an unbounded
    inverse CDF maps to infinity, so for each Gaussian parameter the unit-cube
    coordinate is confined to ``[q, 1 - q]`` (``q = truncation_quantile``)
    before the transform. Elementary effects remain per unit of the original
    grid coordinate; :meth:`MorrisResult.to_physical_units` is unavailable for
    such problems because the transform is nonlinear.

    Args:
        problem: Problem definition with uniform and/or Gaussian marginals.
        n_trajectories: Number of trajectories r (>= 2). Each contributes one
            elementary effect per parameter, so r is the sample size behind
            every screening measure: more trajectories tighten the mu_star
            ranking (and any bootstrap CIs) at proportionally more model
            evaluations. Typical screening uses 10-50.
        num_levels: Grid levels ``p`` for the trajectory design (default 4,
            step ``delta = p / (2 * (p - 1))``). Even values make all levels
            equally probable; odd values trigger a warning. Ignored by the
            radial design.
        method: ``"trajectory"`` (Morris 1991 grid walks, default) or
            ``"radial"`` (Campolongo 2011 star designs around scrambled-Sobol'
            base points). The radial design spreads points quasi-randomly
            instead of on a coarse grid and has no ``num_levels`` to choose,
            at the cost of fewer duplicate rows to deduplicate.
        scramble: Whether to Owen-scramble the Sobol' sequence (radial design
            only).
        seed: Random seed or generator for reproducibility. Pass an ``int``
            (or ``None``) to keep the prefix-nesting guarantee of
            :meth:`MorrisSamples.downsample`; a reused
            ``np.random.Generator`` advances its state between calls and breaks
            that nesting.
        truncation_quantile: Tail probability ``q`` excluded on each side of
            every Gaussian marginal's grid (default 0.005, probing the
            0.5%-99.5% quantile range). Applied to truncated Gaussians as
            well for consistency; ignored for uniform marginals.
        verbose: If ``True`` (default), print a short summary including how
            many duplicate rows were removed.

    Returns:
        MorrisSamples with a unique sample matrix plus elementary-effect
        bookkeeping for later analysis.

    Raises:
        ValueError: If ``n_trajectories``, ``num_levels``, ``method``, or
            ``truncation_quantile`` are invalid.
    """
    if not 0.0 < truncation_quantile < 0.5:
        raise ValueError(f"truncation_quantile must be in (0, 0.5), got {truncation_quantile}")
    if n_trajectories < 2:
        raise ValueError(f"n_trajectories must be >= 2, got {n_trajectories}")
    if num_levels < 2:
        raise ValueError(f"num_levels must be >= 2, got {num_levels}")
    if method not in ("trajectory", "radial"):
        raise ValueError(f"method must be 'trajectory' or 'radial', got {method!r}")
    if method == "trajectory" and num_levels % 2 != 0:
        warnings.warn(
            f"jaxgsa: num_levels={num_levels} is odd — grid levels are not equally "
            "probable and steps land off-grid; an even value is recommended",
            stacklevel=2,
        )

    D = problem.num_vars
    if method == "trajectory":
        rng = np.random.default_rng(seed)
        expanded_unit, ee_idx_after, ee_idx_before, ee_delta = _build_trajectories(
            n_trajectories, D, num_levels, rng
        )
    else:
        expanded_unit, ee_idx_after, ee_idx_before, ee_delta = _build_radial(
            n_trajectories, D, scramble=scramble, seed=seed
        )

    if problem.has_non_uniform_inputs:
        # Confine unbounded-support dimensions to [q, 1-q] so the inverse CDF
        # stays finite at the grid boundaries. Uniform is the only bounded
        # marginal today, so "not uniform" == "unbounded"; any future bounded
        # distribution must be excluded from this mask. The squash is
        # deterministic, so dedup and prefix-nesting are unaffected.
        q = truncation_quantile
        unbounded_dims = np.array([spec[0] != "uniform" for spec in problem.input_specs])
        expanded_unit[:, unbounded_dims] = q + expanded_unit[:, unbounded_dims] * (1.0 - 2.0 * q)

    expanded_samples = _transform_samples(problem, expanded_unit)
    unique_samples, expanded_to_unique = _stable_unique_rows(expanded_samples)

    if verbose:
        _print_morris_summary(
            n_params=D,
            method=method,
            n_trajectories=n_trajectories,
            num_levels=num_levels,
            n_runs=unique_samples.shape[0],
            n_expanded=expanded_samples.shape[0],
        )

    return MorrisSamples(
        samples=unique_samples,
        n_expanded=expanded_samples.shape[0],
        expanded_to_unique=expanded_to_unique,
        n_trajectories=n_trajectories,
        num_levels=num_levels,
        method=method,
        ee_idx_after=ee_idx_after,
        ee_idx_before=ee_idx_before,
        ee_delta=ee_delta,
        n_params=D,
        problem=problem,
    )
