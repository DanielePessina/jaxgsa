"""Morris elementary-effects sampling (trajectory and radial designs).

Follows the same two-layer contract as :mod:`gsax.sampling`:

1. ``sample()`` returns only the unique rows that a user should evaluate.
2. ``MorrisSamplingResult`` carries enough metadata to reconstruct the full
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
from typing import Literal, overload

import numpy as np
from scipy.stats.qmc import Sobol

from gsax.problem import Problem
from gsax.sampling import _next_power_of_2, _stable_unique_rows, _transform_samples

# Offset between the Sobol' draws used for radial base points (a) and
# auxiliary points (b). Reusing draw i for both would give delta = 0;
# Campolongo et al. (2011) recommend a shift of 4 positions.
_RADIAL_SHIFT = 4

# Smallest |delta| accepted for a radial elementary effect; below this the
# finite difference is numerically meaningless.
_MIN_RADIAL_DELTA = 1e-9


@dataclass(frozen=True)
class MorrisSamplingResult:
    """Unique Morris samples plus metadata to locate elementary effects.

    Attributes:
        samples: Unique rows to evaluate with the user's model, shape
            ``(n_unique, D)`` in the problem's physical units.
        sample_ids: Stable integer identifiers aligned 1:1 with ``samples``.
        expanded_n_total: Row count of the full expanded design before
            deduplication, always ``n_trajectories * (D + 1)``.
        expanded_to_unique: Integer index map of shape ``(expanded_n_total,)``
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
    sample_ids: np.ndarray
    expanded_n_total: int
    expanded_to_unique: np.ndarray
    n_trajectories: int
    num_levels: int
    method: Literal["trajectory", "radial"]
    ee_idx_after: np.ndarray
    ee_idx_before: np.ndarray
    ee_delta: np.ndarray
    n_params: int
    problem: Problem

    @property
    def n_total(self) -> int:
        """Number of unique rows in ``samples``."""
        return self.samples.shape[0]

    @overload
    def downsample(self, n_trajectories: int) -> MorrisSamplingResult: ...

    @overload
    def downsample(
        self, n_trajectories: int, Y: np.ndarray
    ) -> tuple[MorrisSamplingResult, np.ndarray]: ...

    def downsample(
        self, n_trajectories: int, Y: np.ndarray | None = None
    ) -> MorrisSamplingResult | tuple[MorrisSamplingResult, np.ndarray]:
        """Return a smaller result by prefix-slicing to fewer trajectories.

        Trajectories are generated sequentially from independent draws
        (trajectory design) or from prefix-nested Sobol' points (radial
        design), so the first *m* trajectories of an *r*-trajectory run are
        identical to drawing *m* trajectories directly with the same seed.
        Simulate once at the largest ``n_trajectories`` and slice down —
        no re-simulation needed.

        Optionally pass ``Y`` (model outputs aligned with ``samples``) to get
        the corresponding output slice back.

        Args:
            n_trajectories: Target trajectory count (``2 <= m <= r``).
            Y: Model outputs with shape ``(n_total, ...)``. When provided,
                the matching prefix is returned alongside the new result.

        Returns:
            ``MorrisSamplingResult`` when called without ``Y``, or
            ``(MorrisSamplingResult, Y_small)`` when ``Y`` is provided.

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
        if Y is not None and Y.shape[0] != self.n_total:
            raise ValueError(f"Y.shape[0]={Y.shape[0]} does not match n_total={self.n_total}")
        if n_trajectories == self.n_trajectories:
            return (self, Y) if Y is not None else self

        new_expanded_n = n_trajectories * (self.n_params + 1)
        new_exp2uniq = self.expanded_to_unique[:new_expanded_n]
        # First-occurrence dedup order makes the unique index set of any
        # expanded prefix itself a prefix, so max+1 recovers the unique count.
        n_unique_new = int(new_exp2uniq.max()) + 1

        sr_small = MorrisSamplingResult(
            samples=self.samples[:n_unique_new].copy(),
            sample_ids=np.arange(n_unique_new, dtype=np.int64),
            expanded_n_total=new_expanded_n,
            expanded_to_unique=new_exp2uniq.copy(),
            n_trajectories=n_trajectories,
            num_levels=self.num_levels,
            method=self.method,
            ee_idx_after=self.ee_idx_after[:n_trajectories].copy(),
            ee_idx_before=self.ee_idx_before[:n_trajectories].copy(),
            ee_delta=self.ee_delta[:n_trajectories].copy(),
            n_params=self.n_params,
            problem=self.problem,
        )

        if Y is not None:
            return sr_small, Y[:n_unique_new].copy()
        return sr_small


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
        ValueError: If any |delta| falls below ``_MIN_RADIAL_DELTA``.
    """
    D = n_params
    sampler = Sobol(d=2 * D, scramble=scramble, seed=seed)
    # Draw a power-of-2 count (scipy warns otherwise) and slice; Sobol'
    # prefixes are bit-identical, so the extra rows change nothing.
    draws = sampler.random(_next_power_of_2(n_trajectories + _RADIAL_SHIFT))
    a = draws[:n_trajectories, :D]
    b = draws[_RADIAL_SHIFT : n_trajectories + _RADIAL_SHIFT, D:]

    ee_delta = b - a
    tiny = np.abs(ee_delta) < _MIN_RADIAL_DELTA
    if np.any(tiny):
        j, i = np.argwhere(tiny)[0]
        raise ValueError(
            f"Radial design produced a near-zero step |delta|={abs(ee_delta[j, i]):.2e} "
            f"for trajectory {j}, parameter {i}; the elementary effect would be "
            "numerically meaningless. Use scramble=True or a different seed."
        )

    expanded_unit = np.empty((n_trajectories * (D + 1), D), dtype=np.float64)
    ee_idx_after = np.empty((n_trajectories, D), dtype=np.int64)
    ee_idx_before = np.empty((n_trajectories, D), dtype=np.int64)

    for j in range(n_trajectories):
        offset = j * (D + 1)
        expanded_unit[offset] = a[j]
        for i in range(D):
            row = a[j].copy()
            row[i] = b[j, i]
            expanded_unit[offset + 1 + i] = row
            ee_idx_before[j, i] = offset
            ee_idx_after[j, i] = offset + 1 + i

    return expanded_unit, ee_idx_after, ee_idx_before, ee_delta


def _print_morris_summary(
    *,
    n_params: int,
    method: str,
    n_trajectories: int,
    num_levels: int,
    unique_n: int,
    expanded_n_total: int,
) -> None:
    """Print a compact summary of the generated unique Morris design."""
    duplicates_removed = expanded_n_total - unique_n
    duplicate_fraction = duplicates_removed / expanded_n_total if expanded_n_total else 0.0
    levels_label = f", num_levels={num_levels}" if method == "trajectory" else ""
    print(
        "gsax.sample_morris: "
        f"D={n_params}, method={method}, n_trajectories={n_trajectories}{levels_label}, "
        f"expanded_rows={expanded_n_total}, returned_unique={unique_n}, "
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
) -> MorrisSamplingResult:
    """Generate unique Morris elementary-effects samples for model evaluation.

    Builds ``n_trajectories`` one-at-a-time paths of ``D + 1`` points each
    (``n_trajectories * (D + 1)`` expanded rows), removes exact duplicate rows
    while preserving first-occurrence order, and returns only the unique rows
    for the user to evaluate. :func:`gsax.morris.analyze` reconstructs the
    expanded layout internally.

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
            elementary effect per parameter; typical screening uses 10-50.
        num_levels: Grid levels ``p`` for the trajectory design (default 4,
            step ``delta = p / (2 * (p - 1))``). Even values make all levels
            equally probable; odd values trigger a warning. Ignored by the
            radial design.
        method: ``"trajectory"`` (Morris 1991 grid walks, default) or
            ``"radial"`` (Campolongo 2011 star designs around scrambled-Sobol'
            base points).
        scramble: Whether to Owen-scramble the Sobol' sequence (radial design
            only).
        seed: Random seed or generator for reproducibility.
        truncation_quantile: Tail probability ``q`` excluded on each side of
            every Gaussian marginal's grid (default 0.005, probing the
            0.5%-99.5% quantile range). Applied to truncated Gaussians as
            well for consistency; ignored for uniform marginals.
        verbose: If ``True`` (default), print a short summary including how
            many duplicate rows were removed.

    Returns:
        MorrisSamplingResult with a unique sample matrix plus elementary-effect
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
            f"gsax: num_levels={num_levels} is odd — grid levels are not equally "
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
        # Confine Gaussian coordinates to [q, 1-q] so the inverse CDF stays
        # finite at the grid boundaries. The squash is deterministic, so
        # bitwise deduplication and prefix-nesting are unaffected.
        q = truncation_quantile
        gaussian_dims = np.array([spec[0] != "uniform" for spec in problem.input_specs])
        expanded_unit[:, gaussian_dims] = q + expanded_unit[:, gaussian_dims] * (1.0 - 2.0 * q)

    expanded_samples = _transform_samples(problem, expanded_unit)
    unique_samples, expanded_to_unique = _stable_unique_rows(expanded_samples)

    if verbose:
        _print_morris_summary(
            n_params=D,
            method=method,
            n_trajectories=n_trajectories,
            num_levels=num_levels,
            unique_n=unique_samples.shape[0],
            expanded_n_total=expanded_samples.shape[0],
        )

    return MorrisSamplingResult(
        samples=unique_samples,
        sample_ids=np.arange(unique_samples.shape[0], dtype=np.int64),
        expanded_n_total=expanded_samples.shape[0],
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
