"""Morris design generation for the trajectory and radial schemes.

This module follows the same two-layer contract as :mod:`jaxgsa.sampling`:

1. :func:`sample` returns only the unique rows that a user should evaluate.
2. :class:`MorrisSamples` carries enough metadata to rebuild the full expanded
   design and to locate each elementary effect inside it.

Both designs cost ``n_trajectories * (D + 1)`` expanded rows. Trajectory points
live on a coarse ``num_levels`` grid. Exact duplicate rows across trajectories
are therefore common in low dimensions, and deduplication saves real model
evaluations. A pair of expanded-row indices plus a signed unit-cube step
describes every elementary effect, so the analysis reduces to one
gather-subtract-divide whichever design produced the points.

References:
    Morris (1991). Technometrics 33(2):161-174.
    Campolongo, Cariboni & Saltelli (2007). Environ. Model. Softw. 22:1509-1518.
    Campolongo, Cariboni & Saltelli (2011). Comput. Phys. Commun. 182:978-988.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
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
from jaxgsa._core.validation import _raise_categorical_design, _raise_correlated_design
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.problem import Problem

# Offset between the Sobol' draws used for the radial base points (a) and the
# auxiliary points (b). Reusing draw i for both would give delta = 0.
# Campolongo et al. (2011) recommend a shift of 4 positions.
_RADIAL_SHIFT = 4


def _min_radial_delta() -> float:
    """Return the smallest |delta| a radial step may have to stay measurable.

    A radial elementary effect is ``(f(a with b_i) - f(a)) / (b_i - a_i)``.
    Suppose ``|b_i - a_i|`` sits near the floating-point resolution the model
    actually runs at. The two evaluation rows then round to the same value, and
    the effect degenerates to ``0`` or to amplified rounding noise. JAX
    defaults to float32, and uses float64 only when x64 is enabled, so the
    guard tracks the JAX default dtype rather than the float64 design array.

    Returns:
        ``10 * eps`` of the JAX default floating dtype.
    """
    return 10.0 * float(np.finfo(jnp.zeros(1).dtype).eps)


@dataclass(frozen=True)
class MorrisSamples(UniqueDesignSamples):
    """Unique Morris samples plus metadata to locate elementary effects.

    The class carries two row counts. ``n_runs`` is the number of unique rows
    you must evaluate, one model run per row. ``n_expanded`` is the size of the
    full Morris design before deduplication, which the analysis rebuilds
    internally. ``n_expanded`` is greater than or equal to ``n_runs``.

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
        ee_idx_after: Expanded-row index of the perturbed point of each
            elementary effect, shape ``(r, D)``, one entry per trajectory and
            parameter.
        ee_idx_before: Expanded-row index of the reference point of each
            elementary effect, shape ``(r, D)``.
        ee_delta: Signed unit-cube step of each elementary effect, shape
            ``(r, D)``, so that ``EE = (Y[after] - Y[before]) / delta``.
        n_params: Number of problem dimensions ``D``.
        problem: Problem definition used to transform the samples.
        n_blocks_dropped: Number of blocks that the design lost at
            construction because their step was not measurable. This is 0 for
            a design that :func:`jaxgsa.morris.sample` built. It can be
            positive only for a design derived from another method, such as
            :meth:`jaxgsa.sobol.SobolSamples.to_morris`. The analysis reports
            the loss, because the user did not ask for a smaller design.
            :meth:`downsample` resets it to 0, whatever size is asked for:
            the caller names the size and gets it, so no trajectory is
            missing. A design that came through a lossy conversion is still
            warned about at the conversion itself.
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
    n_blocks_dropped: int = 0

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

        The generators build trajectories in sequence, from independent draws
        in the trajectory design and from prefix-nested Sobol' points in the
        radial design. The first m trajectories of an r-trajectory run are
        therefore identical to drawing m trajectories directly with the same
        integer seed. Simulate once at the largest ``n_trajectories`` and slice
        down, with no re-simulation. This holds for an ``int`` or ``None``
        seed. A reused ``np.random.Generator`` advances its state between
        calls, so it is not prefix-nested.

        Pass ``Y``, the model outputs aligned with ``samples``, to get the
        matching output slice back as well.

        Args:
            n_trajectories: Target trajectory count, ``2 <= m <= r``.
            Y: Model outputs, shape ``(n_runs, ...)``. When given, the method
                returns the matching prefix alongside the new result.

        Returns:
            A :class:`MorrisSamples` when called without ``Y``, or
            ``(MorrisSamples, Y_small)`` when ``Y`` is given.

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
            # Same size, so every array is already right. Only the dropped-block
            # count can differ, and it is reset here for the same reason it is
            # reset below: the caller named this size and received it. Returning
            # ``self`` unchanged would make the count depend on whether the
            # requested size happened to equal the current one.
            same = self if self.n_blocks_dropped == 0 else replace(self, n_blocks_dropped=0)
            return (same, Y) if Y is not None else same

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
            # The count answers one question: how many trajectories did the
            # user ask for but not get? A downsample answers it afresh. The
            # caller named ``n_trajectories`` here and receives exactly that,
            # so nothing is missing from this design and the earlier loss no
            # longer applies. Carrying the old count forward would make
            # analyze() report a "requested" total the user never asked for.
            n_blocks_dropped=0,
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
            "n_blocks_dropped": self.n_blocks_dropped,
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
        """Rebuild a :class:`MorrisSamples` from a loaded NPZ payload."""
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
            # Files written before this field existed have no entry for it.
            n_blocks_dropped=int(meta.get("n_blocks_dropped", 0)),
        )


def _radial_samples_from_blocks(
    *,
    samples: np.ndarray,
    block_rows: np.ndarray,
    problem: Problem,
) -> MorrisSamples:
    """Assemble a radial :class:`MorrisSamples` from an evaluated design.

    The function does not care where the points came from. The caller supplies
    the unique sample matrix, plus, for each radial block, the row index of the
    block's base point followed by the ``D`` points that each perturb one
    parameter. A design built for another method can therefore be reread as a
    Morris design at zero extra model cost. See
    :meth:`jaxgsa.sobol.SobolSamples.to_morris`.

    ``samples`` passes through unchanged, so outputs already computed for it
    stay aligned and need no re-evaluation.

    Args:
        samples: Unique rows already evaluated, shape ``(n_runs, D)``, in the
            problem's physical units.
        block_rows: Integer indices into ``samples``, shape
            ``(n_blocks, D + 1)``. Column 0 is the block's base point. Column
            ``1 + j`` is the point that differs from it in parameter ``j``
            only.
        problem: Problem definition the samples were drawn for.

    Returns:
        A radial :class:`MorrisSamples` ready for
        :func:`jaxgsa.morris.analyze`.

    Raises:
        ValueError: If fewer than two blocks are left with a measurable step.

    Warns:
        JaxgsaWarning: If any block is dropped because its base and perturbed
            points coincide at the model's floating-point resolution.
    """
    D = problem.num_vars
    # The elementary-effect denominator must be the unit-cube step, matching
    # what ``sample()`` records, so recover unit coordinates in float64.
    samples_unit = _inverse_transform_samples(problem, samples)

    param_idx = np.arange(D)
    # Read each step off the coordinate that actually differs between the base
    # row and the row perturbed in that parameter. This needs no knowledge of
    # the source layout. It collapses to exactly 0 when deduplication has
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
            category=JaxgsaWarning,
        )

    n_blocks = int(block_rows.shape[0])
    if n_blocks < 2:
        raise ValueError(
            f"Only {n_blocks} radial block(s) have a measurable step; "
            "Morris measures need at least 2"
        )

    # Block b occupies expanded rows [b*(D+1), (b+1)*(D+1)): the base point
    # first, then the D perturbed points in parameter order. That is exactly
    # the row order of block_rows, so flattening it gives the expansion map.
    offsets = (np.arange(n_blocks) * (D + 1)).astype(np.int64)
    return MorrisSamples(
        samples=samples,
        n_expanded=n_blocks * (D + 1),
        expanded_to_unique=np.ascontiguousarray(block_rows, dtype=np.int64).reshape(-1),
        n_trajectories=n_blocks,
        # Unused by the radial design, so hold it at the sample() default.
        num_levels=4,
        method="radial",
        ee_idx_after=offsets[:, None] + 1 + param_idx.astype(np.int64),
        ee_idx_before=np.broadcast_to(offsets[:, None], (n_blocks, D)).copy(),
        ee_delta=ee_delta,
        n_params=D,
        problem=problem,
        n_blocks_dropped=n_dropped,
    )


def _build_trajectories(
    n_trajectories: int,
    n_params: int,
    num_levels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate Morris trajectories on the ``num_levels`` grid.

    The function builds the points on an integer half-level grid, in units of
    ``1 / (2 * (p - 1))``. The Morris step ``delta = p / (2 * (p - 1))`` is
    exactly ``p`` half-levels there. Converting to float once at the end makes
    equal grid points bitwise identical across trajectories, which is what lets
    exact deduplication collapse them.

    The function consumes randomness strictly per trajectory. The first m
    trajectories of an r-trajectory draw therefore match a direct m-trajectory
    draw.

    Returns:
        ``(expanded_unit, ee_idx_after, ee_idx_before, ee_delta)``.
        ``expanded_unit`` holds unit-cube points of shape ``(r * (D + 1), D)``,
        and the three bookkeeping arrays have shape ``(r, D)``.
    """
    D = n_params
    p = num_levels
    # delta = p / (2(p-1)) expressed in half-level units of 1/(2(p-1)).
    delta_int = p
    delta = p / (2.0 * (p - 1))
    # A base level l/(p-1) must leave room for a +delta step: l <= p/2 - 1.
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

    Each trajectory uses a base point ``a`` and an auxiliary point ``b``, both
    drawn from a ``2D``-dimensional Sobol' sequence. ``b`` is shifted
    ``_RADIAL_SHIFT`` draws ahead of ``a``. Star row i swaps coordinate i of
    ``a`` with ``b_i``. Every elementary effect therefore compares one star row
    against the shared base row, with the per-step delta ``b_i - a_i``.

    Returns:
        The same tuple layout as :func:`_build_trajectories`.

    Raises:
        ValueError: If any ``|delta|`` falls below the working float-precision
            floor (see :func:`_min_radial_delta`).
    """
    D = n_params
    r = n_trajectories
    sampler = Sobol(d=2 * D, scramble=scramble, seed=seed)
    # Draw a power-of-2 count, which scipy warns about otherwise, and slice.
    # Sobol' prefixes are bit-identical, so the extra rows change nothing.
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

    # Star block j: row 0 is the base point a[j], and row 1+i is a[j] with
    # coordinate i alone swapped to b[j, i]. This consumes no randomness, so
    # the whole design is three vectorized assignments.
    block = np.repeat(a[:, None, :], D + 1, axis=1)  # (r, D+1, D)
    diag = np.arange(D)
    block[:, 1 + diag, diag] = b  # swap coordinate i of star row 1+i
    expanded_unit = block.reshape(r * (D + 1), D)

    offsets = (np.arange(r) * (D + 1)).astype(np.int64)
    ee_idx_before = np.broadcast_to(offsets[:, None], (r, D)).copy()
    ee_idx_after = offsets[:, None] + 1 + diag.astype(np.int64)

    return expanded_unit, ee_idx_after, ee_idx_before, ee_delta


def _squash_open_sides(
    expanded_unit: np.ndarray,
    ee_delta: np.ndarray,
    problem: Problem,
    truncation_quantile: float,
) -> None:
    """Pull the design away from the unit-cube faces on open marginal sides.

    A Morris design touches the unit-cube boundaries exactly, and an unbounded
    inverse CDF maps 0 and 1 to -inf and +inf. The function therefore pulls in
    every side of a marginal whose support is open by the tail probability
    ``q``, which keeps the transform finite.

    Only genuinely open sides move. Uniform and categorical marginals are
    bounded, so they never move. A Gaussian marginal with an explicit ``low``
    is bounded below, and one with an explicit ``high`` is bounded above. A
    two-sided truncated Gaussian therefore gets no squash at all. Truncating it
    again would silently narrow the input model the user declared.

    The squash is an affine map on each dimension, so it also rescales the step
    that each elementary effect actually takes. The function multiplies
    ``ee_delta`` by the same per-dimension factor. That keeps the divisor equal
    to the coordinate difference the design really uses.

    The function modifies both arrays in place. The map is deterministic, so it
    leaves exact deduplication and prefix-nested downsampling unaffected.

    Args:
        expanded_unit: Unit-cube design, shape ``(r * (D + 1), D)``, modified
            in place.
        ee_delta: Signed unit-cube steps, shape ``(r, D)``, modified in place.
        problem: Problem definition whose marginals decide which sides are open.
        truncation_quantile: Tail probability ``q`` removed from each open side.
    """
    q = truncation_quantile
    for idx, spec in enumerate(problem.input_specs):
        # Index rather than unpack: the normalized spec carries a trailing
        # categorical payload slot, so its width is not fixed.
        dist, low, high = spec[0], spec[3], spec[4]
        # Only a Gaussian marginal can be unbounded. Uniform and categorical
        # marginals are bounded, so they never move.
        if dist != "gaussian":
            continue
        lo_target = 0.0 if low is not None else q
        hi_target = 1.0 if high is not None else 1.0 - q
        scale = hi_target - lo_target
        if scale == 1.0:  # both sides already bounded, so nothing to squash
            continue
        expanded_unit[:, idx] = lo_target + expanded_unit[:, idx] * scale
        ee_delta[:, idx] *= scale


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
    truncation_quantile: float = 1e-4,
    verbose: bool = True,
) -> MorrisSamples:
    """Generate unique Morris elementary-effects samples for model evaluation.

    Morris screening ranks parameters by one-at-a-time finite differences,
    called elementary effects, spread across the whole input domain. It is the
    usual first step for weeding out unimportant parameters before a more
    expensive variance-based analysis. Each trajectory perturbs every parameter
    exactly once, so the full design costs at most
    ``n_trajectories * (D + 1)`` model evaluations.

    This function builds those ``n_trajectories`` paths of ``D + 1`` points
    each. It removes exact duplicate rows while it keeps first-occurrence
    order, and returns only the unique rows for the user to evaluate.
    :func:`jaxgsa.morris.analyze` rebuilds the expanded layout internally.

    Gaussian marginals are supported through a truncated-quantile grid. The
    Morris design touches the unit-cube boundaries, and an unbounded inverse
    CDF maps 0 and 1 to infinity. Each open side of a Gaussian marginal is
    therefore pulled in by ``q = truncation_quantile`` before the transform. A
    side that the problem already bounds with an explicit ``low`` or ``high``
    is left alone, so a two-sided truncated Gaussian is sampled exactly as
    declared. Elementary effects use the step the design really takes, so the
    squash does not bias them. :meth:`MorrisResult.to_physical_units` is
    unavailable for Gaussian problems because the transform is nonlinear.

    On an unbounded marginal ``mu_star`` has no ``q -> 0`` limit. The design
    always includes unit levels 0 and 1 exactly. A smaller ``q`` therefore
    always reaches further into the tail, and the elementary effects grow with
    it. ``mu_star`` magnitudes are scale-dependent by construction on an
    unbounded marginal, so only rankings are comparable across truncation
    settings. Use :meth:`jaxgsa.Problem.from_dict` with ``truncate_gaussians``
    if you want one bounded input model that every method shares.

    Args:
        problem: Problem definition with uniform and/or Gaussian marginals.
        n_trajectories: Number of trajectories r (>= 2). Each one contributes
            one elementary effect per parameter, so r is the sample size behind
            every screening measure. More trajectories tighten the mu_star
            ranking and any bootstrap intervals, at proportionally more model
            evaluations. Typical screening uses 10-50.
        num_levels: Grid levels ``p`` for the trajectory design (default 4,
            step ``delta = p / (2 * (p - 1))``). An even value makes all levels
            equally probable. An odd value triggers a warning. The radial
            design ignores this argument.
        method: ``"trajectory"`` for Morris 1991 grid walks (default), or
            ``"radial"`` for Campolongo 2011 star designs around
            scrambled-Sobol' base points. The radial design spreads points
            quasi-randomly instead of on a coarse grid, and it has no
            ``num_levels`` to choose. In exchange it leaves fewer duplicate
            rows to deduplicate.
        scramble: Whether to Owen-scramble the Sobol' sequence. Radial design
            only.
        seed: Random seed or generator for reproducibility. Pass an ``int`` or
            ``None`` to keep the prefix-nesting guarantee of
            :meth:`MorrisSamples.downsample`. A reused ``np.random.Generator``
            advances its state between calls, which breaks that nesting.
        truncation_quantile: Tail probability ``q`` removed from each open side
            of every Gaussian marginal's grid (default 1e-4, probing the
            0.01%-99.99% quantile range). At this default the grid drops 0.29%
            of the marginal variance and 5.0% of its fourth moment. The former
            default of 5e-3 dropped 7.5% and 24%, and it visibly perturbed
            rankings. Sides that the marginal already bounds with ``low`` or
            ``high`` are not squashed. Ignored for uniform marginals.
        verbose: If ``True`` (default), print a short summary that includes how
            many duplicate rows were removed.

    Returns:
        A :class:`MorrisSamples` with a unique sample matrix plus
        elementary-effect bookkeeping for the later analysis.

    Raises:
        ValueError: If ``n_trajectories``, ``num_levels``, ``method``, or
            ``truncation_quantile`` is invalid. It also raises when
            ``problem.correlation`` declares a dependence structure, because
            the one-at-a-time design assumes independent inputs. It raises
            again when ``problem`` has categorical parameters, because the
            design steps each parameter along a grid and a grid has no meaning
            for unordered level codes.
    """
    _raise_correlated_design(problem, "jaxgsa.morris.sample")
    _raise_categorical_design(problem, "jaxgsa.morris.sample")
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
            category=JaxgsaWarning,
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
        _squash_open_sides(expanded_unit, ee_delta, problem, truncation_quantile)

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
