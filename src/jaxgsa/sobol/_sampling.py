"""Sobol/Saltelli sampling with unique user-facing rows.

The public contract of this module is intentionally split in two layers:

1. ``sample()`` returns only the unique rows that a user should evaluate.
2. ``SobolSamples`` also carries enough metadata to reconstruct the full
   expanded Saltelli layout later inside :func:`jaxgsa.sobol.analyze`.

This avoids wasted model evaluations in low-dimensional cases where the
expanded Saltelli design contains exact duplicate rows.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, overload

import numpy as np
from scipy.stats.qmc import Sobol

from jaxgsa._core.samples import UniqueDesignSamples
from jaxgsa._core.sampling import (
    _is_power_of_2,
    _next_power_of_2,
    _power_of_2_error,
    _stable_unique_rows,
    _transform_samples,
)
from jaxgsa._core.validation import _raise_categorical_design, _raise_correlated_design
from jaxgsa.problem import Problem, _categorical_dims

if TYPE_CHECKING:
    from jaxgsa.morris import MorrisSamples


@dataclass(frozen=True)
class SobolSamples(UniqueDesignSamples):
    """Unique Sobol samples plus metadata for Saltelli reconstruction.

    Returned by :func:`jaxgsa.sobol.sample`. Evaluate your model at every row of
    ``samples`` (in order) and pass this object together with the outputs to
    :func:`jaxgsa.sobol.analyze`.

    Note the two row counts: ``n_runs`` is the number of *unique* rows you
    must evaluate (one model run per row), while ``n_expanded`` is the
    (larger or equal) pre-deduplication size of the full Saltelli design that
    the analysis reconstructs internally.

    ``samples`` is a deduplicated evaluation set, not a distributional
    sample. Duplicate rows are collapsed, so the empirical marginal of a
    column in ``samples`` does not match the declared input distribution.
    The effect is strong for a categorical parameter, where whole rows
    repeat often: with ``probs = [0.9, 0.1]`` the ``samples`` column shows
    about ``[0.84, 0.16]``. The declared marginal is recovered only in the
    expanded design, which ``analyze`` rebuilds through
    ``expanded_to_unique``, so the indices are correct. Evaluate
    ``samples`` and pass the outputs to ``analyze``; do not reuse
    ``samples`` on its own as a Monte Carlo design.

    Attributes:
        samples: Unique rows to evaluate with the user's model. Shape
            ``(n_runs, D)`` where ``D`` is the number of parameters, in
            physical units (each Sobol marginal transformed into the
            problem's declared input distribution). Deduplicated, so its
            empirical marginal is distorted -- see the note above.
        sample_ids: Stable integer identifiers aligned 1:1 with ``samples``.
            Useful for joining model outputs back onto the sampling table.
        n_expanded: Row count of the full expanded Saltelli layout before
            deduplication. This is the number of rows analyzed internally.
        expanded_to_unique: Integer index map of shape ``(n_expanded,)``.
            For each expanded Saltelli row, gives the corresponding row index in
            ``samples``.
        base_n: Number of base Sobol points used to construct the Saltelli
            design. Always a power of 2.
        n_params: Number of problem dimensions ``D``.
        calc_second_order: Whether the expanded design includes the extra
            cross-matrices needed for second-order Sobol indices.
        problem: Problem definition used to transform the samples.
    """

    samples: np.ndarray  # shape (n_unique, D), scaled to bounds
    sample_ids: np.ndarray
    n_expanded: int
    expanded_to_unique: np.ndarray
    base_n: int
    n_params: int
    calc_second_order: bool
    problem: Problem

    @overload
    def downsample(self, base_n: int) -> SobolSamples: ...

    @overload
    def downsample(self, base_n: int, Y: np.ndarray) -> tuple[SobolSamples, np.ndarray]: ...

    def downsample(
        self, base_n: int, Y: np.ndarray | None = None
    ) -> SobolSamples | tuple[SobolSamples, np.ndarray]:
        """Return a smaller ``SobolSamples`` by prefix-slicing to a lower ``base_n``.

        Sobol sequences are prefix-nested: the first *K* base points of a
        draw with *N > K* base points are bit-identical to drawing *K*
        base points directly (same seed and scramble).  This means you can
        simulate the model once at the largest ``base_n`` and recover exact
        results for any smaller power-of-2 ``base_n`` by slicing — no
        re-simulation needed.

        Optionally pass ``Y`` (model outputs aligned with ``samples``) to
        get the corresponding output slice back — similar to how
        ``sklearn.model_selection.train_test_split`` accepts both *X* and
        *y*.

        This property does **not** hold for Latin Hypercube Sampling (LHS),
        whose stratification depends on *N*.

        Args:
            base_n: Target base size (must be a power of 2 and
                ``<= self.base_n``).
            Y: Model outputs with shape ``(n_runs, ...)``.  When provided,
                the matching prefix is returned alongside the new result.

        Returns:
            ``SobolSamples`` when called without ``Y``, or
            ``(SobolSamples, Y_small)`` when ``Y`` is provided.

        Raises:
            ValueError: If ``base_n`` is not a power of 2, exceeds
                ``self.base_n``, or ``Y`` has too few rows.
        """
        if not _is_power_of_2(base_n):
            raise ValueError(_power_of_2_error("base_n", base_n, reason="Sobol' sequence balance"))
        if base_n > self.base_n:
            raise ValueError(
                f"Cannot upsample: requested base_n={base_n} > current base_n={self.base_n}"
            )
        self._validate_downsample_Y(Y)
        if base_n == self.base_n:
            return (self, Y) if Y is not None else self

        step = _saltelli_step(self.n_params, self.calc_second_order)
        new_expanded_n = base_n * step
        samples_small, new_exp2uniq, n_unique_new, Y_small = self._prefix_slice(new_expanded_n, Y)

        sr_small = SobolSamples(
            samples=samples_small,
            sample_ids=np.arange(n_unique_new, dtype=np.int64),
            n_expanded=new_expanded_n,
            expanded_to_unique=new_exp2uniq,
            base_n=base_n,
            n_params=self.n_params,
            calc_second_order=self.calc_second_order,
            problem=self.problem,
        )

        if Y_small is not None:
            return sr_small, Y_small
        return sr_small

    def to_morris(self, *, verbose: bool = True) -> MorrisSamples:
        """Reinterpret this Saltelli design as a radial Morris design.

        A Saltelli design already *is* a Morris radial (star) design: within
        each base point, the row ``A`` and the ``D`` rows ``AB_j`` differ in
        exactly one parameter, which is what an elementary effect needs.
        Campolongo et al. (2011) build the radial design from a ``2D``-dimensional
        Sobol' sequence for precisely this reason, and
        :func:`jaxgsa.sobol.sample` draws the same sequence the same way.

        The two methods then weight the same increments differently. Writing
        ``EE_j = (f(AB_j) - f(A)) / delta_j`` with ``delta_j = B_j - A_j``,
        Jansen's total-order estimator is ``E[(delta_j * EE_j)^2] / (2 Var Y)``
        while Morris reports ``mu_star = E|EE_j|``. So screening measures come
        out of a design you have already paid for — **no extra model
        evaluations**. Pass the returned object and your existing ``Y`` (the
        same array you would pass to :func:`jaxgsa.sobol.analyze`) to
        :func:`jaxgsa.morris.analyze`.

        **Which estimand this is.** The derived design is a *radial* design,
        so it estimates the radial quantity
        ``E|f(A with B_j) - f(A)| / |B_j - A_j|``, in which the step varies
        from block to block. That is not the classical Morris quantity, which
        uses one fixed grid step ``Delta``. The two differ by much more than
        sampling noise: on Ishigami at ``r = 8192`` the derived ``mu_star`` is
        ``[8.68, 15.01, 6.62]`` against ``[8.69, 15.02, 6.64]`` from
        ``morris.sample(..., method="radial")``, but ``[7.59, 7.88, 6.39]``
        from the default ``method="trajectory"`` — a factor 1.9 on ``x2``, and
        2.5 on its ``sigma``. ``morris.sample`` defaults to
        ``method="trajectory"``, so compare these measures against
        ``morris.sample(..., method="radial")``, never against the default.

        ``n_trajectories`` is ``base_n`` for both design variants: one radial
        block per base point, based at ``A``. Second-order designs also hold a
        block based at ``B`` (``B`` with its ``BA_j`` rows), which this method
        deliberately does not harvest. The reason is that pooling it buys
        nothing measurable. The two blocks are *not* algebraically the same
        effect in general — that equality holds only for additive
        contributions, and the measured paired-effect correlations on Ishigami
        are 0.50 / 1.00 / -0.06, so only ``x2`` (from the purely additive
        ``7 sin^2(x2)`` term) is a genuine duplicate. But over 150 seeds at
        ``base_n = 128`` the pooled estimator's variance ratio against the
        A-only estimator is ``[1.07, 1.00, 1.59]``: no reduction, and worse on
        ``x3``. Pooling would also need a cluster bootstrap over base points to
        keep the confidence intervals honest, because the two blocks in a base
        point are dependent. That is real machinery for no gain.

        Because the derived measures reuse the very same model outputs as the
        Sobol indices, agreement between ``mu_star`` and ``ST`` is not an
        independent check of either.

        Args:
            verbose: If ``True`` (default), print a short summary of the
                derived design.

        Returns:
            A ``MorrisSamples`` whose ``samples`` is this object's ``samples``
            unchanged, so ``n_runs`` and any outputs computed for it stay valid.

        Raises:
            ValueError: If fewer than two blocks are left with a measurable
                step (see below).

        Warns:
            UserWarning: If any parameter has an *unbounded* Gaussian
                marginal. ``mu_star`` then has no fixed scale, because how far
                the design reaches into the tail sets its magnitude, and the
                Saltelli design and :func:`jaxgsa.morris.sample` reach
                different distances (the Saltelli design bounds support only at
                the library's own clip, +/-7.03 sigma). Only *rankings* are
                comparable across designs. Bound the marginals with
                ``Problem.from_dict(..., truncate_gaussians=q)`` if magnitudes
                must match. Once both sides are bounded the derived and native
                radial measures agree: measured ratios 0.999 (linear), 0.997
                (``x^2``), 0.988 (``x^4``), 0.987 (``exp(x^2/3)``), each within
                its own seed-to-seed spread.
            UserWarning: If any block is dropped for having a near-zero step.
                Unlike :func:`jaxgsa.morris.sample`'s radial design, which
                offsets the auxiliary points by four draws, Saltelli takes
                ``A`` and ``B`` from the *same* Sobol' row, so the two can
                coincide. This is a non-issue at the default
                ``scramble=True``: 0 of 65536 blocks were dropped across 8
                seeds at ``D = 3``. With ``scramble=False`` the drop rate is
                real but falls off with ``base_n`` — measured 21.9% at
                ``base_n=64``, 9.4% at 256, 2.3% at 1024 and 1.2% at 4096 — and
                the survivors are a *biased* subsequence: ``mu_star`` comes out
                ``[8.34, 14.88, 5.55]`` at ``base_n=64`` against
                ``[8.68, 15.01, 6.62]`` scrambled, so ``x3`` reads 16% low.
                Keep ``scramble=True``.

        References:
            Campolongo, Cariboni & Saltelli (2011). Comput. Phys. Commun.
                182:978-988.
            Jansen (1999). Comput. Phys. Commun. 117:35-43.
        """
        # An elementary effect divides by a step along one input axis, which
        # has no meaning for unordered level codes.
        _raise_categorical_design(self.problem, "jaxgsa.sobol.SobolSamples.to_morris")
        # Imported lazily: morris knows nothing about sobol, and this keeps the
        # dependency one-directional and free of an import cycle.
        from jaxgsa.morris._sampling import _radial_samples_from_blocks

        D = self.n_params
        step = _saltelli_step(D, self.calc_second_order)
        # First expanded row of each base point's group, and a column vector of
        # the same for broadcasting one row per parameter.
        starts = np.arange(self.base_n) * step
        offsets = starts[:, None]
        params = np.arange(D)

        # Layout per base point: [A, AB_0..AB_{D-1}, (BA_0..BA_{D-1},) B].
        # Only the A-based block is harvested. Second-order designs also hold a
        # radial block based at B (B with its BA_j rows). The two blocks are
        # algebraically the same effect only for additive contributions:
        # measured paired-effect correlations on Ishigami are 0.50 / 1.00 /
        # -0.06, so only x2 (the additive 7 sin^2(x2) term) is a true
        # duplicate. The reason to skip the B block is not duplication, it is
        # that pooling gives no measured variance reduction — the pooled /
        # A-only variance ratio over 150 seeds at base_n=128 is
        # [1.07, 1.00, 1.59] — while requiring a cluster bootstrap over base
        # points to keep the CIs honest, since the two blocks share a base
        # point. Real machinery, no gain.
        block_rows = np.empty((self.base_n, D + 1), dtype=np.int64)
        block_rows[:, 0] = self.expanded_to_unique[starts]
        block_rows[:, 1:] = self.expanded_to_unique[offsets + 1 + params]

        _warn_unbounded_gaussian(self.problem)
        derived = _radial_samples_from_blocks(
            samples=self.samples,
            block_rows=block_rows,
            problem=self.problem,
        )
        if verbose:
            _print_to_morris_summary(
                n_params=D,
                base_n=self.base_n,
                n_blocks=derived.n_trajectories,
                n_runs=derived.n_runs,
                calc_second_order=self.calc_second_order,
            )
        return derived

    def _extra_arrays(self) -> dict[str, np.ndarray]:
        """Persist the sample identifiers alongside the base arrays."""
        return {"sample_ids": self.sample_ids}

    def _extra_metadata(self) -> dict[str, Any]:
        """Persist the Saltelli design parameters in the metadata blob."""
        return {"base_n": self.base_n, "calc_second_order": self.calc_second_order}

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
    ) -> SobolSamples:
        """Rebuild a ``SobolSamples`` from a loaded NPZ payload."""
        return cls(
            samples=samples,
            sample_ids=arrays["sample_ids"],
            n_expanded=n_expanded,
            expanded_to_unique=expanded_to_unique,
            base_n=int(meta["base_n"]),
            n_params=problem.num_vars,
            calc_second_order=bool(meta["calc_second_order"]),
            problem=problem,
        )


def _saltelli_step(n_params: int, calc_second_order: bool) -> int:
    """Return the number of expanded Saltelli rows per base Sobol point."""
    # Saltelli step = A + D*AB [+ D*BA] + B = D+2 [or 2D+2]
    return 2 * n_params + 2 if calc_second_order else n_params + 2


# Inflation-loop safety valve: never let a doubling materialize more candidate
# rows than this floor (or 8x the requested unique count, whichever is
# larger). The cap is checked BEFORE the doubling is built, so the warning
# fires before any huge allocation.
_CANDIDATE_ROW_CAP = 1 << 22


def _max_distinct_rows(problem: Problem) -> int | None:
    """Upper bound on the number of distinct sample rows, if one exists.

    An all-categorical problem can only produce ``prod(L_d)`` distinct rows
    (each column takes one of its ``L_d`` level codes), no matter how large
    the design grows. Any continuous column makes the count unbounded, in
    which case ``None`` is returned.
    """
    dims_levels = _categorical_dims(problem)
    if len(dims_levels) != problem.num_vars:
        return None
    return math.prod(n_levels for _, n_levels in dims_levels)


def _build_expanded_samples(
    n_params: int,
    base_n: int,
    *,
    calc_second_order: bool,
    scramble: bool,
    seed: int | np.random.Generator | None,
) -> np.ndarray:
    """Generate the full expanded Saltelli matrix for a fixed ``base_n``.

    The returned matrix still includes exact duplicate rows when the Saltelli
    construction collapses in low dimensions. Deduplication happens later.
    """
    D = n_params
    step = _saltelli_step(D, calc_second_order)
    # Draw from a 2D-dimensional Sobol sequence; first D dims become matrix A, last D become B
    sampler = Sobol(d=2 * D, scramble=scramble, seed=seed)
    base = sampler.random(base_n)

    # Split the 2D-dimensional Sobol draw into the standard Saltelli base
    # matrices A and B, each with shape (base_n, D).
    A = base[:, :D]
    B = base[:, D:]
    # Saltelli interleaved layout per base point i:
    #   [A_i, AB_0, ..., AB_{D-1}, BA_0, ..., BA_{D-1}, B_i]
    # AB_j = A with column j replaced by B's column j (and vice-versa for BA_j).
    # Built fully vectorized into one preallocated array: a per-row Python
    # list would transiently hold base_n * step small arrays plus object
    # overhead, which dwarfs the design itself at large sizes.
    diag = np.arange(D)
    out = np.empty((base_n, step, D), dtype=base.dtype)
    out[:, 0, :] = A
    AB = np.repeat(A[:, None, :], D, axis=1)  # (base_n, D, D)
    AB[:, diag, diag] = B
    out[:, 1 : D + 1, :] = AB
    if calc_second_order:
        BA = np.repeat(B[:, None, :], D, axis=1)
        BA[:, diag, diag] = A
        out[:, D + 1 : 2 * D + 1, :] = BA
    out[:, -1, :] = B
    return out.reshape(base_n * step, D)


def _warn_unbounded_gaussian(problem: Problem) -> None:
    """Warn that an unbounded Gaussian gives ``mu_star`` no fixed scale.

    ``GaussianInputSpec`` accepts ``low`` and/or ``high``, so a one-sided
    truncation still leaves the opposite tail unbounded and is reported. Only
    uniforms and two-sided-truncated Gaussians stay silent.
    """
    # Only Gaussian marginals can be unbounded; categorical codes are bounded.
    unbounded = [
        name
        for name, spec in zip(problem.names, problem.input_specs)
        if spec[0] == "gaussian" and (spec[3] is None or spec[4] is None)
    ]
    if not unbounded:
        return
    warnings.warn(
        f"jaxgsa: parameters {unbounded} have unbounded gaussian marginals. An elementary "
        "effect on an unbounded marginal has no fixed scale: how far the design reaches "
        "into the tail sets the magnitude of mu_star, and the Saltelli design and "
        "morris.sample reach different distances. Rankings are unaffected. Use "
        "Problem.from_dict(..., truncate_gaussians=q) if magnitudes must be comparable "
        "across designs",
        # Reached from SobolSamples.to_morris, so the user's frame is two up.
        stacklevel=3,
    )


def _print_to_morris_summary(
    *,
    n_params: int,
    base_n: int,
    n_blocks: int,
    n_runs: int,
    calc_second_order: bool,
) -> None:
    """Print a compact summary of the Morris design derived from a Saltelli design."""
    order_label = "second-order" if calc_second_order else "first/total-order"
    print(
        "jaxgsa.sobol.SobolSamples.to_morris: "
        f"D={n_params}, mode={order_label}, base_n={base_n}, "
        f"blocks={n_blocks}, effects={n_blocks * n_params}, "
        f"reusing n_runs={n_runs} existing evaluations (0 new model runs)"
    )


def _print_sampling_summary(
    *,
    n_params: int,
    target_n: int,
    n_runs: int,
    n_expanded: int,
    base_n: int,
    calc_second_order: bool,
    scramble: bool,
) -> None:
    """Print a compact summary of the generated unique Sobol design."""
    duplicates_removed = n_expanded - n_runs
    duplicate_fraction = duplicates_removed / n_expanded if n_expanded else 0.0
    order_label = "second-order" if calc_second_order else "first/total-order"
    print(
        "jaxgsa.sobol.sample: "
        f"D={n_params}, mode={order_label}, base_n={base_n}, "
        f"requested_runs>={target_n}, n_runs={n_runs}, "
        f"n_expanded={n_expanded}, duplicates_removed={duplicates_removed} "
        f"({duplicate_fraction:.1%}), scramble={scramble}"
    )


def sample(
    problem: Problem,
    n_samples: int,
    *,
    base_n: int | None = None,
    calc_second_order: bool = True,
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
    verbose: bool = True,
) -> SobolSamples:
    """Generate the input samples needed for Sobol analysis with ``jaxgsa.sobol.analyze``.

    Typical usage: call this once, evaluate your model at every row of the
    returned ``result.samples`` (shape ``(n_runs, D)``), then pass the result
    object and your outputs to :func:`jaxgsa.sobol.analyze`.

    Internally this builds a Saltelli design — the structured layout of base
    and column-swapped sample matrices that Sobol index estimators require —
    for a candidate ``base_n``, then removes exact duplicate rows while
    preserving first-occurrence order (in low dimensions the Saltelli
    construction repeats rows, and evaluating a model twice on the same input
    is wasted work). If the unique matrix is still smaller than the requested
    evaluation budget, ``base_n`` is doubled and the process repeats until
    enough unique rows are available.

    Args:
        problem: Problem definition with parameter names and distributions.
        n_samples: Minimum desired number of unique model evaluations. The
            returned design may contain somewhat more rows than this, never
            fewer. Ignored when ``base_n`` is provided.
        base_n: If given, use this exact Sobol base size (must be a power
            of 2) instead of searching for one. The expanded Saltelli
            design will have ``base_n * (2*D + 2)`` rows (second order)
            or ``base_n * (D + 2)`` rows (first/total only). This gives
            direct control over the sampling budget.
        calc_second_order: If ``True`` (default), include the extra
            cross-matrices needed to estimate second-order (pairwise
            interaction) Sobol indices.  Set to ``False`` if you only need
            first/total-order indices — it nearly halves the evaluation
            budget (Saltelli step ``D + 2`` instead of ``2*D + 2``).
        scramble: Whether to apply Owen scrambling to the Sobol sequence
            (recommended; randomizes the sequence so different seeds give
            statistically independent designs).
        seed: Random seed or generator for reproducibility.
        verbose: If ``True`` (default), print a short summary describing the
            requested unique count, returned unique count, expanded Saltelli
            size, and how many duplicate rows were removed.

    Returns:
        SobolSamples with a unique sample matrix plus expansion metadata for
        later Sobol analysis.

    Raises:
        ValueError: If ``base_n`` is not a power of 2, or
            ``problem.correlation`` declares a dependence structure (the
            Saltelli design and its estimators assume independent inputs).
    """
    _raise_correlated_design(problem, "jaxgsa.sobol.sample")
    D = problem.num_vars
    # Rows per base point -- determines the ratio between base_n and total expanded rows
    step = _saltelli_step(D, calc_second_order)

    if base_n is not None:
        if not _is_power_of_2(base_n):
            raise ValueError(_power_of_2_error("base_n", base_n, reason="Sobol' sequence balance"))
        target_n = None
    else:
        target_n = max(1, n_samples)
        # Estimate initial base_n from requested unique count,
        # rounding up to next power of 2 (Sobol sequence requirement).
        base_n = _next_power_of_2(math.ceil(target_n / step))

    expanded_samples_unit = _build_expanded_samples(
        D,
        base_n,
        calc_second_order=calc_second_order,
        scramble=scramble,
        seed=seed,
    )
    expanded_samples = _transform_samples(problem, expanded_samples_unit)
    unique_samples, expanded_to_unique = _stable_unique_rows(expanded_samples)

    if target_n is not None:
        # Deduplication may reduce the unique count below the target; double
        # base_n and rebuild until enough unique rows exist. Finite-support
        # marginals (today: categorical level codes) collapse whole
        # probability bins onto one value, so the unique count can saturate
        # and the loop would otherwise inflate the design forever. Two
        # guards stop it: a saturation predicate (the known distinct-row
        # bound is reached, or a doubling added no new unique rows) and a
        # candidate-row cap checked BEFORE the next doubling is
        # materialized, so the warning fires before any huge allocation.
        # The returned design then keeps duplicate rows — they are
        # legitimate Saltelli samples; the dedup exists only to save model
        # evaluations.
        max_unique = _max_distinct_rows(problem)
        max_candidate_rows = max(_CANDIDATE_ROW_CAP, 8 * target_n)
        prev_unique = -1
        while unique_samples.shape[0] < target_n:
            n_unique = unique_samples.shape[0]
            # One saturation predicate feeds both the break and the warning
            # reason, so the two can never drift apart.
            bound_reached = max_unique is not None and n_unique >= max_unique
            saturated = bound_reached or n_unique == prev_unique
            next_rows = 2 * base_n * step
            if saturated or next_rows > max_candidate_rows:
                if bound_reached:
                    reason = f"the problem has only {max_unique} possible distinct rows"
                elif saturated:
                    reason = "the last base_n doubling added no new unique rows"
                else:
                    reason = (
                        f"the next doubling would materialize {next_rows} candidate "
                        f"rows, above the cap of {max_candidate_rows}"
                    )
                warnings.warn(
                    f"jaxgsa.sobol.sample: the requested n_samples={target_n} "
                    f"unique rows cannot be reached because {reason}. The "
                    f"design is returned with {n_unique} unique rows and "
                    "keeps its duplicate rows. Duplicates are valid Saltelli "
                    "samples; deduplication only saves model evaluations",
                    stacklevel=2,
                )
                break
            prev_unique = n_unique
            base_n *= 2
            expanded_samples_unit = _build_expanded_samples(
                D,
                base_n,
                calc_second_order=calc_second_order,
                scramble=scramble,
                seed=seed,
            )
            expanded_samples = _transform_samples(problem, expanded_samples_unit)
            unique_samples, expanded_to_unique = _stable_unique_rows(expanded_samples)

    sample_ids = np.arange(unique_samples.shape[0], dtype=np.int64)
    if verbose:
        _print_sampling_summary(
            n_params=D,
            target_n=target_n if target_n is not None else unique_samples.shape[0],
            n_runs=unique_samples.shape[0],
            n_expanded=expanded_samples.shape[0],
            base_n=base_n,
            calc_second_order=calc_second_order,
            scramble=scramble,
        )

    return SobolSamples(
        samples=unique_samples,
        sample_ids=sample_ids,
        n_expanded=expanded_samples.shape[0],
        expanded_to_unique=expanded_to_unique,
        base_n=base_n,
        n_params=D,
        calc_second_order=calc_second_order,
        problem=problem,
    )
