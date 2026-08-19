"""Sobol/Saltelli sampling with unique user-facing rows.

The public contract of this module has two layers:

1. ``sample()`` returns only the unique rows that a user should evaluate.
2. ``SobolSamples`` also carries enough metadata to reconstruct the full
   expanded Saltelli layout later inside :func:`jaxgsa.sobol.analyze`.

This split avoids wasted model evaluations. In low dimensions the expanded
Saltelli design contains exact duplicate rows, and evaluating a model twice
on the same input buys nothing.

Deduplication runs on the unit cube, before the marginal transform, so that
``SobolSamples`` can also keep the theta-free unit-cube design alongside the
physical one. ``SobolSamples.transform`` uses it to re-derive the physical
design for other distribution parameters, differentiably. See
:func:`_dedupe_design` for why the reordering is free.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple, overload

import numpy as np
from scipy.stats.qmc import Sobol

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.samples import UniqueDesignSamples
from jaxgsa._core.sampling import (
    Theta,
    _is_power_of_2,
    _jax_transform_samples,
    _next_power_of_2,
    _power_of_2_error,
    _stable_unique_rows,
    _transform_samples,
)
from jaxgsa._core.validation import _raise_categorical_design, _raise_correlated_design
from jaxgsa._core.warning_types import JaxgsaWarning
from jaxgsa.problem import CategoricalSpec, GaussianSpec, Problem, _categorical_dims

if TYPE_CHECKING:
    from jax import Array

    from jaxgsa.morris import MorrisSamples


@dataclass(frozen=True)
class SobolSamples(UniqueDesignSamples):
    """Unique Sobol samples plus metadata for Saltelli reconstruction.

    Returned by :func:`jaxgsa.sobol.sample`. Evaluate your model at every row of
    ``samples`` (in order) and pass this object together with the outputs to
    :func:`jaxgsa.sobol.analyze`.

    The object carries two row counts. ``n_runs`` is the number of unique rows
    you must evaluate, one model run per row. ``n_expanded`` is the size of the
    full Saltelli design before deduplication, which the analysis reconstructs
    internally. ``n_expanded`` is always greater than or equal to ``n_runs``.

    ``samples`` is a deduplicated evaluation set, not a distributional
    sample. Duplicate rows are collapsed, so the empirical marginal of a
    column in ``samples`` does not match the declared input distribution.
    The effect is strong for a categorical parameter, where whole rows
    repeat often: with ``probs = [0.9, 0.1]`` the ``samples`` column shows
    about ``[0.84, 0.16]``. The declared marginal is recovered only in the
    expanded design, which ``analyze`` rebuilds through
    ``expanded_to_unique``, so the indices are correct. Evaluate
    ``samples`` and pass the outputs to ``analyze``. Do not reuse
    ``samples`` on its own as a Monte Carlo design.

    Attributes:
        samples: Unique rows to evaluate with the user's model, shape
            ``(n_runs, D)``. The values are in physical units: each Sobol
            marginal is transformed into the problem's declared input
            distribution. The rows are deduplicated, so the empirical marginal
            is distorted (see the paragraph above).
        sample_ids: Stable integer identifiers aligned 1:1 with ``samples``.
            Use them to join model outputs back onto the sampling table.
        n_expanded: Row count of the full expanded Saltelli layout before
            deduplication. This is the number of rows analyzed internally.
        expanded_to_unique: Integer index map, shape ``(n_expanded,)``. For
            each expanded Saltelli row, it gives the matching row index in
            ``samples``.
        base_n: Number of base Sobol points used to build the Saltelli design.
            Always a power of 2.
        n_params: Number of problem dimensions ``D``.
        calc_second_order: Whether the expanded design includes the extra
            cross-matrices needed for second-order Sobol indices.
        problem: Problem definition used to transform the samples.
        unit: The same design on the unit cube, before any marginal
            transform, shape ``(n_unit, D)``. These points depend only on the
            Sobol' sequence, never on the input distributions, which is what
            :meth:`transform` needs to re-derive the physical design for other
            distribution parameters. For a continuous problem the rows
            correspond 1:1 with ``samples``. For a problem with a categorical
            parameter they do not, and ``n_unit`` is larger: many unit values
            map to one level code, and those rows are collapsed after the
            transform. ``None`` only on a design that :func:`sample` did not
            build, such as one constructed by hand or loaded from a file
            written before jaxgsa 0.9.
        expanded_to_unit: Integer index map, shape ``(n_expanded,)``, from
            each expanded Saltelli row to its row in ``unit``. Equal to
            ``expanded_to_unique`` unless the categorical collapse pass merged
            rows. ``None`` under the same conditions as ``unit``.
    """

    samples: np.ndarray  # shape (n_unique, D), scaled to bounds
    sample_ids: np.ndarray
    n_expanded: int
    expanded_to_unique: np.ndarray
    base_n: int
    n_params: int
    calc_second_order: bool
    problem: Problem
    # Defaulted so a design built before these fields existed -- by hand, or
    # by an older version of save() -- still constructs. Everything that
    # sample() returns has them set.
    unit: np.ndarray | None = None
    expanded_to_unit: np.ndarray | None = None

    def transform(self, theta: Theta | None = None) -> Array:
        """Re-derive the physical design from the unit cube, differentiably.

        The Sobol' points in :attr:`unit` carry no distribution information.
        This method pushes them through the problem's marginals to produce the
        physical design, in pure JAX, so the result is a differentiable
        function of the distribution parameters. Composing it with a JAX model
        and :func:`jaxgsa.sobol.indices` gives a chain that ``jax.grad`` and
        ``jax.jacrev`` can differentiate end to end, which answers questions
        of the form "how does S1 move if I widen this input's range?".

        ``theta`` overrides distribution parameters by name. Anything it does
        not mention keeps the value the problem declares, so a caller
        differentiating one bound does not have to restate the others. The
        field names are those of :class:`jaxgsa.problem.UniformInputSpec` and
        :class:`jaxgsa.problem.GaussianInputSpec`::

            theta = {"a": {"low": 0.0, "high": 1.0},
                     "b": {"mean": 3.0, "variance": 9.0}}
            X = sampling_result.transform(theta)

        Tier T4 (behavioural contract): the returned array must reproduce
        :attr:`samples` when ``theta`` is ``None``, and its Jacobian with
        respect to ``theta`` must match central finite differences. Both are
        checked in ``tests/test_sobol_gradients.py``.

        Args:
            theta: Distribution parameters keyed by parameter name, then by
                field name. Values may be Python floats or JAX scalars.
                ``None`` (the default) means "use the problem's own values",
                which reproduces :attr:`samples`.

        Returns:
            The physical design as a JAX array, shape ``(n_runs, D)``.
            :attr:`samples` is always built in float64 by SciPy, so this
            reproduces it to float32 precision by default (measured max
            absolute difference 6e-6) and to 9e-16 with ``jax_enable_x64``
            on. Enable x64 when the difference matters, which it does for a
            finite-difference check.

        Raises:
            ValueError: If the problem has a categorical parameter, if
                :attr:`unit` is ``None``, or if ``theta`` names a parameter or
                field the problem does not have.
        """
        # A categorical inverse CDF is a step function: its derivative is zero
        # almost everywhere and undefined on the jumps, so a gradient through
        # it would be silently useless rather than wrong-looking. The row
        # counts of `unit` and `samples` also differ for such a problem, so
        # there is no design to return either.
        if self.problem.has_categorical_inputs:
            categorical = [
                name
                for name, spec in zip(self.problem.names, self.problem.input_specs)
                if isinstance(spec, CategoricalSpec)
            ]
            raise ValueError(
                "jaxgsa.sobol.SobolSamples.transform cannot transform a categorical "
                f"problem (parameters {categorical}). A categorical inverse CDF is a "
                "step function, so it has no useful derivative, and its many-to-one "
                "collapse also makes `unit` and `samples` different row counts. Use "
                "`samples` directly, and differentiate a design whose parameters are "
                "all continuous."
            )
        if self.unit is None:
            raise ValueError(
                "This SobolSamples carries no unit-cube design, so it cannot be "
                "re-transformed. Only a design returned by jaxgsa.sobol.sample has "
                "one; a hand-built design, or one loaded from a file written before "
                "jaxgsa 0.9, does not."
            )
        return _jax_transform_samples(self.problem, self.unit, theta)

    @overload
    def downsample(self, base_n: int) -> SobolSamples: ...

    @overload
    def downsample(self, base_n: int, Y: np.ndarray) -> tuple[SobolSamples, np.ndarray]: ...

    def downsample(
        self, base_n: int, Y: np.ndarray | None = None
    ) -> SobolSamples | tuple[SobolSamples, np.ndarray]:
        """Return a smaller ``SobolSamples`` by prefix-slicing to a lower ``base_n``.

        Sobol sequences are prefix-nested. The first ``K`` base points of a
        draw with ``N > K`` base points are bit-identical to drawing ``K`` base
        points directly, given the same seed and scramble. So you can simulate
        the model once at the largest ``base_n``, then slice to recover exact
        results for any smaller power-of-2 ``base_n``. No re-simulation is
        needed.

        Pass ``Y`` (model outputs aligned with ``samples``) to get the matching
        output slice back. This mirrors how
        ``sklearn.model_selection.train_test_split`` accepts both ``X`` and
        ``y``.

        Latin Hypercube Sampling (LHS) lacks this property, because its
        stratification depends on ``N``.

        Args:
            base_n: Target base size. Must be a power of 2 and
                ``<= self.base_n``.
            Y: Model outputs, shape ``(n_runs, ...)``. When given, the matching
                prefix is returned alongside the new result.

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

        # The unit design is prefix-sliced on its own map, not on the physical
        # one: the categorical collapse pass can leave the two with different
        # row counts, so `n_unique_new` does not index `unit`.
        unit_small = None
        exp2unit_small = None
        if self.unit is not None and self.expanded_to_unit is not None:
            exp2unit_small = self.expanded_to_unit[:new_expanded_n].copy()
            unit_small = self.unit[: int(exp2unit_small.max()) + 1].copy()

        sr_small = SobolSamples(
            samples=samples_small,
            sample_ids=np.arange(n_unique_new, dtype=np.int64),
            n_expanded=new_expanded_n,
            expanded_to_unique=new_exp2uniq,
            base_n=base_n,
            n_params=self.n_params,
            calc_second_order=self.calc_second_order,
            problem=self.problem,
            unit=unit_small,
            expanded_to_unit=exp2unit_small,
        )

        if Y_small is not None:
            return sr_small, Y_small
        return sr_small

    def to_morris(self, *, verbose: bool = True) -> MorrisSamples:
        """Reinterpret this Saltelli design as a radial Morris design.

        A Saltelli design is already a Morris radial (star) design. Within one
        base point, the row ``A`` and the ``D`` rows ``AB_j`` differ in exactly
        one parameter, which is what an elementary effect needs. Campolongo
        et al. (2011) build the radial design from a ``2D``-dimensional Sobol'
        sequence for this reason, and :func:`jaxgsa.sobol.sample` draws the
        same sequence the same way.

        The two methods then weight the same increments differently. Write
        ``EE_j = (f(AB_j) - f(A)) / delta_j`` with ``delta_j = B_j - A_j``.
        Jansen's total-order estimator is ``E[(delta_j * EE_j)^2] / (2 Var Y)``,
        while Morris reports ``mu_star = E|EE_j|``. So the screening measures
        come out of a design you have already paid for, and they cost no extra
        model evaluations. Pass the returned object and your existing ``Y``
        (the same array you would pass to :func:`jaxgsa.sobol.analyze`) to
        :func:`jaxgsa.morris.analyze`.

        The derived measures are the radial estimand, not the classical Morris
        one. The derived design is a radial design, so it estimates the radial
        quantity ``E|f(A with B_j) - f(A)| / |B_j - A_j|``, in which the step
        varies from block to block. The classical Morris quantity instead uses
        one fixed grid step ``Delta``. The two differ by much more than
        sampling noise. On Ishigami at ``r = 8192`` the derived ``mu_star`` is
        ``[8.68, 15.01, 6.62]`` against ``[8.69, 15.02, 6.64]`` from
        ``morris.sample(..., method="radial")``. The default
        ``method="trajectory"`` gives ``[7.59, 7.88, 6.39]`` instead: a factor
        1.9 on ``x2``, and 2.5 on its ``sigma``. ``morris.sample`` defaults to
        ``method="trajectory"``, so compare these measures against
        ``morris.sample(..., method="radial")``, never against the default.

        ``n_trajectories`` is ``base_n`` for both design variants: one radial
        block per base point, based at ``A``. Second-order designs also hold a
        block based at ``B``, that is ``B`` with its ``BA_j`` rows. This method
        deliberately does not harvest that block, because pooling it buys
        nothing measurable. In general the two blocks are not algebraically the
        same effect. That equality holds only for additive contributions, and
        the measured paired-effect correlations on Ishigami are
        0.50 / 1.00 / -0.06. So only ``x2`` is a genuine duplicate, and it
        comes from the purely additive ``7 sin^2(x2)`` term. Even for the rest,
        over 150 seeds at ``base_n = 128`` the pooled estimator's variance
        ratio against the A-only estimator is ``[1.07, 1.00, 1.59]``: no
        reduction, and worse on ``x3``. Pooling would also need a cluster
        bootstrap over base points to keep the confidence intervals honest,
        because the two blocks in a base point are dependent. That is real
        machinery for no gain.

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
                step (see below), or if ``problem`` has categorical
                parameters. An elementary effect divides by a step along one
                input axis, which has no meaning for unordered level codes.
                The Saltelli design itself stays valid, so
                :func:`jaxgsa.sobol.sample` and :func:`jaxgsa.sobol.analyze`
                still accept a categorical problem.

        Warns:
            JaxgsaWarning: If any parameter has an unbounded Gaussian marginal.
                ``mu_star`` then has no fixed scale, because how far the design
                reaches into the tail sets its magnitude. The Saltelli design
                and :func:`jaxgsa.morris.sample` reach different distances (the
                Saltelli design bounds support only at the library's own clip,
                +/-7.03 sigma). Only rankings are comparable across designs.
                Bound the marginals with
                ``Problem.from_dict(..., truncate_gaussians=q)`` if magnitudes
                must match. Once both sides are bounded the derived and native
                radial measures agree: measured ratios 0.999 (linear), 0.997
                (``x^2``), 0.988 (``x^4``), 0.987 (``exp(x^2/3)``), each within
                its own seed-to-seed spread.
            JaxgsaWarning: If any block is dropped for having a near-zero step.
                The radial design of :func:`jaxgsa.morris.sample` offsets the
                auxiliary points by four draws. Saltelli instead takes ``A``
                and ``B`` from one Sobol' row, so the two can coincide. This is
                a non-issue at the default ``scramble=True``: 0 of 65536 blocks
                were dropped across 8 seeds at ``D = 3``. With
                ``scramble=False`` the drop rate is real, but it falls off with
                ``base_n``: measured 21.9% at ``base_n=64``, 9.4% at 256, 2.3%
                at 1024 and 1.2% at 4096. The survivors are also a biased
                subsequence. ``mu_star`` comes out ``[8.34, 14.88, 5.55]`` at
                ``base_n=64`` against ``[8.68, 15.01, 6.62]`` scrambled, so
                ``x3`` reads 16% low. Keep ``scramble=True``.

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
        # duplicate. The reason to skip the B block is not duplication. It is
        # that pooling gives no measured variance reduction: the pooled /
        # A-only variance ratio over 150 seeds at base_n=128 is
        # [1.07, 1.00, 1.59]. Pooling would also require a cluster bootstrap
        # over base points to keep the CIs honest, since the two blocks share a
        # base point. Real machinery, no gain.
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
        """Persist the sample identifiers and the unit design."""
        arrays = {"sample_ids": self.sample_ids}
        # The unit design cannot be recovered from `samples` bit-for-bit (the
        # inverse CDFs round differently), and transform() needs it exactly,
        # so it is stored rather than recomputed on load.
        if self.unit is not None and self.expanded_to_unit is not None:
            arrays["unit"] = self.unit
            arrays["expanded_to_unit"] = self.expanded_to_unit
        return arrays

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
            # `.get`: files written before jaxgsa 0.9 carry no unit design.
            # Such a design still analyzes; only transform() is unavailable.
            unit=arrays.get("unit"),
            expanded_to_unit=arrays.get("expanded_to_unit"),
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
    """Return an upper bound on the number of distinct sample rows, if one exists.

    An all-categorical problem can only produce ``prod(L_d)`` distinct rows
    (each column takes one of its ``L_d`` level codes), no matter how large
    the design grows. Any continuous column makes the count unbounded, and
    then ``None`` is returned.
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


class _DedupedDesign(NamedTuple):
    """The four arrays that describe one deduplicated Saltelli design.

    Attributes:
        unit: Unique unit-cube rows, shape ``(n_unit, D)``.
        expanded_to_unit: Map from each expanded row to its ``unit`` row.
        samples: Unique rows in physical units, shape ``(n_runs, D)``.
        expanded_to_unique: Map from each expanded row to its ``samples`` row.
    """

    unit: np.ndarray
    expanded_to_unit: np.ndarray
    samples: np.ndarray
    expanded_to_unique: np.ndarray


def _dedupe_design(problem: Problem, expanded_unit: np.ndarray) -> _DedupedDesign:
    """Deduplicate an expanded Saltelli design, on the unit cube first.

    Deduplication happens before the marginal transform, not after, so the
    unit-cube points stay a design in their own right: they are theta-free,
    and :meth:`SobolSamples.transform` can re-derive the physical design from
    them for any distribution parameters. Doing it the other way round would
    tie the retained row set to one particular theta.

    The ordering is free for a continuous problem. Every continuous marginal
    transform here is strictly monotone, hence injective, so it maps distinct
    rows to distinct rows and cannot merge two rows that the unit-cube pass
    kept apart. Measured on uniform, Gaussian and truncated-Gaussian problems,
    scrambled and unscrambled, first- and second-order: both orderings give
    the identical ``samples`` array and the identical index map.

    A categorical marginal breaks injectivity, because its step CDF sends a
    whole probability bin to one level code. So a categorical problem gets a
    second collapse pass after the transform. That saving is large enough to
    be worth the extra pass: at ``base_n=64`` on a two-way categorical
    problem, 256 unit-unique rows collapse to 150 physical rows, which is 71%
    fewer model evaluations. Composing the two maps reproduces exactly what a
    single pass over the transformed design would have produced, because both
    passes keep first-occurrence order.

    Args:
        problem: Problem definition supplying the marginals.
        expanded_unit: Full expanded Saltelli design on the unit cube,
            shape ``(n_expanded, D)``.

    Returns:
        The deduplicated design. ``unit`` and ``samples`` have the same row
        count for a continuous problem, and different row counts when the
        collapse pass merged rows.
    """
    unit, expanded_to_unit = _stable_unique_rows(expanded_unit)
    samples = _transform_samples(problem, unit)
    if not problem.has_categorical_inputs:
        return _DedupedDesign(unit, expanded_to_unit, samples, expanded_to_unit)

    collapsed, unit_to_unique = _stable_unique_rows(samples)
    return _DedupedDesign(unit, expanded_to_unit, collapsed, unit_to_unique[expanded_to_unit])


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
        if isinstance(spec, GaussianSpec) and (spec.low is None or spec.high is None)
    ]
    if not unbounded:
        return
    warnings.warn(
        f"jaxgsa.sobol.to_morris: parameters {unbounded} have unbounded gaussian "
        "marginals. An elementary effect on an unbounded marginal has no fixed "
        "scale: how far the design reaches "
        "into the tail sets the magnitude of mu_star, and the Saltelli design and "
        "morris.sample reach different distances. Rankings are unaffected. Use "
        "Problem.from_dict(..., truncate_gaussians=q) if magnitudes must be comparable "
        "across designs",
        # Reached from SobolSamples.to_morris, so the user's frame is two up.
        stacklevel=3,
        category=JaxgsaWarning,
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
    _verbose.emit(
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
    _verbose.emit(
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

    Internally the function builds a Saltelli design for a candidate
    ``base_n``. A Saltelli design is the structured layout of base and
    column-swapped sample matrices that Sobol index estimators require. The
    function then removes exact duplicate rows and keeps first-occurrence
    order. In low dimensions the Saltelli construction repeats rows, and
    evaluating a model twice on the same input is wasted work. If the unique
    matrix is still smaller than the requested evaluation budget, ``base_n``
    doubles and the process repeats until enough unique rows are available.

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
            interaction) Sobol indices. Set it to ``False`` if you only need
            first/total-order indices. That nearly halves the evaluation
            budget: the Saltelli step becomes ``D + 2`` instead of
            ``2*D + 2``.
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
    # Rows per base point: sets the ratio between base_n and total expanded rows.
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

    expanded_unit = _build_expanded_samples(
        D,
        base_n,
        calc_second_order=calc_second_order,
        scramble=scramble,
        seed=seed,
    )
    n_expanded = expanded_unit.shape[0]
    design = _dedupe_design(problem, expanded_unit)

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
        # The returned design then keeps duplicate rows. They are legitimate
        # Saltelli samples: the dedup exists only to save model evaluations.
        max_unique = _max_distinct_rows(problem)
        max_candidate_rows = max(_CANDIDATE_ROW_CAP, 8 * target_n)
        prev_unique = -1
        while design.samples.shape[0] < target_n:
            n_unique = design.samples.shape[0]
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
                    category=JaxgsaWarning,
                )
                break
            prev_unique = n_unique
            base_n *= 2
            expanded_unit = _build_expanded_samples(
                D,
                base_n,
                calc_second_order=calc_second_order,
                scramble=scramble,
                seed=seed,
            )
            n_expanded = expanded_unit.shape[0]
            design = _dedupe_design(problem, expanded_unit)

    sample_ids = np.arange(design.samples.shape[0], dtype=np.int64)
    if verbose:
        _print_sampling_summary(
            n_params=D,
            target_n=target_n if target_n is not None else design.samples.shape[0],
            n_runs=design.samples.shape[0],
            n_expanded=n_expanded,
            base_n=base_n,
            calc_second_order=calc_second_order,
            scramble=scramble,
        )

    return SobolSamples(
        samples=design.samples,
        sample_ids=sample_ids,
        n_expanded=n_expanded,
        expanded_to_unique=design.expanded_to_unique,
        base_n=base_n,
        n_params=D,
        calc_second_order=calc_second_order,
        problem=problem,
        unit=design.unit,
        expanded_to_unit=design.expanded_to_unit,
    )
