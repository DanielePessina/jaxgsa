"""Joint and conditional input draws shared by Kucherenko and VKOGA.

Both methods need the same three operations on the input distribution: a
joint draw, a redraw of every input except ``x_i`` given ``x_i``, and a redraw
of ``x_i`` given the rest. :class:`Draws` names those operations. Two
implementations exist:

* :class:`GaussianDraws` for the built-in Gaussian copula (independent inputs
  are the identity-matrix case). Its rows live in the latent normal space.
* :class:`CustomDraws` for a user :class:`InputSampler` declared on the
  problem. Its rows are physical inputs.

Every operation consumes uniform points of width :attr:`Draws.dimension`, so a
caller generates one kind of noise for either implementation. Rows are kept
in the implementation's own space while conditioning, and converted once at
the end with :meth:`Draws.to_physical` (Kucherenko designs) or
:meth:`Draws.to_unit` (VKOGA's surrogate coordinates).
"""

from __future__ import annotations

from collections.abc import Sequence
from operator import index as _as_index
from typing import Protocol

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from jaxgsa._core.copula import (
    _ConditionalPlan,
    _SubsetPlan,
    assemble_latent,
    build_conditional_plan,
    build_subset_plan,
    draw_free_given_fixed,
    draw_rest_given_self,
    draw_self_given_rest,
    independent_correlation,
    latent_to_physical,
)
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa.problem import CategoricalSpec, InputSpec, Problem, UniformSpec

# Keeps the probit and the marginal inverse CDFs finite at exact 0 and 1.
_UNIT_CLIP = 1e-12


def probit(u: np.ndarray) -> np.ndarray:
    """Map unit-interval coordinates to standard normal scores ``Phi^-1(u)``.

    Args:
        u: Coordinates in ``[0, 1]``. Exact 0 and 1 are clipped, so the
            scores stay finite.

    Returns:
        Normal scores, same shape as ``u``.
    """
    return ndtri(np.clip(u, _UNIT_CLIP, 1.0 - _UNIT_CLIP))


class InputSampler(Protocol):
    """Transform uniform points into joint or conditional physical input rows.

    Attach one to a problem with ``Problem(..., sampler=f)`` or
    ``problem.with_sampler(f, sampler_dimension=k)``. jaxgsa supplies the uniform
    points and assembles every design. The sampler only maps points to rows.

    jaxgsa may request any sorted subset of fixed columns. The three most
    common calls are:

    * ``fixed_indices=()``: return joint draws. ``conditioning_rows`` is
      ``None``.
    * ``fixed_indices=(i,)``: keep column ``i`` of ``conditioning_rows`` and
      draw every other input from its conditional given ``x_i``.
    * ``fixed_indices`` = every column except ``i``: keep those columns and
      draw ``x_i`` from its conditional given the rest.

    Other subsets follow the same rule: keep the columns in
    ``fixed_indices`` and draw the remaining columns from their conditional
    given the kept ones.

    The return value must have shape ``(n, D)``, hold finite real values in
    ``problem.names`` order, and copy the fixed columns exactly. Every value
    must lie in the support of its declared marginal: inside ``[low, high]``
    for a uniform or truncated Gaussian input, and an integer level code in
    ``0 .. L-1`` for a categorical input. A continuous value may overshoot a
    bound by float32 rounding (a few units in the last place); jaxgsa clips
    it to the bound. Anything further out raises. An unbounded Gaussian
    input has no limit. The sampler must be deterministic: the same arguments give
    the same rows. It must be a conditional-quantile map from uniform points
    to rows, that is, an inverse Rosenblatt transform. Do not use rejection
    sampling or MCMC.
    Then the same uniform points under different conditioning give coupled
    draws, and jaxgsa relies on this. jaxgsa passes copies, so the sampler may
    modify its arguments.
    """

    def __call__(
        self,
        uniform_points: np.ndarray,
        /,
        *,
        conditioning_rows: np.ndarray | None = None,
        fixed_indices: tuple[int, ...] = (),
    ) -> npt.ArrayLike:
        """Return one physical input row per uniform point.

        Args:
            uniform_points: ``(n, dimension)`` points in ``[0, 1]``.
            conditioning_rows: ``(n, D)`` physical rows holding the fixed
                values, or ``None`` for a joint draw.
            fixed_indices: Columns of ``conditioning_rows`` to keep.

        Returns:
            ``(n, D)`` physical input rows.
        """
        ...


def sobol_uniforms(
    n: int,
    dimension: int,
    *,
    seed: int | np.random.Generator | np.random.SeedSequence | None,
    scramble: bool = True,
) -> np.ndarray:
    """Draw ``(n, dimension)`` points from a Sobol' sequence.

    Args:
        n: Number of points, a power of two. Callers round up before calling.
        dimension: Coordinates per point.
        seed: Seed for the scrambling.
        scramble: Scramble the sequence. SciPy applies a linear matrix
            scramble and a digital random shift, not Owen nested scrambling.
            Unscrambled, the origin is
            skipped, because it maps to the clipped extreme of every marginal.

    Returns:
        ``(n, dimension)`` uniform points.
    """
    engine = qmc.Sobol(d=dimension, scramble=scramble, seed=seed)
    if not scramble:
        engine.fast_forward(1)
    return engine.random(n)


def _sample_array(values: npt.ArrayLike, shape: tuple[int, int], name: str) -> np.ndarray:
    """Return a validated, independently owned real sample array."""
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{name} must return shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numeric values")
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


# A float32 sampler can overshoot a bound by one unit in the last place:
# float32(0.3) > 0.3. Allow a few float32 ulps, relative to the bound's size.
_SUPPORT_RTOL = 4 * float(np.finfo(np.float32).eps)


def _check_support(
    rows: np.ndarray, input_specs: Sequence[InputSpec], names: Sequence[str]
) -> None:
    """Raise if any entry of ``rows`` lies outside its marginal's support.

    A continuous value within float32 rounding of a bound is clipped to it in
    place, so a sampler written in single precision is not refused for an
    overshoot of one unit in the last place. Categorical codes are checked
    exactly. One min/max pass finds the columns that need a closer look, so
    valid output costs two reductions and no per-row work.
    """
    if rows.shape[0] == 0:
        return
    low = np.full(len(input_specs), -np.inf)
    high = np.full(len(input_specs), np.inf)
    levels: dict[int, int] = {}
    for i, spec in enumerate(input_specs):
        if isinstance(spec, CategoricalSpec):
            low[i], high[i] = 0.0, len(spec.probs) - 1
            levels[i] = len(spec.probs)
        else:
            if spec.low is not None:
                low[i] = spec.low
            if spec.high is not None:
                high[i] = spec.high
    continuous = [i for i in range(len(input_specs)) if i not in levels]
    if continuous:
        lo, hi = low[continuous], high[continuous]
        scale = np.maximum(
            1.0,
            np.fmax(
                np.abs(np.where(np.isfinite(lo), lo, 0.0)),
                np.abs(np.where(np.isfinite(hi), hi, 0.0)),
            ),
        )
        tol = _SUPPORT_RTOL * scale
        rows[:, continuous] = np.where(
            (rows[:, continuous] < lo) & (rows[:, continuous] >= lo - tol), lo, rows[:, continuous]
        )
        rows[:, continuous] = np.where(
            (rows[:, continuous] > hi) & (rows[:, continuous] <= hi + tol), hi, rows[:, continuous]
        )
    bad = np.flatnonzero((rows.min(axis=0) < low) | (rows.max(axis=0) > high))
    for i in levels:
        if i not in bad and np.any(rows[:, i] != np.rint(rows[:, i])):
            bad = np.sort(np.append(bad, i))
    if bad.size == 0:
        return
    i = int(bad[0])
    spec = input_specs[i]
    column = rows[:, i]
    if i in levels:
        wrong = (column < 0) | (column > levels[i] - 1) | (column != np.rint(column))
        support = f"categorical, integer codes 0 to {levels[i] - 1}"
    else:
        wrong = (column < low[i]) | (column > high[i])
        kind = "uniform" if isinstance(spec, UniformSpec) else "truncated Gaussian"
        support = f"{kind} on [{low[i]:g}, {high[i]:g}]"
    first = int(np.flatnonzero(wrong)[0])
    raise ValueError(
        f"sampler returned values outside the support of {names[i]!r} ({support}): "
        f"{int(wrong.sum())} rows, first at row {first} of this sampler call, "
        f"value {column[first]:g}"
    )


def transform_samples(
    sampler: InputSampler,
    uniform_points: np.ndarray,
    n_params: int,
    *,
    conditioning_rows: np.ndarray | None = None,
    fixed_indices: tuple[int, ...] = (),
    input_specs: Sequence[InputSpec] | None = None,
    names: Sequence[str] | None = None,
) -> np.ndarray:
    """Call a user sampler and check what it returns.

    Args:
        sampler: The user transformation.
        uniform_points: ``(n, dimension)`` uniform points.
        n_params: Number of physical input columns ``D``.
        conditioning_rows: ``(n, D)`` physical rows with the fixed values.
        fixed_indices: Distinct columns the sampler must keep exactly.
        input_specs: Marginal spec of each column. When given, every returned
            value must lie in its marginal's support. ``None`` skips that
            check.
        names: Parameter names for the error message. Defaults to
            ``x0, x1, ...``.

    Returns:
        Owned finite float64 rows of shape ``(n, D)``.

    Raises:
        ValueError: If the sampler returns the wrong shape, non-finite or
            non-real values, values outside a marginal's support, or changes
            a fixed column.
    """
    shape = (uniform_points.shape[0], n_params)
    original = None
    if conditioning_rows is not None:
        original = _sample_array(conditioning_rows, shape, "conditioning_rows")
    result = _sample_array(
        sampler(
            uniform_points.copy(),
            conditioning_rows=None if original is None else original.copy(),
            fixed_indices=fixed_indices,
        ),
        shape,
        "sampler",
    )
    if original is not None and not np.array_equal(
        result[:, fixed_indices], original[:, fixed_indices]
    ):
        raise ValueError("sampler must copy the fixed columns of conditioning_rows exactly")
    if input_specs is not None:
        _check_support(result, input_specs, names or [f"x{i}" for i in range(n_params)])
    return result


def check_fixed(fixed: tuple[int, ...], n_params: int) -> tuple[int, ...]:
    """Return ``fixed`` as a tuple of ints after checking it.

    Raises:
        ValueError: If ``fixed`` is not sorted, has repeats, or holds an index
            outside ``range(n_params)``.
    """
    try:
        checked = tuple(_as_index(j) for j in fixed)
    except TypeError:
        raise ValueError(f"fixed must hold integer column indices, got {fixed!r}") from None
    if any(j < 0 or j >= n_params for j in checked):
        raise ValueError(f"fixed indices must lie in [0, {n_params}), got {checked}")
    if any(b <= a for a, b in zip(checked, checked[1:])):
        raise ValueError(f"fixed must be sorted and unique, got {checked}")
    return checked


class Draws(Protocol):
    """Joint and conditional draws of a problem's inputs.

    Rows are ``(n, D)`` arrays in the implementation's own space. Each method
    takes uniform points of its own width: :attr:`dimension` for
    :meth:`joint`, :attr:`rest_dimension` for :meth:`rest_given_self` and
    :attr:`self_dimension` for :meth:`self_given_rest`. The widths differ
    because the Gaussian copula needs one noise column to redraw ``x_i`` but a
    custom sampler always reads ``dimension`` columns. A conditional method
    also takes ``(n, D)`` conditioning rows from the same space and keeps
    their fixed columns exactly.
    """

    @property
    def dimension(self) -> int:
        """Uniform columns for :meth:`joint`."""
        ...

    @property
    def rest_dimension(self) -> int:
        """Uniform columns for :meth:`rest_given_self`."""
        ...

    @property
    def self_dimension(self) -> int:
        """Uniform columns for :meth:`self_given_rest`."""
        ...

    def joint(self, u: np.ndarray) -> np.ndarray:
        """Return joint rows."""
        ...

    def joint_led_by(self, u: np.ndarray, index: int) -> np.ndarray:
        """Return joint rows whose column ``index`` depends on ``u[:, 0]`` alone.

        An outer loop that conditions on ``x_i`` then keeps the 1-D
        stratification of the Sobol' points in that column. An
        implementation that cannot do this returns :meth:`joint`.
        """
        ...

    def rest_given_self(self, u: np.ndarray, rows: np.ndarray, index: int) -> np.ndarray:
        """Keep column ``index`` and redraw the rest from their conditional."""
        ...

    def self_given_rest(self, u: np.ndarray, rows: np.ndarray, index: int) -> np.ndarray:
        """Keep every column except ``index`` and redraw it from its conditional."""
        ...

    def free_given_fixed(
        self, u: np.ndarray, rows: np.ndarray, fixed: tuple[int, ...]
    ) -> np.ndarray:
        """Keep the ``fixed`` columns and redraw the others from their conditional.

        ``fixed`` must be sorted, unique and in range. ``()`` is a joint draw.
        Every column fixed returns a copy of ``rows``. ``u`` has
        :meth:`free_dimension` columns.
        """
        ...

    def free_dimension(self, fixed: tuple[int, ...]) -> int:
        """Return the uniform columns that :meth:`free_given_fixed` consumes."""
        ...

    def to_physical(self, rows: np.ndarray) -> np.ndarray:
        """Map rows to physical input units."""
        ...

    def to_unit(self, rows: np.ndarray) -> np.ndarray:
        """Map rows to marginal CDF coordinates in ``[0, 1]``."""
        ...


class GaussianDraws:
    """Gaussian-copula draws in the latent standard-normal space.

    An independent problem uses the identity matrix, so the conditional
    redraws reduce to fresh independent draws.

    Attributes:
        problem: Problem supplying the marginals and the correlation.
        plan: Precomputed Gaussian conditionals.
        dimension: Uniform coordinates per joint row, one per parameter.
        rest_dimension: Noise columns to redraw ``Z_-i``, ``D - 1``.
        self_dimension: Noise columns to redraw ``Z_i``, one.
    """

    def __init__(self, problem: Problem) -> None:
        """Build the conditional plan for the problem's correlation.

        Args:
            problem: Problem with ``dependence`` ``"independent"`` or
                ``"gaussian"``.
        """
        R = problem.correlation
        if R is None:
            R = independent_correlation(problem.n_params)
        self.problem = problem
        self.plan: _ConditionalPlan = build_conditional_plan(R)
        self.dimension = problem.n_params
        self.rest_dimension = problem.n_params - 1
        self.self_dimension = 1
        self._correlation = R
        self._subset_plans: dict[tuple[int, ...], _SubsetPlan] = {}

    @staticmethod
    def _check_width(u: np.ndarray, width: int, method: str) -> None:
        if u.ndim != 2 or u.shape[1] != width:
            raise ValueError(
                f"GaussianDraws.{method} needs uniform points of shape (n, {width}), got {u.shape}"
            )

    def joint(self, u: np.ndarray) -> np.ndarray:
        """Return correlated latent rows."""
        self._check_width(u, self.dimension, "joint")
        return probit(u) @ self.plan.chol_full.T

    def joint_led_by(self, u: np.ndarray, index: int) -> np.ndarray:
        """Draw ``Z_i`` from ``u[:, 0]``, then ``Z_-i | Z_i`` from the rest of ``u``."""
        self._check_width(u, self.dimension, "joint_led_by")
        z_self = probit(u[:, 0])
        z_rest = draw_rest_given_self(self.plan, index, z_self, probit(u[:, 1:]))
        return assemble_latent(index, self.plan.others[index], z_self, z_rest)

    def rest_given_self(self, u: np.ndarray, rows: np.ndarray, index: int) -> np.ndarray:
        """Redraw ``Z_-i | Z_i`` from ``(n, D - 1)`` uniforms ``u``."""
        self._check_width(u, self.rest_dimension, "rest_given_self")
        others = self.plan.others[index]
        z_self = rows[:, index]
        z_rest = draw_rest_given_self(self.plan, index, z_self, probit(u))
        return assemble_latent(index, others, z_self, z_rest)

    def self_given_rest(self, u: np.ndarray, rows: np.ndarray, index: int) -> np.ndarray:
        """Redraw ``Z_i | Z_-i`` from ``(n, 1)`` uniforms ``u``."""
        self._check_width(u, self.self_dimension, "self_given_rest")
        others = self.plan.others[index]
        z_rest = rows[:, others]
        z_self = draw_self_given_rest(self.plan, index, z_rest, probit(u[:, 0]))
        return assemble_latent(index, others, z_self, z_rest)

    def free_dimension(self, fixed: tuple[int, ...]) -> int:
        """Return ``D - len(fixed)``, one noise column per free parameter."""
        return self.dimension - len(check_fixed(fixed, self.dimension))

    def free_given_fixed(
        self, u: np.ndarray, rows: np.ndarray, fixed: tuple[int, ...]
    ) -> np.ndarray:
        """Keep the ``fixed`` latent columns and redraw ``Z_F | Z_K``.

        Plans are cached per subset, so a design that visits many subsets
        builds each one once. Categorical inputs need no special case: the
        correlation holds the identity on their rows, as in
        :meth:`rest_given_self`.

        Args:
            u: ``(n, D - len(fixed))`` uniform points.
            rows: ``(n, D)`` latent rows holding the fixed values. Ignored
                when ``fixed`` is empty.
            fixed: Sorted, unique column indices to keep.

        Returns:
            ``(n, D)`` latent rows. Fixed columns are copied exactly.

        Raises:
            ValueError: If ``fixed`` is invalid or the shapes do not match.
        """
        fixed = check_fixed(fixed, self.dimension)
        self._check_width(u, self.dimension - len(fixed), "free_given_fixed")
        if not fixed:
            return self.joint(u)
        if rows.ndim != 2 or rows.shape != (u.shape[0], self.dimension):
            raise ValueError(
                f"GaussianDraws.free_given_fixed needs rows of shape "
                f"({u.shape[0]}, {self.dimension}), got {rows.shape}"
            )
        plan = self._subset_plans.get(fixed)
        if plan is None:
            plan = build_subset_plan(self._correlation, fixed)
            self._subset_plans[fixed] = plan
        result = np.array(rows, dtype=np.float64, copy=True)
        if plan.free.shape[0]:
            result[:, plan.free] = draw_free_given_fixed(plan, rows[:, plan.fixed], probit(u))
        return result

    def to_physical(self, rows: np.ndarray) -> np.ndarray:
        """Map latent rows through the marginal inverse CDFs."""
        return latent_to_physical(self.problem, rows)

    def to_unit(self, rows: np.ndarray) -> np.ndarray:
        """Map latent rows to copula coordinates."""
        return ndtr(rows)


class CustomDraws:
    """Draws from the problem's custom :class:`InputSampler`, in physical units.

    Attributes:
        problem: Problem declaring the sampler.
        sampler: The user transformation.
        dimension: Uniform coordinates per row, for every kind of draw.
        rest_dimension: Equal to ``dimension``.
        self_dimension: Equal to ``dimension``.
    """

    def __init__(self, problem: Problem) -> None:
        """Wrap the problem's sampler.

        Args:
            problem: Problem with ``dependence == "custom"``.
        """
        sampler, dimension = problem.sampler, problem.sampler_dimension
        if sampler is None or dimension is None:
            raise ValueError("CustomDraws needs a problem with a sampler")
        self.problem = problem
        self.sampler: InputSampler = sampler
        self.dimension: int = dimension

    @property
    def rest_dimension(self) -> int:
        """Uniform columns for :meth:`rest_given_self`; a sampler reads all of them."""
        return self.dimension

    @property
    def self_dimension(self) -> int:
        """Uniform columns for :meth:`self_given_rest`; a sampler reads all of them."""
        return self.dimension

    def joint(self, u: np.ndarray) -> np.ndarray:
        """Return joint physical rows."""
        return self.free_given_fixed(u, u, ())  # ``rows`` is ignored when nothing is fixed

    def joint_led_by(self, u: np.ndarray, index: int) -> np.ndarray:
        """Return :meth:`joint` rows; a custom sampler fixes its own column map."""
        del index
        return self.joint(u)

    def rest_given_self(self, u: np.ndarray, rows: np.ndarray, index: int) -> np.ndarray:
        """Keep ``x_i`` and redraw the rest through the sampler."""
        return self.free_given_fixed(u, rows, (index,))

    def self_given_rest(self, u: np.ndarray, rows: np.ndarray, index: int) -> np.ndarray:
        """Keep ``x_-i`` and redraw ``x_i`` through the sampler."""
        fixed = tuple(j for j in range(self.problem.n_params) if j != index)
        return self.free_given_fixed(u, rows, fixed)

    def free_dimension(self, fixed: tuple[int, ...]) -> int:
        """Return ``dimension``; a custom sampler always reads every column."""
        check_fixed(fixed, self.problem.n_params)
        return self.dimension

    def free_given_fixed(
        self, u: np.ndarray, rows: np.ndarray, fixed: tuple[int, ...]
    ) -> np.ndarray:
        """Pass ``fixed`` straight to the sampler.

        An empty ``fixed`` is a joint draw: the sampler gets no rows.
        """
        fixed = check_fixed(fixed, self.problem.n_params)
        return transform_samples(
            self.sampler,
            u,
            self.problem.n_params,
            conditioning_rows=rows if fixed else None,
            fixed_indices=fixed,
            input_specs=self.problem.input_specs,
            names=self.problem.names,
        )

    def to_physical(self, rows: np.ndarray) -> np.ndarray:
        """Return the rows unchanged; they are already physical."""
        return rows

    def to_unit(self, rows: np.ndarray) -> np.ndarray:
        """Map physical rows through the declared marginal CDFs.

        Every row comes from the sampler through :func:`transform_samples`,
        which has already checked it against the declared support.
        """
        unit = cdf_to_unit_interval(jnp.asarray(rows), self.problem)
        return np.asarray(unit, dtype=np.float64)


def draws_for(problem: Problem) -> GaussianDraws | CustomDraws:
    """Return the draw strategy matching ``problem.dependence``."""
    if problem.dependence == "custom":
        return CustomDraws(problem)
    return GaussianDraws(problem)
