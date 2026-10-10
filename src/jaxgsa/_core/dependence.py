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

from typing import Protocol

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from jaxgsa._core.copula import (
    _ConditionalPlan,
    assemble_latent,
    build_conditional_plan,
    draw_rest_given_self,
    draw_self_given_rest,
    independent_correlation,
    latent_to_physical,
)
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa.problem import Problem

# Keeps the probit and the marginal inverse CDFs finite at exact 0 and 1.
_UNIT_CLIP = 1e-12


class InputSampler(Protocol):
    """Transform uniform points into joint or conditional physical input rows.

    Attach one to a problem with ``Problem(..., sampler=f)`` or
    ``problem.with_sampler(f, dimension=k)``. jaxgsa supplies the uniform
    points and assembles every design. The sampler only maps points to rows.

    jaxgsa makes exactly three kinds of call:

    * ``fixed_indices=()``: return joint draws. ``conditioning_rows`` is
      ``None``.
    * ``fixed_indices=(i,)``: keep column ``i`` of ``conditioning_rows`` and
      draw every other input from its conditional given ``x_i``.
    * ``fixed_indices`` = every column except ``i``: keep those columns and
      draw ``x_i`` from its conditional given the rest.

    The return value must have shape ``(n, D)``, hold finite real values in
    ``problem.names`` order, and copy the fixed columns exactly. The sampler
    must be deterministic: the same arguments give the same rows. jaxgsa
    passes copies, so the sampler may modify its arguments.
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
        scramble: Owen-scramble the sequence. Unscrambled, the origin is
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


def transform_samples(
    sampler: InputSampler,
    uniform_points: np.ndarray,
    n_params: int,
    *,
    conditioning_rows: np.ndarray | None = None,
    fixed_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Call a user sampler and check what it returns.

    Args:
        sampler: The user transformation.
        uniform_points: ``(n, dimension)`` uniform points.
        n_params: Number of physical input columns ``D``.
        conditioning_rows: ``(n, D)`` physical rows with the fixed values.
        fixed_indices: Distinct columns the sampler must keep exactly.

    Returns:
        Owned finite float64 rows of shape ``(n, D)``.

    Raises:
        ValueError: If the sampler returns the wrong shape, non-finite or
            non-real values, or changes a fixed column.
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
    return result


class Draws(Protocol):
    """Joint and conditional draws of a problem's inputs.

    Rows are ``(n, D)`` arrays in the implementation's own space. Every method
    takes ``(n, dimension)`` uniform points. A conditional method also takes
    ``(n, D)`` conditioning rows from the same space and keeps their fixed
    columns exactly.
    """

    dimension: int

    def joint(self, u: np.ndarray) -> np.ndarray:
        """Return joint rows."""
        ...

    def rest_given_self(self, u: np.ndarray, rows: np.ndarray, index: int) -> np.ndarray:
        """Keep column ``index`` and redraw the rest from their conditional."""
        ...

    def self_given_rest(self, u: np.ndarray, rows: np.ndarray, index: int) -> np.ndarray:
        """Keep every column except ``index`` and redraw it from its conditional."""
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
        dimension: Uniform coordinates per row, one per parameter.
    """

    def __init__(self, problem: Problem) -> None:
        """Build the conditional plan for the problem's correlation.

        Args:
            problem: Problem with ``dependence`` ``"independent"`` or
                ``"gaussian"``.
        """
        R = problem.correlation
        self.problem = problem
        self.plan: _ConditionalPlan = build_conditional_plan(
            independent_correlation(problem.num_vars) if R is None else R
        )
        self.dimension = problem.num_vars

    @staticmethod
    def _normal(u: np.ndarray) -> np.ndarray:
        return ndtri(np.clip(u, _UNIT_CLIP, 1.0 - _UNIT_CLIP))

    def joint(self, u: np.ndarray) -> np.ndarray:
        """Return correlated latent rows."""
        return self._normal(u) @ self.plan.chol_full.T

    def rest_given_self(self, u: np.ndarray, rows: np.ndarray, index: int) -> np.ndarray:
        """Redraw ``Z_-i | Z_i`` from the columns of ``u`` other than ``index``."""
        others = self.plan.others[index]
        z_self = rows[:, index]
        z_rest = draw_rest_given_self(self.plan, index, z_self, self._normal(u[:, others]))
        return assemble_latent(index, others, z_self, z_rest)

    def self_given_rest(self, u: np.ndarray, rows: np.ndarray, index: int) -> np.ndarray:
        """Redraw ``Z_i | Z_-i`` from column ``index`` of ``u``."""
        others = self.plan.others[index]
        z_rest = rows[:, others]
        z_self = draw_self_given_rest(self.plan, index, z_rest, self._normal(u[:, index]))
        return assemble_latent(index, others, z_self, z_rest)

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
        dimension: Uniform coordinates per row.
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

    def joint(self, u: np.ndarray) -> np.ndarray:
        """Return joint physical rows."""
        return transform_samples(self.sampler, u, self.problem.num_vars)

    def rest_given_self(self, u: np.ndarray, rows: np.ndarray, index: int) -> np.ndarray:
        """Keep ``x_i`` and redraw the rest through the sampler."""
        return transform_samples(
            self.sampler, u, self.problem.num_vars, conditioning_rows=rows, fixed_indices=(index,)
        )

    def self_given_rest(self, u: np.ndarray, rows: np.ndarray, index: int) -> np.ndarray:
        """Keep ``x_-i`` and redraw ``x_i`` through the sampler."""
        fixed = tuple(j for j in range(self.problem.num_vars) if j != index)
        return transform_samples(
            self.sampler, u, self.problem.num_vars, conditioning_rows=rows, fixed_indices=fixed
        )

    def to_physical(self, rows: np.ndarray) -> np.ndarray:
        """Return the rows unchanged; they are already physical."""
        return rows

    def to_unit(self, rows: np.ndarray) -> np.ndarray:
        """Map physical rows through the declared marginal CDFs.

        Raises:
            ValueError: If a row falls outside a declared bounded support,
                which means the sampler disagrees with the problem's
                marginals.
        """
        for i, spec in enumerate(self.problem.input_specs):
            low, high = getattr(spec, "low", None), getattr(spec, "high", None)
            column = rows[:, i]
            if (low is not None and np.any(column < low)) or (
                high is not None and np.any(column > high)
            ):
                raise ValueError(
                    f"The custom sampler returned values outside the declared "
                    f"support of {self.problem.names[i]!r}. Declare marginals "
                    "that match the sampler."
                )
        unit = cdf_to_unit_interval(jnp.asarray(rows), self.problem)
        return np.asarray(unit, dtype=np.float64)


def draws_for(problem: Problem) -> GaussianDraws | CustomDraws:
    """Return the draw strategy matching ``problem.dependence``."""
    if problem.dependence == "custom":
        return CustomDraws(problem)
    return GaussianDraws(problem)
