"""Defines the HDMRResult dataclass for RS-HDMR sensitivity analysis results."""

from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import xarray as xr
from jax import Array

from gsax._normalization import _default_output_names
from gsax.problem import Problem


class HDMREmulator(TypedDict):
    """Typed emulator payload returned inside ``HDMRResult``.

    The coefficient arrays are stored on the fitted analysis scale. When
    ``prenormalize`` is ``True``, ``y_mean`` and ``y_std`` are used by
    ``emulate_hdmr()`` to map predictions back to the original output scale.
    """

    C1: Array
    C2: Array | None
    C3: Array | None
    f0: Array
    prenormalize: bool
    y_mean: Array
    y_std: Array
    m: int
    maxorder: int
    c2: list[tuple[int, int]]
    c3: list[tuple[int, int, int]]


@dataclass
class HDMRResult:
    """RS-HDMR (Random Sampling High-Dimensional Model Representation) results.

    Stores ANCOVA-decomposed sensitivity indices. Each *term* is one component
    function of the HDMR expansion -- a single parameter, a pair, or a triple
    (up to the ``maxorder`` used) -- named in ``terms``. Per-term indices
    (Sa, Sb, S) have a trailing ``n_terms`` axis; the per-parameter ST has a
    trailing ``D`` axis.

    Shapes follow ``(T, K, n_terms)`` for time-resolved multi-output analyses.
    Singleton T and/or K dimensions are squeezed when the original Y had fewer
    than 3 dimensions.

    Attributes:
        Sa: Structural (uncorrelated) variance fraction per term, shape
            ``(n_terms,)`` / ``(K, n_terms)`` / ``(T, K, n_terms)``. The part
            of a term's contribution independent of other inputs.
        Sb: Correlative contribution per term, same shape as ``Sa``. Near
            zero when inputs are independent; non-zero values flag variance
            shared through input correlation (and can be negative).
        S: Total contribution per term, ``S = Sa + Sb``, same shape.
        ST: Total-order index per parameter -- its first-order term plus
            every interaction term containing it -- shape ``(D,)`` /
            ``(K, D)`` / ``(T, K, D)``.
        problem: Problem definition used for the analysis.
        terms: Human-readable term labels, e.g. ``("x1", "x2", "x1/x2")``;
            interaction terms join parameter names with ``/``.
        emulator: Fitted surrogate state for ``hdmr.emulate``, or None.
        select: F-test significance count per term, summed over the T*K
            output slices (max value T*K), or None. Low counts mark terms
            the F-test deems insignificant.
        rmse: Emulator fit RMSE per output slice in the units of ``Y``,
            shape ``()`` / ``(K,)`` / ``(T, K)``, or None.
    """

    Sa: Array
    Sb: Array
    S: Array
    ST: Array
    problem: Problem
    terms: tuple[str, ...]
    emulator: HDMREmulator | None = None
    select: Array | None = None
    rmse: Array | None = None
    # True when layout inference inserted the singleton output axis (a 2-D
    # (N, T) Y under a single named output). emulate_hdmr squeezes it back so
    # predictions mirror the training Y's rank.
    _inserted_output_axis: bool = field(default=False, repr=False)

    @property
    def S1(self) -> Array:
        """First-order Sobol indices (structural contribution of first-order terms).

        Equivalent to ``Sa[:D]`` — the uncorrelated variance fraction of each
        single-parameter component function, which matches the definition of
        first-order Sobol indices.

        Returns:
            Array of shape ``(D,)`` / ``(K, D)`` / ``(T, K, D)``.
        """
        D = self.problem.num_vars
        return self.Sa[..., :D]

    def __repr__(self) -> str:
        """Return a concise summary showing index shapes."""
        shapes = {
            "Sa": self.Sa.shape,
            "Sb": self.Sb.shape,
            "S": self.S.shape,
            "ST": self.ST.shape,
        }
        return f"HDMRResult({shapes})"

    def to_dataset(
        self,
        time_coords: np.ndarray | list | None = None,
    ) -> xr.Dataset:
        """Convert results to a labeled xarray Dataset.

        Args:
            time_coords: Coordinate values for the time dimension when
                arrays are 3-D. Defaults to integer indices.

        Returns:
            An ``xr.Dataset`` with variables ``Sa``, ``Sb``, ``S``, ``ST``,
            and optionally ``select`` and ``rmse``.
        """
        param_names = list(self.problem.names)
        term_labels = list(self.terms)
        ndim = self.Sa.ndim

        if ndim == 1:
            dims_term = ("term",)
            dims_param = ("param",)
            coords: dict = {"term": term_labels, "param": param_names}
        elif ndim == 2:
            onames = _default_output_names(self.Sa.shape[0], self.problem)
            dims_term = ("output", "term")
            dims_param = ("output", "param")
            coords = {"term": term_labels, "param": param_names, "output": onames}
        elif ndim == 3:
            T = self.Sa.shape[0]
            onames = _default_output_names(self.Sa.shape[1], self.problem)
            tcoords = list(time_coords) if time_coords is not None else list(range(T))
            dims_term = ("time", "output", "term")
            dims_param = ("time", "output", "param")
            coords = {
                "term": term_labels,
                "param": param_names,
                "output": onames,
                "time": tcoords,
            }
        else:
            msg = f"Unexpected Sa.ndim={ndim}"
            raise ValueError(msg)

        # Sa/Sb/S are indexed by term (including interactions); ST is indexed
        # by parameter, so it needs a different dimension coordinate.
        data_vars: dict = {
            "Sa": (dims_term, np.asarray(self.Sa)),
            "Sb": (dims_term, np.asarray(self.Sb)),
            "S": (dims_term, np.asarray(self.S)),
            "ST": (dims_param, np.asarray(self.ST)),
        }

        if self.select is not None:
            data_vars["select"] = (("term",), np.asarray(self.select))

        if self.rmse is not None:
            # RMSE dimensions mirror the squeezed output layout:
            # scalar -> (), vector -> (output,), matrix -> (time, output).
            rmse_np = np.asarray(self.rmse)
            if rmse_np.ndim == 0:
                data_vars["rmse"] = ((), rmse_np)
            elif rmse_np.ndim == 1:
                data_vars["rmse"] = (("output",), rmse_np)
            else:
                data_vars["rmse"] = (("time", "output"), rmse_np)

        return xr.Dataset(data_vars, coords=coords)
