"""Defines the HDMRResult dataclass for RS-HDMR sensitivity analysis results."""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import xarray as xr
from jax import Array

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

    Stores ANCOVA-decomposed sensitivity indices: structural (Sa), correlative
    (Sb), total per-term (S), and total-order per-parameter (ST). Each term
    corresponds to a first-, second-, or third-order component function in the
    HDMR expansion.

    Shapes follow ``(T, K, n_terms)`` for time-resolved multi-output analyses.
    Singleton T and/or K dimensions are squeezed when the original Y had fewer
    than 3 dimensions.
    """

    Sa: Array  # (n_terms,) or (K, n_terms) or (T, K, n_terms)
    Sb: Array  # correlative contribution, same shape as Sa
    S: Array  # total contribution per term (Sa + Sb)
    ST: Array  # (D,) or (K, D) or (T, K, D) total-order per parameter
    problem: Problem
    terms: tuple[str, ...]  # ("x1", "x2", "x1/x2", ...) term labels
    emulator: HDMREmulator | None = (
        None  # fitted coefficients, matching scalar/multi-output layout
    )
    select: Array | None = None  # (n_terms,) F-test selection counts
    rmse: Array | None = None  # (), (K,), or (T, K) emulator RMSE

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
        output_names = self.problem.output_names
        ndim = self.Sa.ndim

        if ndim == 1:
            dims_term = ("term",)
            dims_param = ("param",)
            coords: dict = {"term": term_labels, "param": param_names}
        elif ndim == 2:
            K = self.Sa.shape[0]
            if output_names is not None and len(output_names) != K:
                msg = f"output_names length {len(output_names)} != K={K}"
                raise ValueError(msg)
            onames = list(output_names) if output_names else [f"y{i}" for i in range(K)]
            dims_term = ("output", "term")
            dims_param = ("output", "param")
            coords = {"term": term_labels, "param": param_names, "output": onames}
        elif ndim == 3:
            T, K = self.Sa.shape[0], self.Sa.shape[1]
            if output_names is not None and len(output_names) != K:
                msg = f"output_names length {len(output_names)} != K={K}"
                raise ValueError(msg)
            onames = list(output_names) if output_names else [f"y{i}" for i in range(K)]
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

        data_vars: dict = {
            "Sa": (dims_term, np.asarray(self.Sa)),
            "Sb": (dims_term, np.asarray(self.Sb)),
            "S": (dims_term, np.asarray(self.S)),
            "ST": (dims_param, np.asarray(self.ST)),
        }

        if self.select is not None:
            data_vars["select"] = (("term",), np.asarray(self.select))

        if self.rmse is not None:
            rmse_np = np.asarray(self.rmse)
            if rmse_np.ndim == 0:
                data_vars["rmse"] = ((), rmse_np)
            elif rmse_np.ndim == 1:
                data_vars["rmse"] = (("output",), rmse_np)
            else:
                data_vars["rmse"] = (("time", "output"), rmse_np)

        return xr.Dataset(data_vars, coords=coords)
