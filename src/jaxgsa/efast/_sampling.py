"""eFAST (extended FAST) sampling via sinusoidal search curves.

Generates samples along ergodic search curves in the input space using
incommensurate frequencies, one curve per parameter. Each curve assigns
the highest frequency omega_0 to the parameter of interest and lower
complementary frequencies to the remaining parameters.

References:
    Saltelli, Tarantola & Chan (1999). Technometrics 41(1):39-56.
    Cukier et al. (1973). J. Chem. Phys. 59(8):3873-3878.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from jaxgsa._core.sampling import _transform_samples
from jaxgsa._core.validation import _raise_correlated_design
from jaxgsa.problem import Problem


@dataclass(frozen=True)
class EFASTSamples:
    """eFAST search-curve design plus the metadata needed to analyze it.

    Returned by :func:`jaxgsa.efast.sample`. Evaluate your model at every row
    of ``samples`` (in order) and pass this object together with the outputs
    to :func:`jaxgsa.efast.analyze`. Because the interference factor ``M``
    travels inside this object, it can never be mismatched between sampling
    and analysis.

    Terms:

    - ``n_per_curve`` — number of points along each sinusoidal search curve.
      There is one curve per parameter, so the design has ``D`` curves.
    - ``n_runs`` — total number of rows to evaluate, ``n_per_curve * D``
      (the package-wide meaning: unique rows you run the model on).

    Row layout: rows ``i*n_per_curve:(i+1)*n_per_curve`` of ``samples`` form
    the search curve for parameter ``i`` — the block along which parameter
    ``i`` oscillates at the primary frequency ``omega_0`` while all other
    parameters oscillate at lower complementary frequencies.

    Attributes:
        samples: Rows to evaluate with the user's model. Shape
            ``(n_per_curve * D, D)`` in the problem's physical units (each
            uniform search-curve marginal transformed into the problem's
            declared input distribution).
        n_per_curve: Number of samples along each search curve.
        M: Interference factor used to build the design — how many harmonics
            of ``omega_0`` are credited to the focal parameter during
            analysis.
        problem: Problem definition used to transform the samples.
    """

    samples: np.ndarray  # shape (n_per_curve * D, D), physical units
    n_per_curve: int
    M: int
    problem: Problem

    def __post_init__(self) -> None:
        """Validate design consistency.

        Raises:
            ValueError: If ``M < 1``, ``n_per_curve < 4*M^2*(D-1) + 1``, or
                ``samples`` does not have shape ``(n_per_curve * D, D)``.
        """
        if self.M < 1:
            raise ValueError(f"M must be >= 1, got {self.M}")
        D = self.problem.num_vars
        _check_n_per_curve(D, self.M, self.n_per_curve)
        expected = (self.n_per_curve * D, D)
        if self.samples.shape != expected:
            raise ValueError(
                f"samples has shape {self.samples.shape} but the design requires "
                f"(n_per_curve * D, D) = {expected}"
            )

    @property
    def n_runs(self) -> int:
        """Number of rows in ``samples`` (model runs to evaluate)."""
        return self.n_per_curve * self.problem.num_vars


def _min_n_per_curve(D: int, M: int) -> int:
    """Smallest ``n_per_curve`` admitting D distinct search-curve frequencies.

    The focal parameter runs at ``omega_0 = (n_per_curve - 1) // (2*M)`` and
    the ``D-1`` others must fit as *distinct* integers in
    ``[1, omega_0 // (2*M)]`` — that upper limit keeps their harmonics clear
    of the focal parameter's ``M`` harmonics. Requiring
    ``omega_0 // (2*M) >= D-1`` gives ``n_per_curve >= 4*M^2*(D-1) + 1``.

    Args:
        D: Number of parameters.
        M: Interference factor.

    Returns:
        Minimum admissible ``n_per_curve``. For ``D == 1`` there are no
        complementary frequencies, so the bound is the plain ``4*M^2 + 1``.
    """
    return 4 * M**2 * max(D - 1, 1) + 1


def _check_n_per_curve(D: int, M: int, n_per_curve: int) -> None:
    """Reject designs too short to give every parameter its own frequency.

    Args:
        D: Number of parameters.
        M: Interference factor.
        n_per_curve: Proposed number of samples per search curve.

    Raises:
        ValueError: If ``n_per_curve`` is below :func:`_min_n_per_curve`.
    """
    minimum = _min_n_per_curve(D, M)
    if n_per_curve >= minimum:
        return
    if D == 1:
        raise ValueError(f"n_per_curve must be > 4*M^2 = {4 * M**2}, got {n_per_curve}")
    raise ValueError(
        f"n_per_curve must be >= 4*M^2*(D-1) + 1 = {minimum} for D = {D} "
        f"parameters at M = {M}, got {n_per_curve}. Below that the D-1 "
        f"complementary frequencies cannot all be distinct, so non-focal "
        f"parameters would share a frequency and become indistinguishable "
        f"along the curve. Raise n_per_curve to at least {minimum} "
        f"(costing {minimum * D} model runs) or lower M."
    )


def _assign_frequencies(D: int, omega_0: int, M: int) -> np.ndarray:
    """Assign complementary frequencies for D-1 non-focal parameters.

    Args:
        D: Number of parameters.
        omega_0: Primary frequency for the focal parameter.
        M: Interference factor.

    Returns:
        (D-1,) array of distinct complementary frequencies.

    Raises:
        ValueError: If ``omega_0`` is too low to give the ``D-1`` non-focal
            parameters distinct frequencies. Callers must enforce
            :func:`_min_n_per_curve` before reaching this point.
    """
    if D == 1:
        return np.array([], dtype=np.int64)
    # Spread D-1 complementary frequencies evenly over the integers in
    # [1, omega_0/(2M)]. Fewer slots than parameters would force duplicates,
    # which is a silent-bias failure, not a degradation: two non-focal
    # parameters sharing a frequency also share the curve's phase, so their
    # columns are identical and the model cannot separate them.
    m = omega_0 // (2 * M)
    if m < D - 1:
        raise ValueError(
            f"omega_0 = {omega_0} leaves only {m} distinct complementary "
            f"frequencies for {D - 1} non-focal parameters"
        )
    return np.floor(np.linspace(1, m, D - 1)).astype(np.int64)


def sample(
    problem: Problem,
    n_per_curve: int,
    *,
    M: int = 4,
    seed: int | np.random.Generator | None = None,
) -> EFASTSamples:
    """Generate eFAST samples along sinusoidal search curves.

    For each of the D parameters, generates ``n_per_curve`` samples along a
    search curve where the focal parameter oscillates at the highest
    frequency omega_0 and the others at lower complementary frequencies.
    Evaluate the model on every row of the returned object's ``samples``
    (``n_per_curve * D`` runs total, keeping the rows in order) and pass
    the returned object together with the outputs to ``efast.analyze``.

    Args:
        problem: Problem definition with parameter distributions.
        n_per_curve: Number of samples per search curve. Must satisfy
            ``n_per_curve >= 4*M^2*(D-1) + 1`` so that all ``D`` parameters
            get distinct frequencies (for ``D == 1`` this is the usual
            ``n_per_curve > 4*M^2``). Larger values raise
            omega_0 = (n_per_curve-1)//(2M), separating the focal
            parameter's harmonics further from the complementary
            frequencies and improving index accuracy, at the cost of
            proportionally more model runs.
        M: Interference factor -- how many harmonics of omega_0 are
            credited to the focal parameter during analysis. Default 4
            (the standard choice; rarely needs changing).
        seed: Random seed for phase-shift reproducibility.

    Returns:
        EFASTSamples carrying the ``(n_per_curve * D, D)`` sample array in
        the problem's physical units plus the design metadata
        (``n_per_curve``, ``M``, ``problem``). Rows
        ``i*n_per_curve:(i+1)*n_per_curve`` form the search curve for
        parameter ``i``.

    Raises:
        ValueError: If ``M < 1``, ``n_per_curve < 4*M^2*(D-1) + 1``, or
            ``problem.correlation`` declares a dependence structure (the
            search-curve design assumes independent inputs).
    """
    _raise_correlated_design(problem, "jaxgsa.efast.sample")
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")

    D = problem.num_vars
    _check_n_per_curve(D, M, n_per_curve)

    rng = np.random.default_rng(seed)

    # Max integer frequency fitting n_per_curve samples while keeping M
    # harmonics below Nyquist.
    omega_0 = (n_per_curve - 1) // (2 * M)
    omega_compl = _assign_frequencies(D, omega_0, M)

    # Parametric variable s in [0, 2pi) — uniform grid along the search curve.
    s = (2 * math.pi / n_per_curve) * np.arange(n_per_curve)
    X = np.zeros((n_per_curve * D, D))

    for i in range(D):
        # Focal param i gets omega_0 (highest freq = most variation = identifiable);
        # remaining params get lower complementary frequencies.
        omega = np.zeros(D, dtype=np.int64)
        omega[i] = omega_0
        idx = [j for j in range(D) if j != i]
        omega[idx] = omega_compl

        # Random phase breaks symmetry so each curve samples a different cross-section.
        phi = 2 * math.pi * rng.random()

        row_slice = slice(i * n_per_curve, (i + 1) * n_per_curve)
        for j in range(D):
            # Cukier's transform: arcsin(sin(w*s+phi))/pi + 0.5 maps sinusoidal
            # oscillation to uniform [0,1] marginals (otherwise arcsine-shaped).
            X[row_slice, j] = 0.5 + (1.0 / math.pi) * np.arcsin(np.sin(omega[j] * s + phi))

    # CDF-based transform: map [0,1] samples to the problem's physical parameter space.
    X = _transform_samples(problem, X)

    return EFASTSamples(samples=X, n_per_curve=n_per_curve, M=M, problem=problem)
