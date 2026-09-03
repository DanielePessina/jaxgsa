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

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jaxgsa._core import verbose as _verbose
from jaxgsa._core.samples import (
    _jaxgsa_version,
    _npz_path,
    _problem_from_meta,
    _problem_to_meta,
    _save_target,
)
from jaxgsa._core.sampling import _transform_samples
from jaxgsa._core.validation import _raise_categorical_design, _raise_correlated_design
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

    - ``n_per_curve``: number of points along each sinusoidal search curve.
      There is one curve per parameter, so the design has ``D`` curves.
    - ``n_runs``: total number of rows to evaluate, ``n_per_curve * D``. This
      is the package-wide meaning of unique rows you run the model on.

    Rows ``i*n_per_curve:(i+1)*n_per_curve`` of ``samples`` form the search
    curve for parameter ``i``. Along that block, parameter ``i`` oscillates at
    the primary frequency ``omega_0`` while every other parameter oscillates
    at a lower complementary frequency.

    Attributes:
        samples: Rows to evaluate with the user's model, shape
            ``(n_per_curve * D, D)``, in the problem's physical units. Each
            uniform search-curve marginal is transformed into the problem's
            declared input distribution.
        n_per_curve: Number of samples along each search curve.
        M: Interference factor used to build the design. It sets how many
            harmonics of ``omega_0`` are credited to the focal parameter
            during analysis.
        problem: Problem definition used to transform the samples.
    """

    samples: np.ndarray  # (n_per_curve * D, D), physical units
    n_per_curve: int
    M: int
    problem: Problem

    def __post_init__(self) -> None:
        """Check that the design metadata and the sample array agree.

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
        """Number of rows in ``samples``, which is the number of model runs."""
        return self.n_per_curve * self.problem.num_vars

    def save(self, path: str | Path) -> None:
        """Save the full design to one compressed NPZ file.

        The on-disk layout follows the same conventions as the other design
        objects (``jaxgsa.sobol.SobolSamples.save`` and friends): one
        compressed file holding the sample matrix plus a JSON ``metadata``
        blob that records the problem definition, the design parameters
        (``n_per_curve``, ``M``), and the jaxgsa version that wrote the file.
        eFAST has no deduplication, so there is no expansion map to store.

        Args:
            path: Destination file path. If it does not already end in
                ``.npz``, the suffix is appended, matching NumPy's ``savez``
                convention. The file written to disk may therefore differ
                from the exact string passed: ``"run.A"`` is saved as
                ``"run.A.npz"``.

        Raises:
            FileNotFoundError: If the parent directory of ``path`` does not
                exist. The directory is never created for you.
        """
        meta = {
            "jaxgsa_version": _jaxgsa_version(),
            "problem": _problem_to_meta(self.problem),
            "n_per_curve": self.n_per_curve,
            "M": self.M,
        }
        np.savez_compressed(
            _save_target(path),
            allow_pickle=False,
            samples=self.samples,
            metadata=np.asarray(json.dumps(meta)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EFASTSamples":
        """Load a design saved by :meth:`save`.

        Args:
            path: Path to the saved design. If it does not already end in
                ``.npz``, the suffix is appended before opening, mirroring
                the convention used by :meth:`save`. The file actually read
                may therefore differ from the exact string passed.

        Returns:
            The reconstructed design, equal to the instance that was saved.
            Analyzing it gives the same result as analyzing the original.

        Raises:
            FileNotFoundError: If no file exists at the resolved path.
            KeyError: If the file is missing required arrays or metadata
                entries (i.e. it was not written by :meth:`save`).
        """
        with np.load(_npz_path(path), allow_pickle=False) as data:
            meta = json.loads(data["metadata"].item())
            return cls(
                samples=data["samples"],
                n_per_curve=int(meta["n_per_curve"]),
                M=int(meta["M"]),
                problem=_problem_from_meta(meta["problem"]),
            )


@dataclass(frozen=True)
class _FrequencyPlan:
    """The frequency layout of one eFAST design, computed once.

    ``sample()`` and ``analyze()`` both need to know how the search curve
    spends the frequency axis. This object is the single place that decides
    it, so the two sides cannot drift apart.

    It holds **two different bands**, and they are not two versions of one
    quantity:

    - The *assigned* band ``[1, assigned_max]`` with
      ``assigned_max = omega_0 // (2*M)``. ``sample()`` draws the ``D-1``
      complementary carrier frequencies from it. The upper limit keeps those
      carriers clear of the focal parameter's ``M`` harmonics.
    - The *analysis* band ``[1, analysis_max]`` with
      ``analysis_max = omega_0 // 2``. ``analyze()`` sums the power over all
      of it to get the complementary variance, following Saltelli, Tarantola
      & Chan (1999). Everything below ``omega_0 / 2`` comes from the non-focal
      parameters: the assigned carriers, and also the interference between
      them, which lands on frequencies nobody was assigned.

    The analysis band is therefore deliberately wider than the assigned band.
    The relation ``1 <= assigned_max <= analysis_max`` is the invariant the
    two sides rely on, and ``__post_init__`` checks it.

    Attributes:
        omega_0: Primary frequency given to the focal parameter.
        omega_compl: Complementary frequencies for the ``D-1`` non-focal
            parameters, shape ``(D-1,)``, distinct, all inside the assigned
            band.
        assigned_max: Highest frequency ``sample()`` may assign.
        analysis_max: Highest frequency ``analyze()`` counts as complementary.
    """

    omega_0: int
    omega_compl: np.ndarray
    assigned_max: int
    analysis_max: int

    def __post_init__(self) -> None:
        """Check that the assigned band lies inside the analysis band.

        Raises:
            ValueError: If the invariant
                ``1 <= assigned_max <= analysis_max`` fails, or if an assigned
                frequency falls outside the assigned band. Either would mean
                the sampler put carrier power where the analyzer does not look
                for it, or below a floor it treats as noise.
        """
        if not 1 <= self.assigned_max <= self.analysis_max:
            raise ValueError(
                f"eFAST frequency plan is inconsistent: assigned band "
                f"[1, {self.assigned_max}] does not lie inside the analysis "
                f"band [1, {self.analysis_max}] for omega_0 = {self.omega_0}"
            )
        if self.omega_compl.size and (
            int(self.omega_compl.min()) < 1 or int(self.omega_compl.max()) > self.assigned_max
        ):
            raise ValueError(
                f"eFAST complementary frequencies {self.omega_compl.tolist()} fall "
                f"outside the assigned band [1, {self.assigned_max}]"
            )


def _min_n_per_curve(D: int, M: int) -> int:
    """Smallest ``n_per_curve`` admitting D distinct search-curve frequencies.

    This is the frequency plan read backwards. The focal parameter runs at
    ``omega_0 = (n_per_curve - 1) // (2*M)``, and the ``D-1`` other parameters
    must fit as distinct integers in the assigned band
    ``[1, omega_0 // (2*M)]``. Requiring ``omega_0 // (2*M) >= D-1`` gives
    ``n_per_curve >= 4*M^2*(D-1) + 1``.

    Args:
        D: Number of parameters.
        M: Interference factor.

    Returns:
        Minimum admissible ``n_per_curve``. For ``D == 1`` there are no
        complementary frequencies, so the bound is the plain ``4*M^2 + 1``,
        which is also what the assigned band needs to hold one slot.
    """
    return 4 * M**2 * max(D - 1, 1) + 1


def _frequency_plan(D: int, n_per_curve: int, M: int) -> _FrequencyPlan:
    """Compute the frequency layout of an eFAST design, once, for both sides.

    ``sample()`` calls this to build the design and ``analyze()`` calls it to
    read the design back. Every frequency-axis number either module needs comes
    from here.

    Args:
        D: Number of parameters.
        n_per_curve: Number of samples along each search curve.
        M: Interference factor.

    Returns:
        The :class:`_FrequencyPlan` for this design.

    Raises:
        ValueError: If ``n_per_curve`` is below :func:`_min_n_per_curve`, or if
            the resulting bands violate the plan's invariant.
    """
    _check_n_per_curve(D, M, n_per_curve)
    # Largest integer frequency that fits n_per_curve samples while keeping all
    # M harmonics of omega_0 below Nyquist.
    omega_0 = (n_per_curve - 1) // (2 * M)
    assigned_max = omega_0 // (2 * M)
    analysis_max = omega_0 // 2
    return _FrequencyPlan(
        omega_0=omega_0,
        omega_compl=_assign_frequencies(D, assigned_max),
        assigned_max=assigned_max,
        analysis_max=analysis_max,
    )


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


def _assign_frequencies(D: int, assigned_max: int) -> np.ndarray:
    """Assign complementary frequencies for D-1 non-focal parameters.

    Args:
        D: Number of parameters.
        assigned_max: Top of the assigned band, from the frequency plan.

    Returns:
        Distinct complementary frequencies, shape ``(D-1,)``, spread evenly
        over ``[1, assigned_max]``. Empty for ``D == 1``.
    """
    if D == 1:
        return np.array([], dtype=np.int64)
    # Spread D-1 complementary frequencies evenly over the integers in
    # [1, assigned_max]. _check_n_per_curve guarantees at least D-1 slots.
    # Fewer slots than parameters would force duplicates, which is a
    # silent-bias failure rather than a degradation: two non-focal parameters
    # sharing a frequency also share the curve's phase, so their columns are
    # identical and the model cannot separate them.
    return np.floor(np.linspace(1, assigned_max, D - 1)).astype(np.int64)


def sample(
    problem: Problem,
    n_per_curve: int,
    *,
    M: int = 4,
    seed: int | np.random.Generator | None = None,
    verbose: bool = True,
) -> EFASTSamples:
    """Generate eFAST samples along sinusoidal search curves.

    For each of the D parameters, this builds ``n_per_curve`` samples along one
    search curve. On that curve the focal parameter oscillates at the highest
    frequency omega_0, and the others oscillate at lower complementary
    frequencies. Evaluate the model on every row of the returned object's
    ``samples``, keeping the rows in order, for ``n_per_curve * D`` runs in
    total. Then pass the returned object and the outputs to ``efast.analyze``.

    Args:
        problem: Problem definition with parameter distributions.
        n_per_curve: Number of samples per search curve. It must satisfy
            ``n_per_curve >= 4*M^2*(D-1) + 1``, so that all ``D`` parameters
            get distinct frequencies. For ``D == 1`` this is the usual
            ``n_per_curve > 4*M^2``. A larger value raises
            omega_0 = (n_per_curve-1)//(2M). That separates the focal
            parameter's harmonics further from the complementary frequencies
            and improves index accuracy, at the cost of proportionally more
            model runs.
        M: Interference factor, the number of harmonics of omega_0 credited to
            the focal parameter during analysis. Default 4, the standard
            choice, which rarely needs changing.
        seed: Random seed, for reproducible phase shifts.
        verbose: If ``True`` (default), print a one-line summary of the
            design: curves, rows per curve, total runs, and the focal
            frequency. Pass ``False`` for a silent run.

    Returns:
        An ``EFASTSamples`` carrying the ``(n_per_curve * D, D)`` sample array
        in the problem's physical units, plus the design metadata
        ``n_per_curve``, ``M``, and ``problem``. Rows
        ``i*n_per_curve:(i+1)*n_per_curve`` form the search curve for
        parameter ``i``.

    Raises:
        ValueError: In any of these cases. ``M < 1``.
            ``n_per_curve < 4*M^2*(D-1) + 1``. ``problem.correlation`` declares
            a dependence structure, and the search-curve design assumes
            independent inputs. ``problem`` has categorical parameters, and the
            search curve sweeps each input continuously, which has no meaning
            for unordered level codes.
    """
    _raise_correlated_design(problem, "jaxgsa.efast.sample")
    _raise_categorical_design(problem, "jaxgsa.efast.sample")
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")

    D = problem.num_vars

    # One plan, shared with analyze(). It validates n_per_curve on the way.
    plan = _frequency_plan(D, n_per_curve, M)
    omega_0 = plan.omega_0
    omega_compl = plan.omega_compl

    rng = np.random.default_rng(seed)

    # Parametric variable s in [0, 2pi): a uniform grid along the search curve.
    s = (2 * math.pi / n_per_curve) * np.arange(n_per_curve)
    X = np.zeros((n_per_curve * D, D))

    for i in range(D):
        # Focal parameter i gets omega_0. The highest frequency carries the
        # most variation, which makes that parameter identifiable. The
        # remaining parameters get lower complementary frequencies.
        omega = np.zeros(D, dtype=np.int64)
        omega[i] = omega_0
        idx = [j for j in range(D) if j != i]
        omega[idx] = omega_compl

        # A random phase breaks the symmetry, so each curve samples a
        # different cross-section.
        phi = 2 * math.pi * rng.random()

        row_slice = slice(i * n_per_curve, (i + 1) * n_per_curve)
        for j in range(D):
            # Cukier's transform: arcsin(sin(w*s+phi))/pi + 0.5 maps the
            # sinusoidal oscillation to uniform [0, 1] marginals. Without it
            # the marginals are arcsine-shaped.
            X[row_slice, j] = 0.5 + (1.0 / math.pi) * np.arcsin(np.sin(omega[j] * s + phi))

    # CDF-based transform: map the [0, 1] samples into the problem's physical
    # parameter space.
    X = _transform_samples(problem, X)

    if verbose:
        # A search curve is an ordered sweep, so there is nothing to
        # deduplicate: every row is kept, and n_runs is exact.
        _verbose.sample_summary(
            method="jaxgsa.efast.sample",
            problem=problem,
            entries=[
                ("n_curves", D),
                ("n_per_curve", n_per_curve),
                ("n_runs", n_per_curve * D),
                ("M", M),
                ("omega_0", omega_0),
            ],
        )

    return EFASTSamples(samples=X, n_per_curve=n_per_curve, M=M, problem=problem)
