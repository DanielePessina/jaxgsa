"""Sobol/Saltelli sampling with unique user-facing rows.

The public contract of this module is intentionally split in two layers:

1. ``sample()`` returns only the unique rows that a user should evaluate.
2. ``SamplingResult`` also carries enough metadata to reconstruct the full
   expanded Saltelli layout later inside :func:`gsax.analyze`.

This avoids wasted model evaluations in low-dimensional cases where the
expanded Saltelli design contains exact duplicate rows.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm
from scipy.stats.qmc import Sobol

from gsax.problem import Problem, _normalized_input_to_dict

# Supported serialization formats for SamplingResult.save/load
_SAMPLE_FORMATS = {"csv", "txt", "xlsx", "parquet", "pkl"}


@dataclass(frozen=True)
class SamplingResult:
    """Unique Sobol samples plus metadata for Saltelli reconstruction.

    Attributes:
        samples: Unique rows to evaluate with the user's model. Shape
            ``(n_total, D)`` after transforming each Sobol marginal into the
            problem's declared input distribution.
        sample_ids: Stable integer identifiers aligned 1:1 with ``samples``.
            Useful for joining model outputs back onto the sampling table.
        expanded_n_total: Row count of the full expanded Saltelli layout before
            deduplication. This is the number of rows analyzed internally.
        expanded_to_unique: Integer index map of shape ``(expanded_n_total,)``.
            For each expanded Saltelli row, gives the corresponding row index in
            ``samples``.
        base_n: Number of base Sobol points used to construct the Saltelli
            design. Always a power of 2.
        n_params: Number of problem dimensions ``D``.
        calc_second_order: Whether the expanded design includes BA blocks for
            second-order Sobol indices.
        problem: Problem definition used to transform the samples.
    """

    samples: np.ndarray  # shape (n_unique, D), scaled to bounds
    sample_ids: np.ndarray
    expanded_n_total: int
    expanded_to_unique: np.ndarray
    base_n: int
    n_params: int
    calc_second_order: bool
    problem: Problem

    @property
    def n_total(self) -> int:
        """Number of unique rows in ``samples``."""
        return self.samples.shape[0]

    @property
    def samples_df(self) -> pd.DataFrame:
        """Return the unique sample matrix as a DataFrame with ``SampleID``.

        The DataFrame is intended as a convenience view for export, inspection,
        or joining with model outputs. The underlying canonical representation
        remains the NumPy array in ``samples``.
        """
        data = {"SampleID": self.sample_ids}
        for idx, name in enumerate(self.problem.names):
            data[name] = self.samples[:, idx]
        return pd.DataFrame(data, copy=False)

    def save(self, path: str | Path, *, format: str = "csv") -> None:
        """Serialize samples and metadata to disk.

        Args:
            path: File stem (no extension), e.g. ``"experiment"`` or
                ``"data/experiment"``.
            format: One of ``"csv"``, ``"txt"``, ``"xlsx"``, ``"parquet"``,
                ``"pkl"``.
        """
        if format not in _SAMPLE_FORMATS:
            raise ValueError(
                f"Unsupported format {format!r}. Choose from {sorted(_SAMPLE_FORMATS)}."
            )

        stem = Path(path)
        # DataFrame without SampleID — just parameter columns
        df = pd.DataFrame(self.samples, columns=list(self.problem.names))

        # --- write samples file ---
        sample_path = stem.with_suffix(f".{format}")
        _write_samples(df, sample_path, format)

        # --- identity mapping check ---
        # If expanded_to_unique is just 0..N-1, no duplicates exist;
        # skip writing the .npz file in that case.
        identity = bool(
            np.array_equal(
                self.expanded_to_unique,
                np.arange(self.expanded_n_total, dtype=self.expanded_to_unique.dtype),
            )
        )

        # --- write JSON metadata ---
        meta = {
            "gsax_version": _pkg_version("gsax"),
            "problem": {
                "names": list(self.problem.names),
                "bounds": [list(b) for b in self.problem.bounds]
                if self.problem.bounds is not None
                else None,
                "input_specs": [
                    _normalized_input_to_dict(spec) for spec in self.problem.input_specs
                ],
                "output_names": list(self.problem.output_names)
                if self.problem.output_names is not None
                else None,
            },
            "base_n": self.base_n,
            "calc_second_order": self.calc_second_order,
            "expanded_n_total": self.expanded_n_total,
            "identity_mapping": identity,
            "sample_format": format,
        }
        json_path = stem.with_suffix(".json")
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # --- write expanded_to_unique (skip for identity mappings) ---
        if not identity:
            npz_path = stem.with_suffix(".npz")
            np.savez_compressed(npz_path, expanded_to_unique=self.expanded_to_unique)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= *n*."""
    if n <= 0:
        return 1
    # Bit-length trick: (n-1).bit_length() gives the position of the highest
    # set bit, so 1 << that yields the smallest power of 2 >= n.
    return 1 << (n - 1).bit_length()


def _saltelli_step(n_params: int, calc_second_order: bool) -> int:
    """Return the number of expanded Saltelli rows per base Sobol point."""
    # Saltelli step = A + D*AB [+ D*BA] + B = D+2 [or 2D+2]
    return 2 * n_params + 2 if calc_second_order else n_params + 2


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
    # .copy() prevents aliasing: without it, overwriting element j would mutate
    # the original A[i] or B[i] row shared across iterations.
    rows = []
    for i in range(base_n):
        rows.append(A[i])
        for j in range(D):
            AB_j = A[i].copy()
            AB_j[j] = B[i, j]
            rows.append(AB_j)
        if calc_second_order:
            for j in range(D):
                BA_j = B[i].copy()
                BA_j[j] = A[i, j]
                rows.append(BA_j)
        rows.append(B[i])

    return np.array(rows)


def _transform_uniform(unit_values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Affine-map unit-interval samples into a finite uniform range."""
    return unit_values * (high - low) + low


def _transform_gaussian(
    unit_values: np.ndarray,
    mean: float,
    variance: float,
    *,
    low: float | None,
    high: float | None,
) -> np.ndarray:
    """Transform unit-interval samples into Gaussian or truncated Gaussian values."""
    # Inverse-CDF sampling (probability integral transform): if U ~ Uniform(0,1)
    # then F^{-1}(U) ~ F.  Clipping to (1e-12, 1-1e-12) prevents ppf (percent-
    # point function = quantile = inverse CDF) from returning +/-inf at boundaries.
    clipped = np.clip(unit_values, 1e-12, 1.0 - 1e-12)
    std = math.sqrt(variance)
    if low is None and high is None:
        return mean + std * norm.ppf(clipped)

    # Standardised truncation bounds a=(lo-mu)/sigma, b=(hi-mu)/sigma follow
    # scipy's truncnorm convention (standard-normal scale).
    a = -np.inf if low is None else (low - mean) / std
    b = np.inf if high is None else (high - mean) / std
    return truncnorm.ppf(clipped, a=a, b=b, loc=mean, scale=std)


def _transform_samples(problem: Problem, samples_unit: np.ndarray) -> np.ndarray:
    """Transform unit-cube Sobol samples into the problem's declared marginals."""
    # Pre-allocate output; each column is filled independently by its marginal's inverse CDF
    transformed = np.empty_like(samples_unit, dtype=np.float64)

    for idx, spec in enumerate(problem.input_specs):
        dist, first, second, low, high = spec
        if dist == "uniform":
            transformed[:, idx] = _transform_uniform(samples_unit[:, idx], first, second)
        else:
            transformed[:, idx] = _transform_gaussian(
                samples_unit[:, idx],
                first,
                second,
                low=low,
                high=high,
            )

    return transformed


def _stable_unique_rows(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate rows while preserving first-occurrence order.

    Returns:
        ``(unique_samples, expanded_to_unique)`` where ``expanded_to_unique``
        maps each original row position in ``samples`` back to the retained
        unique row index.
    """
    # Ensure C-contiguous layout so tobytes() gives a consistent byte representation
    samples = np.ascontiguousarray(samples)
    unique_rows: list[np.ndarray] = []
    expanded_to_unique = np.empty(samples.shape[0], dtype=np.int64)
    seen: dict[bytes, int] = {}

    for idx, row in enumerate(samples):
        # ``row.tobytes()`` gives a stable exact-match key for the already
        # scaled floating-point row. Exact deduplication is what we want here:
        # if two rows are bitwise equal, evaluating the model twice is wasteful.
        key = row.tobytes()
        unique_idx = seen.get(key)
        if unique_idx is None:
            unique_idx = len(unique_rows)
            seen[key] = unique_idx
            unique_rows.append(row.copy())
        expanded_to_unique[idx] = unique_idx

    if unique_rows:
        unique_samples = np.vstack(unique_rows)
    else:
        unique_samples = np.empty((0, samples.shape[1]), dtype=samples.dtype)
    return unique_samples, expanded_to_unique


def _print_sampling_summary(
    *,
    n_params: int,
    target_n: int,
    unique_n: int,
    expanded_n_total: int,
    base_n: int,
    calc_second_order: bool,
    scramble: bool,
) -> None:
    """Print a compact summary of the generated unique Sobol design."""
    duplicates_removed = expanded_n_total - unique_n
    duplicate_fraction = duplicates_removed / expanded_n_total if expanded_n_total else 0.0
    order_label = "second-order" if calc_second_order else "first/total-order"
    print(
        "gsax.sample: "
        f"D={n_params}, mode={order_label}, base_n={base_n}, "
        f"requested_unique>={target_n}, returned_unique={unique_n}, "
        f"expanded_rows={expanded_n_total}, duplicates_removed={duplicates_removed} "
        f"({duplicate_fraction:.1%}), scramble={scramble}"
    )


def sample_mc(
    problem: Problem,
    N: int,
    *,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Generate plain Monte Carlo samples from the input distributions.

    Unlike Saltelli/Sobol sampling, these have no quasi-random structure.
    Suitable for methods that need i.i.d. draws (e.g. DGSM).

    Args:
        problem: Problem definition with parameter distributions.
        N: Number of samples. Must be >= 1.
        seed: Random seed or generator for reproducibility.

    Returns:
        (N, D) sample array in the problem's physical units.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    rng = np.random.default_rng(seed)
    samples_unit = rng.random((N, problem.num_vars))
    return _transform_samples(problem, samples_unit)


def sample(
    problem: Problem,
    n_samples: int,
    *,
    base_n: int | None = None,
    calc_second_order: bool = True,
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
    verbose: bool = True,
) -> SamplingResult:
    """Generate unique Sobol/Saltelli samples for model evaluation.

    The function first builds the standard expanded Saltelli design for a
    candidate ``base_n``. It then removes exact duplicate rows while
    preserving first-occurrence order. If the resulting unique matrix is still
    smaller than the requested evaluation budget, ``base_n`` is doubled and
    the process repeats until enough unique rows are available.

    Args:
        problem: Problem definition with parameter names and bounds.
        n_samples: Minimum desired number of unique model evaluations.
            Ignored when ``base_n`` is provided.
        base_n: If given, use this exact Sobol base size (must be a power
            of 2) instead of searching for one. The expanded Saltelli
            design will have ``base_n * (2*D + 2)`` rows (second order)
            or ``base_n * (D + 2)`` rows (first/total only). This gives
            direct control over the sampling budget.
        calc_second_order: If ``True``, include BA cross-matrices so that
            second-order Sobol indices can be computed.  This increases
            the expanded Saltelli step from ``D + 2`` to ``2*D + 2``.
        scramble: Whether to apply Owen scrambling to the Sobol sequence.
        seed: Random seed or generator for reproducibility.
        verbose: If ``True`` (default), print a short summary describing the
            requested unique count, returned unique count, expanded Saltelli
            size, and how many duplicate rows were removed.

    Returns:
        SamplingResult with a unique sample matrix plus expansion metadata for
        later Sobol analysis.
    """
    D = problem.num_vars
    # Rows per base point -- determines the ratio between base_n and total expanded rows
    step = _saltelli_step(D, calc_second_order)

    if base_n is not None:
        # Power-of-2 check via bit trick (only one bit set)
        if base_n < 1 or (base_n & (base_n - 1)) != 0:
            raise ValueError(f"base_n must be a power of 2, got {base_n}")
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
        # Deduplication may reduce unique count below target;
        # double base_n and rebuild until we have enough.
        while unique_samples.shape[0] < target_n:
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
            unique_n=unique_samples.shape[0],
            expanded_n_total=expanded_samples.shape[0],
            base_n=base_n,
            calc_second_order=calc_second_order,
            scramble=scramble,
        )

    return SamplingResult(
        samples=unique_samples,
        sample_ids=sample_ids,
        expanded_n_total=expanded_samples.shape[0],
        expanded_to_unique=expanded_to_unique,
        base_n=base_n,
        n_params=D,
        calc_second_order=calc_second_order,
        problem=problem,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_samples(df: pd.DataFrame, path: Path, fmt: str) -> None:
    """Write the samples DataFrame to disk in the requested format."""
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "txt":
        np.savetxt(
            path,
            df.values,
            header=" ".join(df.columns),
            comments="",
        )
    elif fmt == "xlsx":
        try:
            df.to_excel(path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Writing xlsx requires openpyxl. Install it with: uv add openpyxl"
            ) from exc
    elif fmt == "parquet":
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Writing parquet requires pyarrow. Install it with: uv add pyarrow"
            ) from exc
    elif fmt == "pkl":
        df.to_pickle(path)


def _read_samples(path: Path, fmt: str) -> np.ndarray:
    """Read samples back from disk and return as a NumPy array."""
    if fmt == "csv":
        return pd.read_csv(path).values
    elif fmt == "txt":
        with open(path) as f:
            n_cols = len(f.readline().split())
        arr = np.loadtxt(path, skiprows=1)
        # loadtxt returns 1-D for single-row or single-column files; reshape based on header count
        if arr.ndim == 1:
            if n_cols == 1:
                arr = arr.reshape(-1, 1)
            else:
                arr = arr.reshape(1, -1)
        return arr
    elif fmt == "xlsx":
        try:
            return pd.read_excel(path).values
        except ImportError as exc:
            raise ImportError(
                "Reading xlsx requires openpyxl. Install it with: uv add openpyxl"
            ) from exc
    elif fmt == "parquet":
        try:
            return pd.read_parquet(path).values
        except ImportError as exc:
            raise ImportError(
                "Reading parquet requires pyarrow. Install it with: uv add pyarrow"
            ) from exc
    elif fmt == "pkl":
        return pd.read_pickle(path).values
    raise ValueError(f"Unsupported format {fmt!r}")


def load(path: str | Path, *, format: str = "csv") -> SamplingResult:
    """Load a previously saved :class:`SamplingResult` from disk.

    Args:
        path: File stem (no extension) matching what was passed to
            :meth:`SamplingResult.save`.
        format: Sample file format (must match the format used when saving).

    Returns:
        Reconstructed :class:`SamplingResult`.
    """
    stem = Path(path)
    json_path = stem.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")

    meta = json.loads(json_path.read_text(encoding="utf-8"))

    # Reconstruct Problem
    prob_meta = meta["problem"]
    output_names = (
        tuple(prob_meta["output_names"]) if prob_meta["output_names"] is not None else None
    )
    # Prefer rich input_specs (supports Gaussian); fall back to legacy bounds-only format
    if "input_specs" in prob_meta and prob_meta["input_specs"] is not None:
        from gsax.problem import _normalize_input_spec

        problem = Problem._from_normalized_inputs(
            names=tuple(prob_meta["names"]),
            input_specs=tuple(_normalize_input_spec(spec) for spec in prob_meta["input_specs"]),
            output_names=output_names,
        )
    else:
        problem = Problem(
            names=tuple(prob_meta["names"]),
            bounds=tuple(tuple(b) for b in prob_meta["bounds"]),
            output_names=output_names,
        )

    # Read samples
    sample_path = stem.with_suffix(f".{format}")
    samples = _read_samples(sample_path, format)

    n_unique = samples.shape[0]
    sample_ids = np.arange(n_unique, dtype=np.int64)

    # Reconstruct expanded_to_unique
    if meta["identity_mapping"]:
        expanded_to_unique = np.arange(meta["expanded_n_total"], dtype=np.int64)
    else:
        npz_path = stem.with_suffix(".npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"Expected mapping file not found: {npz_path}")
        expanded_to_unique = np.load(npz_path)["expanded_to_unique"]

    return SamplingResult(
        samples=samples,
        sample_ids=sample_ids,
        expanded_n_total=meta["expanded_n_total"],
        expanded_to_unique=expanded_to_unique,
        base_n=meta["base_n"],
        n_params=problem.num_vars,
        calc_second_order=meta["calc_second_order"],
        problem=problem,
    )
