"""Fixed-seed numerical baseline for jaxgsa.

This script runs every public method on a small fixed set of problems and
writes every numeric field of every result to a JSON file. A later refactor
that is declared "plumbing only" must reproduce that file exactly. See
``scripts/baseline/README.md``.

Run it with::

    uv run scripts/baseline_dump.py

Add ``--out PATH`` to write somewhere else, and ``--twice`` to build the dump
two times in one process and confirm the two are byte-identical.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import jaxgsa
from jaxgsa import (
    borgonovo,
    dgsm,
    efast,
    hdmr,
    hsic,
    kucherenko,
    morris,
    optimal_transport,
    pawn,
    pce,
    shapley,
    sobol,
    vkoga,
)
from jaxgsa.benchmarks import ishigami, sobol_g

# --------------------------------------------------------------------------
# Policy constants
# --------------------------------------------------------------------------

SEED = 20260818
"""The one seed used by every sampler, bootstrap and permutation test."""

DEFAULT_OUT = Path(__file__).resolve().parent / "baseline" / "baseline-1.0.0.json"

INLINE_ARRAY_LIMIT = 1024
"""Arrays with at most this many elements are written out element by element.

A larger array is recorded as a SHA-256 digest of its exact bytes. The
comparison stays bit-for-bit either way; only the diff report is coarser.
"""

EXCLUDED_FIELDS: dict[str, tuple[str, ...]] = {
    # "<problem>/<method>": ("field", ...)
    #
    # Fields that are not reproducible at a fixed seed. Each entry must carry
    # a comment saying why. The list is empty: every method in jaxgsa 0.8.0
    # reproduces bit-for-bit on a fixed seed (verified 2026-08-18 by running
    # this script with --twice).
}
"""Documented exclusions from the comparison. Keep this empty if you can."""


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def _digest(array: np.ndarray) -> str:
    """Return a SHA-256 digest of an array's exact bytes.

    Args:
        array: Array to digest. It is made contiguous first, so the digest
            depends on values, dtype and shape but not on strides.

    Returns:
        Hexadecimal digest string.
    """
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _encode_array(array: np.ndarray) -> dict[str, Any]:
    """Encode an array with full float precision.

    Args:
        array: The array to encode.

    Returns:
        A JSON-safe mapping with the shape, the dtype, and either every
        value or a digest of the bytes.
    """
    out: dict[str, Any] = {
        "__array__": True,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size <= INLINE_ARRAY_LIMIT:
        # ``tolist`` gives Python floats, and ``json`` writes those with
        # ``repr``, which round-trips exactly.
        out["values"] = np.asarray(array).tolist()
    else:
        out["sha256"] = _digest(array)
    return out


def encode(value: Any) -> Any:
    """Turn any result field into a JSON-safe, exactly comparable value.

    Args:
        value: Field value taken from a result object.

    Returns:
        A JSON-safe structure. Unsupported objects become a marker holding
        their type name, so a field is never dropped without a trace.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, enum.Enum):
        # By value, not by type name. `InvalidUnit` says which block of data a
        # method treats as indivisible, so recording only "InvalidUnit" would
        # let a wiring error that reports a Saltelli group where a trajectory
        # belongs pass this gate unnoticed.
        return {"__enum__": f"{type(value).__name__}.{value.name}", "value": encode(value.value)}
    if isinstance(value, jaxgsa.Problem):
        # Inputs, not outputs. The problem is echoed once per problem entry.
        return {"__problem__": list(value.names)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, jax.Array | np.ndarray):
        return _encode_array(np.asarray(value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return encode_fields(value)
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        # NamedTuple, such as the fitted VKOGA state.
        return {str(k): encode(v) for k, v in value._asdict().items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [encode(v) for v in value]
    return {"__unencoded__": type(value).__name__}


def encode_fields(obj: Any) -> dict[str, Any]:
    """Encode every dataclass field of a result object.

    Traversal uses :func:`dataclasses.fields`, so a field added later is
    picked up without editing this script.

    Args:
        obj: A dataclass instance.

    Returns:
        Mapping from field name to encoded value.
    """
    return {f.name: encode(getattr(obj, f.name, None)) for f in dataclasses.fields(obj)}


# --------------------------------------------------------------------------
# Problems
# --------------------------------------------------------------------------

_TIME_FACTORS = jnp.array([1.0, 0.5, -0.25])
_OUTPUT_OFFSETS = jnp.array([0.0, 3.0])
_CATEGORICAL_PROBS = (0.5, 0.3, 0.2)
_CATEGORICAL_OFFSETS = jnp.array([0.0, 2.0, -1.0])
_CORRELATION = np.array(
    [
        [1.0, 0.5, 0.0],
        [0.5, 1.0, -0.3],
        [0.0, -0.3, 1.0],
    ]
)


def _ishigami_scalar(X: jax.Array) -> jax.Array:
    """Ishigami with a scalar output, shape ``(N,)``."""
    return ishigami.evaluate(X)


def _sobol_g_multi(X: jax.Array) -> jax.Array:
    """Sobol g-function widened to two outputs, shape ``(N, 2)``."""
    g = sobol_g.evaluate(X)
    return jnp.stack([g, 2.0 * g + _OUTPUT_OFFSETS[1]], axis=-1)


def _ishigami_series(X: jax.Array) -> jax.Array:
    """Ishigami widened to a time series, shape ``(N, 3, 2)``."""
    g = ishigami.evaluate(X)
    return g[:, None, None] * _TIME_FACTORS[None, :, None] + _OUTPUT_OFFSETS[None, None, :]


def _categorical_model(X: jax.Array) -> jax.Array:
    """One uniform input plus a per-level offset, shape ``(N,)``."""
    codes = jnp.asarray(X[:, 1], dtype=jnp.int32)
    return X[:, 0] + jnp.take(_CATEGORICAL_OFFSETS, codes)


def _as_point_model(
    model: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], jax.Array]:
    """Adapt a batch model to the one-sample convention ``dgsm.analyze`` wants.

    Args:
        model: Callable on an ``(N, D)`` matrix.

    Returns:
        Callable on a single ``(D,)`` row, differentiable by ``jax.jacrev``.
    """

    def point(x: jax.Array) -> jax.Array:
        return model(x[None, :])[0]

    return point


@dataclasses.dataclass(frozen=True)
class Case:
    """One problem, its model, and the sample size used for it.

    Attributes:
        name: Key used in the dump and in the diff report.
        problem: The jaxgsa problem.
        model: Callable mapping an ``(N, D)`` input matrix to output.
        n: Base sample size. Design methods multiply it by their own factor.
        shape: Human-readable output shape, recorded in the dump.
        efast_n: Points per eFAST search curve. eFAST needs at least
            ``4 M^2 (D - 1) + 1`` of them, so it grows with the dimension.
    """

    name: str
    problem: jaxgsa.Problem
    model: Callable[[jax.Array], jax.Array]
    n: int
    shape: str
    efast_n: int = 257


def _categorical_problem() -> jaxgsa.Problem:
    """One uniform and one three-level categorical parameter."""
    return jaxgsa.Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": list(_CATEGORICAL_PROBS)},
        }
    )


def build_cases() -> list[Case]:
    """Build the fixed set of baseline problems.

    Returns:
        One case per output shape and per input feature that the refactors
        touch: scalar, multi-output, time series, declared correlation, and
        a categorical input.
    """
    return [
        Case("ishigami_scalar", ishigami.PROBLEM, _ishigami_scalar, 512, "(N,)"),
        Case("sobol_g_multi", sobol_g.PROBLEM, _sobol_g_multi, 512, "(N, 2)", efast_n=513),
        Case("ishigami_series", ishigami.PROBLEM, _ishigami_series, 512, "(N, 3, 2)"),
        Case(
            "ishigami_correlated",
            ishigami.PROBLEM.with_correlation(_CORRELATION),
            _ishigami_scalar,
            512,
            "(N,)",
        ),
        Case("categorical_mixed", _categorical_problem(), _categorical_model, 512, "(N,)"),
    ]


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------
#
# Every method gets a fixed, explicit set of keyword arguments. Defaults are
# not relied on: a later release may change a default, and that would change
# the numbers for a reason unrelated to wiring.


def _m_sobol(case: Case) -> Any:
    sr = sobol.sample(case.problem, n_samples=case.n, seed=SEED, verbose=False)
    Y = case.model(jnp.asarray(sr.samples))
    return sobol.analyze(sr, Y, num_resamples=20, key=jax.random.key(SEED))


def _m_morris(case: Case) -> Any:
    sr = morris.sample(case.problem, n_trajectories=32, seed=SEED, verbose=False)
    Y = case.model(jnp.asarray(sr.samples))
    return morris.analyze(sr, Y, num_resamples=20, key=jax.random.key(SEED))


def _m_efast(case: Case) -> Any:
    sr = efast.sample(case.problem, n_per_curve=case.efast_n, seed=SEED)
    Y = case.model(jnp.asarray(sr.samples))
    return efast.analyze(sr, Y)


def _m_kucherenko(case: Case) -> Any:
    sr = kucherenko.sample(case.problem, n_samples=case.n, seed=SEED)
    Y = case.model(jnp.asarray(sr.samples))
    return kucherenko.analyze(sr, Y)


def _m_borgonovo(case: Case, X: jax.Array, Y: jax.Array) -> Any:
    return borgonovo.analyze(case.problem, X, Y, n_bootstrap=20, seed=SEED)


def _m_pawn(case: Case, X: jax.Array, Y: jax.Array) -> Any:
    return pawn.analyze(case.problem, X, Y, n_bins=8, n_bootstrap=20, seed=SEED)


def _m_hsic(case: Case, X: jax.Array, Y: jax.Array) -> Any:
    return hsic.analyze(case.problem, X, Y, n_perms=50, seed=SEED)


def _m_optimal_transport(case: Case, X: jax.Array, Y: jax.Array) -> Any:
    return optimal_transport.analyze(
        case.problem, X, Y, n_partitions=8, n_bootstrap=10, dummy=True, seed=SEED
    )


def _m_pce(case: Case, X: jax.Array, Y: jax.Array) -> Any:
    return pce.analyze(case.problem, X, Y, order=3)


def _m_hdmr(case: Case, X: jax.Array, Y: jax.Array) -> Any:
    return hdmr.analyze(case.problem, X, Y, maxorder=2, maxiter=50)


def _m_shapley(case: Case, X: jax.Array, Y: jax.Array) -> Any:
    return shapley.analyze(case.problem, X, Y, backend="pce", order=3)


def _m_vkoga(case: Case, X: jax.Array, Y: jax.Array) -> Any:
    return vkoga.analyze(
        case.problem,
        X,
        Y,
        max_centers=32,
        n_folds=3,
        n_outer=64,
        n_inner=32,
        n_variance=512,
        seed=SEED,
    )


def _m_dgsm(case: Case, X: jax.Array, _Y: jax.Array) -> Any:
    # dgsm.analyze differentiates a one-sample function, not a batch one.
    return dgsm.analyze(case.problem, _as_point_model(case.model), X)


DESIGN_METHODS: dict[str, Callable[[Case], Any]] = {
    "efast": _m_efast,
    "kucherenko": _m_kucherenko,
    "morris": _m_morris,
    "sobol": _m_sobol,
}

GIVEN_DATA_METHODS: dict[str, Callable[[Case, jax.Array, jax.Array], Any]] = {
    "borgonovo": _m_borgonovo,
    "dgsm": _m_dgsm,
    "hdmr": _m_hdmr,
    "hsic": _m_hsic,
    "optimal_transport": _m_optimal_transport,
    "pawn": _m_pawn,
    "pce": _m_pce,
    "shapley": _m_shapley,
    "vkoga": _m_vkoga,
}


def _run(label: str, fn: Callable[[], Any], verbose: bool) -> dict[str, Any]:
    """Run one method and encode its result, or record why it refused.

    A method that raises is recorded by exception type only. The message is
    left out on purpose: wording changes are not numerical changes.

    Args:
        label: ``"<problem>/<method>"``, used for the exclusion list.
        fn: Zero-argument callable returning a result object.
        verbose: Print progress to stderr.

    Returns:
        Mapping with a ``status`` key of ``"ok"``, ``"raised"`` or
        ``"failed"``.
    """
    if verbose:
        print(f"  {label}", file=sys.stderr, flush=True)
    try:
        result = fn()
    except (ValueError, TypeError, NotImplementedError) as exc:
        return {"status": "raised", "exception": type(exc).__name__}
    except Exception as exc:  # noqa: BLE001 - a crash must not lose the rest of the dump
        print(traceback.format_exc(), file=sys.stderr)
        return {"status": "failed", "exception": type(exc).__name__}

    fields = encode_fields(result)
    for name in EXCLUDED_FIELDS.get(label, ()):
        fields.pop(name, None)
    return {"status": "ok", "type": type(result).__name__, "fields": fields}


def build_dump(verbose: bool = True) -> dict[str, Any]:
    """Run every method on every case and encode the results.

    Args:
        verbose: Print progress to stderr.

    Returns:
        The comparable body of the dump. The header is added separately, so
        this mapping holds numbers only.
    """
    body: dict[str, Any] = {}
    for case in build_cases():
        if verbose:
            print(f"{case.name}", file=sys.stderr, flush=True)
        entry: dict[str, Any] = {
            "output_shape": case.shape,
            "n": case.n,
            "names": list(case.problem.names),
            "has_correlated_inputs": bool(case.problem.has_correlated_inputs),
            "has_categorical_inputs": bool(case.problem.has_categorical_inputs),
            "methods": {},
        }

        X_np = jaxgsa.sampling.monte_carlo(case.problem, case.n * 4, seed=SEED)
        X = jnp.asarray(X_np)
        Y = case.model(X)
        entry["monte_carlo_X"] = encode(X_np)
        entry["monte_carlo_Y"] = encode(np.asarray(Y))

        for name, fn in sorted(DESIGN_METHODS.items()):
            entry["methods"][name] = _run(f"{case.name}/{name}", lambda fn=fn: fn(case), verbose)
        for name, gfn in sorted(GIVEN_DATA_METHODS.items()):
            entry["methods"][name] = _run(
                f"{case.name}/{name}", lambda gfn=gfn: gfn(case, X, Y), verbose
            )

        body[case.name] = entry
    return body


def _git_commit() -> str:
    """Return the current commit hash, or ``"unknown"`` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _package_version() -> str:
    """Return the installed jaxgsa version, or ``"unknown"``."""
    try:
        return importlib.metadata.version("jaxgsa")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_header() -> dict[str, Any]:
    """Return the environment header. It is never compared."""
    return {
        "jaxgsa_version": _package_version(),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "jax_default_backend": jax.default_backend(),
        "x64_enabled": bool(getattr(jax.config, "jax_enable_x64")),
        "git_commit": _git_commit(),
        "seed": SEED,
        "inline_array_limit": INLINE_ARRAY_LIMIT,
        "excluded_fields": {k: list(v) for k, v in EXCLUDED_FIELDS.items()},
    }


def to_json(body: dict[str, Any], header: dict[str, Any] | None = None) -> str:
    """Serialize a dump body with full float precision.

    Args:
        body: The comparable body from :func:`build_dump`.
        header: Optional environment header.

    Returns:
        The JSON text, with sorted keys so the bytes are reproducible.
    """
    doc = {"header": header if header is not None else build_header(), "results": body}
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the baseline file.

    Args:
        argv: Command-line arguments, or ``None`` for ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSON path")
    parser.add_argument(
        "--twice", action="store_true", help="build twice and check the two dumps agree"
    )
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    args = parser.parse_args(argv)
    verbose = not args.quiet

    body = build_dump(verbose=verbose)
    if args.twice:
        second = build_dump(verbose=verbose)
        if to_json(body, {}) != to_json(second, {}):
            print("NON-DETERMINISTIC: the two dumps differ", file=sys.stderr)
            return 2
        print("determinism check passed: two dumps are identical", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_json(body))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
