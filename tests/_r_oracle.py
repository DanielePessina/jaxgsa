"""Run a short R script as an out-of-process oracle for a test.

Some estimators have a published R implementation. A test can run that
implementation as a black box on the same data and compare the numbers. This
module is the one place that knows how to do so. The R code is never read or
copied: the test only calls the package's public functions.

Data goes to R through temporary CSV files, and results come back the same
way. A test calls :func:`run_r` and gets NumPy arrays. If ``Rscript`` or a
required R package is missing, the test is skipped, not failed.

Example::

    from _r_oracle import run_r

    out = run_r(
        '''
        x <- sobolrank(model = NULL, X = as.data.frame(X))
        tell(x, as.numeric(Y))
        S <- as.numeric(unlist(x$S))
        ''',
        arrays={"X": X, "Y": Y},
        packages=("sensitivity",),
        outputs=("S",),
    )
    out["S"]  # a 1-D NumPy array
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

# Exit code the R prelude uses to say "a required package is not installed".
_MISSING_PACKAGE = 77

_TIMEOUT_SECONDS = 300


def run_r(
    script: str,
    *,
    arrays: Mapping[str, np.ndarray],
    outputs: Sequence[str],
    packages: Sequence[str] = (),
) -> dict[str, np.ndarray]:
    """Run an R script on NumPy arrays and return named NumPy arrays.

    Each input array becomes an R matrix with the same name. A 1-D array
    becomes a one-column matrix. The script must assign every name in
    ``outputs`` before it ends. Each output must be a vector or a matrix of
    numbers.

    Args:
        script: R code to run. It sees one matrix per entry of ``arrays``.
        arrays: Input arrays by R variable name. Values are written with 17
            significant digits, so the round trip loses nothing in float64.
        outputs: Names of the R variables to read back.
        packages: R packages to load with ``library()`` before the script. If
            one is not installed, the calling test is skipped.

    Returns:
        A dict from output name to a float64 array. An output with one column
        comes back 1-D, and any other output comes back 2-D.

    Raises:
        RuntimeError: If R fails for a reason other than a missing package.
            The message carries R's standard error.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed")

    with tempfile.TemporaryDirectory(prefix="jaxgsa_r_oracle_") as tmp:
        workdir = Path(tmp)
        lines = []
        for package in packages:
            lines.append(
                f"if (!requireNamespace({package!r}, quietly = TRUE)) "
                f"quit(status = {_MISSING_PACKAGE})"
            )
            lines.append(
                f"suppressPackageStartupMessages(library({package!r}, character.only = TRUE))"
            )
        for name, array in arrays.items():
            path = workdir / f"in_{name}.csv"
            np.savetxt(
                path,
                np.asarray(array, dtype=np.float64).reshape(len(array), -1),
                fmt="%.17g",
                delimiter=",",
            )
            lines.append(f"{name} <- as.matrix(read.csv({str(path)!r}, header = FALSE))")
            lines.append(f"dimnames({name}) <- NULL")
        lines.append(script)
        for name in outputs:
            path = workdir / f"out_{name}.csv"
            lines.append(
                f"write.table(as.matrix({name}), {str(path)!r}, sep = ',', "
                "row.names = FALSE, col.names = FALSE)"
            )
        script_path = workdir / "script.R"
        script_path.write_text("\n".join(lines) + "\n")

        done = subprocess.run(
            [rscript, "--vanilla", str(script_path)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
        if done.returncode == _MISSING_PACKAGE:
            pytest.skip(f"an R package from {list(packages)} is not installed")
        if done.returncode != 0:
            raise RuntimeError(f"R script failed (exit {done.returncode}):\n{done.stderr}")

        results = {}
        for name in outputs:
            values = np.loadtxt(workdir / f"out_{name}.csv", delimiter=",", ndmin=2)
            results[name] = values[:, 0] if values.shape[1] == 1 else values
        return results
