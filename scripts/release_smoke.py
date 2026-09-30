"""Check public workflows against an installed wheel, outside the checkout."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np
import xarray as xr

import jaxgsa


def main() -> None:
    """Verify installation, persisted sampling, analytical indices, and export."""
    assert jaxgsa.__file__ is not None
    package_path = Path(jaxgsa.__file__).resolve()
    assert package_path.is_relative_to(Path(sys.prefix).resolve()), package_path
    problem = jaxgsa.Problem.from_dict({"x1": (-1.0, 1.0), "x2": (-1.0, 1.0)})
    expected = np.array([0.2, 0.8])  # Var(x1 + 2*x2), independent equal variances.

    with TemporaryDirectory() as directory:
        design = jaxgsa.sobol.sample(problem, n_samples=2048, seed=17, verbose=False)
        design_path = Path(directory) / "design.npz"
        design.save(design_path)
        loaded = jaxgsa.sobol.SobolSamples.load(design_path)
        np.testing.assert_array_equal(loaded.samples, design.samples)
        outputs = jnp.asarray(loaded.samples[:, 0] + 2 * loaded.samples[:, 1])
        result = jaxgsa.sobol.analyze(loaded, outputs, verbose=False)
        assert isinstance(result, jaxgsa.sobol.SobolResult)
        np.testing.assert_allclose(result.S1, expected, atol=0.015, rtol=0)
        np.testing.assert_allclose(result.ST, expected, atol=0.015, rtol=0)
        dataset = result.to_dataset()
        dataset_path = Path(directory) / "indices.nc"
        dataset.to_netcdf(dataset_path, engine="scipy")
        with xr.open_dataset(dataset_path, engine="scipy") as restored:
            xr.testing.assert_allclose(restored, dataset)

    inputs = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 64, seed=19))
    outputs = inputs[:, 0] + 2 * inputs[:, 1]
    surrogate = jaxgsa.pce.analyze(problem, inputs, outputs, order=1, verbose=False)
    assert isinstance(surrogate, jaxgsa.pce.PCEResult)
    held_out = jnp.asarray(jaxgsa.sampling.monte_carlo(problem, 16, seed=23))
    np.testing.assert_allclose(
        surrogate.predict(held_out), held_out[:, 0] + 2 * held_out[:, 1], atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(surrogate.S1, expected, atol=1e-5, rtol=0)
    print(f"Installed-wheel workflows passed: {package_path}")


if __name__ == "__main__":
    main()
