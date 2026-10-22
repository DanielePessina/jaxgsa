"""Tests for the 1.1.0 error-message and result-field fixes (T4, internal)."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.morris import MorrisSamples
from jaxgsa.problem import Problem
from jaxgsa.sobol import SobolSamples


@pytest.fixture
def problem() -> Problem:
    return Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)})


def _model(x):
    return x[0] + 2.0 * x[1] * x[2]


def test_dgsm_model_in_y_position(problem: Problem) -> None:
    X = jaxgsa.sampling.monte_carlo(problem, 64, seed=0)
    with pytest.raises(TypeError, match="Pass the model as fn="):
        jaxgsa.dgsm.analyze(problem, X, _model)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="Pass the model as fn="):
        jaxgsa.dgsm.indices(problem, X, _model)  # ty: ignore[invalid-argument-type]


def test_samples_load_wrong_class(problem: Problem, tmp_path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = jaxgsa.sobol.sample(problem, 16, seed=0)
    path = tmp_path / "s.npz"
    samples.save(path)
    with pytest.raises(ValueError, match="holds a SobolSamples; load it with jaxgsa.sobol"):
        MorrisSamples.load(path)
    assert isinstance(SobolSamples.load(path), SobolSamples)


def test_samples_load_old_file_without_class(problem: Problem, tmp_path) -> None:
    import json

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = jaxgsa.sobol.sample(problem, 16, seed=0)
    path = tmp_path / "s.npz"
    samples.save(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(arrays["metadata"].item())
    del meta["samples_class"], meta["samples_class_path"]
    arrays["metadata"] = np.asarray(json.dumps(meta))
    old = tmp_path / "old.npz"
    np.savez_compressed(old, **arrays)
    assert isinstance(SobolSamples.load(old), SobolSamples)


def test_to_dataset_time_coords_length(problem: Problem) -> None:
    rng = np.random.default_rng(0)
    X = jaxgsa.sampling.monte_carlo(problem, 256, seed=0)
    Y = jnp.asarray(np.stack([X[:, 0] + t * X[:, 1] for t in range(4)], axis=1)[:, :, None])
    res = jaxgsa.pawn.analyze(problem, X, Y, verbose=False)
    assert res.to_dataset(time_coords=[0, 1, 2, 3]).sizes["time"] == 4
    with pytest.raises(ValueError, match="time_coords has 3 values.*4 time steps"):
        res.to_dataset(time_coords=rng.random(3))


@pytest.fixture
def sobol_result(problem: Problem):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = jaxgsa.sobol.sample(problem, 64, seed=0)
        Y = jnp.asarray([_model(np.asarray(x)) for x in samples.samples])
        return jaxgsa.sobol.analyze(samples, Y, verbose=False)


def test_dataset_attrs_method_and_version(sobol_result) -> None:
    attrs = sobol_result.to_dataset().attrs
    assert attrs["method"] == "sobol"
    assert attrs["jaxgsa_version"] == jaxgsa.__version__


def test_sampling_swapped_order(problem: Problem) -> None:
    X = jaxgsa.sampling.monte_carlo(problem, 32, seed=0)
    with pytest.raises(TypeError, match=r"correlate\(problem, X\)"):
        jaxgsa.sampling.correlate(X, problem)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"monte_carlo\(problem, n_samples\)"):
        jaxgsa.sampling.monte_carlo(32, problem)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"transform\(problem, U\)"):
        jaxgsa.sampling.transform(X, problem)  # ty: ignore[invalid-argument-type]


def test_row_count_message_names_x(problem: Problem) -> None:
    X = jaxgsa.sampling.monte_carlo(problem, 100, seed=0)
    Y = jnp.zeros(150)
    with pytest.raises(ValueError, match="Y has 150 rows but X has 100"):
        jaxgsa.pawn.analyze(problem, X, Y, verbose=False)


def test_hdmr_order_is_clamped_to_d() -> None:
    p = Problem.from_dict({"a": (0.0, 1.0)})
    X = jaxgsa.sampling.monte_carlo(p, 500, seed=0)
    Y = jnp.asarray(X[:, 0] + X[:, 0] ** 2)
    res = jaxgsa.hdmr.analyze(p, X, Y, order=2, verbose=False)
    assert res.order == 1
    assert "order=1" in repr(res)
    assert res.to_dataset().attrs["order"] == 1


def test_kucherenko_shapley_has_no_surrogate_fields() -> None:
    p = Problem.from_dict({"a": (0.0, 1.0), "b": (0.0, 1.0)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = jaxgsa.kucherenko.sample(p, 512, subsets="all", seed=0)
        Y = jnp.asarray(samples.samples[:, 0] + samples.samples[:, 0] * samples.samples[:, 1])
        res = jaxgsa.kucherenko.analyze(samples, Y, verbose=False)
    sh = res.shapley()
    assert sh.backend == "exact"
    assert sh.explained_variance is None
    assert sh.order is None
    ds = sh.to_dataset()
    assert "explained_variance" not in ds
    assert "order" not in ds.attrs
    assert ds.attrs["method"] == "shapley"
    assert "explained_variance" not in repr(sh)


def test_sobol_row_count_error_names_the_design(problem: Problem) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = jaxgsa.sobol.sample(problem, 16, seed=0)
    n = samples.samples.shape[0]
    with pytest.raises(ValueError, match=rf"the design has {n}; evaluate the model once per row"):
        jaxgsa.sobol.analyze(samples, jnp.zeros(n + 3), verbose=False)


def test_morris_row_count_error_names_the_design(problem: Problem) -> None:
    samples = jaxgsa.morris.sample(problem, 4, seed=0)
    n = samples.samples.shape[0]
    with pytest.raises(ValueError, match=rf"the design has {n}; evaluate the model once per row"):
        jaxgsa.morris.analyze(samples, jnp.zeros(n + 2), verbose=False)


@pytest.mark.parametrize("entry", ["indices", "analyze"])
def test_efast_row_count_error_names_the_design(problem: Problem, entry: str) -> None:
    design = jaxgsa.efast.sample(problem, 129, seed=0)
    n = design.n_runs
    fn = getattr(jaxgsa.efast, entry)
    kwargs = {"verbose": False} if entry == "analyze" else {}
    pattern = rf"the design has {n} \(n_runs = n_per_curve \* D"
    with pytest.raises(ValueError, match=pattern) as info:
        fn(design, jnp.zeros(n + 5), **kwargs)
    assert "evaluate the model once per row of design.samples" in str(info.value)
    assert "samples.samples" not in str(info.value)


def test_given_data_row_count_error_still_names_x(problem: Problem) -> None:
    X = jaxgsa.sampling.monte_carlo(problem, 64, seed=0)
    with pytest.raises(ValueError, match="Y has 60 rows but X has 64"):
        jaxgsa.pawn.analyze(problem, jnp.asarray(X), jnp.zeros(60), verbose=False)


def test_dgsm_precomputed_row_counts_must_agree(problem: Problem) -> None:
    X = jaxgsa.sampling.monte_carlo(problem, 10, seed=0)
    with pytest.raises(ValueError, match="X has 10 rows, Y has 16 rows and dfdx has 16 rows"):
        jaxgsa.dgsm.analyze(
            problem, jnp.asarray(X), jnp.zeros(16), dfdx=jnp.zeros((16, 3)), verbose=False
        )


def test_efast_load_checks_samples_class(problem: Problem, tmp_path) -> None:
    samples = jaxgsa.efast.sample(problem, 129, seed=0)
    path = tmp_path / "e.npz"
    samples.save(path)
    assert isinstance(jaxgsa.efast.EFASTSamples.load(path), jaxgsa.efast.EFASTSamples)
    with pytest.raises(ValueError, match="holds a EFASTSamples; load it with jaxgsa.efast"):
        SobolSamples.load(path)
    with pytest.raises(ValueError, match="holds a EFASTSamples"):
        jaxgsa.kucherenko.KucherenkoSamples.load(path)


def test_efast_load_reads_files_without_class_key(problem: Problem, tmp_path) -> None:
    import json

    samples = jaxgsa.efast.sample(problem, 129, seed=0)
    path = tmp_path / "e.npz"
    samples.save(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(arrays["metadata"].item())
    del meta["samples_class"], meta["samples_class_path"]
    arrays["metadata"] = np.asarray(json.dumps(meta))
    old = tmp_path / "old.npz"
    np.savez_compressed(old, **arrays)
    loaded = jaxgsa.efast.EFASTSamples.load(old)
    np.testing.assert_array_equal(loaded.samples, samples.samples)


def test_result_method_attr_for_user_subclass() -> None:
    from jaxgsa.sobol import SobolResult

    class MySobol(SobolResult):
        pass

    assert MySobol.__module__ == __name__
    from jaxgsa._core.result import _method_name

    assert _method_name(MySobol) == "sobol"
    assert _method_name(SobolResult) == "sobol"
