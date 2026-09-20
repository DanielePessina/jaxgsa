"""End-to-end checks for automatically bucketed irregular outputs.

The public contract is intentionally small: a ragged ``Y`` is inferred from
its ``(times, values)`` channels, analyzed one channel at a time, and exported
without padding the channels onto a fabricated common grid.
"""

from __future__ import annotations

from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import jaxgsa
from jaxgsa._core.irregular import IrregularResult
from jaxgsa.problem import Problem
from jaxgsa.sampling import monte_carlo

N = 128
OUTPUT_NAMES = ("conc", "d43")
PROBLEM = Problem.from_dict(
    {"x0": (0.0, 1.0), "x1": (0.0, 1.0), "x2": (0.0, 1.0)},
    output_names=OUTPUT_NAMES,
)
UNLABELED = Problem.from_dict({"x0": (0.0, 1.0), "x1": (0.0, 1.0)})
T_CONC = jnp.array([0.0, 1.0, 2.0, 2.5, 6.0])
T_D43 = jnp.array([0.0, 2.0, 5.5, 6.0])


def _ragged_y(n: int = N, *, seed: int = 0) -> list[tuple[Array, Array]]:
    """Build two deterministic channels on different time grids."""
    rng = np.random.default_rng(seed)
    return [
        (T_CONC, jnp.asarray(rng.normal(size=(n, T_CONC.shape[0])))),
        (T_D43, jnp.asarray(rng.normal(size=(n, T_D43.shape[0])))),
    ]


def _sobol_design():
    return jaxgsa.sobol.sample(PROBLEM, N, seed=0)


def _irregular(result: Any) -> IrregularResult:
    """Narrow the public return type for a ragged-output call."""
    assert isinstance(result, IrregularResult)
    return result


def test_sobol_ragged_output_is_bucketed_and_exported_without_padding():
    """The primary feature path preserves each channel's own coordinate grid."""
    design = _sobol_design()
    result = _irregular(jaxgsa.sobol.analyze(design, _ragged_y(design.n_runs)))

    assert list(result.channels) == list(OUTPUT_NAMES)
    dataset = result.to_dataset()
    assert dataset["conc_S1"].dims == ("time_conc", "param")
    assert dataset["d43_S1"].dims == ("time_d43", "param")
    assert dataset["conc_S2"].dims == ("time_conc", "param_i", "param_j")
    np.testing.assert_array_equal(dataset["time_conc"], T_CONC)
    np.testing.assert_array_equal(dataset["time_d43"], T_D43)


def test_named_channels_follow_problem_output_order():
    """A mapping may be supplied in any order but result labels stay stable."""
    design = _sobol_design()
    ragged = _ragged_y(design.n_runs)
    named = {"d43": ragged[1], "conc": ragged[0]}

    result = _irregular(jaxgsa.sobol.analyze(design, named))

    assert list(result.channels) == ["conc", "d43"]
    np.testing.assert_array_equal(result.times["conc"], T_CONC)


def test_unsorted_times_and_single_time_values_are_normalized_publicly():
    """The public path sorts time/value columns and accepts a one-point series."""
    X = jnp.asarray(monte_carlo(UNLABELED, n=N, seed=1))
    rng = np.random.default_rng(0)
    values = jnp.asarray(rng.normal(size=(N, 3)))
    single = jnp.asarray(rng.normal(size=N))
    result = _irregular(
        jaxgsa.hsic.analyze(
            UNLABELED,
            X,
            [(jnp.array([2.0, 0.0, 1.0]), values), (jnp.array([4.0]), single)],
            n_perms=10,
            key=jax.random.key(0),
        )
    )

    np.testing.assert_array_equal(result.times["y0"], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(result.times["y1"], [4.0])
    assert result.channels["y1"].R2_HSIC.shape == (1, 1, 2)


def test_shared_grid_matches_the_regular_analysis():
    """Bucketing is numerically the same as regular analysis when grids agree."""
    design = _sobol_design()
    times = jnp.array([0.0, 1.0, 2.0])
    rng = np.random.default_rng(2)
    values = jnp.asarray(rng.normal(size=(N, 3, 2)))
    regular = jaxgsa.sobol.analyze(design, values)
    assert not isinstance(regular, IrregularResult)
    ragged = [(times, values[:, :, 0]), (times, values[:, :, 1])]
    irregular = _irregular(jaxgsa.sobol.analyze(design, ragged))

    for index, name in enumerate(OUTPUT_NAMES):
        np.testing.assert_allclose(
            irregular.channels[name].S1[:, 0, :], regular.S1[:, index, :], atol=1e-6
        )
        np.testing.assert_allclose(
            irregular.channels[name].ST[:, 0, :], regular.ST[:, index, :], atol=1e-6
        )


def test_given_data_channel_matches_direct_regular_analysis():
    """One representative given-data method checks per-channel routing."""
    X = jnp.asarray(monte_carlo(PROBLEM, n=N, seed=1))
    ragged = _ragged_y(N)
    key = jax.random.key(0)
    irregular = _irregular(jaxgsa.hsic.analyze(PROBLEM, X, ragged, n_perms=20, key=key))
    direct = jaxgsa.hsic.analyze(
        PROBLEM.with_output_names(("conc",)), X, ragged[0][1][:, :, None], n_perms=20, key=key
    )
    assert not isinstance(direct, IrregularResult)
    np.testing.assert_allclose(irregular.channels["conc"].R2_HSIC, direct.R2_HSIC)
    np.testing.assert_allclose(irregular.channels["conc"].T_HSIC, direct.T_HSIC)


def test_bootstrap_is_forwarded_per_channel_and_keeps_replicates():
    design = _sobol_design()
    result = _irregular(
        jaxgsa.sobol.analyze(
            design,
            _ragged_y(design.n_runs),
            n_bootstrap=10,
            key=jax.random.key(3),
            keep_replicates=True,
        )
    )

    for channel in result.channels.values():
        assert channel.ci is not None and channel.ci.n_bootstrap == 10
        assert channel.ci.replicates is not None
        assert channel.S1_conf is not None
        assert channel.ST_conf is not None


@pytest.mark.parametrize(
    ("ragged", "pattern"),
    [
        (
            [(T_CONC, jnp.ones((N + 1, T_CONC.shape[0]))), (T_D43, jnp.ones((N, 4)))],
            "sample rows",
        ),
        (
            [(jnp.array([0.0, 0.0]), jnp.ones((N, 2))), (T_D43, jnp.ones((N, 4)))],
            "not repeat",
        ),
        (
            {"conc": (T_CONC, jnp.ones((N, 5))), "wrong": (T_D43, jnp.ones((N, 4)))},
            "dict keys",
        ),
        (
            [(T_CONC, jnp.ones((N, 5, 1))), (T_D43, jnp.ones((N, 4)))],
            "values must be",
        ),
    ],
)
def test_malformed_ragged_inputs_are_rejected_by_the_public_api(ragged, pattern):
    design = _sobol_design()
    with pytest.raises(ValueError, match=pattern):
        jaxgsa.sobol.analyze(design, ragged)


def test_regular_array_stays_on_the_regular_path():
    design = _sobol_design()
    result = jaxgsa.sobol.analyze(design, jnp.ones((design.n_runs, 3, 2)))
    assert not isinstance(result, IrregularResult)


def test_dgsm_explicitly_rejects_irregular_outputs():
    X = jnp.asarray(monte_carlo(PROBLEM, n=N, seed=1))
    with pytest.raises(NotImplementedError, match="irregular"):
        jaxgsa.dgsm.analyze(PROBLEM, X=X, Y=_ragged_y(N))


def test_removed_selector_is_not_part_of_the_public_contract():
    design = _sobol_design()
    with pytest.raises(TypeError, match="irregular"):
        cast(Callable[..., Any], jaxgsa.sobol.analyze)(
            design, _ragged_y(design.n_runs), irregular="bucket"
        )


@pytest.mark.verbose_output
def test_verbose_summary_mentions_each_channel(capsys):
    design = _sobol_design()
    jaxgsa.sobol.analyze(design, _ragged_y(design.n_runs), verbose=True)
    output = capsys.readouterr().out
    assert "irregular analysis: 2 output channel(s)" in output
    assert "conc: SobolResult" in output
    assert "d43: SobolResult" in output


_GIVEN_DATA: dict[str, tuple[Callable[..., Any], dict[str, Any], int]] = {
    "borgonovo": (jaxgsa.borgonovo.analyze, {}, N),
    "hdmr": (jaxgsa.hdmr.analyze, {"maxorder": 1, "maxiter": 10}, 320),
    "hsic": (jaxgsa.hsic.analyze, {"n_perms": 10, "key": jax.random.key(0)}, N),
    "optimal_transport": (jaxgsa.optimal_transport.analyze, {}, N),
    "pawn": (jaxgsa.pawn.analyze, {}, N),
    "pce": (jaxgsa.pce.analyze, {"order": 2}, N),
    "shapley": (jaxgsa.shapley.analyze, {"backend": "pce"}, N),
    "vkoga": (
        jaxgsa.vkoga.analyze,
        {"n_outer": 32, "n_inner": 16, "n_variance": 64, "key": jax.random.key(0)},
        N,
    ),
}


@pytest.mark.parametrize("name", sorted(_GIVEN_DATA))
def test_every_given_data_method_supports_ragged_outputs(name):
    analyze, kwargs, n = _GIVEN_DATA[name]
    X = jnp.asarray(monte_carlo(PROBLEM, n=n, seed=1))
    result = _irregular(analyze(PROBLEM, X, _ragged_y(n), **kwargs))
    assert list(result.channels) == list(OUTPUT_NAMES)


_DESIGN_BASED: dict[
    str, tuple[Callable[..., Any], Callable[..., Any], dict[str, Any], dict[str, Any]]
] = {
    "efast": (jaxgsa.efast.sample, jaxgsa.efast.analyze, {"n_per_curve": 129}, {}),
    "kucherenko": (
        jaxgsa.kucherenko.sample,
        jaxgsa.kucherenko.analyze,
        {"n_samples": N},
        {},
    ),
    "morris": (jaxgsa.morris.sample, jaxgsa.morris.analyze, {"n_trajectories": 10}, {}),
    "sobol": (jaxgsa.sobol.sample, jaxgsa.sobol.analyze, {"n_samples": N}, {}),
}


@pytest.mark.parametrize("name", sorted(_DESIGN_BASED))
def test_every_design_based_method_supports_ragged_outputs(name):
    sample, analyze, sample_kwargs, analyze_kwargs = _DESIGN_BASED[name]
    design = sample(PROBLEM, **sample_kwargs, seed=0)
    result = _irregular(analyze(design, _ragged_y(design.n_runs), **analyze_kwargs))
    assert list(result.channels) == list(OUTPUT_NAMES)
