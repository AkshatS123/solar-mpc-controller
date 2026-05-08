"""Tests for the synthetic noise generator."""

import numpy as np

from solar_mpc.noise import NoiseSpec, generate_scenarios, realize_trace


def test_generate_scenarios_shapes_and_seed_reproducibility() -> None:
    rng_input = np.random.default_rng(0)
    s = rng_input.uniform(0, 6, 24)
    load = rng_input.uniform(0.5, 2.0, 24)
    p = np.full(24, 0.30)

    a = generate_scenarios(s, load, p, n_scenarios=10, seed=42)
    b = generate_scenarios(s, load, p, n_scenarios=10, seed=42)

    np.testing.assert_array_equal(a.solar_kw_scenarios, b.solar_kw_scenarios)
    np.testing.assert_array_equal(a.load_kw_scenarios, b.load_kw_scenarios)
    assert a.n_scenarios == 10 and a.horizon == 24


def test_clipping_keeps_solar_in_zero_to_three_x_point() -> None:
    s = np.array([2.0, 4.0, 6.0])
    load = np.array([1.0, 1.0, 1.0])
    p = np.full(3, 0.30)
    bundle = generate_scenarios(s, load, p, n_scenarios=200, seed=7)
    assert (bundle.solar_kw_scenarios >= 0).all()
    assert (bundle.solar_kw_scenarios <= 3.0 * s[None, :] + 1e-6).all()


def test_load_scenarios_are_nonnegative() -> None:
    s = np.full(6, 3.0)
    load = np.full(6, 0.5)
    p = np.full(6, 0.30)
    bundle = generate_scenarios(s, load, p, n_scenarios=100, seed=11)
    assert (bundle.load_kw_scenarios >= 0).all()


def test_zero_solar_stays_zero_under_multiplicative_noise() -> None:
    s = np.array([0.0, 0.0, 5.0, 5.0])
    load = np.full(4, 1.0)
    p = np.full(4, 0.30)
    bundle = generate_scenarios(s, load, p, n_scenarios=50, seed=3)
    np.testing.assert_array_equal(bundle.solar_kw_scenarios[:, :2], 0.0)


def test_realize_trace_uses_independent_seed_from_scenarios() -> None:
    s = np.full(10, 4.0)
    load = np.full(10, 1.0)
    realized_a = realize_trace(s, load, seed=1)
    realized_b = realize_trace(s, load, seed=2)
    assert not np.allclose(realized_a[0], realized_b[0])


def test_higher_sigma_widens_solar_distribution() -> None:
    s = np.full(96, 4.0)
    load = np.full(96, 1.0)
    p = np.full(96, 0.30)
    low = generate_scenarios(s, load, p, n_scenarios=200, seed=5, noise_spec=NoiseSpec(0.1, 0.1))
    high = generate_scenarios(s, load, p, n_scenarios=200, seed=5, noise_spec=NoiseSpec(0.5, 0.1))
    assert high.solar_kw_scenarios.std() > low.solar_kw_scenarios.std()
