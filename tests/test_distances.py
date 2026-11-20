"""
Known-answer tests for the exact quantities and the two distances.
Run with:  python -m pytest -q
"""

import numpy as np
import pytest
from src import ground_truth
from src.scoring import (grid_distance, grid_histogram, mode_share_distance,mode_shares, total_variation)

#  the exact targets 
def test_exact_grid_masses_sum_to_one():
    masses, outside = ground_truth.exact_grid_masses_p0()
    assert masses.min() >= 0 and 0 <= outside < 1e-6  # p0 barely leaves [-3, 3]^2
    assert masses.sum() + outside == pytest.approx(1.0, abs=1e-12)


def test_numerical_masses_match_exact_cdf_masses():
    """Two independent routes to the same numbers: closed-form Gaussian CDFs
    versus numerical integration. If either is wrong they will disagree."""
    exact, _ = ground_truth.exact_grid_masses_p0()
    numeric, _ = ground_truth.numerical_grid_masses(ground_truth.density_p0)
    assert total_variation(exact, numeric) < 1e-3


def test_exact_mode_shares_p0_are_one_eighth():
    shares = ground_truth.exact_mode_shares(ground_truth.density_p0)
    assert np.allclose(shares, 1 / 8, atol=1e-4)


#  total variation itself 
def test_tv_basic_cases():
    assert total_variation([0.5, 0.5], [0.5, 0.5]) == 0.0          # identical
    assert total_variation([1, 0], [0, 1]) == 1.0                  # no overlap
    assert total_variation([0.7, 0.3], [0.4, 0.6]) == pytest.approx(0.3)  # by hand


#  grid distance 
def test_grid_distance_zero_when_target_is_the_histogram():
    rng = np.random.default_rng(0)
    x = ground_truth.sample_p0(5000, rng)
    frac, out = grid_histogram(x)
    assert grid_distance(x, frac, out) == pytest.approx(0.0, abs=1e-12)


def test_grid_distance_all_samples_in_one_cell_vs_uniform():
    x = np.full((1000, 2), 0.05)  # every point in the same cell
    uniform = np.full((60, 60), 1 / 3600)
    assert grid_distance(x, uniform, 0.0) == pytest.approx(1 - 1 / 3600)


def test_grid_distance_counts_points_outside_the_grid():
    x = np.array([[0.05, 0.05]] * 50 + [[10.0, 10.0]] * 50)
    target = np.zeros((60, 60))
    frac, _ = grid_histogram(x[:1])
    target[frac > 0] = 1.0
    assert grid_distance(x, target, 0.0) == pytest.approx(0.5)


def test_grid_distance_one_mode_vs_p0_is_seven_eighths():
    rng = np.random.default_rng(1)
    x = ground_truth.CENTRES[0] + ground_truth.STD * rng.standard_normal((200_000, 2))
    masses, outside = ground_truth.exact_grid_masses_p0()
    assert grid_distance(x, masses, outside) == pytest.approx(7 / 8, abs=0.01)


#  mode-share distance 
def test_mode_shares_of_exact_centres():
    x = np.repeat(ground_truth.CENTRES, [3, 1, 1, 1, 1, 1, 1, 1], axis=0)  # 3 copies of centre 0
    assert np.allclose(mode_shares(x), [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


def test_mode_share_distance_one_mode_is_seven_eighths():
    x = ground_truth.CENTRES[[0]] + np.zeros((100, 2))
    assert mode_share_distance(x, ground_truth.EXACT_MODE_SHARES_P0) == pytest.approx(7 / 8)


def test_mode_share_distance_exact_samples_are_close():
    rng = np.random.default_rng(2)
    x = ground_truth.sample_p0(200_000, rng)
    assert mode_share_distance(x, ground_truth.EXACT_MODE_SHARES_P0) < 0.01
