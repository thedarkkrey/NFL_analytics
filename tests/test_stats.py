"""Tests der Statistik-Werkzeuge."""

import numpy as np
import pandas as pd
import pytest

from nfl import stats


def games_frame():
    """Zwei Saisons: 2022 mit drei Spielen, 2023 mit zwei Spielen, je zwei Zeilen pro Spiel."""
    games = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]
    seasons = [2022] * 6 + [2023] * 4
    return pd.DataFrame({"game_id": games, "season": seasons, "x": np.arange(10.0)})


def test_brier_perfect_and_worst():
    y = np.array([0, 1, 1, 0])
    assert stats.brier(y, y).sum() == 0
    assert stats.brier(y, 1 - y).mean() == 1


def test_log_loss_is_finite_at_certainty():
    loss = stats.log_loss([1, 0], [0.0, 1.0])
    # Clipping auf 1e-8 begrenzt den Verlust auf -ln(1e-8) ≈ 18,4
    assert np.all(np.isfinite(loss)) and loss == pytest.approx(-np.log(1e-8), rel=1e-6)
    assert stats.log_loss([1], [0.5])[0] == pytest.approx(np.log(2))


def test_game_counts_resample_within_season():
    frame = games_frame()
    rng = np.random.default_rng(1)
    for _ in range(200):
        counts = pd.Series(stats.game_counts(frame, rng), index=frame.game_id)
        per_game = counts.groupby(level=0).first()
        # Zeilen desselben Spiels tragen dasselbe Gewicht
        assert (counts.groupby(level=0).nunique() == 1).all()
        # Jede Saison behält ihre Spielanzahl: Ziehungen bleiben in der Saison
        assert per_game[["A", "B", "C"]].sum() == 3
        assert per_game[["D", "E"]].sum() == 2


def test_game_counts_independent_of_row_order():
    frame = games_frame()
    shuffled = frame.sample(frac=1, random_state=0)
    a = stats.game_counts(frame, np.random.default_rng(5))
    b = stats.game_counts(shuffled, np.random.default_rng(5))
    assert np.array_equal(pd.Series(a, index=frame.index).loc[shuffled.index].to_numpy(), b)


def test_bootstrap_means_reproducible_and_covers_estimate():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"game_id": np.repeat(np.arange(200), 5), "season": 2024,
                          "v": rng.normal(size=1000)})
    first = stats.bootstrap_means(frame, "v", reps=200, seed=3)
    second = stats.bootstrap_means(frame, "v", reps=200, seed=3)
    pd.testing.assert_frame_equal(first, second)
    row = first.loc["v"]
    assert row.lower < row.estimate < row.upper


def test_bootstrap_means_ignores_missing_values():
    frame = games_frame()
    values = {"half": np.where(frame.x < 5, frame.x, np.nan)}
    result = stats.bootstrap_means(frame, values, reps=50, seed=1)
    assert result.loc["half", "estimate"] == pytest.approx(2.0)
    assert np.isfinite(result.loc["half", ["lower", "upper"]]).all()


def test_bootstrap_draws_are_paired_across_values():
    frame = games_frame()
    draws = stats.bootstrap_draws(frame, {"a": frame.x, "b": frame.x + 1}, reps=100)
    assert np.allclose(draws.b - draws.a, 1)  # gleiche Resamples -> Differenz exakt konstant
    lower, upper = stats.interval(draws.a)
    assert lower <= frame.x.mean() <= upper


def test_constant_values_give_degenerate_interval():
    frame = games_frame().assign(c=1.0)
    row = stats.bootstrap_means(frame, "c", reps=50).loc["c"]
    assert row.lower == row.estimate == row.upper == 1.0


def test_calibration_barely_changes_calibrated_prediction():
    rng = np.random.default_rng(42)
    p = rng.uniform(0.05, 0.95, 50_000)
    y = rng.uniform(size=p.size) < p
    model = stats.SigmoidCalibration().fit(p, y)
    assert np.abs(model.predict(p) - p).max() < 0.02
    slope, intercept = stats.calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1, abs=0.05) and intercept == pytest.approx(0, abs=0.05)


def test_calibration_repairs_overconfident_prediction():
    rng = np.random.default_rng(7)
    true = rng.uniform(0.1, 0.9, 50_000)
    y = rng.uniform(size=true.size) < true
    overconfident = 1 / (1 + np.exp(-2 * stats.logit(true)))  # Logits verdoppelt
    slope, _ = stats.calibration_slope_intercept(y, overconfident)
    assert slope == pytest.approx(0.5, abs=0.05)
    repaired = stats.SigmoidCalibration().fit(overconfident, y).predict(overconfident)
    assert stats.brier(y, repaired).mean() < stats.brier(y, overconfident).mean()


def test_calibration_weights_equal_duplicated_rows():
    p = np.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7])
    y = np.array([0, 1, 1, 1, 0, 0])
    w = np.array([1, 2, 1, 3, 1, 2])
    weighted = stats.SigmoidCalibration().fit(p, y, w)
    duplicated = stats.SigmoidCalibration().fit(np.repeat(p, w), np.repeat(y, w))
    assert weighted.slope == pytest.approx(duplicated.slope, rel=1e-4)


def test_calibration_table_equal_width_bins():
    p = np.array([0.05, 0.15, 0.12, 0.95, 1.0])
    y = np.array([0, 1, 0, 1, 1])
    table = stats.calibration_table(y, p, bins=10)
    assert list(table.bin) == [0, 1, 9]  # leere Klassen fallen weg, p = 1 in Klasse 9
    assert list(table.n) == [1, 2, 2]
    assert table.loc[1, "observed_rate"] == 0.5
    assert table.loc[1, "lower"] == pytest.approx(0.1)
