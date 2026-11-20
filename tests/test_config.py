import pytest

from nfl import config


def test_root_is_project_folder():
    assert (config.ROOT / "ANLEITUNG.md").exists()
    assert config.RAW_DIR == config.ROOT / "data" / "raw"


def test_splits_are_disjoint_and_complete():
    train, cal, ev = map(set, (config.TRAIN_SEASONS, config.CALIBRATION_SEASONS, config.EVALUATION_SEASONS))
    assert not (train & cal or train & ev or cal & ev)
    assert train | cal | ev == set(range(2014, 2026))
    # zeitlich geordnet: jeder Block liegt vollständig vor dem nächsten
    assert max(train) < min(cal) and max(cal) < min(ev)


@pytest.mark.parametrize(
    "season, expected",
    [(2014, "train"), (2021, "train"), (2022, "calibration"), (2023, "calibration"),
     (2024, "evaluation"), (2025, "evaluation")],
)
def test_split_of(season, expected):
    assert config.split_of(season) == expected


@pytest.mark.parametrize(
    "season, expected",
    [(2018, "2018-2023"), (2023, "2018-2023"), (2024, "2024"), (2025, "2025")],
)
def test_kickoff_period(season, expected):
    assert config.kickoff_period(season) == expected


@pytest.mark.parametrize("season", [2013, 2026])
def test_split_of_rejects_unknown_season(season):
    with pytest.raises(ValueError):
        config.split_of(season)


@pytest.mark.parametrize("season", [2017, 2026])
def test_kickoff_period_rejects_unknown_season(season):
    with pytest.raises(ValueError):
        config.kickoff_period(season)


def test_features():
    assert isinstance(config.FEATURES, list)  # pandas: df[FEATURES] wählt Spalten nur mit Liste
    assert len(config.FEATURES) == 8 == len(set(config.FEATURES))
    assert config.SPREAD not in config.FEATURES
    assert config.SEED == 20261111


def test_repetitions():
    assert config.BOOTSTRAP_REPS == 1000
    assert config.REFIT_REPS == 100
    assert config.FOURTH_DOWN_REFITS == 200

