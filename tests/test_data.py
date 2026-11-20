"""Tests der Datenaufbereitung auf kleinen, von Hand gebauten Spielen."""

import numpy as np
import pandas as pd

from nfl import data


def make_raw(rows):
    """Minimaler Rohdatenrahmen: Standardwerte für alle Spalten, überschrieben durch `rows`."""
    base = {
        "season": 2024, "season_type": "REG", "game_id": "G1", "home_team": "HOM",
        "away_team": "AWY", "result": 7, "spread_line": 3.0, "posteam": "HOM", "defteam": "AWY",
        "qtr": 1, "game_half": "Half1", "down": 1, "ydstogo": 10, "yardline_100": 75,
        "goal_to_go": 0, "score_differential": 0, "game_seconds_remaining": 3600,
        "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3, "play_type": "pass",
        "desc": "", "ep": 0.5, "epa": 0.0, "penalty": 0, "qb_dropback": 1, "qb_scramble": 0,
        "kickoff_attempt": 0, "safety": 0, "touchdown": 0, "fixed_drive": 1,
        "fixed_drive_result": "Punt",
    }
    frame = pd.DataFrame([{**base, **row} for row in rows])
    frame["play_id"] = np.arange(1, len(frame) + 1)
    for column in data.RAW_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame


def kickoff(**row):
    return {"play_type": "kickoff", "kickoff_attempt": 1, "down": np.nan, "posteam": "AWY",
            "defteam": "HOM", "desc": "K kicks 65 yards, Touchback.", "touchback": 1, **row}


def prepared(rows):
    return data.add_next_situation(data.add_game_context(make_raw(rows)))


def test_tie_leaves_posteam_win_missing():
    raw = data.add_game_context(make_raw([{"result": 0}, {"result": 7}, {"result": 7, "posteam": "AWY"}]))
    assert np.isnan(raw.posteam_win[0])
    assert raw.posteam_win[1] == 1.0  # Heimteam hat den Ball und gewinnt
    assert raw.posteam_win[2] == 0.0  # Auswärtsteam hat den Ball und verliert


def test_spread_from_possession_view():
    raw = data.add_game_context(make_raw([{"posteam": "HOM"}, {"posteam": "AWY"}]))
    assert list(raw.posteam_spread) == [3.0, -3.0]


def test_next_situation_skips_rows_without_state():
    raw = prepared([{}, {"play_type": None, "down": np.nan}, {"yardline_100": 60}])
    assert raw.next_yardline_100[0] == 60  # Timeout-Zeile ohne Zustand wird übersprungen


def test_next_situation_does_not_cross_half():
    raw = prepared([{"game_half": "Half1", "qtr": 2}, {"game_half": "Half2", "qtr": 3}])
    assert not raw.next_exists[0]
    assert np.isnan(raw.next_yardline_100[0])


def test_next_situation_does_not_cross_game():
    raw = prepared([{"game_id": "G1"}, {"game_id": "G2"}])
    assert not raw.next_exists[0]


def test_onside_kick_is_invalid():
    raw = prepared([kickoff(desc="K kicks onside 12 yards, recovered by HOM.", touchback=0), {}])
    kicks = data.build_kickoffs(raw)
    assert kicks.onside[0] and not kicks.valid[0]


def test_kick_after_safety_is_invalid():
    raw = prepared([{"safety": 1}, {"play_type": None, "down": np.nan}, kickoff(posteam="HOM"), {}])
    kicks = data.build_kickoffs(raw)
    assert kicks.safety_kick[0] and not kicks.valid[0]


def test_repeated_kick_is_invalid_but_its_retake_is_valid():
    raw = prepared([kickoff(), kickoff(), {"posteam": "AWY", "yardline_100": 70}])
    kicks = data.build_kickoffs(raw)
    assert list(kicks.repeated) == [True, False]
    assert kicks.start_own_yards[1] == 30


def test_overtime_kickoff_is_excluded():
    raw = prepared([kickoff(qtr=5, game_half="Overtime"), {"qtr": 5, "game_half": "Overtime"}])
    assert data.build_kickoffs(raw).empty


def test_one_red_zone_entry_per_drive():
    raw = prepared([{"yardline_100": 30}, {"yardline_100": 18}, {"yardline_100": 9},
                    {"yardline_100": 15, "fixed_drive": 2, "fixed_drive_result": "Touchdown"}])
    entries = data.red_zone_entries(data.build_plays(raw))
    assert list(entries.yardline_100) == [18, 15]
    assert list(entries.touchdown_drive) == [0, 1]


def test_early_downs_scramble_variant():
    raw = prepared([{"play_type": "run", "qb_dropback": 1, "qb_scramble": 1},
                    {"play_type": "run", "qb_dropback": 0}, {"penalty": 1}])
    early = data.early_downs(data.build_plays(raw))
    assert len(early) == 2  # Strafe fällt heraus
    assert list(early.dropback) == [1, 0]
    assert list(early.dropback_no_scramble) == [0, 0]
