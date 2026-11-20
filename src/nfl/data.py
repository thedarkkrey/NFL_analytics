"""Datenaufbereitung: Rohdaten -> plays.parquet, kickoffs.parquet und vier Populationen.

Ablauf (siehe prepare()):
    load_raw -> add_game_context -> add_next_situation -> build_plays / build_kickoffs
Die Filter der Fallstudien (valid_kickoffs, fourth_down_decisions, red_zone_entries,
early_downs) arbeiten auf den gespeicherten Tabellen.
"""

import hashlib

import numpy as np
import pandas as pd

from nfl import config

RAW_COLUMNS = [
    # Spiel
    "season", "season_type", "game_id", "play_id", "home_team", "away_team", "result",
    "spread_line", "home_opening_kickoff",
    # Situation vor dem Snap
    "posteam", "defteam", "qtr", "game_half", "down", "ydstogo", "yardline_100", "goal_to_go",
    "score_differential", "game_seconds_remaining", "half_seconds_remaining",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "total_home_score", "total_away_score",
    # Spielzug und Ergebnis
    "play_type", "desc", "ep", "epa", "yards_gained", "penalty", "qb_dropback", "qb_scramble",
    "sack", "rush_attempt", "fourth_down_converted", "fourth_down_failed", "touchdown", "td_team", "safety", "fumble_lost", "field_goal_result", "kick_distance",
    "fixed_drive", "fixed_drive_result",
    # Kickoff
    "kickoff_attempt", "touchback", "kickoff_fair_catch", "kickoff_downed", "kickoff_out_of_bounds",
    "kickoff_returner_player_id", "return_touchdown", "own_kickoff_recovery", "return_yards",
]

SCRIMMAGE_TYPES = ["pass", "run", "punt", "field_goal"]
# Spielzugtypen, deren Zeile eine verwertbare Spielsituation beschreibt (für add_next_situation).
STATE_TYPES = SCRIMMAGE_TYPES + ["qb_kneel", "qb_spike", "no_play"]

PLAYS_FILE = config.PROCESSED_DIR / "plays.parquet"
KICKOFFS_FILE = config.PROCESSED_DIR / "kickoffs.parquet"


def _flag(series: pd.Series) -> pd.Series:
    """0/1-Spalte mit fehlenden Werten -> bool (fehlend zählt als 0)."""
    return series.fillna(0).eq(1)


# --- Laden und Anreichern -------------------------------------------------------------

def load_raw(seasons=config.ALL_SEASONS) -> pd.DataFrame:
    """Benötigte Spalten aller Saisons, nur reguläre Saison, in Quellreihenfolge."""
    frames = [pd.read_parquet(config.RAW_DIR / f"play_by_play_{s}.parquet", columns=RAW_COLUMNS)
              for s in seasons]
    raw = pd.concat(frames, ignore_index=True)
    return raw.loc[raw.season_type.eq("REG")].reset_index(drop=True)


def add_game_context(raw: pd.DataFrame) -> pd.DataFrame:
    """Spielausgang und Wettlinie aus Sicht des Ballbesitzers, Zeitblock je Saison."""
    out = raw.copy()
    home = out.posteam.eq(out.home_team)
    away = out.posteam.eq(out.away_team)
    # result = Heimpunkte - Auswärtspunkte; aus Sicht des Ballbesitzers ggf. umdrehen.
    margin = out.result.where(home, -out.result).where(home | away)
    # Unentschieden (margin == 0) und Zeilen ohne Ballbesitzer bleiben fehlend.
    out["posteam_win"] = (margin > 0).astype(float).where(margin.notna() & margin.ne(0))
    # spread_line > 0: Heimteam favorisiert. posteam_spread > 0: Ballbesitzer favorisiert.
    out["posteam_spread"] = out.spread_line.where(home, -out.spread_line).where(home | away)
    out["split"] = out.season.map(config.split_of)
    return out


def add_next_situation(raw: pd.DataFrame) -> pd.DataFrame:
    """Hängt an jede Zeile die nächste verwertbare Situation im selben Spiel und derselben Halbzeit.

    Verwertbar ist ein Kickoff oder ein gültiger Zustand (down und yardline_100 vorhanden,
    play_type in STATE_TYPES). Erwartet die Quellreihenfolge (Spiele zusammenhängend).
    """
    out = raw.copy()
    n = len(out)
    is_kickoff = _flag(out.kickoff_attempt)
    is_state = out.down.notna() & out.yardline_100.notna() & out.play_type.isin(STATE_TYPES)
    candidate = (is_kickoff | is_state).to_numpy()

    # Position der nächsten Kandidatenzeile *nach* i: rückwärts laufendes Minimum.
    pos = np.where(candidate, np.arange(n), n)
    at_or_after = np.minimum.accumulate(pos[::-1])[::-1]
    nxt = np.append(at_or_after[1:], n)  # strikt nach i

    group = pd.Series(list(zip(out.game_id, out.game_half))).factorize()[0]
    group = np.append(group, -1)  # Sentinel für "keine nächste Zeile"
    exists = group[nxt] == group[:n]
    take = np.where(exists, nxt, 0)

    def pick(column):
        return out[column].to_numpy()[take]

    out["next_exists"] = exists
    out["next_is_kickoff"] = exists & is_kickoff.to_numpy()[take]
    out["next_posteam"] = pd.Series(pick("posteam"), index=out.index).where(exists)
    for column in ["yardline_100", "game_seconds_remaining", "ep"]:
        out[f"next_{column}"] = pd.Series(pick(column), index=out.index, dtype=float).where(exists)
    return out


# --- Haupttabellen --------------------------------------------------------------------

def build_plays(raw: pd.DataFrame) -> pd.DataFrame:
    """Spielzüge aus der regulären Spielzeit mit vollständiger Situation."""
    keep = (raw.qtr.le(4) & raw.play_type.isin(SCRIMMAGE_TYPES)
            & raw[config.FEATURES].notna().all(axis=1))
    plays = raw.loc[keep].copy()
    plays["possession_changed"] = plays.next_exists & plays.next_posteam.ne(plays.posteam)
    plays["seconds_used"] = (plays.game_seconds_remaining
                             - plays.next_game_seconds_remaining).where(plays.next_exists)
    return plays.reset_index(drop=True)


def build_kickoffs(raw: pd.DataFrame) -> pd.DataFrame:
    """Kickoffs der regulären Spielzeit mit Kategorie, Gültigkeit und Startposition.

    Auf nflverse-Kickoffzeilen ist posteam das empfangende Team.
    """
    is_kickoff = _flag(raw.kickoff_attempt)
    kicks = raw.loc[is_kickoff & raw.qtr.le(4)].copy()
    text = kicks.desc.fillna("")

    # Gründe für Ungültigkeit
    kicks["nullified"] = kicks.play_type.eq("no_play")
    kicks["onside"] = text.str.contains(r"\bonside\b", case=False)
    # Safety-Kick: Der zuletzt tatsächlich ausgeführte Spielzug vor dem Kick war ein Safety.
    executed = raw.play_type.notna() & raw.play_type.ne("no_play")
    last_safety = raw.safety.eq(1).where(executed).groupby(raw.game_id).ffill()
    before = last_safety.groupby(raw.game_id).shift(1).eq(True)
    kicks["safety_kick"] = (before.loc[kicks.index]
                            | text.str.contains(r"\bfree kick\b|\bsafety kick\b", case=False))
    # Wiederholt: Direkt danach folgt noch ein Kickoff, ohne dass dieser Kick gepunktet hat.
    scored = _flag(kicks.touchdown) | _flag(kicks.safety)
    kicks["repeated"] = kicks.next_is_kickoff & ~scored
    kicks["valid"] = ~(kicks.nullified | kicks.onside | kicks.safety_kick | kicks.repeated)

    # Kategorie: return hat Vorrang vor touchback vor other, Rest unknown.
    touchback = _flag(kicks.touchback) | text.str.contains(r"\btouchback\b", case=False)
    other = (_flag(kicks.kickoff_fair_catch) | _flag(kicks.kickoff_downed)
             | _flag(kicks.kickoff_out_of_bounds))
    has_returner = kicks.kickoff_returner_player_id.fillna("").ne("")
    return_text = text.str.contains(r"for (?:no gain|[+-]?\d+ yards?)", case=False)
    returned = (has_returner | return_text | _flag(kicks.return_touchdown)) & ~touchback & ~other
    kicks["category"] = np.select([returned, touchback, other], ["return", "touchback", "other"],
                                  default="unknown")

    # Startposition nur, wenn eindeutig: nächste Situation existiert, ist kein Kickoff,
    # der Empfänger hat den Ball, kein Return-TD und kein Ballverlust.
    kicks["return_touchdown"] = _flag(kicks.return_touchdown)
    kicks["ball_lost"] = _flag(kicks.fumble_lost) | _flag(kicks.own_kickoff_recovery)
    scrimmage_next = kicks.next_exists & ~kicks.next_is_kickoff
    kicks["start_known"] = (scrimmage_next & kicks.next_posteam.eq(kicks.posteam)
                            & ~kicks.return_touchdown & ~kicks.ball_lost)
    kicks["start_own_yards"] = (100 - kicks.next_yardline_100).where(kicks.start_known)
    kicks["next_ep"] = kicks.next_ep.where(scrimmage_next)
    kicks["next_opponent"] = scrimmage_next & kicks.next_posteam.ne(kicks.posteam)
    kicks["penalty"] = _flag(kicks.penalty)
    kicks["period"] = kicks.season.map(lambda s: config.kickoff_period(s)
                                       if s in config.KICKOFF_SEASONS else None)

    columns = ["game_id", "play_id", "season", "period", "posteam", "defteam", "category",
               "penalty", "return_touchdown", "ball_lost", "start_known", "start_own_yards",
               "next_ep", "next_opponent", "nullified", "onside", "safety_kick", "repeated", "valid"]
    return kicks[columns].reset_index(drop=True)


def prepare() -> None:
    """Baut plays.parquet, kickoffs.parquet und results/tables/populations.csv."""
    raw = add_next_situation(add_game_context(load_raw()))
    plays = build_plays(raw)
    kickoffs = build_kickoffs(raw)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    plays.to_parquet(PLAYS_FILE, index=False)
    kickoffs.to_parquet(KICKOFFS_FILE, index=False)
    population_summary(raw, plays, kickoffs).to_csv(config.TABLES_DIR / "populations.csv", index=False)
    # Prüfsummen belegen, auf welchem Datenstand die Ergebnisse beruhen.
    files = sorted(config.RAW_DIR.glob("*.parquet"))
    pd.DataFrame({"file": [f.name for f in files],
                  "sha256": [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]}
                 ).to_csv(config.TABLES_DIR / "raw_checksums.csv", index=False)


def load_plays() -> pd.DataFrame:
    return pd.read_parquet(PLAYS_FILE)


def load_kickoffs() -> pd.DataFrame:
    return pd.read_parquet(KICKOFFS_FILE)


# --- Populationen der Fallstudien -------------------------------------------------------

def valid_kickoffs(kickoffs: pd.DataFrame) -> pd.DataFrame:
    """Fallstudie 1: gültige Kickoffs 2018–2025."""
    keep = kickoffs.valid & kickoffs.season.isin(config.KICKOFF_SEASONS)
    return kickoffs.loc[keep].reset_index(drop=True)


def fourth_down_decisions(plays: pd.DataFrame) -> pd.DataFrame:
    """Fallstudie 2: 4. Versuch, 1–5 Yards zu gehen, 10–60 Yards vor der Endzone."""
    keep = (plays.down.eq(4) & plays.ydstogo.between(1, 5) & plays.yardline_100.between(10, 60))
    out = plays.loc[keep].copy()
    out["action"] = out.play_type.map({"pass": "go", "run": "go", "punt": "punt",
                                       "field_goal": "field_goal"})
    return out.reset_index(drop=True)


def red_zone_entries(plays: pd.DataFrame) -> pd.DataFrame:
    """Fallstudie 3: erster Spielzug je Drive innerhalb der 20-Yard-Linie."""
    inside = plays.loc[plays.yardline_100.le(20)]
    first = inside.drop_duplicates(["game_id", "posteam", "fixed_drive"], keep="first").copy()
    first["touchdown_drive"] = first.fixed_drive_result.eq("Touchdown").astype(int)
    return first.reset_index(drop=True)


def early_downs(plays: pd.DataFrame) -> pd.DataFrame:
    """Fallstudie 4: 1./2. Versuch, Dropback oder geplanter Lauf, ohne Strafe, EPA vorhanden."""
    dropback = _flag(plays.qb_dropback)
    designed_run = plays.play_type.eq("run") & ~dropback
    keep = (plays.down.isin([1, 2]) & (dropback | designed_run) & ~_flag(plays.penalty)
            & plays.epa.notna())
    out = plays.loc[keep].copy()
    out["dropback"] = dropback.loc[keep].astype(int)
    # Variante für Phase 8: Scrambles zählen als Lauf.
    out["dropback_no_scramble"] = (out.dropback.eq(1) & ~_flag(out.qb_scramble)).astype(int)
    return out.reset_index(drop=True)


def population_summary(raw: pd.DataFrame, plays: pd.DataFrame,
                       kickoffs: pd.DataFrame) -> pd.DataFrame:
    """Größe jeder Population je Zeitblock (bzw. Kickoff-Periode), Langformat."""
    rows = [{"population": "raw_rows", "group": "all", "n": len(raw), "games": raw.game_id.nunique()}]
    populations = {
        "win_model_plays": plays,
        "fourth_down_decisions": fourth_down_decisions(plays),
        "red_zone_entries": red_zone_entries(plays),
        "early_down_plays": early_downs(plays),
    }
    for name, frame in populations.items():
        for split in ["train", "calibration", "evaluation"]:
            part = frame.loc[frame.split.eq(split)]
            rows.append({"population": name, "group": split, "n": len(part),
                         "games": part.game_id.nunique()})
    kicks = valid_kickoffs(kickoffs)
    for period in ["2018-2023", "2024", "2025"]:
        part = kicks.loc[kicks.period.eq(period)]
        rows.append({"population": "valid_kickoffs", "group": period, "n": len(part),
                     "games": part.game_id.nunique()})
    return pd.DataFrame(rows)
