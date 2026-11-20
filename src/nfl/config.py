"""Zentrale Konfiguration: Pfade, Saisonblöcke, Merkmale und Seed.

Alle anderen Module beziehen Pfade und Konstanten von hier, damit Datensplit und
Merkmalsliste an genau einer Stelle definiert sind.
"""

from pathlib import Path

# --- Pfade (relativ zum Projektordner, unabhängig vom Arbeitsverzeichnis) ---------
ROOT = Path(__file__).resolve().parents[2]  # src/nfl/config.py -> Projektordner

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
BOOTSTRAP_DIR = RESULTS_DIR / "bootstrap"

# --- Zeitliche Aufteilung -----------------------------------------------------------
# Drei disjunkte, zeitlich geordnete Blöcke: Anpassung -> Modellwahl/Kalibrierung ->
# einmalige Bewertung. Keine zufällige Aufteilung, damit kein Wissen aus späteren
# Saisons in frühere Modelle fließt.
TRAIN_SEASONS = tuple(range(2014, 2022))  # 2014–2021: Modelle anpassen
CALIBRATION_SEASONS = (2022, 2023)  # Modellwahl, Kalibrierung
EVALUATION_SEASONS = (2024, 2025)  # berichtete Ergebnisse, nur einmal ansehen
ALL_SEASONS = TRAIN_SEASONS + CALIBRATION_SEASONS + EVALUATION_SEASONS

KICKOFF_SEASONS = tuple(range(2018, 2026))  # Fallstudie 1: 2018–2025

# --- Merkmale -----------------------------------------------------------------------
# Situation vor dem Snap. Reihenfolge ist fest, weil Modelle Spalten positionsweise sehen.
# Liste, kein Tupel: df[("a", "b")] wäre in pandas ein einzelner MultiIndex-Schlüssel.
FEATURES = [
    "yardline_100",
    "down",
    "ydstogo",
    "goal_to_go",
    "score_differential",  # Ballbesitzer minus Gegner
    "game_seconds_remaining",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]
SPREAD = "posteam_spread"  # Teamstärke, nur in Sensitivitätsanalysen

SEED = 20261111

# --- Wiederholungen ---------------------------------------------------------------------
BOOTSTRAP_REPS = 1000  # Spiel-Bootstrap über feste Vorhersagen (Intervalle)
REFIT_REPS = 100  # vollständige Neuanpassungen: Red Zone, Run/Pass, Simulationen
FOURTH_DOWN_REFITS = 200  # Fourth Down braucht mehr: "sicher" hängt an den Intervallgrenzen


def split_of(season: int) -> str:
    """Ordnet eine Saison ihrem Zeitblock zu: "train", "calibration" oder "evaluation"."""
    if season in TRAIN_SEASONS:
        return "train"
    if season in CALIBRATION_SEASONS:
        return "calibration"
    if season in EVALUATION_SEASONS:
        return "evaluation"
    raise ValueError(f"Saison {season} liegt außerhalb von {ALL_SEASONS[0]}–{ALL_SEASONS[-1]}.")


def kickoff_period(season: int) -> str:
    """Regelperiode für die Kickoff-Fallstudie: "2018-2023", "2024" oder "2025"."""
    if season not in KICKOFF_SEASONS:
        raise ValueError(
            f"Saison {season} liegt außerhalb von {KICKOFF_SEASONS[0]}–{KICKOFF_SEASONS[-1]}."
        )
    if season <= 2023:
        return "2018-2023"
    return str(season)
