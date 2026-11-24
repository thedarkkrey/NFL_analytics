"""Statistik-Werkzeuge: Bewertungsmaße, Spiel-Bootstrap und Kalibrierung.

Grundidee des Spiel-Bootstraps: Spielzüge desselben Spiels sind nicht unabhängig
(gleiche Teams, gleicher Spielverlauf, gleiches Wetter). Gezogen werden deshalb ganze
Spiele mit Zurücklegen, getrennt je Saison, damit jede Saison mit ihrer echten
Spielanzahl im Resample vertreten ist. Die Ziehungshäufigkeit eines Spiels wird zum
Gewicht aller seiner Zeilen.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nfl import config

# Clipping-Grenzen wie in der Referenz, damit Log-Loss und Kalibrierung vergleichbar bleiben.
LOSS_EPS = 1e-8  # Log-Loss: gegen log(0)
LOGIT_EPS = 1e-6  # Kalibrierung: begrenzt Logits auf ±13,8


# --- Bewertungsmaße (ein Wert pro Beobachtung) --------------------------------------------

def brier(y, p) -> np.ndarray:
    """Quadratischer Fehler (p - y)²; der Mittelwert ist der Brier-Score."""
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def log_loss(y, p) -> np.ndarray:
    """Negative Log-Likelihood je Beobachtung, p auf [LOSS_EPS, 1 - LOSS_EPS] begrenzt."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), LOSS_EPS, 1 - LOSS_EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p / (1 - p))


# --- Spiel-Bootstrap ----------------------------------------------------------------------

def _season_games(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Eindeutige Spiele in fester Reihenfolge (Saison, game_id) und ihre Saison."""
    games = frame[["season", "game_id"]].drop_duplicates().sort_values(["season", "game_id"])
    return games.game_id.to_numpy(), games.season.to_numpy()


def _draw_game_counts(seasons: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ziehungshäufigkeit je Spiel: je Saison so viele Ziehungen wie Spiele, mit Zurücklegen."""
    counts = np.zeros(len(seasons))
    for season in np.unique(seasons):
        members = np.flatnonzero(seasons == season)
        drawn = rng.integers(0, len(members), size=len(members))
        counts[members] = np.bincount(drawn, minlength=len(members))
    return counts


def game_counts(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Ein Spiel-Bootstrap-Resample: pro Zeile, wie oft ihr Spiel gezogen wurde."""
    games, seasons = _season_games(frame)
    counts = _draw_game_counts(seasons, rng)
    position = pd.Series(np.arange(len(games)), index=games)
    return counts[position.loc[frame.game_id].to_numpy()]


def _value_table(frame: pd.DataFrame, values) -> pd.DataFrame:
    """values: Spaltenname, Liste von Spaltennamen, dict {name: Werte je Zeile} oder DataFrame."""
    if isinstance(values, pd.DataFrame):
        return values.astype(float)
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, dict):
        values = {name: frame[name] for name in values}
    return pd.DataFrame({name: np.asarray(v, dtype=float) for name, v in values.items()},
                        index=frame.index)


def bootstrap_draws(frame: pd.DataFrame, values, reps: int = config.BOOTSTRAP_REPS,
                    seed: int = config.SEED) -> pd.DataFrame:
    """Spiel-Bootstrap-Mittelwerte: eine Zeile je Resample, eine Spalte je Größe.

    Fehlende Werte werden je Größe ignoriert (Zähler und Nenner). Alle Größen stammen aus
    denselben Resamples, Differenzen zwischen ihnen sind also gepaart.
    Ziehschema wie in der Referenz (je Saison eine Matrix reps × Spiele), damit die
    Intervalle bei gleichem Seed übereinstimmen.
    """
    table = _value_table(frame, values)
    games, seasons = _season_games(frame)
    # Summen und Anzahlen je Spiel: ein Resample ist dann nur noch eine Summe über Spiele.
    grouped = table.groupby(frame.game_id.to_numpy())
    sums = grouped.sum().reindex(games).to_numpy()  # sum() überspringt NaN
    sizes = grouped.count().reindex(games).to_numpy()

    rng = np.random.default_rng(seed)
    total_sums = np.zeros((reps, table.shape[1]))
    total_sizes = np.zeros((reps, table.shape[1]))
    for season in np.unique(seasons):
        members = np.flatnonzero(seasons == season)
        drawn = members[rng.integers(0, len(members), size=(reps, len(members)))]
        total_sums += sums[drawn].sum(axis=1)
        total_sizes += sizes[drawn].sum(axis=1)
    return pd.DataFrame(total_sums / total_sizes, columns=table.columns)


def interval(draws) -> tuple[float, float]:
    """95-%-Perzentilintervall aus Bootstrap-Ziehungen."""
    lower, upper = np.quantile(np.asarray(draws, dtype=float), [0.025, 0.975])
    return float(lower), float(upper)


def bootstrap_means(frame: pd.DataFrame, values, reps: int = config.BOOTSTRAP_REPS,
                    seed: int = config.SEED) -> pd.DataFrame:
    """Mittelwerte mit 95-%-Spiel-Bootstrap-Intervall (Perzentile 2,5 / 97,5)."""
    table = _value_table(frame, values)
    draws = bootstrap_draws(frame, table, reps, seed)
    return pd.DataFrame({
        "estimate": table.mean().to_numpy(),
        "lower": draws.quantile(0.025).to_numpy(),
        "upper": draws.quantile(0.975).to_numpy(),
    }, index=table.columns)


# --- Kalibrierung -------------------------------------------------------------------------

class SigmoidCalibration:
    """Platt-Skalierung: logistische Regression von y auf logit(p).

    Eine bereits kalibrierte Vorhersage ergibt Steigung ≈ 1 und Achsenabschnitt ≈ 0,
    die Abbildung ist dann nahezu die Identität.
    """

    def fit(self, p, y, weights=None) -> "SigmoidCalibration":
        # C = 1e6: praktisch unregularisiert, wie in der Referenz.
        self.model_ = LogisticRegression(C=1e6, max_iter=1000)
        self.model_.fit(logit(p).reshape(-1, 1), np.asarray(y), sample_weight=weights)
        return self

    def predict(self, p) -> np.ndarray:
        return self.model_.predict_proba(logit(p).reshape(-1, 1))[:, 1]

    @property
    def slope(self) -> float:
        return float(self.model_.coef_[0, 0])

    @property
    def intercept(self) -> float:
        return float(self.model_.intercept_[0])


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Gleich breite Klassen auf [0, 1]: Anzahl, mittlere Vorhersage, beobachtete Quote."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    # Klasse i umfasst [i/bins, (i+1)/bins); p = 1 kommt in die letzte Klasse.
    index = np.minimum((p * bins).astype(int), bins - 1)
    table = (pd.DataFrame({"bin": index, "p": p, "y": y})
             .groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                 observed_rate=("y", "mean"))
             .reset_index())
    table.insert(1, "lower", edges[table.bin])
    table.insert(2, "upper", edges[table.bin + 1])
    return table


def calibration_slope_intercept(y, p, weights=None) -> tuple[float, float]:
    """Kalibrierungssteigung und -achsenabschnitt (ideal 1 und 0)."""
    model = SigmoidCalibration().fit(p, y, weights)
    return model.slope, model.intercept
