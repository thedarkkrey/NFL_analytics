"""Validierung: Stimmen die einfachen Filter aus data.py mit den Referenzdateien überein?

Referenzdateien in data/reference/:
    red_zone_reviewed_entries.csv   von Hand geprüfte Red-Zone-Eintritte
                                    (game_id, play_id, season, split, drive_td_target, included)
    kickoffs_previous_pipeline.csv  Kickoffs der früheren, aufwendigeren Rekonstruktion
                                    (game_id, play_id, season, category, start_status,
                                     start_position_own_yards, eligible)

Checkpoint (ANLEITUNG.md, Phase 4):
    Red Zone: 20.383 gemeinsame Eintritte, 1 nur neu, 2 abweichende Labels.
    Kickoffs: alle 21.037 alten enthalten, 7 nur neu, Kategorien identisch.
"""

import pandas as pd

from nfl import config, data

RED_ZONE_REFERENCE = config.REFERENCE_DIR / "red_zone_reviewed_entries.csv"
KICKOFF_REFERENCE = config.REFERENCE_DIR / "kickoffs_previous_pipeline.csv"
VALIDATION_FILE = config.TABLES_DIR / "validation.csv"
KEYS = ["game_id", "play_id"]


def _agreement(new: pd.DataFrame, old: pd.DataFrame, new_label: str, old_label: str) -> dict:
    """Abgleich über KEYS: gemeinsam, nur neu, nur alt, abweichendes Label bei gemeinsamen."""
    for name, frame in [("neu", new), ("alt", old)]:
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Doppelte Schlüssel in der Tabelle '{name}'.")
    merged = _merge(new, old, new_label, old_label)
    common = merged.loc[merged._merge.eq("both")]
    return {
        "common": len(common),
        "only_new": int(merged._merge.eq("left_only").sum()),
        "only_reference": int(merged._merge.eq("right_only").sum()),
        "label_mismatch": int(common.label_new.ne(common.label_old).sum()),
    }


def _merge(new: pd.DataFrame, old: pd.DataFrame, new_label: str, old_label: str) -> pd.DataFrame:
    """Outer-Join über KEYS mit den Spalten label_new, label_old und _merge."""
    return (new[KEYS + [new_label]].rename(columns={new_label: "label_new"})
            .merge(old[KEYS + [old_label]].rename(columns={old_label: "label_old"}),
                   on=KEYS, how="outer", indicator=True))


def red_zone_agreement(entries: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Vergleicht die neuen Red-Zone-Eintritte mit den von Hand geprüften.

    entries:   Ergebnis von data.red_zone_entries(plays)
    reference: red_zone_reviewed_entries.csv (nur included == True)
    Abgleich über (game_id, play_id).

    Rückgabe: Anzahl gemeinsamer Eintritte, nur neu, nur alt, und wie viele
    gemeinsame Eintritte ein abweichendes Label haben (touchdown_drive vs. drive_td_target).
    """
    reviewed = reference.loc[reference.included.astype(bool)]
    return _agreement(entries, reviewed, "touchdown_drive", "drive_td_target")


def kickoff_agreement(kickoffs: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Vergleicht die neuen gültigen Kickoffs mit der früheren Rekonstruktion.

    kickoffs:  Ergebnis von data.valid_kickoffs(kickoffs)
    reference: kickoffs_previous_pipeline.csv (nur eligible == True)
    Abgleich über (game_id, play_id).

    Rückgabe: Anzahl gemeinsamer Kicks, nur neu, nur alt, und wie viele
    gemeinsame Kicks eine abweichende Kategorie haben.
    """
    eligible = reference.loc[reference.eligible.astype(bool)]
    return _agreement(kickoffs, eligible, "category", "category")


def main() -> None:
    """Lädt Tabellen und Referenzen, ruft beide Vergleiche auf und schreibt validation.csv.

    Eigener Vorschlag (nicht in der Anleitung benannt): ein Einstiegspunkt pro Modul,
    damit run_all.py in Phase 11 jeden Schritt einzeln aufrufen kann.
    """
    checks = {
        "red_zone": red_zone_agreement(data.red_zone_entries(data.load_plays()),
                                       pd.read_csv(RED_ZONE_REFERENCE)),
        "kickoffs": kickoff_agreement(data.valid_kickoffs(data.load_kickoffs()),
                                      pd.read_csv(KICKOFF_REFERENCE)),
    }
    rows = [{"check": check, "metric": metric, "n": n}
            for check, result in checks.items() for metric, n in result.items()]
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(VALIDATION_FILE, index=False)
