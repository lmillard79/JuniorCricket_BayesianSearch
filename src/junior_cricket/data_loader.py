"""Data loading and aggregation for the junior cricket model.

Handles three data sources and merges them into the per-player,
per-period aggregates consumed by the PyMC model:

1. Manual scorebook CSVs (paper-scored games; the primary source
   for BNJCA U10/U11 where e-scoring is optional).
2. PlayHQ GraphQL fixture/team exports (schedule, results and,
   where e-scored, per-player statistics).
3. A player registry CSV with dates of birth (drives the
   relative-age effect) and wicketkeeper flags (drive bowling
   allocation tiers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

BATTING_COLUMNS = [
    "date",
    "opponent",
    "player_name",
    "balls_faced",
    "runs_scored",
    "fours",
    "sixes",
    "dismissed",
    "retired",
    "source",
]

BOWLING_COLUMNS = [
    "date",
    "opponent",
    "player_name",
    "balls_bowled",
    "runs_conceded",
    "wickets",
    "wides",
    "no_balls",
    "source",
]

REGISTRY_COLUMNS = [
    "player_name",
    "date_of_birth",
    "is_wicketkeeper",
]


class DataValidationError(ValueError):
    """Raised when an input CSV does not match the expected schema."""


def load_batting_csv(path: Path) -> pd.DataFrame:
    """Load and validate a manual batting scorebook CSV.

    Args:
        path: Path to a CSV matching the batting template.

    Returns:
        DataFrame with validated and typed batting records.

    Raises:
        DataValidationError: On missing files, missing columns or
            negative numeric entries.
    """
    return _load_and_validate(path, BATTING_COLUMNS, "batting")


def load_bowling_csv(path: Path) -> pd.DataFrame:
    """Load and validate a manual bowling scorebook CSV.

    Args:
        path: Path to a CSV matching the bowling template.

    Returns:
        DataFrame with validated and typed bowling records.

    Raises:
        DataValidationError: On missing files, missing columns or
            negative numeric entries.
    """
    return _load_and_validate(path, BOWLING_COLUMNS, "bowling")


def load_player_registry(path: Path) -> pd.DataFrame:
    """Load and validate the player registry CSV.

    Args:
        path: Path to a CSV matching the registry template.

    Returns:
        DataFrame with player names, dates of birth parsed to
        monthly ages, and wicketkeeper flags.

    Raises:
        DataValidationError: On missing files, columns or bad dates.
    """
    if not path.exists():
        raise DataValidationError(f"Player registry not found: {path}")

    frame = pd.read_csv(path)
    missing = set(REGISTRY_COLUMNS) - set(frame.columns)
    if missing:
        raise DataValidationError(
            f"Player registry {path} missing columns: {sorted(missing)}"
        )

    frame["player_name"] = frame["player_name"].str.strip()
    frame["date_of_birth"] = pd.to_datetime(
        frame["date_of_birth"], errors="coerce"
    )
    if frame["date_of_birth"].isna().any():
        bad = frame.loc[frame["date_of_birth"].isna(), "player_name"]
        raise DataValidationError(
            f"Unparseable date_of_birth for players: {list(bad)}"
        )
    keeper = frame["is_wicketkeeper"].astype(str).str.lower()
    frame["is_wicketkeeper"] = keeper.isin(["yes", "y", "true", "1"])
    return frame


def _load_and_validate(
    path: Path, columns: List[str], label: str
) -> pd.DataFrame:
    """Shared loader/validator for scorebook CSVs.

    Args:
        path: CSV file to load.
        columns: Required column names.
        label: Human-readable label for error messages.

    Returns:
        Validated DataFrame with a parsed ``date`` column.

    Raises:
        DataValidationError: On schema or value violations.
    """
    if not path.exists():
        raise DataValidationError(f"{label} CSV not found: {path}")

    frame = pd.read_csv(path)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise DataValidationError(
            f"{label} CSV {path} missing columns: {sorted(missing)}"
        )

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise DataValidationError(
            f"{label} CSV {path} has unparseable dates in the date column"
        )

    numeric = [c for c in columns if c not in ("date", "opponent",
                                               "player_name", "source")]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise DataValidationError(
                f"{label} CSV {path} has non-numeric values in {column}"
            )
        if (frame[column] < 0).any():
            raise DataValidationError(
                f"{label} CSV {path} has negative values in {column}"
            )

    frame["player_name"] = frame["player_name"].str.strip()
    frame["source"] = frame["source"].fillna("manual").str.lower()
    return frame


def define_periods(
    period_bounds: List[Tuple[str, str, str]]
) -> pd.DataFrame:
    """Build a period lookup table from (label, start, end) tuples.

    Args:
        period_bounds: Ordered list of period definitions, oldest
            first. Each item is (label, start_date, end_date) with
            ISO date strings; end dates are inclusive.

    Returns:
        DataFrame with columns period, start, end.
    """
    return pd.DataFrame(
        period_bounds, columns=["period", "start", "end"]
    ).astype({"start": "datetime64[ns]", "end": "datetime64[ns]"})


DEFAULT_PERIODS = define_periods(
    [
        ("2024-25 U10 season", "2024-10-01", "2025-03-31"),
        ("2025-26 U10 pre-Xmas", "2025-10-01", "2025-12-31"),
        ("2025-26 U10 post-Xmas", "2026-01-01", "2026-09-30"),
    ]
)


def _assign_periods(
    frame: pd.DataFrame, periods: pd.DataFrame
) -> pd.DataFrame:
    """Attach a period label to each row based on its date.

    Args:
        frame: Scorebook DataFrame with a parsed ``date`` column.
        periods: Period table from ``define_periods``.

    Returns:
        Copy of ``frame`` with ``period`` and ``period_idx`` columns.
    """
    frame = frame.copy()
    frame["period"] = ""
    frame["period_idx"] = -1
    for idx, row in periods.iterrows():
        mask = (frame["date"] >= row["start"]) & (frame["date"] <= row["end"])
        frame.loc[mask, "period"] = row["period"]
        frame.loc[mask, "period_idx"] = idx
    return frame


def aggregate_batting(
    batting: pd.DataFrame, periods: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Aggregate batting records to player x period model inputs.

    Computes the quantities the Beta-Binomial and Gamma-Poisson
    likelihoods need: balls faced and dismissals (hazard), balls
    faced and boundaries (boundary rate), and non-boundary runs
    against non-boundary balls (scoring rate).

    Args:
        batting: Validated batting records.
        periods: Period table; defaults to ``DEFAULT_PERIODS``.

    Returns:
        DataFrame indexed by (player_name, period_idx) with columns
        balls_faced, outs, boundaries, non_boundary_runs and
        non_boundary_balls. Rows with zero balls faced are dropped.
    """
    periods = DEFAULT_PERIODS if periods is None else periods
    tagged = _assign_periods(batting, periods)
    tagged = tagged[tagged["period_idx"] >= 0]

    tagged["boundaries"] = tagged["fours"] + tagged["sixes"]
    tagged["boundary_runs"] = (
        4 * tagged["fours"] + 6 * tagged["sixes"]
    )
    tagged["non_boundary_runs"] = (
        tagged["runs_scored"] - tagged["boundary_runs"]
    ).clip(lower=0)
    tagged["non_boundary_balls"] = (
        tagged["balls_faced"] - tagged["boundaries"]
    ).clip(lower=0)

    grouped = tagged.groupby(["player_name", "period_idx"], sort=True).agg(
        balls_faced=("balls_faced", "sum"),
        outs=("dismissed", "sum"),
        boundaries=("boundaries", "sum"),
        non_boundary_runs=("non_boundary_runs", "sum"),
        non_boundary_balls=("non_boundary_balls", "sum"),
    )
    return grouped[grouped["balls_faced"] > 0].reset_index()


def aggregate_bowling(
    bowling: pd.DataFrame, periods: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Aggregate bowling records to player x period model inputs.

    Args:
        bowling: Validated bowling records.
        periods: Period table; defaults to ``DEFAULT_PERIODS``.

    Returns:
        DataFrame indexed by (player_name, period_idx) with columns
        deliveries, fair_balls, wickets, runs_off_bat and extras.
        Rows with zero deliveries are dropped.
    """
    periods = DEFAULT_PERIODS if periods is None else periods
    tagged = _assign_periods(bowling, periods)
    tagged = tagged[tagged["period_idx"] >= 0]

    tagged["extras"] = tagged["wides"] + tagged["no_balls"]
    tagged["fair_balls"] = (
        tagged["balls_bowled"] - tagged["extras"]
    ).clip(lower=0)
    tagged["runs_off_bat"] = (
        tagged["runs_conceded"] - tagged["extras"]
    ).clip(lower=0)

    grouped = tagged.groupby(["player_name", "period_idx"], sort=True).agg(
        deliveries=("balls_bowled", "sum"),
        fair_balls=("fair_balls", "sum"),
        wickets=("wickets", "sum"),
        runs_off_bat=("runs_off_bat", "sum"),
        extras=("extras", "sum"),
    )
    return grouped[grouped["deliveries"] > 0].reset_index()


def player_months_old(
    registry: pd.DataFrame, as_of: str = "2026-10-10"
) -> Dict[str, float]:
    """Compute each player's age in months at the season start.

    Args:
        registry: Player registry DataFrame.
        as_of: ISO date at which ages are measured; defaults to
            the first round of the 2026/27 U11 season.

    Returns:
        Mapping from player name to age in months.
    """
    as_of_ts = pd.Timestamp(as_of)
    ages: Dict[str, float] = {}
    for _, row in registry.iterrows():
        delta = as_of_ts - row["date_of_birth"]
        ages[row["player_name"]] = delta.days / 30.4375
    return ages
