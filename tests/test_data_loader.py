"""Tests for scorebook CSV loading and aggregation."""

from pathlib import Path

import pandas as pd
import pytest

from junior_cricket.data_loader import (
    DataValidationError,
    aggregate_batting,
    aggregate_bowling,
    define_periods,
    load_batting_csv,
    load_bowling_csv,
    load_player_registry,
    player_months_old,
)

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "data" / "templates"


def make_batting_csv(tmp_path: Path) -> Path:
    """Write a small batting CSV spanning two periods."""
    rows = [
        # 2025-26 pre-Xmas period.
        "2025-11-01,Northside,Alfie,20,18,2,0,1,0,manual",
        "2025-11-08,Southbank,Alfie,24,10,1,0,0,1,manual",
        # 2026 post-Xmas period.
        "2026-02-07,Eastgrove,Alfie,30,26,3,0,1,0,manual",
        "2026-02-14,Westend,Bert,12,4,0,0,1,0,manual",
    ]
    path = tmp_path / "batting.csv"
    path.write_text(
        "date,opponent,player_name,balls_faced,runs_scored,fours,"
        "sixes,dismissed,retired,source\n" + "\n".join(rows) + "\n",
        encoding="ascii",
    )
    return path


def test_load_batting_csv(tmp_path: Path) -> None:
    """Loading parses dates and validates the schema."""
    frame = load_batting_csv(make_batting_csv(tmp_path))
    assert len(frame) == 4
    assert frame["date"].notna().all()


def test_load_batting_rejects_negative(tmp_path: Path) -> None:
    """Negative run counts are rejected."""
    path = tmp_path / "bad.csv"
    path.write_text(
        "date,opponent,player_name,balls_faced,runs_scored,fours,"
        "sixes,dismissed,retired,source\n"
        "2026-02-14,X,Alfie,12,-4,0,0,1,0,manual\n",
        encoding="ascii",
    )
    with pytest.raises(DataValidationError):
        load_batting_csv(path)


def test_load_batting_rejects_missing_column(tmp_path: Path) -> None:
    """A missing column is rejected with a clear message."""
    path = tmp_path / "bad.csv"
    path.write_text(
        "date,opponent,player_name,balls_faced\n"
        "2026-02-14,X,Alfie,12\n",
        encoding="ascii",
    )
    with pytest.raises(DataValidationError):
        load_batting_csv(path)


def test_aggregate_batting_periods(tmp_path: Path) -> None:
    """Aggregation splits records into the configured periods."""
    frame = load_batting_csv(make_batting_csv(tmp_path))
    agg = aggregate_batting(frame)
    alfie = agg[agg["player_name"] == "Alfie"].set_index("period_idx")
    # Pre-Xmas 2025: 44 balls, 1 out, 3 boundaries.
    row = alfie.loc[1]
    assert row["balls_faced"] == 44
    assert row["outs"] == 1
    assert row["boundaries"] == 3
    assert row["non_boundary_runs"] == 18 + 10 - 4 * 3
    # Post-Xmas 2026: 30 balls, 3 boundaries.
    row = alfie.loc[2]
    assert row["balls_faced"] == 30
    assert row["boundaries"] == 3


def test_aggregate_bowling_runs() -> None:
    """Bowling aggregation separates extras from fair-ball figures."""
    rows = pd.DataFrame(
        [
            {
                "date": "2026-02-07",
                "opponent": "Eastgrove",
                "player_name": "Alfie",
                "balls_bowled": 24,
                "runs_conceded": 21,
                "wickets": 2,
                "wides": 3,
                "no_balls": 1,
                "source": "manual",
            }
        ]
    )
    agg = aggregate_bowling(rows)
    row = agg.iloc[0]
    assert row["deliveries"] == 24
    assert row["extras"] == 4
    assert row["fair_balls"] == 20
    assert row["runs_off_bat"] == 17
    assert row["wickets"] == 2


def test_registry_months_old(tmp_path: Path) -> None:
    """The registry converts dates of birth to ages in months."""
    path = tmp_path / "registry.csv"
    path.write_text(
        "player_name,date_of_birth,is_wicketkeeper,notes\n"
        "Alfie,2015-06-15,no,\n"
        "Bert,2015-02-01,yes,\n",
        encoding="ascii",
    )
    registry = load_player_registry(path)
    ages = player_months_old(registry, as_of="2016-06-15")
    assert ages["Alfie"] == pytest.approx(12.0, abs=0.1)
    assert ages["Bert"] == pytest.approx(16.5, abs=0.6)
    assert registry.loc[
        registry["player_name"] == "Bert", "is_wicketkeeper"
    ].iloc[0]


def test_define_periods_ordering() -> None:
    """Periods are defined oldest-first with inclusive bounds."""
    periods = define_periods(
        [("p0", "2024-10-01", "2025-03-31"),
         ("p1", "2025-10-01", "2025-12-31")]
    )
    assert periods.iloc[0]["period"] == "p0"
    assert periods.iloc[1]["period"] == "p1"
