"""Tests for the pool scouting module."""

from pathlib import Path

import pandas as pd
import pytest

from junior_cricket import scouting as S


def stat(**values):
    return [{"type": {"value": k}, "count": v} for k, v in values.items()]


def player(name, pid, batting=None, bowling=None, fielding=None, status="OUT"):
    entries = []
    if batting is not None:
        entries.append({"status": status, "displayOrder": 1, "statistics": stat(**batting)})
    if bowling is not None:
        entries.append({"status": "", "displayOrder": 1, "statistics": stat(**bowling)})
    if fielding is not None:
        entries.append({"status": "", "statistics": stat(**fielding)})
    return {"id": f"id-{name}", "profileID": pid, "name": name, "lineupOrder": 1,
            "periodStatistics": entries}


def card(home, away):
    return {"data": {"game": {"id": "g1", "status": "FINAL",
                              "statistics": {"home": {"players": home}, "away": {"players": away}}}}}


META = S.GameMeta("g1", "Round 1", "2026-10-10", "Home XI", "Away XI",
                  {"TOTAL_SCORE": 100, "TOTAL_OUTS": 5, "TOTAL_OVERS": 22},
                  {"TOTAL_SCORE": 80, "TOTAL_OUTS": 8, "TOTAL_OVERS": 17.1})


def result_side(score, outs, overs):
    return {"periods": [{"period": {"value": "FIRST_INNINGS"}, "statistics": [
        {"type": {"value": "TOTAL_SCORE"}, "count": score},
        {"type": {"value": "TOTAL_OUTS"}, "count": outs},
        {"type": {"value": "TOTAL_OVERS"}, "count": overs}]}]}


def fixture():
    def game(gid, status, home, away, result=None):
        return {"id": gid, "status": {"value": status}, "date": "2026-10-10", "home": {"name": home},
                "away": {"name": away}, "result": result}
    return {"discoverTeam": {"name": "Home XI"}, "discoverTeamFixture": [
        {"name": "Round 1", "fixture": {"games": [
            game("g1", "FINAL", "Home XI", "Away XI",
                 {"home": result_side(100, 5, 22), "away": result_side(80, 8, 17.1)})]}},
        {"name": "Round 2", "fixture": {"games": [game("g2", "UPCOMING", "Away XI", "Home XI")]}},
        {"name": "Round 3", "fixture": {"games": []}},
    ]}


def test_final_games_keeps_finished_games_with_both_totals() -> None:
    games = S.final_games(fixture())
    assert [g.game_id for g in games] == ["g1"]
    assert games[0].home_stats["TOTAL_SCORE"] == 100 and games[0].away_stats["TOTAL_OVERS"] == 17.1


def test_team_schedule_marks_venue_and_byes() -> None:
    rows = S.team_schedule(fixture(), "Home XI")
    assert [(r["round"], r["opponent"], r["venue"]) for r in rows] == [
        ("Round 1", "Away XI", "home"), ("Round 2", "Away XI", "away"), ("Round 3", "bye", "")]


def test_overs_are_read_as_overs_and_balls() -> None:
    assert S.overs_to_balls(22) == 132 and S.overs_to_balls(17.1) == 103 and S.overs_to_balls(0.5) == 5


def test_a_player_game_carries_batting_bowling_and_fielding() -> None:
    home = [
        player("A Batter", "pa", batting=dict(BALLS_FACED=10, TOTAL_RUNS=14, FOURS=2)),
        player("B Bowler", "pb", bowling=dict(OVERS=3, RUNS=12, WICKETS=1, WIDES=3, NO_BALLS=1),
               fielding=dict(TOTAL_CATCHES=1)),
        player("C Absent", "pc", batting=dict(BALLS_FACED=0), status="DID_NOT_BAT"),
        player("D Nameonly", "", batting=dict(BALLS_FACED=4, TOTAL_RUNS=1), status="NOT_OUT"),
    ]
    rows = {r["name"]: r for r in S.player_games(card(home, []), META)}
    a, b, d = rows["A Batter"], rows["B Bowler"], rows["D Nameonly"]
    assert (a["balls"], a["runs"], a["fours"], a["out"], a["team"], a["opponent"]) == (10, 14, 2, 1, "Home XI", "Away XI")
    assert (b["bowled_balls"], b["conceded"], b["wickets"], b["wides"], b["no_balls"], b["catches"]) == (18, 12, 1, 3, 1, 1)
    assert "C Absent" not in rows                       # did not bat, bowl or field
    assert d["key"] == "D Nameonly" and d["out"] == 0   # no profile ID: the name is the key


def test_retired_batters_are_not_out_and_marked() -> None:
    home = [player("R Retiree", "pr", batting=dict(BALLS_FACED=20, TOTAL_RUNS=11), status="RETIRED_NOT_OUT")]
    row = S.player_games(card(home, []), META)[0]
    assert row["retired"] == 1 and row["out"] == 0


def test_aggregate_adds_games_and_keeps_every_team() -> None:
    first = S.player_games(card([player("A", "pa", batting=dict(BALLS_FACED=10, TOTAL_RUNS=14))], []), META)
    other = S.GameMeta("g9", "Round 2", "2026-10-17", "Away XI", "Home XI")
    second = S.player_games(card([player("A", "pa", batting=dict(BALLS_FACED=6, TOTAL_RUNS=2))], []), other)
    out = S.aggregate(first + second)
    row = out.iloc[0]
    assert (row["balls"], row["runs"], row["games"], row["innings"]) == (16, 16, 2, 2)
    assert row["team"] == "Home XI / Away XI"


def _players(rows):
    return S.add_ratings(S.aggregate(rows))


def test_a_few_lucky_balls_are_pulled_toward_the_pool_average() -> None:
    """20 off 10 balls is not worth 30 balls at that rate; a longer good record is worth more."""
    typical = [player(f"Typical {i}", f"t{i}", batting=dict(BALLS_FACED=20, TOTAL_RUNS=12)) for i in range(8)]
    short = player("Short", "s", batting=dict(BALLS_FACED=10, TOTAL_RUNS=20), status="NOT_OUT")
    long = player("Long", "l", batting=dict(BALLS_FACED=100, TOTAL_RUNS=150), status="NOT_OUT")
    out = _players(S.player_games(card(typical + [short, long], []), META)).set_index("name")
    raw_short = 30 * 2.0                                   # two runs a ball, never out
    assert out.loc["Short", "runs_per_100"] > out.loc["Long", "runs_per_100"]          # hotter, but on 10 balls
    assert out.loc["Short", "bat_value"] < 0.8 * raw_short                             # pulled toward the pool
    assert out.loc["Long", "bat_value"] > out.loc["Short", "bat_value"] > out.loc["Typical 0", "bat_value"]


def test_bowlers_are_rated_on_runs_and_wickets_with_the_same_pull() -> None:
    home = [player("Tight", "b1", bowling=dict(OVERS=3, RUNS=6, WICKETS=2)),
            player("Loose", "b2", bowling=dict(OVERS=3, RUNS=30, WICKETS=0)),
            player("Typical", "b3", bowling=dict(OVERS=3, RUNS=18, WICKETS=1))]
    out = _players(S.player_games(card(home, []), META)).set_index("name")
    assert out.loc["Tight", "bowl_economy"] < out.loc["Typical", "bowl_economy"] < out.loc["Loose", "bowl_economy"]
    assert out.loc["Tight", "bowl_economy"] > out.loc["Tight", "raw_economy"]       # pulled toward the pool
    assert out.loc["Tight", "bowl_wickets"] > out.loc["Loose", "bowl_wickets"]


def test_a_stint_with_no_dismissals_is_the_rate_times_the_balls() -> None:
    assert S.stint_value(0.5, 0.0, balls=30) == pytest.approx(15.0)
    assert S.stint_value(0.5, 0.1, balls=30) < 15.0


def test_the_team_table_counts_results_and_run_rates() -> None:
    other = S.GameMeta("g2", "Round 2", "2026-10-17", "Away XI", "Home XI",
                       {"TOTAL_SCORE": 90, "TOTAL_OUTS": 4, "TOTAL_OVERS": 22},
                       {"TOTAL_SCORE": 85, "TOTAL_OUTS": 3, "TOTAL_OVERS": 22})
    table = S.team_table([META, other]).set_index("team")
    home = table.loc["Home XI"]
    assert (home["played"], home["won"], home["lost"], home["runs_for"], home["runs_against"]) == (2, 1, 1, 185, 170)
    assert home["rpo_for"] == pytest.approx(6 * 185 / (132 + 132))
    assert home["wickets_lost"] == 5 + 3


def test_last_seasons_record_is_joined_by_profile_key(tmp_path: Path) -> None:
    folder = tmp_path / "earlier"
    (folder / "raw").mkdir(parents=True)
    (folder / "raw" / "opposition_map.csv").write_text("alias,key,name\nO01,pa,A Batter\nO02,other,Someone\n")
    (folder / "bat.csv").write_text(
        "date,opponent,player_name,balls_faced,runs_scored,fours,sixes,dismissed,retired,source\n"
        "2025-10-11,X,O01,15,9,1,0,1,0,playhq\n2025-11-01,X,O01,14,11,0,0,0,0,playhq\n"
        "2025-10-11,X,O02,15,40,0,0,0,0,playhq\n")
    (folder / "bowl.csv").write_text(
        "date,opponent,player_name,balls_bowled,runs_conceded,wickets,wides,no_balls,source\n"
        "2025-10-11,X,O01,12,10,1,0,0,playhq\n")
    history = S.u10_history(["pa", "unknown"], [(folder / "raw" / "opposition_map.csv", folder / "bat.csv", folder / "bowl.csv")])
    row = history.set_index("key").loc["pa"]
    assert (row["u10_games"], row["u10_balls"], row["u10_runs"], row["u10_out"]) == (2, 29, 20, 1)
    assert (row["u10_bowled"], row["u10_conceded"], row["u10_wickets"]) == (12, 10, 1)
    assert "unknown" not in set(history["key"]) and "other" not in set(history["key"])


def test_no_history_files_gives_an_empty_frame(tmp_path: Path) -> None:
    missing = tmp_path / "nothing.csv"
    assert S.u10_history(["pa"], [(missing, missing, missing)]).empty
