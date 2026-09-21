"""Tests for the pseudonymised PlayHQ scorebook builder."""

from pathlib import Path
from typing import Any, Dict, List

from junior_cricket.data_loader import (
    BATTING_COLUMNS,
    BOWLING_COLUMNS,
    load_batting_csv,
    load_bowling_csv,
)
from junior_cricket.playhq_parse import HOME, parse_events, parse_scorecard
from junior_cricket.playhq_scorebook import (
    GameInfo,
    PlayerAliases,
    alias_names,
    build_ball_rows,
    build_game_rows,
    completed_games,
    load_aliases,
    save_aliases,
    translate_aliases,
    write_rows,
)

NAMES = ("Alex Ng", "Kim Lee", "Sam Poe")


def stats(**values: float) -> List[Dict[str, Any]]:
    return [{"type": {"value": k}, "count": v} for k, v in values.items()]


def player(name: str, pid: str, batting=None, bowling=None, status="RETIRED_NOT_OUT"):
    entries = []
    if batting is not None:
        entries.append({"side": "HOME", "status": status, "displayOrder": 1,
                        "statistics": stats(**batting)})
    if bowling is not None:
        entries.append({"side": "AWAY", "status": "", "displayOrder": 1,
                        "statistics": stats(**bowling)})
    return {"id": f"g-{pid}", "profileID": pid, "name": name,
            "lineupOrder": 1, "periodStatistics": entries}


def make_card():
    """Home: Alex and Kim bat (3 balls each) and bowl one over each."""
    home = [
        player("Alex Ng", "p-alex",
               batting={"BALLS_FACED": 3, "TOTAL_RUNS": 5, "FOURS": 1},
               bowling={"OVERS": 1, "CURRENT_BALLS": 6, "RUNS": 7, "WICKETS": 1}),
        player("Kim Lee", "p-kim",
               batting={"BALLS_FACED": 3, "TOTAL_RUNS": 1},
               bowling={"OVERS": 1, "CURRENT_BALLS": 6, "RUNS": 4}),
    ]
    away = [player("Sam Poe", "p-sam",
                   bowling={"OVERS": 1, "CURRENT_BALLS": 6, "RUNS": 6, "WICKETS": 2},
                   batting={"BALLS_FACED": 1, "TOTAL_RUNS": 0}, status="NOT_OUT")]
    return parse_scorecard({"data": {"game": {
        "id": "g1", "status": "FINAL",
        "statistics": {"home": {"players": home}, "away": {"players": away}}}}})


def event(title, stamp, ts, desc, score=None, icon=None):
    e = {"id": f"e{ts}", "title": title, "description": desc,
         "sportEventStamp": stamp, "timestamp": str(ts), "side": "HOME",
         "period": "FIRST_INNINGS", "visible": True}
    if score is not None:
        e["score"] = score
    if icon:
        e["icon"] = icon
    return e


def own_events():
    """Alex: 4, 1, bowled. Kim: dot, 1, caught."""
    return parse_events([
        event("Alex Ng made 4 runs", "0.1", 1, "Sam Poe to Alex Ng", score="4"),
        event("Alex Ng made 1 run", "0.2", 2, "Sam Poe to Alex Ng", score="1"),
        event("Alex Ng was bowled by Sam Poe", "0.3", 3, "Sam Poe to Alex Ng", icon="W"),
        event("Dot ball", "0.4", 4, "Sam Poe to Kim Lee", score="."),
        event("Kim Lee made 1 run", "0.5", 5, "Sam Poe to Kim Lee", score="1"),
        event("Kim Lee was bowled by Sam Poe", "0.6", 6, "Sam Poe to Kim Lee", icon="W"),
        event("Alex Ng retired not out", "1.1", 7, ""),
    ], HOME)


INFO = GameInfo("g1", "2025-10-11", HOME, "Opposition Team", "Our Team")


def their_events():
    """Away innings: Sam Poe faces one ball and is bowled by Alex Ng."""
    return parse_events([
        event("Sam Poe was bowled by Alex Ng", "0.1", 1, "Alex Ng to Sam Poe", icon="W"),
    ], "AWAY")


def test_rows_use_aliases_and_carry_no_player_names(tmp_path: Path) -> None:
    """Every written value is an alias; no name appears in either CSV."""
    aliases = PlayerAliases()
    rows = build_game_rows(make_card(), INFO, own_events(), None, aliases)
    batting = write_rows(rows.batting, BATTING_COLUMNS, tmp_path / "bat.csv")
    bowling = write_rows(rows.bowling, BOWLING_COLUMNS, tmp_path / "bowl.csv")
    assert set(batting["player_name"]) == {"P01", "P02"}
    assert set(bowling["player_name"]) == {"P01", "P02"}
    text = (tmp_path / "bat.csv").read_text() + (tmp_path / "bowl.csv").read_text()
    for name in NAMES:
        for token in name.split():
            assert token not in text


def test_dismissals_are_counted_per_batter() -> None:
    """Each batter's dismissals come from the events, repeats included."""
    aliases = PlayerAliases()
    rows = build_game_rows(make_card(), INFO, own_events(), None, aliases)
    by_alias = {r["player_name"]: r for r in rows.batting}
    assert by_alias["P01"]["dismissed"] == 1 and by_alias["P01"]["fours"] == 1
    assert by_alias["P02"]["dismissed"] == 1 and by_alias["P02"]["runs_scored"] == 1
    assert by_alias["P01"]["retired"] == 1
    assert rows.warnings == []


def test_bowling_rows_come_from_the_scorecard() -> None:
    """Balls, runs and wickets are read off the bowling lines."""
    rows = build_game_rows(make_card(), INFO, own_events(), None, PlayerAliases())
    by_alias = {r["player_name"]: r for r in rows.bowling}
    assert by_alias["P01"]["balls_bowled"] == 6
    assert by_alias["P01"]["runs_conceded"] == 7 and by_alias["P01"]["wickets"] == 1
    assert by_alias["P02"]["wickets"] == 0


def test_missing_events_skip_batting_but_keep_bowling() -> None:
    """No events means no batting rows (a zero would bias outs down)."""
    rows = build_game_rows(make_card(), INFO, None, None, PlayerAliases())
    assert rows.batting == []
    assert len(rows.bowling) == 2
    assert any("no ball-by-ball events" in w for w in rows.warnings)


def test_disagreeing_events_raise_a_warning() -> None:
    """A scorecard/event mismatch is reported, not hidden."""
    short = parse_events([
        event("Alex Ng made 4 runs", "0.1", 1, "Sam Poe to Alex Ng", score="4"),
    ], HOME)
    rows = build_game_rows(make_card(), INFO, short, None, PlayerAliases())
    assert any("balls faced differ" in w for w in rows.warnings)


def test_opposition_rows_use_their_own_alias_map() -> None:
    """Opposition players get O-aliases and carry our team as opponent."""
    ours, opposition = PlayerAliases(), PlayerAliases(prefix="O")
    rows = build_game_rows(make_card(), INFO, own_events(), their_events(),
                           ours, opposition)
    assert [r["player_name"] for r in rows.opp_batting] == ["O01"]
    assert rows.opp_batting[0]["dismissed"] == 1
    assert rows.opp_batting[0]["opponent"] == "Our Team"
    assert [r["player_name"] for r in rows.opp_bowling] == ["O01"]
    assert rows.opp_bowling[0]["wickets"] == 2
    assert {r["player_name"] for r in rows.batting} == {"P01", "P02"}   # ours unchanged
    assert rows.warnings == []


def test_ball_rows_carry_batter_bowler_and_outcome_as_aliases() -> None:
    """One row per delivery, both innings, aliases only."""
    ours, theirs = PlayerAliases(), PlayerAliases(prefix="O")
    rows, dropped = build_ball_rows(make_card(), INFO, own_events(), their_events(),
                                    ours, theirs)
    assert dropped == 0 and len(rows) == 7             # 6 of ours + 1 of theirs
    first = rows[0]
    assert (first["batter"], first["bowler"]) == ("P01", "O01")
    assert first["runs"] == 4 and first["boundary"] == 1 and first["dismissed"] == 0
    assert first["over"] == 1                            # stamp 0.1 is the first over
    wicket = rows[2]
    assert wicket["dismissed"] == 1 and wicket["run_out"] == 0 and wicket["runs"] == 0
    reply = rows[-1]                                    # their innings: O01 bat, P01 bowl
    assert (reply["batter"], reply["bowler"]) == ("O01", "P01")
    assert reply["dismissed"] == 1
    text = str(rows)
    for name in NAMES:
        for token in name.split():
            assert token not in text


def test_ball_rows_skip_innings_without_events() -> None:
    rows, _ = build_ball_rows(make_card(), INFO, own_events(), None,
                              PlayerAliases(), PlayerAliases(prefix="O"))
    assert len(rows) == 6
    assert all(r["batter"].startswith("P") for r in rows)


def test_opposition_rows_are_off_unless_requested() -> None:
    rows = build_game_rows(make_card(), INFO, own_events(), their_events(), PlayerAliases())
    assert rows.opp_batting == [] and rows.opp_bowling == []


def test_opposition_batting_needs_their_events() -> None:
    """Without the opposition's events their batting rows are skipped."""
    rows = build_game_rows(make_card(), INFO, own_events(), None,
                           PlayerAliases(), PlayerAliases(prefix="O"))
    assert rows.opp_batting == []
    assert len(rows.opp_bowling) == 1


def _fixture_game(gid, date, home, away, status="FINAL", home_total=None, away_total=None):
    def side(total):
        if total is None:
            return {"periods": []}
        return {"periods": [{"period": {"value": "FIRST_INNINGS"}, "statistics": [
            {"type": {"value": "TOTAL_SCORE"}, "count": total},
            {"type": {"value": "TOTAL_OVERS"}, "count": 20}]}]}
    return {"id": gid, "date": date, "status": {"value": status},
            "home": {"name": home}, "away": {"name": away},
            "result": {"home": side(home_total), "away": side(away_total)}}


def test_completed_games_finds_our_side_and_totals() -> None:
    fixture = {
        "discoverTeam": {"name": "Us"},
        "discoverTeamFixture": [
            {"fixture": {"games": [
                _fixture_game("g2", "2025-10-18T08:15", "Them", "Us", home_total=120, away_total=150),
                _fixture_game("g1", "2025-10-11T08:15", "Us", "Them", home_total=109, away_total=108),
                _fixture_game("g3", "2025-10-25T08:15", "Us", "Them", status="ABANDONED"),
                _fixture_game("g4", "2025-11-01T08:15", "A", "B", home_total=1, away_total=2),
            ]}},
        ],
    }
    games = completed_games(fixture)
    assert [g.game_id for g in games] == ["g1", "g2"]          # date order, no abandoned
    first, second = games
    assert (first.our_side, first.our_total, first.their_total) == ("HOME", 109, 108)
    assert (second.our_side, second.our_total, second.their_total) == ("AWAY", 150, 120)
    assert second.opponent == "Them" and second.our_team == "Us"


def test_translate_aliases_replaces_whole_aliases_only() -> None:
    names = {"P01": "Alex Ng", "O12": "Sam Poe"}
    text = "P01 opens; P010 and P99 are unknown; O12 bowls; XP01 stays."
    assert translate_aliases(text, names) == (
        "Alex Ng opens; P010 and P99 are unknown; Sam Poe bowls; XP01 stays."
    )


def test_alias_names_reads_the_maps(tmp_path: Path) -> None:
    ours, theirs = PlayerAliases(), PlayerAliases(prefix="O")
    card = make_card()
    ours.alias_for(card.home[0])
    theirs.alias_for(card.away[0])
    save_aliases(ours, tmp_path / "a.csv")
    save_aliases(theirs, tmp_path / "b.csv")
    names = alias_names(tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "missing.csv")
    assert names == {"P01": "Alex Ng", "O01": "Sam Poe"}


def test_aliases_are_stable_and_round_trip(tmp_path: Path) -> None:
    """A second run reuses aliases and appends new players."""
    aliases = PlayerAliases()
    build_game_rows(make_card(), INFO, own_events(), None, aliases)
    path = tmp_path / "raw" / "player_map.csv"
    save_aliases(aliases, path)
    again = load_aliases(path)
    card = make_card()
    assert again.alias_for(card.home[0]) == "P01"
    assert again.alias_for(card.home[1]) == "P02"
    new = player("Jo Newcomer", "p-new", batting={"BALLS_FACED": 1})
    from junior_cricket.playhq_parse import _appearance
    assert again.alias_for(_appearance(new)) == "P03"
    assert load_aliases(tmp_path / "nothing.csv").entries() == []


def test_written_csvs_pass_the_existing_loaders(tmp_path: Path) -> None:
    """The output is accepted by the pipeline's own validators."""
    rows = build_game_rows(make_card(), INFO, own_events(), None, PlayerAliases())
    write_rows(rows.batting, BATTING_COLUMNS, tmp_path / "bat.csv")
    write_rows(rows.bowling, BOWLING_COLUMNS, tmp_path / "bowl.csv")
    batting = load_batting_csv(tmp_path / "bat.csv")
    bowling = load_bowling_csv(tmp_path / "bowl.csv")
    assert len(batting) == 2 and len(bowling) == 2
    assert batting["source"].unique().tolist() == ["playhq"]
