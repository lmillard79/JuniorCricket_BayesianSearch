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
    build_game_rows,
    load_aliases,
    save_aliases,
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


INFO = GameInfo("g1", "2025-10-11", HOME, "Opposition Team")


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
