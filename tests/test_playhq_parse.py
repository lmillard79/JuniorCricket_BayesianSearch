"""Tests for the PlayHQ scorecard and ball-by-ball parsers."""

from typing import Any, Dict, List

import pytest

from junior_cricket.playhq_parse import (
    HOME,
    balls_bowled,
    link_innings,
    link_names,
    parse_events,
    parse_scorecard,
    reconcile,
)


def stats(**values: float) -> List[Dict[str, Any]]:
    """Build a ``statistics`` list from keyword counts."""
    return [{"type": {"value": k}, "count": v} for k, v in values.items()]


def player(
    name: str,
    pid: str,
    batting: Dict[str, float] = None,
    bowling: Dict[str, float] = None,
    status: str = "RETIRED_NOT_OUT",
) -> Dict[str, Any]:
    """Build one scorecard ``players`` entry."""
    entries = []
    if batting is not None:
        entries.append(
            {"side": "HOME", "status": status, "displayOrder": 2,
             "statistics": stats(**batting)}
        )
    if bowling is not None:
        entries.append(
            {"side": "AWAY", "status": "", "displayOrder": 1,
             "statistics": stats(**bowling)}
        )
    return {"id": f"g-{pid}", "profileID": pid, "name": name,
            "lineupOrder": 3, "periodStatistics": entries}


def scorecard_payload() -> Dict[str, Any]:
    """Home: Alex Ng bats and bowls. Away: Sam Poe bats and bowls."""
    home = [player(
        "Alex Ng", "p-home",
        batting={"BALLS_FACED": 3, "TOTAL_RUNS": 5, "FOURS": 1},
        bowling={"OVERS": 1, "CURRENT_BALLS": 6, "RUNS": 2, "WICKETS": 1},
    )]
    away = [player(
        "Sam Poe", "p-away",
        batting={"BALLS_FACED": 1, "TOTAL_RUNS": 0},
        bowling={"OVERS": 1, "CURRENT_BALLS": 6, "RUNS": 5, "WICKETS": 1},
        status="NOT_OUT",
    )]
    return {"data": {"game": {
        "id": "g1", "status": "FINAL",
        "statistics": {"home": {"players": home}, "away": {"players": away}},
    }}}


def event(title: str, stamp: str, ts: int, desc: str = "", score: str = None,
          icon: str = None) -> Dict[str, Any]:
    """Build one ball-by-ball event."""
    e: Dict[str, Any] = {
        "id": f"e{ts}", "title": title, "description": desc,
        "sportEventStamp": stamp, "timestamp": str(ts), "side": "HOME",
        "period": "FIRST_INNINGS", "visible": True,
    }
    if score is not None:
        e["score"] = score
    if icon:
        e["icon"] = icon
    return e


def test_balls_bowled_complete_partial_and_fractional_overs() -> None:
    """Overs include the latest over; CURRENT_BALLS <6 adds a part over."""
    assert balls_bowled({"OVERS": 3, "CURRENT_BALLS": 6}) == 18
    assert balls_bowled({"OVERS": 2, "CURRENT_BALLS": 0}) == 12
    assert balls_bowled({"OVERS": 2, "CURRENT_BALLS": 3}) == 15
    assert balls_bowled({"OVERS": 2.3}) == 15


def test_parse_scorecard_reads_batting_and_bowling_lines() -> None:
    """Batting and bowling figures land on the right appearance."""
    card = parse_scorecard(scorecard_payload())
    assert card.game_id == "g1" and card.status == "FINAL"
    alex = card.home[0]
    assert alex.profile_id == "p-home"
    assert (alex.batting.balls_faced, alex.batting.runs, alex.batting.fours) == (3, 5, 1)
    assert alex.batting.sixes == 0 and alex.batting.retired is True
    assert (alex.bowling.balls, alex.bowling.runs, alex.bowling.wickets) == (6, 2, 1)
    assert card.away[0].batting.retired is False
    assert card.side(HOME) is card.home


def test_parse_scorecard_rejects_error_payload() -> None:
    """A GraphQL error response is refused rather than half-parsed."""
    with pytest.raises(ValueError):
        parse_scorecard({"errors": [{"message": "boom"}], "data": None})
    with pytest.raises(ValueError):
        parse_scorecard({"data": {"game": None}})


def test_parse_events_decodes_every_known_title() -> None:
    """Dots, runs, 5+, all dismissal forms and retirements decode."""
    events = [
        event("Dot ball", "0.1", 1, "Sam Poe to Alex Ng", score="."),
        event("Alex Ng made 1 run", "0.2", 2, "Sam Poe to Alex Ng", score="1"),
        event("Alex Ng made 4 runs", "0.3", 3, "Sam Poe to Alex Ng", score="4"),
        event("Alex Ng made 5 runs", "0.4", 4, "Sam Poe to Alex Ng", score="5+"),
        event("Alex Ng was bowled by Sam Poe", "0.5", 5, "Sam Poe to Alex Ng", icon="W"),
        event("Alex Ng was caught by Kim Lee", "0.6", 6, "Sam Poe to Alex Ng", icon="W"),
        event("Alex Ng made 1 run and was run out by Kim Lee and assisted by Sam Poe",
              "1.1", 7, "Sam Poe to Alex Ng", icon="W"),
        event("Alex Ng hit their wicket", "1.2", 8, "Sam Poe to Alex Ng", icon="W"),
        event("Alex Ng was run out", "1.3", 9, "Sam Poe to Alex Ng", icon="W"),
        event("Alex Ng retired not out", "1.3", 10),
    ]
    innings = parse_events(events, HOME)
    assert innings.unparsed == []
    assert [d.runs for d in innings.deliveries] == [0, 1, 4, 5, 0, 0, 1, 0, 0]
    kinds = [d.dismissal for d in innings.deliveries]
    assert kinds == [None, None, None, None, "bowled", "caught", "run_out",
                     "hit_wicket", "run_out"]
    run_out = innings.deliveries[6]
    assert run_out.dismissed == "Alex Ng"
    assert run_out.fielders == ("Kim Lee", "Sam Poe")
    assert innings.deliveries[0].over == 1 and innings.deliveries[6].over == 2
    assert innings.deliveries[6].ball == 1
    assert len(innings.retirements) == 1
    assert innings.retirements[0].batter == "Alex Ng"


def test_parse_events_orders_by_time_and_skips_hidden() -> None:
    """Events sort by timestamp, and hidden (undone) ones are dropped."""
    hidden = event("Alex Ng made 6 runs", "0.2", 5, "Sam Poe to Alex Ng", score="6")
    hidden["visible"] = False
    events = [
        event("Alex Ng made 2 runs", "0.2", 20, "Sam Poe to Alex Ng", score="2"),
        event("Dot ball", "0.1", 10, "Sam Poe to Alex Ng", score="."),
        hidden,
    ]
    innings = parse_events(events, HOME)
    assert [d.runs for d in innings.deliveries] == [0, 2]


def test_parse_events_reports_unknown_titles() -> None:
    """Titles the parser does not understand are surfaced."""
    events = [event("Something odd happened", "0.1", 1, "Sam Poe to Alex Ng", score="?")]
    innings = parse_events(events, HOME)
    assert innings.unparsed == ["Something odd happened"]
    assert innings.deliveries == []


def _matching_events() -> List[Dict[str, Any]]:
    """Home batting events matching ``scorecard_payload``."""
    return [
        event("Alex Ng made 4 runs", "0.1", 1, "Sam Poe to Alex Ng", score="4"),
        event("Alex Ng made 1 run", "0.2", 2, "Sam Poe to Alex Ng", score="1"),
        event("Alex Ng was bowled by Sam Poe", "0.3", 3, "Sam Poe to Alex Ng", icon="W"),
    ]


def test_reconcile_agrees_when_events_match_scorecard() -> None:
    """Balls, runs and bowler wickets agree, so no problems."""
    card = parse_scorecard(scorecard_payload())
    innings = parse_events(_matching_events(), HOME)
    assert reconcile(card.home, innings, card.away) == []


def test_link_names_exact_case_and_single_leftover() -> None:
    """Exact, then case-insensitive, then a lone leftover pair link."""
    card = ["Alex Ng", "Sam Poe", "Kim Lee"]
    assert link_names(["Alex Ng"], card) == {"Alex Ng": "Alex Ng"}
    assert link_names(["alex  ng"], card) == {"alex  ng": "Alex Ng"}
    # One event name and one scorecard name left over: paired.
    assert link_names(["Alex Ng", "Sam Poe", "Kimberley Lee"], card) == {
        "Alex Ng": "Alex Ng", "Sam Poe": "Sam Poe", "Kimberley Lee": "Kim Lee",
    }


def test_link_names_never_guesses_when_ambiguous() -> None:
    """Two unmatched names on each side stay unlinked."""
    card = ["Alex Ng", "Sam Poe"]
    assert link_names(["Alec Ng", "Sammy Poe"], card) == {}


def test_link_innings_repairs_a_misspelt_name() -> None:
    """A striker spelt differently in the events is linked and reconciles."""
    card = parse_scorecard(scorecard_payload())
    events = [
        event("Alexander Ng made 4 runs", "0.1", 1, "Sam Poe to Alexander Ng", score="4"),
        event("Alexander Ng made 1 run", "0.2", 2, "Sam Poe to Alexander Ng", score="1"),
        event("Alexander Ng was bowled by Sam Poe", "0.3", 3,
              "Sam Poe to Alexander Ng", icon="W"),
    ]
    linked = link_innings(parse_events(events, HOME), card.home, card.away)
    assert {d.striker for d in linked.deliveries} == {"Alex Ng"}
    assert linked.deliveries[2].dismissed == "Alex Ng"
    assert reconcile(card.home, parse_events(events, HOME), card.away) == []


def test_reconcile_reports_disagreement() -> None:
    """A missing ball is reported rather than silently accepted."""
    card = parse_scorecard(scorecard_payload())
    innings = parse_events(_matching_events()[:2], HOME)
    problems = reconcile(card.home, innings, card.away)
    assert any("balls faced differ" in p for p in problems)
    assert any("wickets differ" in p for p in problems)
