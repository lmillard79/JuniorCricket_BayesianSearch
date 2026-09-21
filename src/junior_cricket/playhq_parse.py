"""Parsing of PlayHQ spectator payloads (scorecards and ball-by-ball).

Everything here is network-free: the functions take the JSON documents
returned by ``PlayHQClient.game_scorecard`` and
``PlayHQClient.game_events`` (or cached copies of them) and turn them
into plain records. PlayHQ publishes player names, and ball events
refer to players by name only, so names are used here to link events
to scorecard lines. They must not reach the scorebook CSVs; see
``playhq_scorebook`` for the pseudonymising layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

HOME = "HOME"
AWAY = "AWAY"

# Dismissal kinds a bowler is credited with (Rule 16.8(v): every
# dismissal except a run out).
BOWLER_CREDITED = ("bowled", "caught", "caught_and_bowled", "stumped",
                   "hit_wicket")


@dataclass(frozen=True)
class BattingLine:
    """One batter's innings as printed on the scorecard.

    Attributes:
        balls_faced: Balls faced (wides and no balls included).
        runs: Runs scored, sundries included for U10.
        fours: Fours hit.
        sixes: Sixes hit.
        retired: True when the batter retired not out.
        order: Position in the batting order as displayed.
    """

    balls_faced: int
    runs: int
    fours: int
    sixes: int
    retired: bool
    order: int


@dataclass(frozen=True)
class BowlingLine:
    """One bowler's figures as printed on the scorecard.

    Attributes:
        balls: Balls bowled (see ``balls_bowled``).
        runs: Runs conceded, sundries included.
        wickets: Wickets credited to the bowler.
        maidens: Maiden overs.
    """

    balls: int
    runs: int
    wickets: int
    maidens: int


@dataclass(frozen=True)
class Appearance:
    """One player's line-up entry with batting and bowling figures.

    Attributes:
        name: Player name as PlayHQ prints it.
        profile_id: Stable PlayHQ profile ID, or None when unlinked.
        player_id: PlayHQ's per-game player ID.
        lineup_order: Position in the submitted line-up.
        batting: Batting figures, or None if the player did not bat.
        bowling: Bowling figures, or None if the player did not bowl.
    """

    name: str
    profile_id: Optional[str]
    player_id: str
    lineup_order: int
    batting: Optional[BattingLine]
    bowling: Optional[BowlingLine]


@dataclass(frozen=True)
class Scorecard:
    """Both sides' appearances for one game.

    Attributes:
        game_id: PlayHQ game ID.
        status: Game status, for example ``FINAL``.
        home: Home side appearances.
        away: Away side appearances.
    """

    game_id: str
    status: str
    home: Tuple[Appearance, ...]
    away: Tuple[Appearance, ...]

    def side(self, side: str) -> Tuple[Appearance, ...]:
        """Return the appearances for ``HOME`` or ``AWAY``."""
        return self.home if side == HOME else self.away


@dataclass(frozen=True)
class Delivery:
    """One ball from the ball-by-ball feed.

    Attributes:
        side: Batting side, ``HOME`` or ``AWAY``.
        over: One-based over number.
        ball: One-based ball number within the over as scored.
        bowler: Bowler name.
        striker: Striker name.
        runs: Runs recorded off the delivery.
        dismissal: Dismissal kind, or None.
        dismissed: Name of the dismissed batter, or None.
        fielders: Fielders named on the dismissal.
        timestamp: Event time in milliseconds.
    """

    side: str
    over: int
    ball: int
    bowler: str
    striker: str
    runs: int
    dismissal: Optional[str] = None
    dismissed: Optional[str] = None
    fielders: Tuple[str, ...] = ()
    timestamp: int = 0


@dataclass(frozen=True)
class Retirement:
    """A batter retiring not out after their allotted balls."""

    side: str
    over: int
    ball: int
    batter: str
    timestamp: int


@dataclass
class ParsedInnings:
    """Decoded events for one innings.

    Attributes:
        side: Batting side.
        deliveries: Balls in the order bowled.
        retirements: Retirement events.
        unparsed: Titles that matched no known pattern.
    """

    side: str
    deliveries: List[Delivery] = field(default_factory=list)
    retirements: List[Retirement] = field(default_factory=list)
    unparsed: List[str] = field(default_factory=list)


_STAMP = re.compile(r"^(\d+)\.(\d+)$")
_SCORE = re.compile(r"^(\d+)\+?$")
_TO = re.compile(r"^(?P<bowler>.+?) to (?P<striker>.+)$")
_MADE = re.compile(r"^(?P<batter>.+?) made (?P<runs>\d+) runs?$")
_RETIRED = re.compile(r"^(?P<batter>.+?) retired(?: not out| out)?$")
# Ordered: the run-out and caught-and-bowled patterns must precede the
# looser ones that share a prefix.
_DISMISSALS = (
    ("run_out", re.compile(
        r"^(?P<batter>.+?) (?:made \d+ runs? and )?was run out"
        r"(?: by (?P<fielders>.+))?$")),
    ("bowled", re.compile(r"^(?P<batter>.+?) was bowled by (?P<bowler>.+)$")),
    ("caught_and_bowled", re.compile(
        r"^(?P<batter>.+?) was caught and bowled(?: by (?P<fielders>.+))?$")),
    ("caught", re.compile(
        r"^(?P<batter>.+?) was caught(?: by (?P<fielders>.+))?$")),
    ("stumped", re.compile(
        r"^(?P<batter>.+?) was stumped(?: by (?P<fielders>.+))?$")),
    ("hit_wicket", re.compile(r"^(?P<batter>.+?) hit their wicket$")),
)


def _stats(period_statistics: Dict[str, Any]) -> Dict[str, float]:
    """Flatten a ``periodStatistics`` entry to {type: count}."""
    return {
        item["type"]["value"]: item["count"]
        for item in period_statistics.get("statistics") or []
    }


def balls_bowled(stats: Dict[str, float]) -> int:
    """Work out balls bowled from a bowler's scorecard statistics.

    ``OVERS`` counts overs including the bowler's most recent one, and
    ``CURRENT_BALLS`` is the number of balls in that most recent over
    (6 when it is complete). A fractional ``OVERS`` is read as
    over.balls (2.3 is two overs and three balls).

    Args:
        stats: Statistic type to count for one bowling line.

    Returns:
        Number of balls bowled.
    """
    overs = float(stats.get("OVERS", 0))
    whole = int(overs)
    fraction = round((overs - whole) * 10)
    if fraction:
        return whole * 6 + int(fraction)
    partial = int(stats.get("CURRENT_BALLS", 0))
    return whole * 6 + (partial if 0 < partial < 6 else 0)


def _appearance(player: Dict[str, Any]) -> Appearance:
    """Convert one ``players`` entry to an ``Appearance``."""
    batting: Optional[BattingLine] = None
    bowling: Optional[BowlingLine] = None
    for entry in player.get("periodStatistics") or []:
        stats = _stats(entry)
        if "BALLS_FACED" in stats and batting is None:
            batting = BattingLine(
                balls_faced=int(round(stats["BALLS_FACED"])),
                runs=int(round(stats.get("TOTAL_RUNS", 0))),
                fours=int(round(stats.get("FOURS", 0))),
                sixes=int(round(stats.get("SIXES", 0))),
                retired=entry.get("status") == "RETIRED_NOT_OUT",
                order=int(entry.get("displayOrder") or 0),
            )
        elif "OVERS" in stats and bowling is None:
            bowling = BowlingLine(
                balls=balls_bowled(stats),
                runs=int(round(stats.get("RUNS", 0))),
                wickets=int(round(stats.get("WICKETS", 0))),
                maidens=int(round(stats.get("MAIDENS", 0))),
            )
    return Appearance(
        name=(player.get("name") or "").strip(),
        profile_id=player.get("profileID") or None,
        player_id=str(player.get("id") or ""),
        lineup_order=int(player.get("lineupOrder") or 0),
        batting=batting,
        bowling=bowling,
    )


def parse_scorecard(payload: Dict[str, Any]) -> Scorecard:
    """Decode a ``gameViewSpectator`` response.

    Args:
        payload: The full JSON response (with its ``data`` member).

    Returns:
        The scorecard with one ``Appearance`` per listed player.

    Raises:
        ValueError: If the response carries errors or no game.
    """
    if payload.get("errors"):
        raise ValueError(f"Scorecard response has errors: {payload['errors']}")
    game = (payload.get("data") or {}).get("game")
    if not game:
        raise ValueError("Scorecard response contains no game")
    statistics = game.get("statistics") or {}
    return Scorecard(
        game_id=str(game["id"]),
        status=str(game.get("status") or ""),
        home=tuple(
            _appearance(p) for p in (statistics.get("home") or {}).get("players") or []
        ),
        away=tuple(
            _appearance(p) for p in (statistics.get("away") or {}).get("players") or []
        ),
    )


def _over_ball(stamp: str) -> Tuple[int, int]:
    """Split a ``sportEventStamp`` such as ``16.3`` into (over, ball).

    PlayHQ counts completed overs, so ``16.3`` is the third ball of
    the seventeenth over.
    """
    match = _STAMP.match(stamp or "")
    if not match:
        return 0, 0
    return int(match.group(1)) + 1, int(match.group(2))


def _split_names(text: Optional[str]) -> Tuple[str, ...]:
    """Split ``A and assisted by B`` style fielder text into names."""
    if not text:
        return ()
    parts = re.split(r" and assisted by | and ", text)
    return tuple(p.strip() for p in parts if p.strip())


def parse_events(events: Sequence[Dict[str, Any]], side: str) -> ParsedInnings:
    """Decode the ball-by-ball events of one innings.

    Args:
        events: The ``gameEvents`` list for one (period, side).
        side: The batting side those events belong to.

    Returns:
        Deliveries in bowling order, retirements, and any titles
        that matched no known pattern (reported, never dropped
        silently).
    """
    innings = ParsedInnings(side=side)
    ordered = sorted(
        (e for e in events if e.get("visible") is not False),
        key=lambda e: (int(e.get("timestamp") or 0),
                       _over_ball(e.get("sportEventStamp", ""))),
    )
    for event in ordered:
        title = (event.get("title") or "").strip()
        over, ball = _over_ball(event.get("sportEventStamp", ""))
        stamp_ms = int(event.get("timestamp") or 0)

        retired = _RETIRED.match(title)
        if retired and event.get("score") is None and not event.get("icon"):
            innings.retirements.append(
                Retirement(side, over, ball, retired.group("batter"), stamp_ms)
            )
            continue

        pair = _TO.match(event.get("description") or "")
        if not pair:
            innings.unparsed.append(title)
            continue
        bowler, striker = pair.group("bowler"), pair.group("striker")

        if event.get("icon") == "W":
            delivery = _dismissal(title, side, over, ball, bowler, striker, stamp_ms)
            if delivery is None:
                innings.unparsed.append(title)
            else:
                innings.deliveries.append(delivery)
            continue

        score = event.get("score")
        match = _SCORE.match(score) if isinstance(score, str) else None
        if title == "Dot ball" or score == ".":
            runs = 0
        elif match:
            runs = int(match.group(1))
        else:
            innings.unparsed.append(title)
            continue
        innings.deliveries.append(
            Delivery(side, over, ball, bowler, striker, runs, timestamp=stamp_ms)
        )
    return innings


def _dismissal(
    title: str, side: str, over: int, ball: int, bowler: str, striker: str,
    stamp_ms: int,
) -> Optional[Delivery]:
    """Build a dismissal ``Delivery`` from a wicket title, or None."""
    for kind, pattern in _DISMISSALS:
        match = pattern.match(title)
        if not match:
            continue
        groups = match.groupdict()
        made = re.search(r"\bmade (\d+) runs?\b", title)
        return Delivery(
            side=side,
            over=over,
            ball=ball,
            bowler=bowler,
            striker=striker,
            runs=int(made.group(1)) if made else 0,
            dismissal=kind,
            dismissed=groups["batter"],
            fielders=_split_names(groups.get("fielders")),
            timestamp=stamp_ms,
        )
    return None


def _normalise(name: str) -> str:
    """Lower-case a name and drop everything except letters and digits."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def link_names(event_names: Sequence[str], card_names: Sequence[str]) -> Dict[str, str]:
    """Map names used in the events feed to names on the scorecard.

    PlayHQ occasionally spells a player differently in the two feeds.
    Exact matches come first, then case and punctuation insensitive
    ones. A leftover pair is only matched when exactly one event name
    and one scorecard name remain, so an ambiguous case is left
    unlinked rather than guessed.

    Args:
        event_names: Names as used in the events.
        card_names: Candidate scorecard names.

    Returns:
        Mapping from event name to scorecard name for linked names.
    """
    cards = list(dict.fromkeys(card_names))
    by_normal: Dict[str, str] = {}
    for card in cards:
        by_normal.setdefault(_normalise(card), card)
    mapping: Dict[str, str] = {}
    unmatched: List[str] = []
    for name in dict.fromkeys(event_names):
        if name in cards:
            mapping[name] = name
        elif _normalise(name) in by_normal:
            mapping[name] = by_normal[_normalise(name)]
        else:
            unmatched.append(name)
    leftovers = [c for c in cards if c not in set(mapping.values())]
    if len(unmatched) == 1 and len(leftovers) == 1:
        mapping[unmatched[0]] = leftovers[0]
    return mapping


def link_innings(
    innings: ParsedInnings,
    batting: Sequence[Appearance],
    fielding: Sequence[Appearance],
) -> ParsedInnings:
    """Rename event names in an innings to their scorecard names.

    Args:
        innings: Decoded events for one innings.
        batting: Batting side's scorecard lines.
        fielding: Fielding side's scorecard lines.

    Returns:
        A copy whose strikers, dismissed batters, bowlers, fielders
        and retiring batters use scorecard spellings where a safe link
        exists; names with no safe link are left as they were.
    """
    bat_names = [a.name for a in batting if a.batting is not None] or [
        a.name for a in batting
    ]
    field_names = [a.name for a in fielding if a.bowling is not None] or [
        a.name for a in fielding
    ]
    bat_events = [d.striker for d in innings.deliveries] + [
        d.dismissed for d in innings.deliveries if d.dismissed
    ] + [r.batter for r in innings.retirements]
    bat_map = link_names(bat_events, bat_names)
    field_map = link_names([d.bowler for d in innings.deliveries], field_names)
    fielder_map = link_names(
        [f for d in innings.deliveries for f in d.fielders],
        [a.name for a in fielding],
    )

    def relabel(mapping: Dict[str, str], name: Optional[str]) -> Optional[str]:
        return mapping.get(name, name) if name else name

    deliveries = [
        Delivery(
            side=d.side, over=d.over, ball=d.ball,
            bowler=relabel(field_map, d.bowler),
            striker=relabel(bat_map, d.striker),
            runs=d.runs, dismissal=d.dismissal,
            dismissed=relabel(bat_map, d.dismissed),
            fielders=tuple(relabel(fielder_map, f) for f in d.fielders),
            timestamp=d.timestamp,
        )
        for d in innings.deliveries
    ]
    retirements = [
        Retirement(r.side, r.over, r.ball, relabel(bat_map, r.batter), r.timestamp)
        for r in innings.retirements
    ]
    return ParsedInnings(
        side=innings.side, deliveries=deliveries, retirements=retirements,
        unparsed=list(innings.unparsed),
    )


def reconcile(
    appearances: Sequence[Appearance],
    innings: ParsedInnings,
    fielding: Sequence[Appearance],
) -> List[str]:
    """Compare decoded events with the scorecard for one innings.

    Names are linked to the scorecard first (``link_innings``).

    Args:
        appearances: Batting side's scorecard lines.
        innings: Decoded events for the innings.
        fielding: Fielding side's scorecard lines (the bowlers).

    Returns:
        Human-readable discrepancies; an empty list means the events
        and the scorecard agree on balls faced, runs, and the wickets
        credited to each bowler.
    """
    innings = link_innings(innings, appearances, fielding)
    problems: List[str] = []
    known_batters = {a.name for a in appearances}
    known_bowlers = {a.name for a in fielding}
    balls: Dict[str, int] = {}
    runs: Dict[str, int] = {}
    wickets: Dict[str, int] = {}
    for d in innings.deliveries:
        balls[d.striker] = balls.get(d.striker, 0) + 1
        runs[d.striker] = runs.get(d.striker, 0) + d.runs
        if d.dismissal in BOWLER_CREDITED:
            wickets[d.bowler] = wickets.get(d.bowler, 0) + 1
        for name in (d.striker, d.dismissed):
            if name and name not in known_batters:
                problems.append("event batter not on scorecard")
        if d.bowler not in known_bowlers:
            problems.append("event bowler not on scorecard")
    for a in appearances:
        if a.batting is None:
            continue
        if balls.get(a.name, 0) != a.batting.balls_faced:
            problems.append(
                f"balls faced differ ({balls.get(a.name, 0)} events vs "
                f"{a.batting.balls_faced} scorecard) for batter {a.player_id}"
            )
        if runs.get(a.name, 0) != a.batting.runs:
            problems.append(
                f"runs differ ({runs.get(a.name, 0)} events vs "
                f"{a.batting.runs} scorecard) for batter {a.player_id}"
            )
    for a in fielding:
        if a.bowling is None:
            continue
        if wickets.get(a.name, 0) != a.bowling.wickets:
            problems.append(
                f"wickets differ ({wickets.get(a.name, 0)} events vs "
                f"{a.bowling.wickets} scorecard) for bowler {a.player_id}"
            )
    if innings.unparsed:
        problems.append(f"{len(innings.unparsed)} event titles not understood")
    return sorted(set(problems))
