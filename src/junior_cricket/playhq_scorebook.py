"""Pseudonymised scorebook rows from PlayHQ scorecards and events.

The rows follow the manual scorebook schemas in ``data_loader``
(``BATTING_COLUMNS`` and ``BOWLING_COLUMNS``), so the existing fit
pipeline reads them unchanged. Player names never reach those rows:
each player becomes a stable alias (P01, P02, ...) keyed by PlayHQ
profile ID. The alias-to-name map is written separately, under
``data/raw`` (gitignored), so a coach can translate a report back.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from junior_cricket.data_loader import BATTING_COLUMNS, BOWLING_COLUMNS
from junior_cricket.playhq_parse import (
    Appearance,
    ParsedInnings,
    Scorecard,
    link_innings,
    reconcile,
)

SOURCE = "playhq"


@dataclass(frozen=True)
class GameInfo:
    """Fixture facts about one game from our team's point of view.

    Attributes:
        game_id: PlayHQ game ID.
        date: ISO date of the game.
        our_side: ``HOME`` or ``AWAY``.
        opponent: Opposition team name (a team, not a person).
        our_team: Our team's name, used as the opponent of the
            opposition's rows.
    """

    game_id: str
    date: str
    our_side: str
    opponent: str
    our_team: str = ""


@dataclass
class GameRows:
    """Scorebook rows built for one game.

    Attributes:
        batting: Our side's rows matching ``BATTING_COLUMNS``.
        bowling: Our side's rows matching ``BOWLING_COLUMNS``.
        opp_batting: The opposition's batting rows (empty unless
            opposition aliases were supplied).
        opp_bowling: The opposition's bowling rows.
        warnings: Data-quality notes worth surfacing to the user.
    """

    batting: List[Dict[str, object]] = field(default_factory=list)
    bowling: List[Dict[str, object]] = field(default_factory=list)
    opp_batting: List[Dict[str, object]] = field(default_factory=list)
    opp_bowling: List[Dict[str, object]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PlayerAliases:
    """Stable pseudonyms for players, keyed by PlayHQ profile ID.

    Args:
        entries: Existing (alias, key, name) triples, so aliases stay
            the same from one run to the next.
        prefix: Alias prefix.
    """

    def __init__(
        self,
        entries: Optional[Iterable[Tuple[str, str, str]]] = None,
        prefix: str = "P",
    ) -> None:
        self._prefix = prefix
        self._alias: Dict[str, str] = {}
        self._name: Dict[str, str] = {}
        for alias, key, name in entries or ():
            self._alias[key] = alias
            self._name[key] = name

    @staticmethod
    def key_for(appearance: Appearance) -> str:
        """Identity key: the profile ID, else the normalised name."""
        if appearance.profile_id:
            return appearance.profile_id
        return "name:" + re.sub(r"[^a-z0-9]", "", appearance.name.lower())

    def alias_for(self, appearance: Appearance) -> str:
        """Return the alias for a player, creating one if new."""
        key = self.key_for(appearance)
        if key not in self._alias:
            used = {int(a[len(self._prefix):]) for a in self._alias.values()
                    if a[len(self._prefix):].isdigit()}
            self._alias[key] = f"{self._prefix}{max(used, default=0) + 1:02d}"
            self._name[key] = appearance.name
        return self._alias[key]

    def entries(self) -> List[Tuple[str, str, str]]:
        """All (alias, key, name) triples, ordered by alias."""
        return sorted(
            ((alias, key, self._name[key]) for key, alias in self._alias.items()),
            key=lambda triple: triple[0],
        )


def load_aliases(path: Path, prefix: str = "P") -> PlayerAliases:
    """Load an alias map written by ``save_aliases`` (empty if absent)."""
    if not path.exists():
        return PlayerAliases(prefix=prefix)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [(r["alias"], r["key"], r["name"]) for r in csv.DictReader(handle)]
    return PlayerAliases(rows, prefix=prefix)


def save_aliases(aliases: PlayerAliases, path: Path) -> None:
    """Write the alias map (this file holds names; keep it out of git)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alias", "key", "name"])
        writer.writerows(aliases.entries())


def _side_rows(
    lines: Sequence[Appearance],
    fielding: Sequence[Appearance],
    events: Optional[ParsedInnings],
    date: str,
    opponent: str,
    aliases: PlayerAliases,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], bool]:
    """Batting and bowling rows for one side of one game.

    Args:
        lines: This side's scorecard lines.
        fielding: The other side's lines (the bowlers this side faced).
        events: Decoded events of this side's batting innings, or None.
        date: ISO date of the game.
        opponent: Label for the opposition column.
        aliases: Alias map, extended in place with any new player.

    Returns:
        (batting rows, bowling rows, whether events were available).
    """
    linked = link_innings(events, lines, fielding) if events else None
    dismissals: Dict[str, int] = {}
    if linked is not None:
        for delivery in linked.deliveries:
            if delivery.dismissed:
                dismissals[delivery.dismissed] = dismissals.get(delivery.dismissed, 0) + 1

    batting: List[Dict[str, object]] = []
    bowling: List[Dict[str, object]] = []
    for appearance in lines:
        alias = aliases.alias_for(appearance)
        has_batted = appearance.batting and appearance.batting.balls_faced > 0
        if has_batted and linked is not None:
            line = appearance.batting
            batting.append({
                "date": date,
                "opponent": opponent,
                "player_name": alias,
                "balls_faced": line.balls_faced,
                "runs_scored": line.runs,
                "fours": line.fours,
                "sixes": line.sixes,
                "dismissed": dismissals.get(appearance.name, 0),
                "retired": int(line.retired),
                "source": SOURCE,
            })
        if appearance.bowling and appearance.bowling.balls > 0:
            line = appearance.bowling
            bowling.append({
                "date": date,
                "opponent": opponent,
                "player_name": alias,
                "balls_bowled": line.balls,
                "runs_conceded": line.runs,
                "wickets": line.wickets,
                "wides": 0,
                "no_balls": 0,
                "source": SOURCE,
            })
    return batting, bowling, linked is not None


def alias_names(*paths: Path) -> Dict[str, str]:
    """Read alias -> name from alias-map files (these hold real names)."""
    names: Dict[str, str] = {}
    for path in paths:
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                names.update({r["alias"]: r["name"] for r in csv.DictReader(handle)})
    return names


def translate_aliases(text: str, names: Dict[str, str]) -> str:
    """Replace aliases (P01, O12, ...) in a report with player names.

    Only whole aliases are replaced, so ``P01`` never touches ``P010``;
    aliases with no known name are left as they are. The result holds real
    names: write it only to a private, gitignored location.
    """
    def swap(match: "re.Match[str]") -> str:
        return names.get(match.group(0), match.group(0))

    return re.sub(r"\b[PO]\d{2,3}\b", swap, text)


def build_game_rows(
    card: Scorecard,
    info: GameInfo,
    own_innings: Optional[ParsedInnings],
    their_innings: Optional[ParsedInnings],
    aliases: PlayerAliases,
    opposition_aliases: Optional[PlayerAliases] = None,
) -> GameRows:
    """Build scorebook rows for one game.

    Balls, runs, fours and sixes come from the scorecard; dismissals
    per batter come from the ball-by-ball events (the scorecard has no
    per-batter dismissal field, and U10 batters carry on after being
    out). Without events for an innings its batting rows are skipped
    rather than filled with zero dismissals, which would bias the
    dismissal rate downwards; bowling rows need no events.

    Args:
        card: Parsed scorecard for the game.
        info: Fixture facts (date, our side, opposition).
        own_innings: Decoded events of our batting innings, or None.
        their_innings: Decoded events of the opposition innings, or
            None.
        aliases: Alias map for our players, extended in place.
        opposition_aliases: Alias map for opposition players. When
            given, the opposition's rows are built too; they let the
            model learn the grade's population, not just our squad.

    Returns:
        The rows and any warnings.
    """
    ours = card.side(info.our_side)
    theirs = card.side("AWAY" if info.our_side == "HOME" else "HOME")
    result = GameRows()

    if own_innings is not None:
        for problem in reconcile(ours, own_innings, theirs):
            result.warnings.append(f"{info.game_id} own innings: {problem}")
    if their_innings is not None:
        for problem in reconcile(theirs, their_innings, ours):
            result.warnings.append(f"{info.game_id} opposition innings: {problem}")

    result.batting, result.bowling, had_own = _side_rows(
        ours, theirs, own_innings, info.date, info.opponent, aliases
    )
    if not had_own and any(a.batting for a in ours):
        result.warnings.append(
            f"{info.game_id}: no ball-by-ball events for our innings; "
            "batting rows skipped (bowling rows kept)"
        )
    if opposition_aliases is not None:
        result.opp_batting, result.opp_bowling, _ = _side_rows(
            theirs, ours, their_innings, info.date,
            info.our_team or "our team", opposition_aliases,
        )
    return result


BALL_COLUMNS = [
    "date", "game_id", "batter", "bowler", "runs", "boundary",
    "dismissed", "run_out",
]


def build_ball_rows(
    card: Scorecard,
    info: GameInfo,
    own_innings: Optional[ParsedInnings],
    their_innings: Optional[ParsedInnings],
    aliases: PlayerAliases,
    opposition_aliases: PlayerAliases,
) -> Tuple[List[Dict[str, object]], int]:
    """One row per delivery, with batter and bowler as aliases.

    This is the input to the joint batter-by-bowler model: every ball
    records who batted, who bowled and what happened, so batter and
    bowler effects can be separated instead of each absorbing the
    other's opposition.

    Args:
        card: Parsed scorecard for the game.
        info: Fixture facts.
        own_innings: Decoded events of our batting innings, or None.
        their_innings: Decoded events of the opposition innings, or None.
        aliases: Alias map for our players.
        opposition_aliases: Alias map for the opposition.

    Returns:
        (rows, number of deliveries dropped because a name could not be
        linked to a scorecard player).
    """
    ours = card.side(info.our_side)
    theirs = card.side("AWAY" if info.our_side == "HOME" else "HOME")
    alias_of = {("ours", a.name): aliases.alias_for(a) for a in ours}
    alias_of.update(
        {("theirs", a.name): opposition_aliases.alias_for(a) for a in theirs}
    )
    rows: List[Dict[str, object]] = []
    dropped = 0
    for innings, we_bat in ((own_innings, True), (their_innings, False)):
        if innings is None:
            continue
        batting, fielding = (ours, theirs) if we_bat else (theirs, ours)
        linked = link_innings(innings, batting, fielding)
        bat_key, bowl_key = ("ours", "theirs") if we_bat else ("theirs", "ours")
        for delivery in linked.deliveries:
            batter = alias_of.get((bat_key, delivery.striker))
            bowler = alias_of.get((bowl_key, delivery.bowler))
            if batter is None or bowler is None:
                dropped += 1
                continue
            rows.append({
                "date": info.date,
                "game_id": info.game_id,
                "batter": batter,
                "bowler": bowler,
                "runs": delivery.runs,
                # Fours, sixes and a four plus an overthrow ("5+").
                "boundary": int(delivery.runs in (4, 5, 6)),
                "dismissed": int(delivery.dismissal is not None),
                "run_out": int(delivery.dismissal == "run_out"),
            })
    return rows, dropped


@dataclass
class Fixture:
    """One completed game from a team's fixture.

    Attributes:
        game_id: PlayHQ game ID.
        date: ISO date of the game.
        our_side: ``HOME`` or ``AWAY``.
        opponent: Opposition team name.
        our_team: Our team's name.
        our_total: Our final team total, if published.
        their_total: The opposition's final total, if published.
    """

    game_id: str
    date: str
    our_side: str
    opponent: str
    our_team: str = ""
    our_total: Optional[int] = None
    their_total: Optional[int] = None


def _innings_total(side_result: Optional[Dict[str, object]]) -> Optional[int]:
    """Read TOTAL_SCORE from the first innings of a fixture ``result`` side."""
    for period in (side_result or {}).get("periods") or []:
        if (period.get("period") or {}).get("value") not in (
            "FIRST_INNINGS", "SECOND_INNINGS",
        ):
            continue
        for stat in period.get("statistics") or []:
            if (stat.get("type") or {}).get("value") == "TOTAL_SCORE":
                return int(stat["count"])
    return None


def completed_games(fixture: Dict[str, object]) -> List[Fixture]:
    """List a team's FINAL games with our side, the opposition and totals.

    Args:
        fixture: The ``team_fixture`` payload.

    Returns:
        Completed games in date order. Games where our team name is
        on neither or both sides are left out.
    """
    team_name = (fixture.get("discoverTeam") or {}).get("name")
    games: List[Fixture] = []
    for rnd in fixture.get("discoverTeamFixture") or []:
        for game in (rnd.get("fixture") or {}).get("games") or []:
            if (game.get("status") or {}).get("value") != "FINAL":
                continue
            home = (game.get("home") or {}).get("name")
            away = (game.get("away") or {}).get("name")
            if (home == team_name) == (away == team_name):
                continue
            side = "HOME" if home == team_name else "AWAY"
            result = game.get("result") or {}
            home_total = _innings_total(result.get("home"))
            away_total = _innings_total(result.get("away"))
            games.append(Fixture(
                game_id=game["id"],
                date=str(game["date"])[:10],
                our_side=side,
                opponent=(away if side == "HOME" else home) or "",
                our_team=team_name or "",
                our_total=home_total if side == "HOME" else away_total,
                their_total=away_total if side == "HOME" else home_total,
            ))
    return sorted(games, key=lambda g: g.date)


def write_rows(
    rows: List[Dict[str, object]], columns: List[str], path: Path
) -> pd.DataFrame:
    """Write scorebook rows to CSV, sorted by date then alias.

    Args:
        rows: Rows keyed by column name.
        columns: Column order (``BATTING_COLUMNS`` or ``BOWLING_COLUMNS``).
        path: Output CSV path; parent directories are created.

    Returns:
        The DataFrame that was written.
    """
    frame = pd.DataFrame(rows, columns=columns)
    frame = frame.sort_values(["date", "player_name"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


__all__ = [
    "BATTING_COLUMNS", "BOWLING_COLUMNS", "GameInfo", "GameRows",
    "PlayerAliases", "build_game_rows", "load_aliases", "save_aliases",
    "write_rows",
]
