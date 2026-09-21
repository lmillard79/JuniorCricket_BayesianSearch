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
from typing import Dict, Iterable, List, Optional, Tuple

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
    """

    game_id: str
    date: str
    our_side: str
    opponent: str


@dataclass
class GameRows:
    """Scorebook rows built for one game.

    Attributes:
        batting: Rows matching ``BATTING_COLUMNS``.
        bowling: Rows matching ``BOWLING_COLUMNS``.
        warnings: Data-quality notes worth surfacing to the user.
    """

    batting: List[Dict[str, object]] = field(default_factory=list)
    bowling: List[Dict[str, object]] = field(default_factory=list)
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


def load_aliases(path: Path) -> PlayerAliases:
    """Load an alias map written by ``save_aliases`` (empty if absent)."""
    if not path.exists():
        return PlayerAliases()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [(r["alias"], r["key"], r["name"]) for r in csv.DictReader(handle)]
    return PlayerAliases(rows)


def save_aliases(aliases: PlayerAliases, path: Path) -> None:
    """Write the alias map (this file holds names; keep it out of git)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alias", "key", "name"])
        writer.writerows(aliases.entries())


def build_game_rows(
    card: Scorecard,
    info: GameInfo,
    own_innings: Optional[ParsedInnings],
    their_innings: Optional[ParsedInnings],
    aliases: PlayerAliases,
) -> GameRows:
    """Build scorebook rows for our side of one game.

    Balls, runs, fours and sixes come from the scorecard; dismissals
    per batter come from the ball-by-ball events (the scorecard has no
    per-batter dismissal field, and U10 batters carry on after being
    out). Without events for our innings the batting rows are skipped
    rather than filled with zero dismissals, which would bias the
    dismissal rate downwards; bowling rows need no events.

    Args:
        card: Parsed scorecard for the game.
        info: Fixture facts (date, our side, opposition).
        own_innings: Decoded events of our batting innings, or None.
        their_innings: Decoded events of the opposition innings, or
            None; used only for the consistency check.
        aliases: Alias map, extended in place with any new player.

    Returns:
        The rows and any warnings.
    """
    ours = card.side(info.our_side)
    theirs = card.side("AWAY" if info.our_side == "HOME" else "HOME")
    result = GameRows()

    linked_own = link_innings(own_innings, ours, theirs) if own_innings else None
    if own_innings is not None:
        for problem in reconcile(ours, own_innings, theirs):
            result.warnings.append(f"{info.game_id} own innings: {problem}")
    if their_innings is not None:
        for problem in reconcile(theirs, their_innings, ours):
            result.warnings.append(f"{info.game_id} opposition innings: {problem}")

    dismissals: Dict[str, int] = {}
    if linked_own is not None:
        for delivery in linked_own.deliveries:
            if delivery.dismissed:
                dismissals[delivery.dismissed] = dismissals.get(delivery.dismissed, 0) + 1

    for appearance in ours:
        alias = aliases.alias_for(appearance)
        has_batted = appearance.batting and appearance.batting.balls_faced > 0
        if has_batted and linked_own is not None:
            line = appearance.batting
            result.batting.append({
                "date": info.date,
                "opponent": info.opponent,
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
            result.bowling.append({
                "date": info.date,
                "opponent": info.opponent,
                "player_name": alias,
                "balls_bowled": line.balls,
                "runs_conceded": line.runs,
                "wickets": line.wickets,
                "wides": 0,
                "no_balls": 0,
                "source": SOURCE,
            })
    if linked_own is None and any(a.batting for a in ours):
        result.warnings.append(
            f"{info.game_id}: no ball-by-ball events for our innings; "
            "batting rows skipped (bowling rows kept)"
        )
    return result


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
