"""Scouting a pool: who is strong in the other teams, from public PlayHQ scorecards.

White's opponents are unknown until they play, but every other game in the pool is
public. This reads the cached scorecards of those games (fetched by
``scripts/fetch_scorecards.py``), puts one line per player per game together, adds up
each player's games, and rates batters and bowlers with a prior so that a single
lucky or unlucky game does not top the list:

* A batter's rate of runs per ball and of dismissals per ball are each pulled toward
  the pool's average by ``BAT_PRIOR_BALLS`` balls' worth of average play, then turned
  into the expected runs of a 30-ball stint ended by a dismissal (the same value the
  batting-order work uses).
* A bowler's runs conceded and wickets per ball are pulled toward the pool's average
  by ``BOWL_PRIOR_BALLS`` balls.

With one game a player has 10 to 30 balls, so the prior dominates and the ranking is
a shortlist to watch, not a verdict. Each extra game moves it toward the player's own
record. A player's U10 record from White's and Navy's data (as an opponent last
season) is shown beside the rating but not mixed into it, because U11 scoring differs
from U10.

The scorecards hold children's names. Anything built from this stays under the
gitignored ``data/`` folder.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from junior_cricket.playhq_parse import balls_bowled

BAT_PRIOR_BALLS = 20.0
BOWL_PRIOR_BALLS = 12.0
STINT_BALLS = 30
INNINGS_PERIODS = ("FIRST_INNINGS", "SECOND_INNINGS")


@dataclass(frozen=True)
class GameMeta:
    """One finished game from a fixture.

    Attributes:
        game_id: PlayHQ game ID.
        round_name: For example ``"Round 1"``.
        date: ISO date.
        home: Home team name.
        away: Away team name.
        home_stats: Home innings statistics (``TOTAL_SCORE``, ``TOTAL_OUTS``,
            ``TOTAL_OVERS``, ``OVER_LIMIT``), empty if unpublished.
        away_stats: The same for the away side.
    """

    game_id: str
    round_name: str
    date: str
    home: str
    away: str
    home_stats: Dict[str, float] = field(default_factory=dict)
    away_stats: Dict[str, float] = field(default_factory=dict)


def _innings_stats(side_result: Optional[Dict[str, object]]) -> Dict[str, float]:
    """Flatten the first innings' statistics of a fixture ``result`` side."""
    for period in (side_result or {}).get("periods") or []:
        if (period.get("period") or {}).get("value") in INNINGS_PERIODS:
            return {
                s["type"]["value"]: s["count"]
                for s in period.get("statistics") or []
                if (s.get("type") or {}).get("value")
            }
    return {}


def final_games(fixture: Dict[str, object]) -> List[GameMeta]:
    """Every FINAL game in a team fixture, with both sides' innings figures.

    Args:
        fixture: A cached ``team_fixture`` payload.

    Returns:
        The finished games in date order.
    """
    games: List[GameMeta] = []
    for rnd in fixture.get("discoverTeamFixture") or []:
        for game in (rnd.get("fixture") or {}).get("games") or []:
            if (game.get("status") or {}).get("value") != "FINAL":
                continue
            result = game.get("result") or {}
            games.append(GameMeta(
                game_id=game["id"],
                round_name=str(rnd.get("name") or ""),
                date=str(game.get("date") or "")[:10],
                home=(game.get("home") or {}).get("name") or "",
                away=(game.get("away") or {}).get("name") or "",
                home_stats=_innings_stats(result.get("home")),
                away_stats=_innings_stats(result.get("away")),
            ))
    return sorted(games, key=lambda g: g.date)


def team_schedule(fixture: Dict[str, object], team_name: str) -> List[Dict[str, str]]:
    """A team's rounds in order: opponent, home or away, date and status.

    Args:
        fixture: A cached ``team_fixture`` payload.
        team_name: The team to follow.

    Returns:
        One dict per round; a bye has no opponent.
    """
    rows: List[Dict[str, str]] = []
    for rnd in fixture.get("discoverTeamFixture") or []:
        games = (rnd.get("fixture") or {}).get("games") or []
        if not games:
            rows.append({"round": str(rnd.get("name")), "date": "", "opponent": "bye",
                         "venue": "", "status": ""})
            continue
        for game in games:
            home = (game.get("home") or {}).get("name") or ""
            away = (game.get("away") or {}).get("name") or ""
            if team_name not in (home, away):
                continue
            rows.append({
                "round": str(rnd.get("name")),
                "date": str(game.get("date") or "")[:10],
                "opponent": away if home == team_name else home,
                "venue": "home" if home == team_name else "away",
                "status": (game.get("status") or {}).get("value") or "",
            })
    return rows


def overs_to_balls(overs: float) -> int:
    """Convert PlayHQ's ``17.1`` (17 overs, 1 ball) to a ball count."""
    whole = int(overs)
    return whole * 6 + int(round((overs - whole) * 10))


def player_games(card_json: Dict[str, object], meta: GameMeta) -> List[Dict[str, object]]:
    """One row per player with any batting, bowling or fielding record in a game.

    Args:
        card_json: The cached ``gameViewSpectator`` response.
        meta: The game's fixture entry, for team names.

    Returns:
        Rows keyed by profile ID (or the name when there is none), with the team,
        the opponent and the player's batting, bowling and fielding figures.
    """
    game = ((card_json.get("data") or {}).get("game")) or {}
    statistics = game.get("statistics") or {}
    rows: List[Dict[str, object]] = []
    for side, team, opponent in (("home", meta.home, meta.away), ("away", meta.away, meta.home)):
        for player in (statistics.get(side) or {}).get("players") or []:
            batting: Dict[str, float] = {}
            bowling: Dict[str, float] = {}
            fielding: Dict[str, float] = {}
            status = ""
            for entry in player.get("periodStatistics") or []:
                stats = {s["type"]["value"]: s["count"] for s in entry.get("statistics") or []}
                if "BALLS_FACED" in stats and not batting:
                    batting, status = stats, str(entry.get("status") or "")
                elif "OVERS" in stats and not bowling:
                    bowling = stats
                for name in ("TOTAL_CATCHES", "CATCHES_AS_FIELDER", "CATCHES_AS_WICKET_KEEPER",
                             "TOTAL_RUN_OUTS", "STUMPINGS"):
                    if name in stats:
                        fielding[name] = fielding.get(name, 0) + stats[name]
            if status == "DID_NOT_BAT":
                batting = {}
            if not (batting or bowling or fielding):
                continue
            catches = fielding.get("TOTAL_CATCHES",
                                   fielding.get("CATCHES_AS_FIELDER", 0)
                                   + fielding.get("CATCHES_AS_WICKET_KEEPER", 0))
            rows.append({
                "key": player.get("profileID") or str(player.get("name") or "").strip(),
                "name": str(player.get("name") or "").strip(),
                "team": team,
                "opponent": opponent,
                "game_id": meta.game_id,
                "round": meta.round_name,
                "date": meta.date,
                "batted": bool(batting),
                "balls": int(round(batting.get("BALLS_FACED", 0))),
                "runs": int(round(batting.get("TOTAL_RUNS", 0))),
                "fours": int(round(batting.get("FOURS", 0))),
                "sixes": int(round(batting.get("SIXES", 0))),
                "out": int(status == "OUT"),
                "retired": int(status == "RETIRED_NOT_OUT"),
                "bowled_balls": balls_bowled(bowling) if bowling else 0,
                "conceded": int(round(bowling.get("RUNS", 0))),
                "wickets": int(round(bowling.get("WICKETS", 0))),
                "wides": int(round(bowling.get("WIDES", 0))),
                "no_balls": int(round(bowling.get("NO_BALLS", 0))),
                "catches": int(round(catches)),
                "run_outs": int(round(fielding.get("TOTAL_RUN_OUTS", 0))),
                "stumpings": int(round(fielding.get("STUMPINGS", 0))),
            })
    return rows


def aggregate(rows: Iterable[Dict[str, object]]) -> pd.DataFrame:
    """Add each player's games together.

    Args:
        rows: Output of ``player_games`` over every game.

    Returns:
        One row per player, with the team(s), games played and summed figures.
    """
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    sums = ["balls", "runs", "fours", "sixes", "out", "retired", "bowled_balls", "conceded",
            "wickets", "wides", "no_balls", "catches", "run_outs", "stumpings"]
    grouped = frame.groupby("key", sort=False)
    out = grouped[sums].sum()
    out["name"] = grouped["name"].first()
    out["team"] = grouped["team"].agg(lambda s: " / ".join(dict.fromkeys(s)))
    out["games"] = grouped["game_id"].nunique()
    out["innings"] = grouped["batted"].sum().astype(int)
    return out.reset_index()


def stint_value(runs_per_ball: float, out_per_ball: float, balls: int = STINT_BALLS) -> float:
    """Expected runs of a stint of up to ``balls`` balls that a dismissal ends."""
    if out_per_ball < 1e-9:
        return balls * runs_per_ball
    return runs_per_ball * (1 - (1 - out_per_ball) ** balls) / out_per_ball


def add_ratings(
    players: pd.DataFrame,
    bat_prior: float = BAT_PRIOR_BALLS,
    bowl_prior: float = BOWL_PRIOR_BALLS,
) -> pd.DataFrame:
    """Rate batters and bowlers against the pool's average.

    Args:
        players: Output of ``aggregate``.
        bat_prior: Balls of average batting blended into every batter.
        bowl_prior: Balls of average bowling blended into every bowler.

    Returns:
        ``players`` plus raw rates and the shrunk ``bat_value`` (expected runs of a
        30-ball stint), ``bowl_economy`` (runs an over) and ``bowl_wickets`` (a
        4-over spell); NaN where the player did not bat or bowl.
    """
    if players.empty:
        return players
    p = players.copy()
    balls_total, bowled_total = p["balls"].sum(), p["bowled_balls"].sum()
    mu_runs = p["runs"].sum() / balls_total if balls_total else 0.0
    mu_out = p["out"].sum() / balls_total if balls_total else 0.0
    mu_conceded = p["conceded"].sum() / bowled_total if bowled_total else 0.0
    mu_wkts = p["wickets"].sum() / bowled_total if bowled_total else 0.0

    batted = p["balls"] > 0
    runs_s = (p["runs"] + bat_prior * mu_runs) / (p["balls"] + bat_prior)
    out_s = (p["out"] + bat_prior * mu_out) / (p["balls"] + bat_prior)
    p["runs_per_100"] = (100 * p["runs"] / p["balls"]).where(batted)
    p["bat_value"] = [
        stint_value(r, o) if b else float("nan") for r, o, b in zip(runs_s, out_s, batted)
    ]
    bowled = p["bowled_balls"] > 0
    conceded_s = (p["conceded"] + bowl_prior * mu_conceded) / (p["bowled_balls"] + bowl_prior)
    wkts_s = (p["wickets"] + bowl_prior * mu_wkts) / (p["bowled_balls"] + bowl_prior)
    p["raw_economy"] = (6 * p["conceded"] / p["bowled_balls"]).where(bowled)
    p["bowl_economy"] = (6 * conceded_s).where(bowled)
    p["bowl_wickets"] = (24 * wkts_s).where(bowled)
    return p


def team_table(games: Sequence[GameMeta]) -> pd.DataFrame:
    """Results so far: for each team, its record and run rates.

    Args:
        games: Finished games, each counted once.

    Returns:
        One row per team: played, won, lost, runs for and against, balls faced
        and bowled, wickets lost, and runs per over for and against.
    """
    rows: Dict[str, Dict[str, float]] = {}
    for g in games:
        for team, mine, theirs in ((g.home, g.home_stats, g.away_stats),
                                   (g.away, g.away_stats, g.home_stats)):
            r = rows.setdefault(team, dict(played=0, won=0, lost=0, runs_for=0, runs_against=0,
                                           balls_faced=0, balls_bowled=0, wickets_lost=0))
            r["played"] += 1
            for_, against = mine.get("TOTAL_SCORE"), theirs.get("TOTAL_SCORE")
            if for_ is not None and against is not None:
                r["won"] += int(for_ > against)
                r["lost"] += int(for_ < against)
                r["runs_for"] += for_
                r["runs_against"] += against
            r["balls_faced"] += overs_to_balls(mine.get("TOTAL_OVERS", 0))
            r["balls_bowled"] += overs_to_balls(theirs.get("TOTAL_OVERS", 0))
            r["wickets_lost"] += int(mine.get("TOTAL_OUTS", 0))
    table = pd.DataFrame.from_dict(rows, orient="index")
    if table.empty:
        return table
    table["rpo_for"] = 6 * table["runs_for"] / table["balls_faced"].where(table["balls_faced"] > 0)
    table["rpo_against"] = 6 * table["runs_against"] / table["balls_bowled"].where(table["balls_bowled"] > 0)
    return table.reset_index(names="team")


def u10_history(
    keys: Iterable[str],
    sources: Sequence[Tuple[Path, Path, Path]],
) -> pd.DataFrame:
    """Last season's record against White or Navy for players who were opponents.

    Args:
        keys: Profile IDs to look up.
        sources: For each earlier data folder: its ``opposition_map.csv``, its
            ``playhq_opp_batting.csv`` and its ``playhq_opp_bowling.csv``.

    Returns:
        One row per player found: U10 games, balls, runs, dismissals and bowling
        balls, runs and wickets, summed over the sources.
    """
    wanted = set(keys)
    pieces: List[pd.DataFrame] = []
    for map_path, batting_path, bowling_path in sources:
        if not (map_path.exists() and batting_path.exists() and bowling_path.exists()):
            continue
        with map_path.open(newline="", encoding="utf-8") as handle:
            alias_key = {r["alias"]: r["key"] for r in csv.DictReader(handle) if r["key"] in wanted}
        if not alias_key:
            continue
        bat = pd.read_csv(batting_path)
        bat = bat[bat["player_name"].isin(alias_key)]
        bowl = pd.read_csv(bowling_path)
        bowl = bowl[bowl["player_name"].isin(alias_key)]
        merged = pd.DataFrame({"key": list(alias_key.values()), "alias": list(alias_key)})
        merged = merged.merge(
            bat.groupby("player_name").agg(
                u10_games=("date", "nunique"), u10_balls=("balls_faced", "sum"),
                u10_runs=("runs_scored", "sum"), u10_out=("dismissed", "sum")),
            left_on="alias", right_index=True, how="left")
        merged = merged.merge(
            bowl.groupby("player_name").agg(
                u10_bowled=("balls_bowled", "sum"), u10_conceded=("runs_conceded", "sum"),
                u10_wickets=("wickets", "sum")),
            left_on="alias", right_index=True, how="left")
        pieces.append(merged.drop(columns="alias"))
    if not pieces:
        return pd.DataFrame(columns=["key"])
    return pd.concat(pieces).groupby("key", as_index=False).sum(numeric_only=True)
