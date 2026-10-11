"""Scout a pool: who is strong in the other teams, and what it takes to beat them.

Reads the fixtures and scorecards already cached by ``fetch_scorecards.py`` (no
requests), adds each player's games together, rates batters and bowlers against the
pool's average, rates the teams from their results, and writes a private report that
names the players, so a coach can see who to watch and what total to aim for. Also shows
each player's U10 record against White or Navy last season where there is one.

The report holds children's names. It is written under the gitignored ``data/``
folder; do not commit or post it.

Typical use, after Saturday's games:
    python scripts/fetch_scorecards.py --data-dir smith_pool --refresh \\
        --team-id 44faa9cd --team-id ba4abc43 --team-id 2ca59b41 --team-id 050b5216 --team-id a7cd7e29
    python scripts/fetch_scorecards.py --data-dir u11_white --refresh --team-id 2320bb54
    python scripts/scout_pool.py
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from junior_cricket.logging_setup import setup_logging
from junior_cricket.scouting import (
    GameMeta,
    TeamModel,
    add_ratings,
    aggregate,
    batting_shape,
    club_of,
    final_games,
    fit_team_strengths,
    matchup,
    player_games,
    strength_table,
    team_schedule,
    team_table,
    u10_club_strengths,
    u10_history,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER_NAME = "scout_pool"
OUR_TEAM_ID = "2320bb54"
TOP_BATTERS = 12
TOP_BOWLERS = 10
LONG_GAME_SCALE = 25 / 22


def _raw(data_dir: Optional[str]) -> Path:
    base = REPO_ROOT / "data" / data_dir if data_dir else REPO_ROOT / "data"
    return base / "raw" / "playhq"


def _load_fixtures(dirs: List[str]) -> List[dict]:
    fixtures = []
    for d in dirs:
        for path in sorted((_raw(d) / "fixtures").glob("team_*.json")):
            with path.open(encoding="utf-8") as handle:
                fixtures.append(json.load(handle))
    return fixtures


def _find_card(game_id: str, dirs: List[str]) -> Optional[dict]:
    for d in dirs:
        path = _raw(d) / "games" / game_id / "scorecard.json"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
    return None


def _bat_text(r: pd.Series) -> str:
    if not r["balls"] and not r["innings"]:
        return ""
    mark = " ret" if r["retired"] else ("" if r["out"] else "*")
    return f"{int(r['runs'])} ({int(r['balls'])}){mark}"


def _bowl_text(r: pd.Series) -> str:
    if not r["bowled_balls"]:
        return ""
    overs, rem = divmod(int(r["bowled_balls"]), 6)
    extras = int(r["wides"] + r["no_balls"])
    return f"{int(r['wickets'])}/{int(r['conceded'])} off {overs}.{rem}" + (f" ({extras} extras)" if extras else "")


def _history_text(r: pd.Series) -> str:
    parts = []
    if r.get("u10_balls", 0) and r["u10_balls"] > 0:
        parts.append(f"bat {int(r['u10_runs'])} off {int(r['u10_balls'])}, {int(r['u10_out'])} out")
    if r.get("u10_bowled", 0) and r["u10_bowled"] > 0:
        parts.append(f"bowl {int(r['u10_wickets'])}/{int(r['u10_conceded'])} off {int(r['u10_bowled'])}")
    return "; ".join(parts)


def _table(frame: pd.DataFrame, columns: Dict[str, str], floats: Dict[str, int] = None) -> List[str]:
    """Markdown table from a frame, renaming columns and rounding some."""
    floats = floats or {}
    lines = ["| " + " | ".join(columns.values()) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            v = row[col]
            if isinstance(v, float):
                v = "" if pd.isna(v) else f"{v:.{floats.get(col, 1)}f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _with_threat(players: pd.DataFrame) -> pd.DataFrame:
    """Add how quickly a batter scores and what their wicket is worth."""
    p = players.copy()
    batted = p["balls"] > 0
    p["fours_per_10"] = (10 * p["fours"] / p["balls"]).where(batted)
    p["above_average"] = (p["bat_value"] - p.loc[batted, "bat_value"].mean()).where(batted)
    return p


def build_report(
    games: List[GameMeta], players: pd.DataFrame, teams: pd.DataFrame, schedule: List[Dict[str, str]],
    our_full: str, today: date, shape: pd.DataFrame, informed: TeamModel, results_only: TeamModel,
) -> str:
    """The Markdown scouting report."""
    our_name = our_full.replace("VDCC ", "")
    players = _with_threat(players)
    out: List[str] = [
        "# Pool scouting report (private)", "",
        f"Built {today.isoformat()} from {len(games)} finished game(s) in the pool. **Contains "
        "children's names from public PlayHQ scorecards: keep it local, do not commit or post it.**",
        "",
        "Everyone has played one game or fewer, so read the rankings as a shortlist to watch and the "
        "team numbers as a first guide, not a verdict. Both firm up each week (section 7).", "",
        "## 1. The pool so far", "",
    ]
    t = teams.sort_values(["won", "rpo_for"], ascending=False).copy()
    if "extras_given" not in t:
        t["extras_given"] = float("nan")
    out += _table(t, {"team": "Team", "played": "P", "won": "W", "lost": "L", "runs_for": "Runs for",
                      "runs_against": "Runs against", "rpo_for": "Runs/over for",
                      "rpo_against": "Runs/over against", "wickets_lost": "Wickets lost",
                      "extras_given": "Extras conceded a game"}, {"rpo_for": 2, "rpo_against": 2})
    out += ["", "Round 1 games were cut to 22 overs a side in the Smith pool.", "",
            f"## 2. {our_name}'s draw and what we know", "",
            "| Round | Date | Opponent | Venue | What we know |", "| --- | --- | --- | --- | --- |"]
    by_team = {r["team"]: r for _, r in teams.iterrows()}
    for row in schedule:
        opp = row["opponent"]
        if opp == "bye":
            know = ""
        elif opp in by_team:
            r = by_team[opp]
            know = (f"played {int(r['played'])}, won {int(r['won'])}; scored {r['rpo_for']:.1f} an over, "
                    f"conceded {r['rpo_against']:.1f}")
        else:
            know = "no game yet, so nothing to go on"
        done = " (played)" if row["status"] == "FINAL" else ""
        out.append(f"| {row['round']} | {row['date']} | {opp}{done} | {row['venue']} | {know} |")

    # --- what it takes to beat them
    out += ["", "## 3. What it takes to beat each opponent", "",
            "A rough model on team totals, not on players. Each innings total is the pool's average plus "
            "the batting team's strength minus the fielding team's, plus about 24 runs of game luck. "
            "Strengths are pulled toward average, so a big innings against a weak attack counts for "
            "little, and each club starts from last season's record against White and Navy (the Bulls "
            "about level with us, WNJCC well below). Totals are for a 22-over game, with a 25-over game "
            "in brackets (scaled, so rougher).", "",
            "| Round | Opponent | Their typical total against us | Our chance | Total for an even chance | For 3 in 4 | For 9 in 10 | Our chance on results alone |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    seen = set()
    for row in schedule:
        opp = row["opponent"]
        if opp == "bye" or opp in seen:
            continue
        seen.add(opp)
        m, m25 = matchup(informed, our_full, opp), matchup(informed, our_full, opp, LONG_GAME_SCALE)
        r = matchup(results_only, our_full, opp)
        unplayed = opp not in by_team
        label = f"{opp} (no games yet: an average team)" if unplayed else opp
        if row["status"] == "FINAL":
            label += " (played)"
        out.append(
            f"| {row['round']} | {label} | {m['their_median']:.0f} ({m25['their_median']:.0f}) | {m['p_win']:.0%} | "
            f"{m['need_50']:.0f} ({m25['need_50']:.0f}) | {m['need_75']:.0f} ({m25['need_75']:.0f}) | "
            f"{m['need_90']:.0f} ({m25['need_90']:.0f}) | {r['p_win']:.0%} |")
    st = strength_table(informed).sort_values("strength", ascending=False)
    out += ["", "How strong each team looks (runs a 22-over innings better or worse than an average team, "
            "with the spread of what we do not yet know):", ""]
    out += _table(st, {"team": "Team", "strength": "Strength", "sd": "Plus or minus"})
    out += ["", f"Pool average innings so far: {informed.mean[0]:.0f} runs in 22 overs. The spreads are as big as "
            "the gaps between teams, so after one round the model cannot separate most of them; the one "
            "clear call is that the WNJCC sides are the weakest.", ""]

    # --- players
    out += ["## 4. Players to watch", ""]
    batters = players[players["balls"] > 0].sort_values("bat_value", ascending=False).head(TOP_BATTERS).copy()
    batters["bat"] = batters.apply(_bat_text, axis=1)
    batters["last_season"] = batters.apply(_history_text, axis=1)
    out += ["### Batters: whose wicket is worth most", "",
            "Ranked by the expected runs of a 30-ball stint (pulled toward the pool average, so one lucky "
            "innings does not top the list). \"Worth\" is how many more runs than an average batter that "
            "stint is worth, which is roughly what getting them out early saves. \"Quick\" scorers show in "
            "runs per 100 balls and fours per 10 balls.", ""]
    out += _table(batters, {"name": "Player", "team": "Team", "bat": "Runs (balls)", "fours": "4s",
                            "fours_per_10": "4s per 10 balls", "runs_per_100": "Runs/100",
                            "bat_value": "30-ball value", "above_average": "Worth over average",
                            "last_season": "U10 against White or Navy"}, {"runs_per_100": 0})
    out += ["", "`*` not out, `ret` retired not out.", ""]
    bowlers = players[players["bowled_balls"] > 0].sort_values("bowl_economy").head(TOP_BOWLERS).copy()
    bowlers["bowl"] = bowlers.apply(_bowl_text, axis=1)
    bowlers["last_season"] = bowlers.apply(_history_text, axis=1)
    out += ["### Bowlers (by runs an over, pulled toward the pool average; wickets in a 4-over spell)", ""]
    out += _table(bowlers, {"name": "Player", "team": "Team", "bowl": "Wkts/runs off overs",
                            "raw_economy": "Raw runs/over", "bowl_economy": "Rated runs/over",
                            "bowl_wickets": "Rated wkts/4 overs", "last_season": "U10 against White or Navy"})
    fielders = players[(players["catches"] + players["run_outs"] + players["stumpings"]) > 0]
    if not fielders.empty:
        def credit(r: pd.Series) -> str:
            parts = [f"{int(n)} {word}{'' if n == 1 else 's'}" for n, word in
                     ((r["catches"], "catch" if r["catches"] == 1 else "catche"),
                      (r["run_outs"], "run out"), (r["stumpings"], "stumping")) if n]
            return f"{r['name']} ({r['team']}) " + ", ".join(parts).replace("catchess", "catches")
        out += ["", "Fielding credit so far: " + "; ".join(credit(r) for _, r in fielders.iterrows()) + "."]

    # --- batting shape
    out += ["", "## 5. Batting shape: top-heavy or deep?", ""]
    if shape.empty:
        out += ["No batting innings with player lines yet.", ""]
    else:
        s = shape.copy()
        s["extras_share"] = s["extras"] / s["total"]
        out += _table(s, {"team": "Team", "round": "Round", "total": "Total", "batter_runs": "Runs off the bat",
                          "extras": "Extras", "top3_share": "Top three's share of bat runs",
                          "ten_plus": "Batters on 10 or more", "retired": "Retired"}, {"top3_share": 2})
        out += ["", "A high top-three share with few batters on 10 means the runs came from a handful, so "
                "getting those few out early matters most. Real consistency needs several games per player: "
                "this is one innings per team so far, and where a player has a U10 record it is in section 4 "
                "and 6.", ""]

    # --- team by team
    out += ["## 6. Team by team, in the order White meets them", ""]
    order = []
    for row in schedule:
        if row["opponent"] != "bye" and row["opponent"] not in order:
            order.append(row["opponent"])
    for team in order:
        meets = [f"{r['round']} ({r['venue']})" for r in schedule if r["opponent"] == team]
        out += [f"### {team}", "", f"White meets them in {', '.join(meets)}.", ""]
        roster = players[players["team"].str.contains(team, regex=False)].copy()
        if roster.empty:
            played = int(by_team[team]["played"]) if team in by_team else 0
            out += [(f"Played {played} game(s), but PlayHQ holds no player lines for them yet (the scorer "
                     "did not use full scoring). " if played else "No finished games yet. ")
                    + "This section fills in as lines appear.", ""]
            continue
        game_lines = [f"- {g.round_name}: {g.home} {int(g.home_stats.get('TOTAL_SCORE', 0))}/"
                      f"{int(g.home_stats.get('TOTAL_OUTS', 0))} v {g.away} "
                      f"{int(g.away_stats.get('TOTAL_SCORE', 0))}/{int(g.away_stats.get('TOTAL_OUTS', 0))}"
                      for g in games if team in (g.home, g.away)]
        out += game_lines + [""]
        roster["bat"] = roster.apply(_bat_text, axis=1)
        roster["bowl"] = roster.apply(_bowl_text, axis=1)
        roster["last_season"] = roster.apply(_history_text, axis=1)
        roster = roster.sort_values("bat_value", ascending=False)
        out += _table(roster, {"name": "Player", "bat": "Bat", "fours": "4s", "bowl": "Bowl",
                               "catches": "Catches", "bat_value": "30-ball value",
                               "last_season": "U10 against White or Navy"})
        out += [""]

    out += ["## 7. How to read this", "",
            "- **30-ball value**: the expected runs from a stint of up to 30 balls that a dismissal ends, "
            "from the batter's runs and dismissals per ball, each blended with 20 balls of average play. "
            "It is the same value the batting-order work uses. Balls include wides and no-balls "
            "(rule 17.9(v)).",
            "- **Rated runs/over and wickets**: the bowler's runs conceded and wickets per ball, each blended "
            "with 12 balls (two overs) of average bowling. Runs conceded include wides and no-balls.",
            "- **U10 against White or Navy**: the player's record as an opponent in 2025/26. It is shown "
            "beside the rating, not mixed into it, because U11 scoring is lower than U10 scoring.",
            "- **Team strength and the totals in section 3**: innings totals only, so it knows nothing about "
            "who played. Assumes about 24 runs of game-to-game luck and about 14 runs of real difference "
            "between teams (the spread seen in the first eight U11 innings); the chance of winning moves "
            "only a few points if those are changed. It adds one number per team for batting and bowling "
            "together, and starts each club from its record against White and Navy last season. 25-over "
            "figures are the 22-over ones scaled by 25/22.",
            "- A player missing from a list may simply not have been in a game yet."]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="smith_pool", help="Folder under data/ holding the pool's cache")
    parser.add_argument("--also-dir", action="append", default=None,
                        help="Other folders whose cache to include (default: u11_white)")
    parser.add_argument("--our-team-id", default=OUR_TEAM_ID)
    parser.add_argument("--history-dir", action="append", default=None,
                        help="Earlier-season folders for U10 history; '' is data/ itself (default: '' and navy)")
    parser.add_argument("--tau", type=float, default=14.0, help="Spread of true team strengths (runs)")
    parser.add_argument("--sigma", type=float, default=24.0, help="Game-to-game luck in an innings total (runs)")
    parser.add_argument("--out", default=None, help="Default: data/<data-dir>/outputs/scouting")
    args = parser.parse_args()

    dirs = [args.data_dir] + (args.also_dir if args.also_dir is not None else ["u11_white"])
    history_dirs = args.history_dir if args.history_dir is not None else ["", "navy"]
    out = Path(args.out) if args.out else REPO_ROOT / "data" / args.data_dir / "outputs" / "scouting"
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOGGER_NAME, out)
    logger.info("Input arguments: %s", vars(args))

    fixtures = _load_fixtures(dirs)
    games: Dict[str, GameMeta] = {}
    for fx in fixtures:
        for g in final_games(fx):
            games.setdefault(g.game_id, g)
    ordered = sorted(games.values(), key=lambda g: (g.date, g.game_id))
    rows = []
    n_cards = 0
    for g in ordered:
        card = _find_card(g.game_id, dirs)
        if card is None:
            logger.warning("No cached scorecard for game %s", g.game_id)
            continue
        n_cards += 1
        rows += player_games(card, g)
    logger.info("%d finished games, %d scorecards, %d player-game rows", len(ordered), n_cards, len(rows))

    players = add_ratings(aggregate(rows))
    if players.empty:
        logger.error("No player lines found; fetch the pool's games first")
        return
    sources = [(_raw(d) / "opposition_map.csv",
                (REPO_ROOT / "data" / d / "processed" if d else REPO_ROOT / "data" / "processed") / "playhq_opp_batting.csv",
                (REPO_ROOT / "data" / d / "processed" if d else REPO_ROOT / "data" / "processed") / "playhq_opp_bowling.csv")
               for d in history_dirs]
    history = u10_history(players["key"], sources)
    players = players.merge(history, on="key", how="left")
    for col in ("u10_games", "u10_balls", "u10_runs", "u10_out", "u10_bowled", "u10_conceded", "u10_wickets"):
        if col not in players:
            players[col] = 0
        players[col] = players[col].fillna(0)

    our_name = next((fx["discoverTeam"]["name"] for fx in fixtures
                     if (fx.get("discoverTeam") or {}).get("id") == args.our_team_id), "Our team")
    our_fixture = next((fx for fx in fixtures if (fx.get("discoverTeam") or {}).get("id") == args.our_team_id), {})
    schedule = team_schedule(our_fixture, our_name) if our_fixture else []
    teams = team_table(ordered)
    if rows:
        frame = pd.DataFrame(rows)
        given = (frame["wides"] + frame["no_balls"]).groupby(frame["team"]).sum()
        teams["extras_given"] = teams["team"].map(given) / teams["played"]

    # Team strengths: from results alone, and starting each club from last season's record.
    pool_teams = [our_name] + [r["opponent"] for r in schedule if r["opponent"] != "bye"]
    history_fixtures = []
    for d in history_dirs:
        folder = _raw(d) / "fixtures"
        for path in sorted(folder.glob("team_*.json")):
            with path.open(encoding="utf-8") as handle:
                history_fixtures.append(json.load(handle))
    clubs = u10_club_strengths(history_fixtures)
    priors = {team: clubs[club_of(team)][0] for team in pool_teams
              if team != our_name and club_of(team) in clubs}
    informed = fit_team_strengths(ordered, pool_teams, tau=args.tau, sigma=args.sigma, strength_priors=priors)
    results_only = fit_team_strengths(ordered, pool_teams, tau=args.tau, sigma=args.sigma)
    logger.info("Club priors from last season: %s", {k: round(v, 1) for k, v in priors.items()})
    shape = batting_shape(rows, ordered)

    players.to_csv(out / "players.csv", index=False)
    teams.to_csv(out / "teams.csv", index=False)
    strength_table(informed).to_csv(out / "team_strengths.csv", index=False)
    report = build_report(ordered, players, teams, schedule, our_name, date.today(), shape, informed, results_only)
    (out / "scouting_report.md").write_text(report, encoding="utf-8")
    logger.info("Wrote %s (%d players, %d with U10 history)", out / "scouting_report.md",
                len(players), int((players["u10_balls"] + players["u10_bowled"] > 0).sum()))


if __name__ == "__main__":
    main()
