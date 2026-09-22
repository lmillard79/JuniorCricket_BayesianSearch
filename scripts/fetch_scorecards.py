"""Fetch a team's PlayHQ scorecards and build pseudonymised scorebook CSVs.

For each team ID: the fixture (with results), then for every completed
game the per-player scorecard and the ball-by-ball events of both
innings. Everything fetched is cached under ``data/raw/playhq/``
(which holds player names and is gitignored) and is never requested
twice, so a run that is interrupted or blocked simply resumes. The
scorebook CSVs written to ``data/processed/`` carry aliases (P01, ...)
instead of names, in the same schema as the manual scorebook.

PlayHQ rate limits these public endpoints: a burst earns a CloudFront
403. Requests are spaced out (``--min-interval``), a per-run budget
(``--max-games``) keeps runs small, and the first 403 or 429 stops all
fetching for the run (the CSVs are still built from what is cached).
For anything ongoing, PlayHQ's official API with a key is the proper
route (https://docs.playhq.com/tech/).

Examples:
    python scripts/fetch_scorecards.py --team-id 75cdae66 --team-id fafdb4c4
    python scripts/fetch_scorecards.py --team-id 75cdae66 --offline
    python scripts/fetch_scorecards.py --team-id 75cdae66 --max-games 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from junior_cricket.data_loader import BATTING_COLUMNS, BOWLING_COLUMNS
from junior_cricket.logging_setup import setup_logging
from junior_cricket.playhq_client import (
    MIN_REQUEST_INTERVAL_SECONDS,
    PlayHQAPIError,
    PlayHQBlockedError,
    PlayHQClient,
)
from junior_cricket.playhq_parse import AWAY, HOME, parse_events, parse_scorecard
from junior_cricket.playhq_scorebook import (
    BALL_COLUMNS,
    GameInfo,
    build_ball_rows,
    build_game_rows,
    completed_games,
    load_aliases,
    save_aliases,
    write_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "playhq"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

LOGGER_NAME = "fetch_scorecards"


def _read(path: Path) -> Optional[Any]:
    """Load a cached JSON file, or None when absent."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, payload: Any) -> None:
    """Cache one JSON payload verbatim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=True)


def main() -> None:
    """Fetch (or load cached) scorecards and write the scorebook CSVs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--team-id", action="append", required=True,
                        help="PlayHQ team ID (repeat for several teams)")
    parser.add_argument("--offline", action="store_true",
                        help="Build from cached files only; no requests")
    parser.add_argument("--min-interval", type=float,
                        default=MIN_REQUEST_INTERVAL_SECONDS,
                        help="Minimum seconds between requests")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Fetch at most this many games this run")
    parser.add_argument("--data-dir", default=None,
                        help="Use data/<name>/ instead of data/ (a second team, kept "
                             "fully separate: its own raw cache, alias numbering and "
                             "processed CSVs, so it cannot overwrite the default team's)")
    args = parser.parse_args()

    global RAW_DIR, PROCESSED_DIR
    if args.data_dir:
        base = REPO_ROOT / "data" / args.data_dir
        RAW_DIR, PROCESSED_DIR = base / "raw" / "playhq", base / "processed"

    logger = setup_logging(LOGGER_NAME, PROCESSED_DIR)
    logger.info("Input arguments: %s", vars(args))
    client = PlayHQClient(min_interval=args.min_interval, logger=logger)
    blocked = False
    fetch_budget = args.max_games

    def can_fetch() -> bool:
        return not args.offline and not blocked

    aliases = load_aliases(RAW_DIR / "player_map.csv")
    opposition_aliases = load_aliases(RAW_DIR / "opposition_map.csv", prefix="O")
    batting_rows: List[Dict[str, object]] = []
    bowling_rows: List[Dict[str, object]] = []
    opp_batting_rows: List[Dict[str, object]] = []
    opp_bowling_rows: List[Dict[str, object]] = []
    ball_rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    n_cached = n_fetched = n_partial = n_missing = 0

    for team_id in args.team_id:
        fixture_path = RAW_DIR / "fixtures" / f"team_{team_id}.json"
        fixture = _read(fixture_path)
        if fixture is None and can_fetch():
            try:
                fixture = client.team_fixture(team_id)
                _write(fixture_path, fixture)
                logger.info("Fetched fixture for team %s", team_id)
            except PlayHQBlockedError as exc:
                blocked = True
                logger.error("Blocked: %s", exc)
            except PlayHQAPIError as exc:
                logger.error("Fixture for team %s failed: %s", team_id, exc)
        if fixture is None:
            logger.warning("No fixture for team %s; skipped", team_id)
            continue

        for game in completed_games(fixture):
            game_dir = RAW_DIR / "games" / game.game_id
            card_json = _read(game_dir / "scorecard.json")
            events: Dict[str, Optional[List[Dict[str, Any]]]] = {
                side: _read(game_dir / f"events_{side}.json")
                for side in (HOME, AWAY)
            }
            needs_fetch = card_json is None or any(v is None for v in events.values())
            if needs_fetch and can_fetch() and (fetch_budget is None or fetch_budget > 0):
                try:
                    if card_json is None:
                        card_json = client.game_scorecard(game.game_id)
                        _write(game_dir / "scorecard.json", card_json)
                    for side in (HOME, AWAY):
                        if events[side] is None:
                            events[side] = client.game_events(game.game_id, side)
                            _write(game_dir / f"events_{side}.json", events[side])
                    if fetch_budget is not None:
                        fetch_budget -= 1
                    logger.info("Fetched game %s (%s)", game.game_id, game.date)
                except PlayHQBlockedError as exc:
                    blocked = True
                    logger.error("Blocked on game %s: %s", game.game_id, exc)
                except PlayHQAPIError as exc:
                    logger.warning("Game %s fetch failed: %s", game.game_id, exc)

            incomplete = card_json is None or any(v is None for v in events.values())
            if incomplete:
                n_partial += 1
            elif needs_fetch:
                n_fetched += 1
            else:
                n_cached += 1

            if card_json is None:
                n_missing += 1
                warnings.append(f"{game.game_id}: no scorecard; game skipped")
                continue
            other = AWAY if game.our_side == HOME else HOME
            own = (parse_events(events[game.our_side], game.our_side)
                   if events[game.our_side] else None)
            theirs = (parse_events(events[other], other)
                      if events[other] else None)
            rows = build_game_rows(
                parse_scorecard(card_json),
                GameInfo(game.game_id, game.date, game.our_side,
                         game.opponent, game.our_team),
                own, theirs, aliases, opposition_aliases,
            )
            batting_rows.extend(rows.batting)
            bowling_rows.extend(rows.bowling)
            opp_batting_rows.extend(rows.opp_batting)
            opp_bowling_rows.extend(rows.opp_bowling)
            warnings.extend(rows.warnings)
            game_balls, dropped = build_ball_rows(
                parse_scorecard(card_json),
                GameInfo(game.game_id, game.date, game.our_side,
                         game.opponent, game.our_team),
                own, theirs, aliases, opposition_aliases,
            )
            ball_rows.extend(game_balls)
            if dropped:
                warnings.append(
                    f"{game.game_id}: {dropped} deliveries dropped (name not linked)"
                )

    batting = write_rows(batting_rows, BATTING_COLUMNS, PROCESSED_DIR / "playhq_batting.csv")
    bowling = write_rows(bowling_rows, BOWLING_COLUMNS, PROCESSED_DIR / "playhq_bowling.csv")
    # Our squad plus the opposition, so the model learns the whole grade.
    write_rows(opp_batting_rows, BATTING_COLUMNS, PROCESSED_DIR / "playhq_opp_batting.csv")
    write_rows(opp_bowling_rows, BOWLING_COLUMNS, PROCESSED_DIR / "playhq_opp_bowling.csv")
    write_rows(batting_rows + opp_batting_rows, BATTING_COLUMNS,
               PROCESSED_DIR / "playhq_all_batting.csv")
    write_rows(bowling_rows + opp_bowling_rows, BOWLING_COLUMNS,
               PROCESSED_DIR / "playhq_all_bowling.csv")
    save_aliases(aliases, RAW_DIR / "player_map.csv")
    save_aliases(opposition_aliases, RAW_DIR / "opposition_map.csv")
    balls = pd.DataFrame(ball_rows, columns=BALL_COLUMNS).sort_values(
        ["date", "game_id"], kind="stable"
    )
    balls.to_csv(PROCESSED_DIR / "playhq_balls.csv", index=False)
    logger.info("Ball rows: %d deliveries across %d games -> playhq_balls.csv",
                len(balls), balls["game_id"].nunique() if len(balls) else 0)

    logger.info(
        "Games: %d complete from cache, %d fetched now, %d incomplete "
        "(some events or the scorecard missing), of which %d have no scorecard",
        n_cached, n_fetched, n_partial, n_missing)
    logger.info("Batting rows: %d (%d players); bowling rows: %d (%d players)",
                len(batting), batting["player_name"].nunique() if len(batting) else 0,
                len(bowling), bowling["player_name"].nunique() if len(bowling) else 0)
    logger.info("Opposition rows: %d batting, %d bowling (%d players)",
                len(opp_batting_rows), len(opp_bowling_rows),
                len(opposition_aliases.entries()))
    logger.info("Alias maps (hold names, keep out of git): %s and %s",
                RAW_DIR / "player_map.csv", RAW_DIR / "opposition_map.csv")
    for warning in warnings[:25]:
        logger.warning("Data note: %s", warning)
    if len(warnings) > 25:
        logger.warning("... and %d more data notes", len(warnings) - 25)
    if blocked:
        logger.error(
            "PlayHQ blocked a request, so fetching stopped early. CSVs were "
            "built from cached data only. Re-run later to resume; do not retry "
            "immediately."
        )


if __name__ == "__main__":
    main()
