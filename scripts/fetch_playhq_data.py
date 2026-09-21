"""Fetch BNJCA data from the PlayHQ public GraphQL API.

Saves raw JSON responses under ``data/raw/playhq/`` and, where
applicable, flattened CSV extracts under ``data/processed/``.

Examples:
    python scripts/fetch_playhq_data.py --game-id f52ca224
    python scripts/fetch_playhq_data.py --grade-id 2bb24d79
    python scripts/fetch_playhq_data.py --stats-grade <grade-id>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from junior_cricket.logging_setup import setup_logging
from junior_cricket.playhq_client import PlayHQClient, PlayHQAPIError

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "playhq"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

LOGGER_NAME = "fetch_playhq_data"


def save_json(name: str, payload: dict, logger: logging.Logger) -> Path:
    """Write one raw JSON payload with a stable filename.

    Args:
        name: Filename stem.
        payload: JSON-serialisable dict.
        logger: Logger for the audit trail.

    Returns:
        Path of the written file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    logger.info("Saved raw payload: %s", path)
    return path


def fetch_game(client: PlayHQClient, game_id: str) -> None:
    """Fetch and store one game's discover payload."""
    logger = logging.getLogger(LOGGER_NAME)
    game = client.discover_game(game_id)
    if game is None:
        raise PlayHQAPIError(f"Game not found: {game_id}")
    save_json(f"game_{game_id}", game, logger)
    logger.info("Game %s: %s", game_id, json.dumps(game)[:200])


def fetch_grade(client: PlayHQClient, grade_id: str) -> None:
    """Fetch and store one grade's discover payload."""
    logger = logging.getLogger(LOGGER_NAME)
    grade = client.discover_grade(grade_id)
    if grade is None:
        raise PlayHQAPIError(f"Grade not found: {grade_id}")
    save_json(f"grade_{grade_id}", grade, logger)
    season_id = grade["season"]["id"]
    logger.info("Grade %s -> season %s", grade_id, season_id)
    fetch_season(client, season_id)


def fetch_season(client: PlayHQClient, season_id: str) -> None:
    """Fetch and store one season's discover payload."""
    logger = logging.getLogger(LOGGER_NAME)
    season = client.discover_season(season_id)
    if season is None:
        raise PlayHQAPIError(f"Season not found: {season_id}")
    save_json(f"season_{season_id}", season, logger)
    logger.info(
        "Season %s: %s (%s), %d grades",
        season_id,
        season.get("name"),
        season.get("competition", {}).get("name"),
        len(season.get("grades", [])),
    )


def fetch_team_fixture(client: PlayHQClient, team_id: str) -> None:
    """Fetch and store one team's fixture with per-round games."""
    logger = logging.getLogger(LOGGER_NAME)
    fixture = client.team_fixture(team_id)
    save_json(f"team_fixture_{team_id}", fixture, logger)
    team = fixture.get("discoverTeam", {})
    logger.info("Team %s: %s", team_id, team.get("name"))


def fetch_grade_statistics(
    client: PlayHQClient, grade_id: str, logger: logging.Logger
) -> None:
    """Fetch all pages of grade player statistics to CSV.

    Args:
        client: Configured PlayHQ client.
        grade_id: Grade whose player statistics should be pulled.
        logger: Logger for the audit trail.

    Raises:
        PlayHQAPIError: When the grade rejects statistics queries
            (statistics are tenant-configurable and may be off).
    """
    first = client.grade_player_statistics(grade_id, page=1)
    meta = first.get("meta", {})
    total_pages = int(meta.get("totalPages", 1))
    logger.info(
        "Grade statistics %s: %s records across %s pages",
        grade_id,
        meta.get("totalRecords"),
        total_pages,
    )

    rows = []
    for page in range(1, max(total_pages, 1) + 1):
        payload = first if page == 1 else client.grade_player_statistics(
            grade_id, page=page
        )
        for result in payload.get("results", []):
            profile = result.get("profile", {})
            stat_values = [
                detail.get("value")
                for stat in result.get("statistics", [])
                for detail in stat.get("details", [])
            ]
            rows.append(
                {
                    "ranking": result.get("ranking"),
                    "profile_id": profile.get("id"),
                    "first_name": profile.get("firstName"),
                    "last_name": profile.get("lastName"),
                    "team": result.get("team", {}).get("name"),
                    "stat_count": len(stat_values),
                    "stat_values": ";".join(map(str, stat_values)),
                }
            )

    frame = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"grade_statistics_{grade_id}.csv"
    frame.to_csv(out_path, index=False)
    logger.info("Saved %d player stat rows: %s", len(frame), out_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", help="PlayHQ game ID")
    parser.add_argument("--grade-id", help="PlayHQ grade ID")
    parser.add_argument("--season-id", help="PlayHQ season ID")
    parser.add_argument("--team-id", help="PlayHQ team ID")
    parser.add_argument(
        "--stats-grade", help="Grade ID for player statistics export"
    )
    return parser


def main() -> None:
    """Run the requested fetches and log the audit trail."""
    args = build_parser().parse_args()
    logger = setup_logging(LOGGER_NAME, RAW_DIR)
    logger.info("Input arguments: %s", vars(args))

    client = PlayHQClient()
    logger.info("PlayHQ endpoint configured for tenant ca")

    if not any(
        [args.game_id, args.grade_id, args.season_id, args.team_id,
         args.stats_grade]
    ):
        logger.error(
            "Nothing to do: pass at least one of --game-id, "
            "--grade-id, --season-id, --team-id, --stats-grade"
        )
        return

    try:
        if args.game_id:
            fetch_game(client, args.game_id)
        if args.grade_id:
            fetch_grade(client, args.grade_id)
        if args.season_id:
            fetch_season(client, args.season_id)
        if args.team_id:
            fetch_team_fixture(client, args.team_id)
        if args.stats_grade:
            fetch_grade_statistics(client, args.stats_grade, logger)
    except PlayHQAPIError as exc:
        logger.error("PlayHQ fetch failed: %s", exc)
        raise

    logger.info("Fetch complete")


if __name__ == "__main__":
    main()
