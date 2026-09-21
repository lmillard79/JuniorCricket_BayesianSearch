"""One-off exploratory probe of the PlayHQ GraphQL schema.

Not wired into the package, and not meant to stay forever: this
answers a specific open question from the data-acquisition
discussion, then can be deleted once we know the answer.

Checks, in order:
1. Whether schema introspection is actually blocked (the existing
   client's docstring assumes it is; that has not been tested).
2. If introspection works, which types look scorecard-shaped
   (innings, batting/bowling performances, results) and what
   fields they expose.
3. Regardless of (1), whether a short list of plausibly named
   fields can be added directly to ``discoverGame``.
4. Whether ``gradePlayerStatistics`` returns real rows for this
   game's own grade, and what the unlabelled ``statistics`` shape
   actually contains for one real player.

Run:
    python scripts/probe_playhq_schema.py --game-id a481b130
"""
from __future__ import annotations

import argparse
import json
import time

from junior_cricket.playhq_client import GRAPHQL_ENDPOINT, PlayHQClient

KEYWORDS = (
    "score", "inning", "batting", "bowling", "performance",
    "result", "stat", "wicket", "over", "delivery", "ball",
)

SCHEMA_QUERY = "query { __schema { types { name kind } } }"

SPECULATIVE_FIELDS = [
    "result",
    "homeScore",
    "awayScore",
    "innings",
    "battingPerformances",
    "bowlingPerformances",
    "scorecard",
    "summary",
    "matchResult",
]


def _post(client: PlayHQClient, query: str, variables: dict) -> dict:
    response = client._session.post(
        GRAPHQL_ENDPOINT,
        json={"query": query, "variables": variables},
        timeout=30,
    )
    time.sleep(0.2)
    return response.json()


def type_fields_query(type_name: str) -> str:
    return (
        'query { __type(name: "%s") { name kind fields { '
        "name type { name kind ofType { name kind } } } } }" % type_name
    )


def discover_game_probe_query(field: str) -> str:
    return (
        "query($gameId: ID!) { discoverGame(gameID: $gameId) { id %s } }"
        % field
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", required=True)
    args = parser.parse_args()
    client = PlayHQClient()

    print("=" * 70)
    print("0. discover_game -> grade id")
    print("=" * 70)
    game = client.discover_game(args.game_id)
    grade_id = None
    if game is None:
        print("Game not found.")
    else:
        grade_id = game.get("round", {}).get("grade", {}).get("id")
        print(f"grade id: {grade_id}")
        print(json.dumps(game, indent=2))

    print("=" * 70)
    print("1. Full schema introspection")
    print("=" * 70)
    payload = _post(client, SCHEMA_QUERY, {})
    hits: list = []
    if "errors" in payload:
        print("BLOCKED:", payload["errors"][0].get("message"))
    else:
        types = payload["data"]["__schema"]["types"]
        hits = sorted(
            {
                t["name"]
                for t in types
                if t.get("name")
                and any(k in t["name"].lower() for k in KEYWORDS)
            }
        )
        print(f"Introspection WORKS. {len(types)} types total.")
        print("Score/innings/stat-shaped types found:")
        for name in hits:
            print(" -", name)

    if hits:
        print("=" * 70)
        print("2. Fields on each match (first 10)")
        print("=" * 70)
        for name in hits[:10]:
            data = _post(client, type_fields_query(name), {}).get(
                "data", {}
            ).get("__type")
            if not data:
                continue
            print(f"\n{name} ({data.get('kind')}):")
            for f in data.get("fields") or []:
                t = f["type"]
                tname = t.get("name") or (t.get("ofType") or {}).get("name")
                print(f"   {f['name']}: {tname}")

    print("=" * 70)
    print("3. Speculative fields directly on discoverGame")
    print("=" * 70)
    for field in SPECULATIVE_FIELDS:
        payload = _post(
            client,
            discover_game_probe_query(field),
            {"gameId": args.game_id},
        )
        if "errors" in payload:
            msg = payload["errors"][0].get("message", "")
            print(f"  {field}: REJECTED - {msg[:120]}")
        else:
            print(f"  {field}: ACCEPTED -> {json.dumps(payload['data'])[:300]}")

    if grade_id:
        print("=" * 70)
        print(f"4. grade_player_statistics for grade {grade_id}")
        print("=" * 70)
        try:
            stats = client.grade_player_statistics(grade_id)
            print(json.dumps(stats, indent=2)[:3000])
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print("FAILED:", exc)


if __name__ == "__main__":
    main()
