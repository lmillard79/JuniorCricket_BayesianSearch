"""PlayHQ public GraphQL API client.

Uses the same two GraphQL services as the PlayHQ public website; no
API key is involved. They take different tenant headers:

* ``https://api.playhq.com/graphql`` (fixtures, results, teams) wants
  ``tenant: cricket-australia``. With the shorter ``ca`` the fixture
  query still answers but the per-innings scores come back empty.
* ``https://spectator.playhq.com/graphql`` (per-game scorecards and
  ball-by-ball events) wants ``x-phq-tenant: ca``.

The query documents below were copied from PlayHQ's own web
bundle (``assets/index.*.js``) so they match the live schema
exactly; GraphQL rejects guessed field names and the schema has
introspection disabled, so verbatim fragments matter.

These are the website's own undocumented endpoints. PlayHQ's
documented route (https://docs.playhq.com/tech/) needs an API key
issued by PlayHQ. The public endpoints sit behind a rate limit: a
burst of requests gets a CloudFront 403. This client therefore spaces
requests out, and stops for good on the first 403 or 429 rather than
retrying (see ``PlayHQBlockedError``).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

GRAPHQL_ENDPOINT = "https://api.playhq.com/graphql"
SPECTATOR_ENDPOINT = "https://spectator.playhq.com/graphql"
DEFAULT_TENANT = "cricket-australia"
SPECTATOR_TENANT = "ca"
WEB_ORIGIN = "https://www.playhq.com"
REQUEST_TIMEOUT_SECONDS = 30
MIN_REQUEST_INTERVAL_SECONDS = 2.5
EVENTS_MAX_PAGES = 12

DISCOVER_GAME_QUERY = """
query gameCentreDiscoverGame($gameId: ID!) {
  discoverGame(gameID: $gameId) {
    id
    alias
    away {
      ... on ProvisionalTeam {
        name
      }
      ... on DiscoverTeam {
        id
        name
      }
    }
    home {
      ... on ProvisionalTeam {
        name
      }
      ... on DiscoverTeam {
        id
        name
      }
    }
    round {
      id
      name
      abbreviatedName
      grade {
        id
        name
        season {
          id
          name
          competition {
            id
            name
            organisation {
              id
              name
            }
          }
        }
      }
    }
  }
}
"""

DISCOVER_GRADE_QUERY = """
query discoverGrade($gradeID: ID!) {
  discoverGrade(gradeID: $gradeID) {
    id
    name
    type
    season {
      id
      name
      competition {
        id
        name
        organisation {
          id
          name
        }
      }
    }
  }
}
"""

DISCOVER_SEASON_QUERY = """
query gradeListDiscoverSeason($id: String!) {
  discoverSeason(seasonID: $id) {
    id
    name
    competition {
      id
      name
      type
      organisation {
        id
        name
      }
    }
    status {
      name
      value
    }
    grades {
      id
      name
      day {
        name
        value
      }
      gender {
        name
        value
      }
      age {
        name
        value
      }
    }
  }
}
"""

DISCOVER_TEAMS_QUERY = """
query discoverOrganisationTeams(
  $seasonCode: String!, $seasonId: ID!,
  $organisationCode: String!, $organisationId: ID!
) {
  discoverSeason(seasonID: $seasonCode) {
    id
    name
    competition {
      id
      name
      organisation {
        id
        name
      }
    }
    status {
      name
      value
    }
    grades {
      id
      name
    }
  }
  discoverTeams(filter: {seasonID: $seasonId, organisationID: $organisationId}) {
    id
    name
    gender {
      name
      value
    }
    ageGroup {
      name
      value
    }
    grade {
      id
      name
    }
  }
  discoverOrganisation(code: $organisationCode) {
    id
    name
    type
  }
}
"""

ROUND_FIXTURE_FRAGMENTS = """
fragment RoundFixtureDiscoverTeamFragment on DiscoverTeam {
  id
  name
  logo {
    sizes {
      url
      dimensions {
        width
        height
      }
    }
  }
  season {
    id
    name
    competition {
      id
      name
    }
  }
  organisation {
    id
    name
    type
  }
}
fragment RoundFixtureTeamFragment on DiscoverPossibleTeam {
  ... on ProvisionalTeam {
    name
    pool {
      id
      name
    }
  }
  ...RoundFixtureDiscoverTeamFragment
}
fragment RoundFixtureFragment on DiscoverRoundFixture {
  byes {
    ...RoundFixtureDiscoverTeamFragment
  }
  games {
    id
    alias
    pool {
      id
      name
    }
    away {
      ...RoundFixtureTeamFragment
    }
    home {
      ...RoundFixtureTeamFragment
    }
    result {
      winner {
        name
        value
      }
      outcome {
        name
        value
      }
      home {
        outcome {
          name
          value
        }
        statistics {
          count
          type {
            value
          }
        }
        periods {
          period {
            label
            value
          }
          type
          closureStatus
          statistics {
            count
            type {
              label
              value
            }
          }
        }
        gameOutcomeDescription
      }
      away {
        outcome {
          name
          value
        }
        statistics {
          count
          type {
            value
          }
        }
        periods {
          period {
            label
            value
          }
          type
          closureStatus
          statistics {
            count
            type {
              label
              value
            }
          }
        }
      }
    }
    status {
      name
      value
    }
    date
    dates
    isStale
    gameType {
      name
      value
      eScoringSettings {
        dismissalsPerBatter
        legalBallsPerOver
      }
    }
  }
}
"""

TEAM_FIXTURE_QUERY = (
    """
query teamFixture($teamID: ID!) {
  discoverTeam(teamID: $teamID) {
    id
    name
    season {
      id
      name
    }
    grade {
      id
      name
    }
    organisation {
      id
      name
    }
  }
  discoverTeamFixture(teamID: $teamID) {
    id
    name
    isStale
    provisionalDates
    grade {
      id
      name
      type
      season {
        id
        name
        competition {
          id
          name
          organisation {
            id
            name
          }
          type
        }
      }
      hideScores
    }
    fixture {
      ...RoundFixtureFragment
    }
  }
}
"""
    + ROUND_FIXTURE_FRAGMENTS
)

GRADE_PLAYER_STATISTICS_QUERY = """
query publicGradeStatistics($gradeID: ID!, $filter: GradePlayerStatisticsFilter) {
  gradePlayerStatistics(gradeID: $gradeID, filter: $filter) {
    meta {
      page
      totalPages
      totalRecords
    }
    results {
      ranking
      profile {
        id
        firstName
        lastName
      }
      team {
        name
      }
      statistics {
        count
        details {
          value
        }
      }
    }
  }
}
"""


# Per-game scorecard (spectator service). Field selection copied from the
# site's gameViewSpectator / GameViewFragment document, trimmed to the
# line-up and per-player statistics the scorecard parser reads.
_PLAYERS = """{
        players {
          id
          profileID
          name
          lineupOrder
          periodStatistics {
            period { value }
            side
            type
            statistics { type { value } count }
            status
            displayOrder
          }
        }
      }"""

GAME_VIEW_SPECTATOR_QUERY = f"""
query gameViewSpectator($id: ID!) {{
  game(id: $id) {{
    id
    status
    statistics {{
      home {_PLAYERS}
      away {_PLAYERS}
    }}
  }}
}}
"""

# Ball-by-ball events (spectator service), copied from the site's
# gameEventsSpectator document. ``after`` is a raw millisecond
# timestamp and ``order`` is EARLIEST_FIRST or LATEST_FIRST.
GAME_EVENTS_SPECTATOR_QUERY = """
query gameEventsSpectator($gameID: ID!, $after: Int, $filters: EventFilter, $order: EventOrder) {
  gameEvents(gameID: $gameID, after: $after, filters: $filters, order: $order) {
    id
    title
    description
    visible
    requireReload
    sportEventStamp
    eventSection
    timestamp
    previousEventID
    side
    period
    ... on ScoreEvent {
      progressiveScore
      score
    }
    ... on FoulEvent {
      type
    }
    ... on DismissalEvent {
      icon
    }
    ... on ExtraEvent {
      icon
    }
    ... on PositionEvent {
      changeDescription
    }
  }
}
"""


class PlayHQAPIError(RuntimeError):
    """Raised when the PlayHQ GraphQL endpoint returns an error."""


class PlayHQBlockedError(PlayHQAPIError):
    """Raised on HTTP 403 or 429: PlayHQ is refusing our requests.

    The client never retries after this. Once raised, every further
    request on the same client raises immediately without touching
    the network, so a caller looping over many games cannot keep
    hitting a service that has told it to stop.
    """


class PlayHQClient:
    """Minimal client for PlayHQ's public GraphQL API.

    Args:
        tenant: Tenant header for ``api.playhq.com``; the site sends
            ``cricket-australia``.
        timeout: Per-request timeout in seconds.
        logger: Optional logger; a module logger is used by default.
        min_interval: Minimum seconds between any two requests.
        spectator_tenant: ``x-phq-tenant`` for the spectator service.
    """

    def __init__(
        self,
        tenant: str = DEFAULT_TENANT,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        logger: Optional[logging.Logger] = None,
        min_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        spectator_tenant: str = SPECTATOR_TENANT,
    ) -> None:
        self._timeout = timeout
        self._logger = logger or logging.getLogger(__name__)
        self._min_interval = min_interval
        self._spectator_tenant = spectator_tenant
        self._last_request = None
        self._blocked = False
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Origin": WEB_ORIGIN,
                "Referer": WEB_ORIGIN + "/",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "tenant": tenant,
            }
        )

    def _throttle(self) -> None:
        """Sleep so consecutive requests are at least ``min_interval`` apart."""
        if self._last_request is not None:
            wait = self._min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
        self._last_request = time.monotonic()

    def _post(
        self,
        url: str,
        body: Dict[str, Any],
        headers: Optional[Dict[str, Optional[str]]] = None,
    ) -> Dict[str, Any]:
        """POST one GraphQL request politely and return the JSON response.

        Args:
            url: GraphQL endpoint.
            body: Request body (query, variables, optional operation name).
            headers: Per-request header overrides; a value of None
                removes that header for this request.

        Returns:
            The decoded JSON response, including any ``errors`` member.

        Raises:
            PlayHQBlockedError: On HTTP 403 or 429, or when an earlier
                request on this client was refused.
            PlayHQAPIError: On transport failure or any other non-200
                status.
        """
        if self._blocked:
            raise PlayHQBlockedError(
                "PlayHQ refused an earlier request; not sending more"
            )
        self._throttle()
        try:
            response = self._session.post(
                url, json=body, headers=headers, timeout=self._timeout
            )
        except requests.RequestException as exc:
            raise PlayHQAPIError(f"PlayHQ request failed: {exc}") from exc

        if response.status_code in (403, 429):
            self._blocked = True
            raise PlayHQBlockedError(
                f"PlayHQ returned HTTP {response.status_code} (request "
                "blocked or rate limited); stopping. Wait before trying "
                "again, or use PlayHQ's official API with a key."
            )
        if response.status_code != 200:
            raise PlayHQAPIError(
                f"PlayHQ returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        return response.json()

    def _execute(self, query: str, variables: Dict[str, Any]) -> Any:
        """Run one GraphQL request and return the ``data`` payload.

        Args:
            query: GraphQL document text.
            variables: Variables for the document.

        Returns:
            The ``data`` member of the response.

        Raises:
            PlayHQAPIError: On transport failure, non-200 status or
                GraphQL-level errors.
        """
        payload = self._post(
            GRAPHQL_ENDPOINT, {"query": query, "variables": variables}
        )
        if "errors" in payload:
            messages = [
                str(error.get("message", error)) for error in payload["errors"]
            ]
            raise PlayHQAPIError("; ".join(messages))

        return payload.get("data")

    def _execute_spectator(
        self, operation: str, query: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run one spectator-service request and return the full response.

        Args:
            operation: GraphQL operation name.
            query: GraphQL document text.
            variables: Variables for the document.

        Returns:
            The full decoded response (with its ``data`` member), so a
            caller can cache it verbatim.

        Raises:
            PlayHQAPIError: On transport failure, non-200 status or
                GraphQL-level errors.
        """
        payload = self._post(
            SPECTATOR_ENDPOINT,
            {"operationName": operation, "query": query,
             "variables": variables},
            headers={"tenant": None, "x-phq-tenant": self._spectator_tenant},
        )
        if payload.get("errors"):
            messages = [
                str(error.get("message", error)) for error in payload["errors"]
            ]
            raise PlayHQAPIError("; ".join(messages))
        return payload

    def discover_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Fetch game, teams, round, grade, season and organisation.

        Args:
            game_id: PlayHQ game ID (short form, e.g. ``"f52ca224"``).

        Returns:
            The ``discoverGame`` dict, or None if not found.
        """
        data = self._execute(DISCOVER_GAME_QUERY, {"gameId": game_id})
        return data.get("discoverGame") if data else None

    def discover_grade(self, grade_id: str) -> Optional[Dict[str, Any]]:
        """Fetch grade details including parent season and organisation.

        Args:
            grade_id: PlayHQ grade ID.

        Returns:
            The ``discoverGrade`` dict, or None if not found.
        """
        data = self._execute(DISCOVER_GRADE_QUERY, {"gradeID": grade_id})
        return data.get("discoverGrade") if data else None

    def discover_season(self, season_id: str) -> Optional[Dict[str, Any]]:
        """Fetch season details including competition and grade list.

        Args:
            season_id: PlayHQ season ID (short or long form).

        Returns:
            The ``discoverSeason`` dict, or None if not found.
        """
        data = self._execute(DISCOVER_SEASON_QUERY, {"id": season_id})
        return data.get("discoverSeason") if data else None

    def discover_teams(
        self, season_id: str, organisation_id: str, season_code: str = "",
        organisation_code: str = "",
    ) -> List[Dict[str, Any]]:
        """List teams for one organisation within one season.

        Args:
            season_id: Season ID used in the ``discoverTeams`` filter.
            organisation_id: Organisation ID used in the filter.
            season_code: Season ID repeated for the season lookup.
            organisation_code: Organisation code; if unknown, pass
                the ID (the lookup result is allowed to be null).

        Returns:
            List of team dicts with id, name, grade and age group.
        """
        data = self._execute(
            DISCOVER_TEAMS_QUERY,
            {
                "seasonCode": season_code or season_id,
                "seasonId": season_id,
                "organisationCode": organisation_code or organisation_id,
                "organisationId": organisation_id,
            },
        )
        return data.get("discoverTeams", []) if data else []

    def team_fixture(self, team_id: str) -> Dict[str, Any]:
        """Fetch a team's fixture with per-round games and results.

        Args:
            team_id: PlayHQ team ID.

        Returns:
            Dict with ``discoverTeam`` and ``discoverTeamFixture``.
        """
        data = self._execute(TEAM_FIXTURE_QUERY, {"teamID": team_id})
        return data or {}

    def grade_player_statistics(
        self, grade_id: str, team_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Fetch per-player statistics for a grade.

        The public endpoint returns the first page with ``meta``
        pagination info; cricket tenants typically expose a single
        page per grade. Filter by team IDs when provided.

        Args:
            grade_id: PlayHQ grade ID.
            team_ids: Optional list of team IDs to filter to.

        Returns:
            Dict with ``meta`` (page, totalPages, totalRecords) and
            ``results`` (ranking, profile, team, statistics).
        """
        filter_value = {"teamIDs": team_ids} if team_ids else None
        data = self._execute(
            GRADE_PLAYER_STATISTICS_QUERY,
            {"gradeID": grade_id, "filter": filter_value},
        )
        return data.get("gradePlayerStatistics", {}) if data else {}

    def game_scorecard(self, game_id: str) -> Dict[str, Any]:
        """Fetch one game's per-player scorecard.

        Args:
            game_id: PlayHQ game ID.

        Returns:
            The full JSON response, ready to cache verbatim and to
            pass to ``playhq_parse.parse_scorecard``.
        """
        return self._execute_spectator(
            "gameViewSpectator", GAME_VIEW_SPECTATOR_QUERY, {"id": game_id}
        )

    def game_events(
        self,
        game_id: str,
        side: str,
        period: str = "FIRST_INNINGS",
        max_pages: int = EVENTS_MAX_PAGES,
    ) -> List[Dict[str, Any]]:
        """Fetch every ball-by-ball event for one innings.

        The service returns the latest 50 events first; older pages
        are requested with ``after`` set to the oldest timestamp seen
        (plus one millisecond, so events sharing that timestamp are
        not skipped) and duplicates are dropped by event ID.

        Args:
            game_id: PlayHQ game ID.
            side: Batting side, ``HOME`` or ``AWAY``.
            period: Innings period; each U10/U11 side bats once, in
                ``FIRST_INNINGS``.
            max_pages: Safety cap on pages fetched.

        Returns:
            Events in time order.
        """
        filters = {"period": period, "side": side}
        seen: Dict[str, Dict[str, Any]] = {}
        variables: Dict[str, Any] = {"gameID": game_id, "filters": filters}
        for _ in range(max_pages):
            payload = self._execute_spectator(
                "gameEventsSpectator", GAME_EVENTS_SPECTATOR_QUERY, variables
            )
            page = (payload.get("data") or {}).get("gameEvents") or []
            fresh = [e for e in page if e["id"] not in seen]
            if not fresh:
                break
            for event in fresh:
                seen[event["id"]] = event
            oldest = min(int(e["timestamp"]) for e in seen.values())
            variables = {
                "gameID": game_id, "filters": filters,
                "after": oldest + 1, "order": "LATEST_FIRST",
            }
        return sorted(seen.values(), key=lambda e: int(e["timestamp"]))
