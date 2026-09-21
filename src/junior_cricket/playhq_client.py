"""PlayHQ public GraphQL API client.

Uses the same GraphQL endpoint as the PlayHQ public website
(``https://api.playhq.com/graphql``) with the ``tenant`` request
header set to the Cricket Australia tenant code ``ca``. No API
key is required for the discover-style queries used here.

The query documents below were copied from PlayHQ's own web
bundle (``assets/index.*.js``) so they match the live schema
exactly; GraphQL rejects guessed field names and the schema has
introspection disabled, so verbatim fragments matter.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

GRAPHQL_ENDPOINT = "https://api.playhq.com/graphql"
DEFAULT_TENANT = "ca"
WEB_ORIGIN = "https://www.playhq.com"
REQUEST_TIMEOUT_SECONDS = 30

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


class PlayHQAPIError(RuntimeError):
    """Raised when the PlayHQ GraphQL endpoint returns an error."""


class PlayHQClient:
    """Minimal client for PlayHQ's public GraphQL discover API.

    Args:
        tenant: PlayHQ tenant code; ``ca`` is Cricket Australia.
        timeout: Per-request timeout in seconds.
        logger: Optional logger; a module logger is used by default.
    """

    def __init__(
        self,
        tenant: str = DEFAULT_TENANT,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._timeout = timeout
        self._logger = logger or logging.getLogger(__name__)
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
        try:
            response = self._session.post(
                GRAPHQL_ENDPOINT,
                json={"query": query, "variables": variables},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise PlayHQAPIError(f"PlayHQ request failed: {exc}") from exc

        if response.status_code != 200:
            raise PlayHQAPIError(
                f"PlayHQ returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        payload = response.json()
        if "errors" in payload:
            messages = [
                str(error.get("message", error)) for error in payload["errors"]
            ]
            raise PlayHQAPIError("; ".join(messages))

        return payload.get("data")

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
