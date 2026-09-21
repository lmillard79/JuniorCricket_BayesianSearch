"""Tests for the PlayHQ client's politeness, block handling and paging."""

import json
from typing import Any, Dict, List

import pytest

from junior_cricket import playhq_client
from junior_cricket.playhq_client import (
    DEFAULT_TENANT,
    SPECTATOR_ENDPOINT,
    TEAM_FIXTURE_QUERY,
    PlayHQAPIError,
    PlayHQBlockedError,
    PlayHQClient,
)


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status: int = 200, payload: Dict[str, Any] = None) -> None:
        self.status_code = status
        self._payload = payload if payload is not None else {"data": {}}
        self.text = json.dumps(self._payload)

    def json(self) -> Dict[str, Any]:
        return self._payload


class FakeTime:
    """Deterministic clock recording the sleeps it is asked for."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: List[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(responder, min_interval: float = 0.0) -> PlayHQClient:
    """Build a client whose HTTP session is replaced by ``responder``."""
    client = PlayHQClient(min_interval=min_interval)
    client._session.post = responder
    return client


def test_defaults_match_what_the_website_sends() -> None:
    """The api.playhq.com tenant is the site's slug, not ``ca``."""
    client = PlayHQClient()
    assert DEFAULT_TENANT == "cricket-australia"
    assert client._session.headers["tenant"] == "cricket-australia"
    assert "result {" in TEAM_FIXTURE_QUERY


def test_requests_are_spaced_by_the_minimum_interval(monkeypatch) -> None:
    """A second request within the interval sleeps for the remainder."""
    fake = FakeTime()
    monkeypatch.setattr(playhq_client, "time", fake)
    client = make_client(lambda *a, **k: FakeResponse(), min_interval=2.5)
    client._execute("query { a }", {})
    fake.now += 1.0                       # only 1 s passes on its own
    client._execute("query { a }", {})
    assert fake.sleeps == [pytest.approx(1.5)]


@pytest.mark.parametrize("status", [403, 429])
def test_block_stops_the_client_for_good(status: int) -> None:
    """After a 403 or 429 no further request touches the network."""
    calls: List[str] = []

    def responder(url, **kwargs):
        calls.append(url)
        return FakeResponse(status=status, payload={})

    client = make_client(responder)
    with pytest.raises(PlayHQBlockedError):
        client._execute("query { a }", {})
    with pytest.raises(PlayHQBlockedError):
        client._execute("query { a }", {})
    with pytest.raises(PlayHQBlockedError):
        client.game_scorecard("g1")
    assert len(calls) == 1                # the later calls never sent


def test_other_http_errors_are_plain_api_errors() -> None:
    """A 500 is an error but not a block, so later calls may proceed."""
    client = make_client(lambda *a, **k: FakeResponse(status=500, payload={}))
    with pytest.raises(PlayHQAPIError) as info:
        client._execute("query { a }", {})
    assert not isinstance(info.value, PlayHQBlockedError)


def test_spectator_requests_use_the_spectator_headers() -> None:
    """The scorecard call goes to the spectator host with x-phq-tenant."""
    seen: Dict[str, Any] = {}

    def responder(url, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json, headers=headers)
        return FakeResponse(payload={"data": {"game": {"id": "g1"}}})

    client = make_client(responder)
    payload = client.game_scorecard("g1")
    assert payload["data"]["game"]["id"] == "g1"
    assert seen["url"] == SPECTATOR_ENDPOINT
    assert seen["headers"] == {"tenant": None, "x-phq-tenant": "ca"}
    assert seen["body"]["operationName"] == "gameViewSpectator"
    assert seen["body"]["variables"] == {"id": "g1"}


def test_graphql_errors_from_the_spectator_service_raise() -> None:
    """An ``errors`` member is surfaced, not returned as data."""
    client = make_client(
        lambda *a, **k: FakeResponse(payload={"errors": [{"message": "nope"}]})
    )
    with pytest.raises(PlayHQAPIError, match="nope"):
        client.game_scorecard("g1")


def _event(event_id: str, ts: int) -> Dict[str, Any]:
    return {"id": event_id, "timestamp": str(ts), "title": "Dot ball"}


def test_game_events_pages_back_without_gaps_or_duplicates() -> None:
    """Older pages use after=oldest+1; overlaps are dropped by ID."""
    pages = [
        [_event("e4", 4000), _event("e3", 3000)],          # latest first
        [_event("e3", 3000), _event("e2", 2000)],          # overlaps e3
        [_event("e1", 1000)],
        [_event("e1", 1000)],                              # only a repeat: stop
    ]
    afters: List[Any] = []

    def responder(url, json=None, headers=None, timeout=None):
        afters.append(json["variables"].get("after"))
        return FakeResponse(payload={"data": {"gameEvents": pages[len(afters) - 1]}})

    client = make_client(responder)
    events = client.game_events("g1", "HOME")
    assert [e["id"] for e in events] == ["e1", "e2", "e3", "e4"]
    assert afters == [None, 3001, 2001, 1001]


def test_game_events_stops_at_the_page_cap() -> None:
    """A service that never runs dry cannot loop forever."""
    counter = {"n": 0}

    def responder(url, json=None, headers=None, timeout=None):
        counter["n"] += 1
        return FakeResponse(payload={"data": {"gameEvents": [
            _event(f"e{counter['n']}", 10_000 - counter["n"])]}})

    client = make_client(responder)
    events = client.game_events("g1", "AWAY", max_pages=3)
    assert counter["n"] == 3 and len(events) == 3
