from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
import respx

from retrievr.fetch import (
    BlockedError,
    DomainNotAllowedError,
    HTTPStatusError,
    RateLimiter,
    RobotsDisallowedError,
    detect_block,
)
from tests.conftest import FIXTURES, FakeClock

SITE = "https://ir.example.com"
HTML = b"<html><body>" + b"<p>ok</p>" * 5000 + b"</body></html>"


def _ok(content: bytes = HTML) -> httpx.Response:
    return httpx.Response(200, content=content, headers={"content-type": "text/html"})


def test_rate_limiter_spaces_requests_per_host(clock: FakeClock) -> None:
    limiter = RateLimiter(1.0, clock=clock, sleep=clock.sleep)
    limiter.wait("a.example")
    limiter.wait("a.example")
    limiter.wait("b.example")
    limiter.wait("a.example")
    assert clock.sleeps == [1.0, 1.0]


def test_get_sends_user_agent_and_respects_rate_limit(
    no_network: respx.MockRouter, make_fetcher: Any, clock: FakeClock
) -> None:
    no_network.get(f"{SITE}/robots.txt").respond(404)
    page = no_network.get(f"{SITE}/docs").mock(return_value=_ok())
    fetcher = make_fetcher(user_agent="retrievr-test/1.0")
    fetcher.get(f"{SITE}/docs")
    fetcher.get(f"{SITE}/docs")
    assert page.calls[0].request.headers["user-agent"] == "retrievr-test/1.0"
    assert clock.sleeps == [1.0, 1.0]  # robots -> page -> page


@pytest.mark.parametrize("status", [403, 429, 503])
def test_block_statuses_raise_blocked(
    no_network: respx.MockRouter, make_fetcher: Any, status: int
) -> None:
    no_network.get(f"{SITE}/robots.txt").respond(404)
    no_network.get(f"{SITE}/docs").respond(status)
    with pytest.raises(BlockedError) as info:
        make_fetcher().get(f"{SITE}/docs")
    assert info.value.status == status


def test_challenge_page_raises_blocked(no_network: respx.MockRouter, make_fetcher: Any) -> None:
    no_network.get(f"{SITE}/robots.txt").respond(404)
    body = b"<html><head><title>Just a moment...</title></head><body></body></html>"
    no_network.get(f"{SITE}/docs").mock(return_value=_ok(body))
    with pytest.raises(BlockedError, match="Cloudflare"):
        make_fetcher().get(f"{SITE}/docs")


def test_http_404_raises_status_error(no_network: respx.MockRouter, make_fetcher: Any) -> None:
    no_network.get(f"{SITE}/robots.txt").respond(404)
    no_network.get(f"{SITE}/missing").respond(404)
    with pytest.raises(HTTPStatusError):
        make_fetcher().get(f"{SITE}/missing")


def test_weak_markers_do_not_flag_normal_pages() -> None:
    """Normal pages behind Imperva load /_Incapsula_Resource; they are not blocks."""
    for page in (FIXTURES / "societegenerale" / "pages").glob("*.html"):
        assert detect_block(200, page.read_text("utf-8")) is None, page.name


def test_weak_markers_flag_small_pages() -> None:
    small = '<html><script src="/_Incapsula_Resource?x=1"></script></html>'
    assert detect_block(200, small) == "challenge page (Imperva)"
    assert detect_block(200, "<html><div class='g-recaptcha'></div></html>") == "captcha page"


def test_robots_disallow_is_enforced(no_network: respx.MockRouter, make_fetcher: Any) -> None:
    no_network.get(f"{SITE}/robots.txt").respond(200, text="User-agent: *\nDisallow: /private/\n")
    no_network.get(f"{SITE}/public").mock(return_value=_ok())
    fetcher = make_fetcher()
    fetcher.get(f"{SITE}/public")
    with pytest.raises(RobotsDisallowedError):
        fetcher.get(f"{SITE}/private/report.pdf")


def test_robots_ignore_is_logged(
    no_network: respx.MockRouter, make_fetcher: Any, caplog: pytest.LogCaptureFixture
) -> None:
    no_network.get(f"{SITE}/private/x").mock(return_value=_ok())
    with caplog.at_level(logging.WARNING, logger="retrievr.fetch"):
        fetcher = make_fetcher(robots="ignore")
    fetcher.get(f"{SITE}/private/x")
    assert "DISABLED" in caplog.text
    assert fetcher.robots_outcome(SITE) == "ignored"


def test_robots_redirect_loop_uses_explicit_policy(
    no_network: respx.MockRouter, make_fetcher: Any, caplog: pytest.LogCaptureFixture
) -> None:
    no_network.get(f"{SITE}/robots.txt").respond(301, headers={"location": f"{SITE}/robots.txt"})
    no_network.get(f"{SITE}/docs").mock(return_value=_ok())
    with caplog.at_level(logging.WARNING, logger="retrievr.fetch"):
        allow = make_fetcher()
        allow.get(f"{SITE}/docs")
    assert allow.robots_outcome(SITE) == "unreachable"
    assert "robots_unreachable=allow" in caplog.text

    deny = make_fetcher(robots_unreachable="deny")
    with pytest.raises(RobotsDisallowedError):
        deny.get(f"{SITE}/docs")


def test_robots_network_error_is_unreachable(
    no_network: respx.MockRouter, make_fetcher: Any
) -> None:
    no_network.get(f"{SITE}/robots.txt").mock(side_effect=httpx.ConnectError("boom"))
    no_network.get(f"{SITE}/docs").mock(return_value=_ok())
    fetcher = make_fetcher()
    fetcher.get(f"{SITE}/docs")
    assert fetcher.robots_outcome(SITE) == "unreachable"


def test_robots_server_error_denies(no_network: respx.MockRouter, make_fetcher: Any) -> None:
    no_network.get(f"{SITE}/robots.txt").respond(500)
    with pytest.raises(RobotsDisallowedError):
        make_fetcher().get(f"{SITE}/docs")


def test_redirect_outside_allowed_domains_is_refused(
    no_network: respx.MockRouter, make_fetcher: Any
) -> None:
    no_network.get(f"{SITE}/robots.txt").respond(404)
    no_network.get(f"{SITE}/docs").respond(302, headers={"location": "https://evil.example/x"})
    with pytest.raises(DomainNotAllowedError):
        make_fetcher().get(f"{SITE}/docs", allowed_domains=["ir.example.com"])


def test_redirect_inside_allowed_domains_is_followed(
    no_network: respx.MockRouter, make_fetcher: Any
) -> None:
    no_network.get(f"{SITE}/robots.txt").respond(404)
    no_network.get("https://files.example.com/robots.txt").respond(404)
    no_network.get(f"{SITE}/docs").respond(
        301, headers={"location": "https://files.example.com/docs"}
    )
    no_network.get("https://files.example.com/docs").mock(return_value=_ok())
    page = make_fetcher().get(
        f"{SITE}/docs", allowed_domains=["ir.example.com", "files.example.com"]
    )
    assert page.final_url == "https://files.example.com/docs"


def test_conditional_get_returns_not_modified(
    no_network: respx.MockRouter, make_fetcher: Any
) -> None:
    no_network.get(f"{SITE}/robots.txt").respond(404)
    route = no_network.get(f"{SITE}/docs").respond(304)
    page = make_fetcher().get(f"{SITE}/docs", etag='"abc"', last_modified="Mon, 01 Jan 2026")
    assert page.not_modified
    assert route.calls[0].request.headers["if-none-match"] == '"abc"'
    assert route.calls[0].request.headers["if-modified-since"] == "Mon, 01 Jan 2026"


def test_proxy_is_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from retrievr.config import FetchSettings
    from retrievr.fetch import _default_client

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    client = _default_client(FetchSettings())
    try:
        assert client._trust_env  # httpx reads HTTP(S)_PROXY / NO_PROXY
        transport = client._transport_for_url(httpx.URL("https://ir.example.com/"))
        assert transport is not client._transport  # a proxy transport was mounted
    finally:
        client.close()


def test_network_errors_become_fetch_errors(
    no_network: respx.MockRouter, make_fetcher: Any
) -> None:
    from retrievr.fetch import NetworkError

    no_network.get(f"{SITE}/robots.txt").respond(404)
    no_network.get(f"{SITE}/docs").mock(side_effect=httpx.RemoteProtocolError("disconnected"))
    with pytest.raises(NetworkError, match="RemoteProtocolError"):
        make_fetcher().get(f"{SITE}/docs")


def test_recipe_run_reports_network_error(no_network: respx.MockRouter, make_fetcher: Any) -> None:
    from datetime import date

    from retrievr.recipe import recipe_from_mapping, run_recipe

    no_network.get(f"{SITE}/robots.txt").respond(404)
    no_network.get(f"{SITE}/docs").mock(side_effect=httpx.ConnectTimeout("slow"))
    recipe = recipe_from_mapping(
        {
            "schema_version": 1,
            "recipe": "x",
            "version": 1,
            "issuer": "x",
            "site": "ir.example.com",
            "allowed_domains": ["ir.example.com"],
            "sources": [{"id": "s", "url": f"{SITE}/docs"}],
        }
    )
    result = run_recipe(recipe, make_fetcher(), today=date(2026, 1, 1))
    assert result.status == "error"
    assert "ConnectTimeout" in result.errors[0]
