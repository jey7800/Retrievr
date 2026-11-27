"""Bot protections answering in place of a page or a document: always a clear block."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import pytest
import respx

from retrievr.api import _answer, blocked_downloads
from retrievr.download import Downloader, DownloadOutcome
from retrievr.fetch import BlockedError, detect_block
from retrievr.recipe import FoundDocument, Recipe, recipe_from_mapping
from retrievr.urls import normalize_url
from tests.conftest import pdf_bytes

SITE = "https://ir.example.com"
FILES = "https://www.example.com"

# What a site behind Imperva serves, with HTTP 200, to a client it wants to challenge:
# a near-empty page whose only content is the script that runs the challenge.
IMPERVA_CHALLENGE = (
    b'<html>\r\n<head>\r\n<META NAME="robots" CONTENT="noindex,nofollow">\r\n'
    b'<script src="/_Incapsula_Resource?SWJIYLWA=719d34d31c8e3a6e6fffd425f7e032f3">\r\n'
    b"</script>\r\n<body>\r\n</body></html>\r\n"
)


@pytest.fixture
def recipe() -> Recipe:
    return recipe_from_mapping(
        {
            "schema_version": 1,
            "recipe": "example-ir",
            "version": 1,
            "issuer": "example",
            "site": "ir.example.com",
            "allowed_domains": ["ir.example.com", "www.example.com"],
            "sources": [{"id": "s", "url": SITE + "/docs"}],
        }
    )


def _doc(name: str) -> FoundDocument:
    url = f"{FILES}/files/{name}"
    return FoundDocument(
        url=url,
        normalized_url=normalize_url(url),
        title=name,
        source_id="s",
        params={},
        page_url=SITE + "/docs",
        extension=name.rsplit(".", 1)[-1],
        date=date(2026, 2, 6),
        type="annual_report",
        period="2025-FY",
        fiscal_year=2025,
        language="en",
        score=1.0,
        justification="test",
    )


@pytest.fixture
def site(no_network: respx.MockRouter) -> respx.MockRouter:
    no_network.get(SITE + "/robots.txt").respond(404)
    no_network.get(FILES + "/robots.txt").respond(404)
    return no_network


def test_the_imperva_challenge_page_is_a_block() -> None:
    assert detect_block(200, IMPERVA_CHALLENGE.decode()) == "challenge page (Imperva)"


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("<html><title>Pardon Our Interruption</title>" + "x" * 30_000, "block page (Imperva)"),
        ("Checking your browser before accessing example.com" + "x" * 30_000, "Cloudflare"),
        (
            "The requested URL was rejected. Please consult with your administrator."
            + "x" * 30_000,
            "block page (F5)",
        ),
        ("Sucuri WebSite Firewall - Access Denied" + "x" * 30_000, "block page (Sucuri)"),
        ("<title>Access to this page has been denied.</title>" + "x" * 30_000, "PerimeterX"),
        ('<script src="https://x.awswaf.com/challenge.js"></script>', "challenge page (AWS WAF)"),
        ('<script src="https://validate.perfdrive.com/x"></script>', "captcha page (Radware)"),
        ("<script>window.KPSDK={}</script>", "challenge page (Kasada)"),
        ("<p>Please verify you are a human</p>", "challenge page (bot protection)"),
    ],
)
def test_other_bot_protections_are_blocks(body: str, reason: str) -> None:
    found = detect_block(200, body)
    assert found is not None and reason in found


def test_weak_markers_on_a_big_normal_page_are_not_blocks() -> None:
    page = "<html><script src='/_Incapsula_Resource?x=1'></script>" + "<p>report</p>" * 5000
    assert detect_block(200, page) is None


def test_a_challenge_served_instead_of_a_file_blocks_and_stops(
    site: respx.MockRouter, make_fetcher: Any, recipe: Recipe, tmp_path: Any
) -> None:
    challenge = site.get(FILES + "/files/a.pdf").respond(
        200, content=IMPERVA_CHALLENGE, headers={"content-type": "text/html"}
    )
    other = site.get(FILES + "/files/b.pdf").respond(200, content=pdf_bytes("b"))
    outcomes = Downloader(make_fetcher(), tmp_path).download_all(
        recipe, [_doc("a.pdf"), _doc("b.pdf")]
    )
    assert [o.status for o in outcomes] == ["blocked"]
    assert "challenge page (Imperva)" in (outcomes[0].reason or "")
    assert challenge.called and not other.called
    assert not any(p.suffix == ".pdf" for p in tmp_path.rglob("*"))


def test_a_challenge_with_a_wrong_content_type_still_blocks(
    site: respx.MockRouter, make_fetcher: Any, recipe: Recipe, tmp_path: Any
) -> None:
    site.get(FILES + "/files/a.pdf").respond(
        200, content=IMPERVA_CHALLENGE, headers={"content-type": "application/pdf"}
    )
    outcomes = Downloader(make_fetcher(), tmp_path).download_all(recipe, [_doc("a.pdf")])
    assert [o.status for o in outcomes] == ["blocked"]


def test_an_ordinary_html_page_instead_of_a_file_is_only_rejected(
    site: respx.MockRouter, make_fetcher: Any, recipe: Recipe, tmp_path: Any
) -> None:
    page = b"<html><title>Page not found</title><p>This report has moved.</p></html>"
    site.get(FILES + "/files/a.pdf").respond(
        200, content=page, headers={"content-type": "text/html"}
    )
    site.get(FILES + "/files/b.pdf").respond(200, content=pdf_bytes("b"))
    outcomes = Downloader(make_fetcher(), tmp_path).download_all(
        recipe, [_doc("a.pdf"), _doc("b.pdf")]
    )
    assert [o.status for o in outcomes] == ["rejected", "downloaded"]


def test_the_downloadable_probe_reads_a_challenge_as_blocked(
    site: respx.MockRouter, make_fetcher: Any, recipe: Recipe
) -> None:
    site.get(FILES + "/files/a.pdf").respond(
        200, content=IMPERVA_CHALLENGE, headers={"content-type": "text/html"}
    )
    site.get(FILES + "/files/b.pdf").respond(200, content=pdf_bytes("b"))
    fetcher = make_fetcher()
    assert _answer(fetcher, recipe, FILES + "/files/a.pdf") == "blocked"
    assert _answer(fetcher, recipe, FILES + "/files/b.pdf") == "ok"


def test_a_refused_download_names_the_block() -> None:
    doc = _doc("a.pdf")
    reason = "blocked: challenge page (Imperva) at https://www.example.com/files/a.pdf"
    downloads = [DownloadOutcome(doc, "blocked", reason=reason)]
    assert blocked_downloads(downloads) == f"the site blocks downloads: {reason}"
    assert blocked_downloads([DownloadOutcome(doc, "rejected", reason="x")]) is None


def test_blocked_error_carries_the_reason() -> None:
    exc = BlockedError("https://www.example.com/a.pdf", "challenge page (Imperva)")
    assert exc.reason == "challenge page (Imperva)" and "blocked:" in str(exc)


def test_a_run_whose_downloads_are_refused_ends_blocked(
    site: respx.MockRouter, make_fetcher: Any, tmp_path: Any
) -> None:
    from retrievr import Retrievr
    from retrievr.config import BrowserSettings, OutputSettings, Settings, StateSettings

    listing = (
        (
            f'<html><body><div id="list"><a href="{FILES}/files/annual-report-2025.pdf">'
            "Annual report 2025</a></div></body></html>"
        ).encode()
        + b"<!-- "
        + b"x" * 25_000
        + b" -->"
    )
    site.get(SITE + "/docs").respond(200, content=listing, headers={"content-type": "text/html"})
    site.get(FILES + "/files/annual-report-2025.pdf").respond(
        200, content=IMPERVA_CHALLENGE, headers={"content-type": "text/html"}
    )
    recipe_path = tmp_path / "example-ir@1.yaml"
    recipe_path.write_text(
        "schema_version: 1\nrecipe: example-ir\nversion: 1\nissuer: example\n"
        "site: ir.example.com\nallowed_domains: [ir.example.com, www.example.com]\n"
        f"sources:\n  - id: s\n    url: {SITE}/docs\n    items:\n      container: '#list'\n"
        "      extensions: [pdf]\n",
        encoding="utf-8",
    )
    events: list[Any] = []
    settings = Settings(
        state=StateSettings(state_dir=tmp_path / "state"),
        output=OutputSettings(output_dir=tmp_path / "out"),
        browser=BrowserSettings(enabled=False),
    )
    r = Retrievr(
        settings=settings,
        fetch_transport=httpx.HTTPTransport(),
        on_event=events.append,
        sleep=lambda _s: None,
        today=date(2026, 3, 1),
    )
    r.install(recipe_path, site="example")
    outcome = r.run("example", download=True, repair=False)
    assert outcome.status == "blocked"
    assert any("the site blocks downloads" in e and "Imperva" in e for e in outcome.errors)
    assert any(e.kind == "blocked" for e in events)
