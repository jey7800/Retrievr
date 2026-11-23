"""Fixes from the fourth round of live cold tests. No network."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from retrievr.api import Retrievr
from retrievr.classify import ClassifyRules, Decision, apply_decisions, classify, notice_type
from retrievr.config import BrowserSettings, Settings, StateSettings
from retrievr.recipe import recipe_from_mapping, run_recipe
from retrievr.validate import Reachability, validate_runs

SITE = "https://ir.example-issuer.com"
FILES = "https://files.example-issuer.com"
TODAY = date(2026, 11, 23)


def _site(file_status: int) -> httpx.MockTransport:
    listing = "".join(
        f'<li><a href="{FILES}/download?id={y}">Annual report {y} (pdf)</a></li>'
        for y in (2023, 2024, 2025)
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/reports":
            return httpx.Response(200, html=f"<html><body><ul>{listing}</ul></body></html>")
        if request.url.host == "files.example-issuer.com" and request.url.path == "/download":
            return httpx.Response(file_status)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _recipe() -> dict[str, object]:
    return {
        "schema_version": 3,
        "recipe": "example-ir",
        "version": 1,
        "issuer": "example-issuer",
        "site": "ir.example-issuer.com",
        "allowed_domains": ["ir.example-issuer.com", "files.example-issuer.com"],
        "sources": [
            {
                "id": "reports",
                "kind": "url_template",
                "url": f"{SITE}/reports",
                "items": {
                    "container": "ul",
                    "item": "li",
                    "link": "a[href]",
                    "document_urls": [r"/download\?id="],
                },
            }
        ],
    }


def _reach(tmp_path: Path, file_status: int) -> tuple[Reachability, bool, str]:
    r = Retrievr(
        settings=Settings(
            state=StateSettings(state_dir=tmp_path / "state"),
            browser=BrowserSettings(enabled=False),
        ),
        fetch_transport=_site(file_status),
        sleep=lambda _s: None,
        today=TODAY,
    )
    recipe = recipe_from_mapping(_recipe())
    with r._fetcher() as fetcher:
        first = run_recipe(recipe, fetcher, today=TODAY)
        assert len(first.documents) == 3
        _chosen, run, reach = r._downloadable(recipe, first, fetcher, None)
    report = validate_runs(recipe, run, run, reachability=reach)
    return reach, report.passed, report.summary()


def test_documents_on_a_host_that_refuses_this_client_are_not_downloadable(
    tmp_path: Path,
) -> None:
    reach, passed, summary = _reach(tmp_path, 403)
    assert reach == Reachability(sampled=3, reachable=0, gone=0, blocked=3)
    assert not passed
    assert "3 blocked (HTTP 403)" in summary


@pytest.mark.parametrize("status", [429, 503])
def test_rate_limits_and_outages_stay_inconclusive(tmp_path: Path, status: int) -> None:
    reach, passed, _summary = _reach(tmp_path, status)
    assert reach.blocked == 0 and reach.ok
    assert passed


def test_one_reachable_document_is_enough() -> None:
    assert Reachability(sampled=3, reachable=1, gone=0, blocked=2).ok


@pytest.mark.parametrize(
    "title",
    [
        "Bank Transaction in Own Shares - 399 KB",
        "Notification of Significant Shareholding",
        "Total Voting Rights",
        "Issuer Call Notice - $1bn Fixed to Fixed Rate Notes",
        "Director/PDMR Shareholding",
        "Déclaration des transactions sur actions propres",
    ],
)
def test_regulatory_notices_are_not_results_press_releases(title: str) -> None:
    assert notice_type("press_release", title) == "other"


@pytest.mark.parametrize(
    "title", ["Q3 2025 results press release", "Annual results 2025", "Share buyback completed"]
)
def test_results_press_releases_stay(title: str) -> None:
    assert notice_type("press_release", title) == "press_release"
    assert notice_type("annual_report", "Transaction in own shares") == "annual_report"


def test_rules_and_llm_decisions_both_apply_the_notice_check() -> None:
    rules = ClassifyRules.model_validate(
        {"type": [{"pattern": "RNS|announcement", "target": "title", "type": "press_release"}]}
    )
    cls = classify(
        rules,
        url=f"{SITE}/uploads/rns-1.pdf",
        fields={"title": "RNS - Transaction in Own Shares"},
    )
    assert cls.type == "other"
    assert "regulatory notice" in " ".join(cls.justification)

    from retrievr.recipe import FoundDocument

    doc = FoundDocument(
        url=f"{SITE}/uploads/rns-2.pdf",
        normalized_url=f"{SITE}/uploads/rns-2.pdf",
        title="Transaction in Own Shares",
        source_id="reports",
        params={},
        page_url=f"{SITE}/reports",
        extension="pdf",
        date=None,
        type=None,
        period="2025",
        fiscal_year=2025,
        language="en",
        score=0.0,
        justification="type undecided by rules",
    )
    decided = apply_decisions(
        [doc],
        {
            doc.normalized_url: Decision(
                type="press_release", period=None, fiscal_year=None, language=None, source="llm"
            )
        },
    )
    assert decided[0].type == "other"


def test_history_from_an_older_library_is_not_read_as_drift(tmp_path: Path) -> None:
    """A run of an older version stored a notice as press_release: after the update, the
    known documents read it as ``other``, so health sees no metadata change."""
    from dataclasses import replace

    from retrievr.recipe import FoundDocument
    from retrievr.store import StateStore

    base = FoundDocument(
        url=f"{SITE}/uploads/rns-3.pdf",
        normalized_url=f"{SITE}/uploads/rns-3.pdf",
        title="Transaction in Own Shares",
        source_id="reports",
        params={},
        page_url=f"{SITE}/reports",
        extension="pdf",
        date=None,
        type="press_release",
        period="2025",
        fiscal_year=2025,
        language="en",
        score=1.0,
        justification="type=press_release by rule",
    )
    results = replace(
        base,
        url=f"{SITE}/uploads/q3.pdf",
        normalized_url=f"{SITE}/uploads/q3.pdf",
        title="Q3 2025 results press release",
    )
    store = StateStore(tmp_path / "state")
    store.record_documents("example", "example-ir@1", [base, results])
    types = {k.title: k.type for k in store.known_documents("example")}
    assert types == {
        "Transaction in Own Shares": "other",
        "Q3 2025 results press release": "press_release",
    }


ROW_FORMAT_PAGE = """<html><body><ul>
<li><article><h3>1Q26 - Press release results</h3><div class="type">pdf, EN, 6 mo</div>
  <a href="/en/document/1q26-pr"><span>View</span></a></article></li>
<li><article><h3>1Q26 - Results presentation</h3><div class="type">PDF (2,4 Mo)</div>
  <a href="/en/document/1q26-slides">View</a></article></li>
<li><article><h3>Our strategy in PDF</h3><a href="/en/strategy">Read more</a></article></li>
</ul><p>Brochure (pdf)</p><a href="/en/contact">Contact</a></body></html>"""


def test_a_format_announced_next_to_the_link_makes_it_a_document() -> None:
    from retrievr.candidates import group_document_links, parse_html, text_format

    groups = group_document_links(parse_html(ROW_FORMAT_PAGE), SITE)
    urls = [c.url for g in groups for c in g.candidates]
    assert urls == [f"{SITE}/en/document/1q26-pr", f"{SITE}/en/document/1q26-slides"]
    assert {c.extension for g in groups for c in g.candidates} == {"pdf"}
    assert text_format("Annual report PDF 3 Mo") == "pdf"  # French size units
