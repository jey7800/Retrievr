"""Recipe schema v4 (ADR 0011): documents on detail pages linked from a listing, and JSON
listings that give file names the page turns into download URLs."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx
import pydantic
import pytest

from retrievr.browser import Exchange
from retrievr.candidates import OPAQUE_DOCUMENT, json_candidates, label_year
from retrievr.config import FetchSettings
from retrievr.endpoints import analyse, file_url_template
from retrievr.fetch import Fetcher
from retrievr.recipe import recipe_from_mapping, run_recipe
from retrievr.snapshots import synthetic_document

SITE = "https://ir.example-issuer.com"
TODAY = date(2026, 11, 23)

LISTING = """<html><body><main>
<a href="/en/financial-results/?fiscal-year=2024">2024</a>
<div class="event"><h3>H1 {y} results</h3><a href="/en/event/h1-{y}-results/">View details</a></div>
<div class="event"><h3>FY {y} results</h3><a href="/en/event/fy-{y}-results/">View details</a></div>
<a href="https://elsewhere.example/event/x/">Partner event</a>
</main></body></html>"""

DETAIL = """<html><head><title>{name}</title></head><body><h1>{name}</h1>
<ul class="downloads">
<li><span>Press release</span> <a href="/en/file/{n}1?download=1">Download pdf</a></li>
<li><span>Presentation</span> <a href="/en/file/{n}2?download=1">Download pdf</a></li>
</ul><footer><a href="/legal.pdf">Legal</a></footer></body></html>"""


def site(request: httpx.Request) -> httpx.Response:
    path, query = request.url.path, request.url.query.decode()
    if path == "/robots.txt":
        return httpx.Response(404)
    if path == "/en/financial-results/":
        year = query.rsplit("=", 1)[-1]
        return httpx.Response(
            200, text=LISTING.format(y=year), headers={"content-type": "text/html"}
        )
    if path.startswith("/en/event/"):
        slug = path.strip("/").rsplit("/", 1)[-1]
        name = slug.replace("-", " ").upper()
        n = "".join(c for c in slug if c.isdigit())
        return httpx.Response(
            200, text=DETAIL.format(name=name, n=n), headers={"content-type": "text/html"}
        )
    if path.startswith("/en/file/"):
        return httpx.Response(200, content=synthetic_document("pdf", str(request.url)) or b"")
    return httpx.Response(404)


def _recipe(**source: Any) -> dict[str, Any]:
    return {
        "schema_version": 4,
        "recipe": "example-ir",
        "version": 1,
        "issuer": "example-issuer",
        "site": "ir.example-issuer.com",
        "allowed_domains": ["ir.example-issuer.com"],
        "sources": [{"id": "results", **source}],
    }


DETAIL_SOURCE = {
    "url": f"{SITE}/en/financial-results/?fiscal-year={{year}}",
    "params": {"year": {"from": 2024, "to": 2025}},
    "detail": {
        "link": "a[href*='/event/']",
        "items": {
            "container": "ul.downloads",
            "item": "li",
            "extensions": ["pdf"],
            "document_urls": [r"/file/\d+\?download=1"],
        },
    },
}


def test_documents_on_detail_pages_are_listed_with_the_page_heading() -> None:
    recipe = recipe_from_mapping(_recipe(**DETAIL_SOURCE))
    fetcher = Fetcher(FetchSettings(), transport=httpx.MockTransport(site), sleep=lambda _s: None)
    with fetcher:
        run = run_recipe(recipe, fetcher, today=TODAY)
    assert run.status == "ok", run.errors
    assert len(run.documents) == 8  # 2 years x 2 events x 2 documents, no legal footer
    assert all("/en/file/" in d.url for d in run.documents)
    assert {d.fiscal_year for d in run.documents} == {2024, 2025}  # from the year param
    headings = {d.fields["heading"] for d in run.documents}
    assert "H1 2025 RESULTS" in headings  # the detail page heading, for the period rules
    visited = [p.url for p in run.pages]
    assert not any("elsewhere.example" in u for u in visited)  # other sites are not opened


def test_detail_pages_require_schema_4() -> None:
    with pytest.raises(pydantic.ValidationError, match="schema_version 4"):
        recipe_from_mapping({**_recipe(**DETAIL_SOURCE), "schema_version": 3})


def test_opaque_download_links_with_a_query_are_documents() -> None:
    assert OPAQUE_DOCUMENT.search("https://ir.example-issuer.com/en/file/29269?download=1")
    assert OPAQUE_DOCUMENT.search("https://ir.example-issuer.com/download/123")


def test_period_labels_name_their_year() -> None:
    assert label_year("Q325") == 2025
    assert label_year("Q3 25") == 2025
    assert label_year("1H24") == 2024
    assert label_year("FY 2023") == 2023
    assert label_year("Annual report") is None


LINKS = [
    f"{SITE}/api/download-file/?path=First_Half_Report_2026_EN_d6129bcfad.pdf",
    f"{SITE}/en/finance/",
]
LISTING_JSON = {
    "data": {
        "publications": [
            {
                "name": "2026 First-half report",
                "publishedDate": "2026-07-28T16:00:00Z",
                "documents": [{"document": {"path": "First_Half_Report_2026_EN_d6129bcfad.pdf"}}],
            },
            {
                "name": "2025 Annual report",
                "publishedDate": "2026-02-11T16:00:00Z",
                "documents": [{"document": {"path": "Annual_Report_2025_EN_ab12.pdf"}}],
            },
        ]
    }
}


def test_file_names_of_a_json_listing_get_the_url_the_page_builds() -> None:
    template = file_url_template("First_Half_Report_2026_EN_d6129bcfad.pdf", LINKS)
    assert template == f"{SITE}/api/download-file/?path={{value}}"
    exchange = Exchange(
        id="X1",
        action="load",
        method="GET",
        url=f"{SITE}/api/dashboard/?page=1",
        request_body=None,
        request_content_type=None,
        requested_with=None,
        status=200,
        content_type="application/json",
        body=json.dumps(LISTING_JSON).encode(),
    )
    assert analyse(exchange).kind == "none"  # without the rendered links: no URL at all
    found = analyse(exchange, LINKS)
    assert found.kind == "json_list"
    lst = found.lists[0]  # the publications, before the nested documents lists
    assert lst.path == "data.publications"
    assert lst.url_templates == {"documents.0.document.path": template}
    assert f"{SITE}/api/download-file/?path=Annual_Report_2025_EN_ab12.pdf" in found.documents
    candidates = json_candidates(
        LISTING_JSON,
        f"{SITE}/api/dashboard/",
        items="data.publications",
        url="documents.0.document.path",
        title="name",
        extensions=["pdf"],
        url_template=template,
    )
    assert [c.url for c in candidates] == [
        f"{SITE}/api/download-file/?path=First_Half_Report_2026_EN_d6129bcfad.pdf",
        f"{SITE}/api/download-file/?path=Annual_Report_2025_EN_ab12.pdf",
    ]
    assert candidates[1].extension == "pdf"  # from the file name in the query


def test_a_url_template_needs_one_value_placeholder() -> None:
    source = {
        "kind": "request",
        "method": "GET",
        "url": f"{SITE}/api/dashboard/",
        "response": {
            "format": "json",
            "items": {
                "path": "data.publications",
                "url": "documents.0.document.path",
                "url_template": f"{SITE}/api/download-file/?path=x",
            },
        },
    }
    with pytest.raises(pydantic.ValidationError, match="exactly one"):
        recipe_from_mapping(_recipe(**source))


def test_nested_download_paths_are_documents() -> None:
    assert OPAQUE_DOCUMENT.search(
        "https://ir.example-issuer.com/downloads/documents/us-en/15db52348"
    )
    assert not OPAQUE_DOCUMENT.search("https://ir.example-issuer.com/investors/documents/overview")


def test_two_digit_years_in_listing_urls() -> None:
    from retrievr.learn import LearnContext, Learner, VisitedPage

    source = {
        "url": f"{SITE}/reports/ar{{yy}}/download-center/",
        "params": {"year": {"from": 2024, "to": 2025}},
    }
    recipe = recipe_from_mapping(_recipe(**source))
    urls = [u for _p, u in recipe.sources[0].expand(TODAY)]  # type: ignore[union-attr]
    assert urls == [
        f"{SITE}/reports/ar24/download-center/",
        f"{SITE}/reports/ar25/download-center/",
    ]
    with pytest.raises(pydantic.ValidationError, match="schema_version 4"):
        recipe_from_mapping({**_recipe(**source), "schema_version": 3})

    context = LearnContext(issuer="example-issuer", start_url=SITE, years=[2025], today=TODAY)
    learner = Learner(Fetcher(FetchSettings()), None, context)  # type: ignore[arg-type]
    seen = f"{SITE}/reports/ar25/download-center/"
    learner.pages["P2"] = VisitedPage("P2", seen, seen, "Download center", groups=[], nav=[])
    template = f"{SITE}/reports/ar{{year}}/download-center/"
    assert learner._year_form(template) == f"{SITE}/reports/ar{{yy}}/download-center/"
    four = f"{SITE}/reports/{{year}}/"
    learner.pages["P3"] = VisitedPage(
        "P3", f"{SITE}/reports/2025/", f"{SITE}/reports/2025/", "x", [], []
    )
    assert learner._year_form(four) == four


def test_web_pages_in_download_folders_are_not_documents() -> None:
    assert not OPAQUE_DOCUMENT.search(
        "https://ir.example-issuer.com/reports/ar02/downloads/default.htm"
    )
    assert not OPAQUE_DOCUMENT.search("https://ir.example-issuer.com/files/index.aspx?id=3")
    assert OPAQUE_DOCUMENT.search("https://ir.example-issuer.com/en/file/29269?download=1")


def test_a_detail_page_with_another_layout_is_read_from_its_main_content() -> None:
    old_layout = (
        '<html><body><h1>FY 2019 results</h1><main><div class="old">'
        '<a href="/en/file/20191?download=1">Download pdf</a></div></main>'
        '<footer><a href="/legal.pdf">Legal</a></footer></body></html>'
    )

    def mixed(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/en/event/fy-2025-results/":
            return httpx.Response(200, text=old_layout, headers={"content-type": "text/html"})
        return site(request)

    recipe = recipe_from_mapping(_recipe(**DETAIL_SOURCE))
    fetcher = Fetcher(FetchSettings(), transport=httpx.MockTransport(mixed), sleep=lambda _s: None)
    with fetcher:
        run = run_recipe(recipe, fetcher, today=TODAY)
    urls = {d.url for d in run.documents}
    assert f"{SITE}/en/file/20191?download=1" in urls  # found in <main> of the old layout
    assert f"{SITE}/legal.pdf" not in urls  # the footer stays out
    assert any("main read on" in w for w in run.warnings)
