"""Recipe schema v5: a listing held as JSON inside the page (a data attribute or a JSON
script) and ``*`` in JSON item paths. No network."""

from __future__ import annotations

import html
import json
from datetime import date

import httpx
import pydantic
import pytest

from retrievr.candidates import json_items, parse_html
from retrievr.config import FetchSettings
from retrievr.endpoints import digest_endpoints, embedded_exchanges, endpoints
from retrievr.fetch import Fetcher
from retrievr.recipe import recipe_from_mapping, run_recipe

SITE = "https://ir.example-issuer.com"
PAGE = f"{SITE}/en/financial-results.html"
TODAY = date(2026, 11, 24)


def _reports() -> dict[str, dict[str, list[dict[str, str]]]]:
    return {
        str(year): {
            f"Q{q}": [
                {
                    "title": f"Q{q} {year} report to shareholders",
                    "link_1_path": f"/content/reports/{year}/q{q}/Q{q}{year % 100}_Report.pdf",
                },
                {
                    "title": f"Q{q} {year} investor presentation",
                    "link_1_path": f"/content/reports/{year}/q{q}/Q{q}{year % 100}_Slides.pdf",
                },
            ]
            for q in (1, 2, 3, 4)
        }
        for year in (2024, 2025)
    }


def _page() -> str:
    data = html.escape(json.dumps(_reports()), quote=True)
    return f"""<html><body>
<select id="year"><option selected>2026</option><option>2025</option><option>2024</option></select>
<div class="cards"><a href="/content/reports/2026/q3/Q326_Report.pdf">Q3 2026 report</a></div>
<report-cards class="cards-data" data-reports="{data}"></report-cards>
<script type="application/json" id="analytics">{{"page": "results", "pad": "{"x" * 300}"}}
</script>
</body></html>"""


def _recipe(**response: object) -> dict[str, object]:
    return {
        "schema_version": 5,
        "recipe": "example-ir",
        "version": 1,
        "issuer": "example-issuer",
        "site": "ir.example-issuer.com",
        "allowed_domains": ["ir.example-issuer.com"],
        "sources": [
            {
                "id": "results",
                "kind": "request",
                "method": "GET",
                "url": PAGE,
                "response": {
                    "format": "json",
                    "embedded": {
                        "selector": "report-cards[data-reports]",
                        "attribute": "data-reports",
                    },
                    "items": {"path": "*.*", "url": "link_1_path", "title": "title"},
                    **response,
                },
                "items": {"extensions": ["pdf"]},
            }
        ],
    }


def _fetcher() -> Fetcher:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/en/financial-results.html":
            return httpx.Response(200, html=_page())
        return httpx.Response(404)

    return Fetcher(FetchSettings(), transport=httpx.MockTransport(handler), sleep=lambda _s: None)


def test_wildcard_paths_walk_every_key_and_keep_them() -> None:
    items = json_items(_reports(), "*.*")
    assert len(items) == 16
    assert items[0][0] == ["2024", "Q1"]
    assert json_items({"a": {"b": [1, 2]}}, "a.b") == [([], 1), ([], 2)]


def test_embedded_json_listing_is_found_and_described() -> None:
    root = parse_html(_page())
    found = embedded_exchanges(root, PAGE)
    assert [ex.embedded for ex in found] == [
        ("script#analytics", None),
        ("report-cards.cards-data[data-reports]", "data-reports"),
    ]
    eps = endpoints(found)  # the analytics JSON lists no document: no endpoint
    assert len(eps) == 1
    analysis = eps[0].documents["J2"]
    assert analysis.lists[0].path == "*.*" and analysis.lists[0].size == 16
    assert len(analysis.documents) == 16
    text = digest_endpoints(eps, TODAY)
    assert "JSON held by the page" in text and "J ids: no call" in text


def test_scripts_holding_json_are_read_too() -> None:
    blob = json.dumps({"docs": [{"url": f"/files/report-{i}.pdf"} for i in range(30)]})
    root = parse_html(f'<html><body><script type="application/json">{blob}</script></body></html>')
    found = embedded_exchanges(root, PAGE)
    assert [ex.embedded for ex in found] == [('script[type="application/json"]', None)]


def test_an_embedded_recipe_replays_from_the_page() -> None:
    recipe = recipe_from_mapping(_recipe())
    with _fetcher() as fetcher:
        result = run_recipe(recipe, fetcher, today=TODAY)
    assert result.status == "ok"
    assert len(result.documents) == 16
    first = min(result.documents, key=lambda d: d.url)
    assert first.url == f"{SITE}/content/reports/2024/q1/Q124_Report.pdf"


def test_schema_v5_features_need_schema_v5() -> None:
    data = _recipe()
    data["schema_version"] = 4
    with pytest.raises(pydantic.ValidationError, match="schema_version 5"):
        recipe_from_mapping(data)
    with pytest.raises(pydantic.ValidationError, match="json responses only"):
        recipe_from_mapping(
            _recipe(format="html", items=None)  # embedded on an HTML response
        )
