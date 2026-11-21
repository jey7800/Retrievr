from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import pydantic
import pytest
import respx
import yaml

from retrievr.classify import ClassifyRules, classify, normalize_period
from retrievr.recipe import UrlTemplateSource, load_recipe, recipe_from_mapping, run_recipe
from tests.conftest import RECIPES

SITE = "https://ir.example.com"


def _recipe(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": 1,
        "recipe": "example-ir",
        "version": 1,
        "issuer": "example",
        "site": "ir.example.com",
        "allowed_domains": ["ir.example.com"],
        "sources": [
            {
                "id": "results",
                "url": SITE + "/docs?year={year}",
                "params": {"year": {"from": 2024, "to": 2025}},
                "pagination": {"selector": "a.next", "max_pages": 3},
                "items": {"container": "#list", "extensions": ["pdf"]},
            }
        ],
    }
    data.update(overrides)
    return data


def test_all_shipped_recipes_validate() -> None:
    recipes = sorted(RECIPES.glob("*.yaml"))
    assert recipes
    for path in recipes:
        recipe = load_recipe(path)
        assert path.name == f"{recipe.key}.yaml"


def test_minimal_recipe_validates() -> None:
    recipe = recipe_from_mapping(_recipe())
    source = recipe.sources[0]
    assert isinstance(source, UrlTemplateSource)
    assert [u for _, u in source.expand(date(2026, 1, 1))] == [
        SITE + "/docs?year=2024",
        SITE + "/docs?year=2025",
    ]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"unknown_key": 1}, "Extra inputs"),
        ({"schema_version": 4}, "schema_version"),
        ({"allowed_domains": ["other.example.com"]}, "allowed_domains"),
        (
            {"sources": [{"id": "s", "url": "https://evil.test/x"}]},
            "outside allowed_domains",
        ),
        ({"sources": [{"id": "s", "url": SITE + "/x?y={year}"}]}, "do not match params"),
        (
            {"sources": [{"id": "s", "url": SITE, "items": {"container": "div[["}}]},
            "invalid CSS selector",
        ),
        ({"classify": {"type": [{"pattern": "(", "type": "x"}]}}, "invalid regular expression"),
        ({"classify": {"period": [{"pattern": "t[1-4]"}]}}, "named group 'year'"),
    ],
)
def test_invalid_recipes_are_rejected(override: dict[str, Any], message: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=message):
        recipe_from_mapping(_recipe(**override))


def test_yaml_on_key_pitfall_is_rejected() -> None:
    """YAML 1.1 reads a bare ``on`` key as boolean True: it must not pass silently."""
    data = yaml.safe_load("type: [{pattern: x, on: text, type: t}]")
    with pytest.raises(pydantic.ValidationError):
        ClassifyRules.model_validate(data)


def test_classify_first_rule_wins_and_defaults() -> None:
    rules = ClassifyRules.model_validate(
        {
            "type": [
                {"pattern": "series-restated", "type": "restated"},
                {"pattern": "series", "type": "series"},
            ],
            "period": [{"pattern": r"(?P<h>[hs][12])[-_](?P<year>20\d{2})"}],
            "language": [{"pattern": r"_(?P<lang>fr|en)\.pdf$"}],
        }
    )
    c = classify(rules, url="https://x/S1-2025-series-restated.pdf", text="", default_language="en")
    assert (c.type, c.period, c.language, c.score) == ("restated", "2025-H1", "en", 1.0)
    u = classify(rules, url="https://x/unknown_FR.pdf", text="")
    assert (u.type, u.language, u.score) == (None, "fr", 0.0)
    assert "type undecided by rules" in u.justification


def test_normalize_period_variants() -> None:
    import re

    assert normalize_period(re.search(r"(?P<q>q3)-(?P<year>2026)", "q3-2026")) == ("2026-Q3", 2026)  # type: ignore[arg-type]
    assert normalize_period(re.search(r"(?P<fy>FY)(?P<year>24)", "FY24")) == ("2024-FY", 2024)  # type: ignore[arg-type]


def _page(items: str, next_href: str | None = None) -> str:
    nxt = f'<a class="next" href="{next_href}">next</a>' if next_href else ""
    return f"<html><body><div id='list'>{items}</div>{nxt}</body></html>"


def test_run_stops_at_max_pages_and_drops_foreign_links(
    no_network: respx.MockRouter, make_fetcher: Any
) -> None:
    no_network.get(SITE + "/robots.txt").respond(404)

    def endless(request: httpx.Request) -> httpx.Response:
        n = int(request.url.params.get("p", "0"))
        items = f'<a href="/f/{n}.pdf">d</a><a href="https://evil.test/x{n}.pdf">x</a>'
        return httpx.Response(200, html=_page(items, f"?year=2025&p={n + 1}"))

    no_network.get(SITE + "/docs").mock(side_effect=endless)
    result = run_recipe(
        recipe_from_mapping(_recipe()), make_fetcher(), years=[2025], today=date(2026, 1, 1)
    )
    assert len(result.pages) == 3
    assert [d.url.rsplit("/", 1)[-1] for d in result.documents] == ["0.pdf", "1.pdf", "2.pdf"]
    assert any("max_pages=3" in w for w in result.warnings)
    assert any("outside domains" in w for w in result.warnings)


def test_run_reports_blocked(no_network: respx.MockRouter, make_fetcher: Any) -> None:
    no_network.get(SITE + "/robots.txt").respond(404)
    no_network.get(SITE + "/docs").respond(429)
    result = run_recipe(recipe_from_mapping(_recipe()), make_fetcher(), today=date(2026, 1, 1))
    assert result.status == "blocked"
    assert result.documents == []
    assert len(no_network.calls) == 2  # robots + first page, then stop


def test_run_never_follows_next_link_outside_domains(
    no_network: respx.MockRouter, make_fetcher: Any
) -> None:
    no_network.get(SITE + "/robots.txt").respond(404)
    no_network.get(SITE + "/docs").respond(
        200, html=_page('<a href="/f/a.pdf">a</a>', "https://evil.test/page2")
    )
    result = run_recipe(
        recipe_from_mapping(_recipe()), make_fetcher(), years=[2025], today=date(2026, 1, 1)
    )
    assert result.status == "ok"
    assert len(result.pages) == 1


def test_rule_can_target_joined_fields() -> None:
    rules = ClassifyRules.model_validate(
        {
            "type": [{"pattern": "^releases$", "target": "column_header", "type": "press_release"}],
            "period": [
                {
                    "pattern": r"^Q(?P<q>[1-4]) \| (?P<year>20\d{2})$",
                    "target": ["row_header", "heading"],
                }
            ],
        }
    )
    c = classify(
        rules,
        url="https://x/a.pdf",
        fields={"column_header": "Releases", "row_header": "Q2", "heading": "2025"},
    )
    assert (c.type, c.period) == ("press_release", "2025-Q2")
    assert "row_header+heading" in "; ".join(c.justification)


@pytest.mark.parametrize(
    "override",
    [
        {"sources": [{"id": "s", "url": SITE, "items": {"table_context": True}}]},
        {"classify": {"type": [{"pattern": "x", "target": "column_header", "type": "t"}]}},
    ],
)
def test_v2_features_need_schema_version_2(override: dict[str, Any]) -> None:
    with pytest.raises(pydantic.ValidationError, match=r"requires? schema_version 2"):
        recipe_from_mapping(_recipe(**override))
    assert recipe_from_mapping(_recipe(schema_version=2, **override)).schema_version == 2


def test_target_list_is_bounded() -> None:
    with pytest.raises(pydantic.ValidationError, match="1 to 4 fields"):
        ClassifyRules.model_validate({"type": [{"pattern": "x", "target": [], "type": "t"}]})
