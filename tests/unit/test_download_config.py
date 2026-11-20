from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pydantic
import pytest
import respx

from retrievr.config import load_settings
from retrievr.download import Downloader, Manifest, check_magic, target_path
from retrievr.recipe import FoundDocument, Recipe, recipe_from_mapping
from retrievr.urls import normalize_url
from tests.conftest import pdf_bytes, xlsx_bytes

SITE = "https://ir.example.com"


@pytest.fixture
def recipe() -> Recipe:
    return recipe_from_mapping(
        {
            "schema_version": 1,
            "recipe": "example-ir",
            "version": 2,
            "issuer": "example",
            "site": "ir.example.com",
            "allowed_domains": ["ir.example.com"],
            "sources": [{"id": "s", "url": SITE + "/docs"}],
        }
    )


def _doc(name: str, **kw: Any) -> FoundDocument:
    url = f"{SITE}/files/{name}"
    defaults: dict[str, Any] = {
        "title": name,
        "source_id": "s",
        "params": {},
        "page_url": SITE + "/docs",
        "extension": name.rsplit(".", 1)[-1],
        "date": date(2026, 2, 6),
        "type": "press_release",
        "period": "2025-Q4",
        "fiscal_year": 2025,
        "language": "fr",
        "score": 1.0,
        "justification": "test",
    }
    defaults.update(kw)
    return FoundDocument(url=url, normalized_url=normalize_url(url), **defaults)


@pytest.fixture
def site(no_network: respx.MockRouter) -> respx.MockRouter:
    no_network.get(SITE + "/robots.txt").respond(404)
    return no_network


def test_target_path_is_safe_and_typed() -> None:
    assert target_path("example", _doc("a.pdf")) == "example/2025/press_release_2025-Q4_fr.pdf"
    odd = _doc("b.PDF", type=None, period=None, language=None, fiscal_year=None, extension="PDF")
    assert target_path("ex/../ample", odd) == "ex-ample/unknown-year/document_noperiod_und.pdf"


def test_magic_bytes() -> None:
    assert check_magic("pdf", b"%PDF-1.7\n....") is None
    assert check_magic("pdf", b"<!DOCTYPE html><html>") == (
        "HTML content instead of a binary document"
    )
    assert check_magic("xlsx", b"%PDF-1.7 ") is not None
    assert check_magic("csv", b"a,b,c\n1,2,3") is None


def test_download_writes_file_and_manifest(
    site: respx.MockRouter, make_fetcher: Any, recipe: Recipe, tmp_path: Path
) -> None:
    site.get(SITE + "/files/a.pdf").respond(200, content=pdf_bytes("a"))
    [outcome] = Downloader(make_fetcher(), tmp_path).download_all(recipe, [_doc("a.pdf")])
    assert outcome.status == "downloaded"
    assert outcome.entry is not None
    assert (tmp_path / outcome.entry.path).read_bytes() == pdf_bytes("a")
    entry = Manifest.load(tmp_path).entries[0]
    assert entry.recipe_version == 2 and entry.sha256 == outcome.entry.sha256
    assert not list((tmp_path / ".retrievr-tmp").iterdir())


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(200, html="<html>login</html>"), "HTML page instead"),
        (httpx.Response(200, content=b"<!doctype html><p>x</p>"), "HTML content"),
        (httpx.Response(200, content=b"GIF89a......"), "bad magic"),
        (httpx.Response(200, content=b""), "empty file"),
    ],
)
def test_non_conforming_downloads_are_rejected(
    site: respx.MockRouter,
    make_fetcher: Any,
    recipe: Recipe,
    tmp_path: Path,
    response: httpx.Response,
    reason: str,
) -> None:
    site.get(SITE + "/files/a.pdf").mock(return_value=response)
    [outcome] = Downloader(make_fetcher(), tmp_path).download_all(recipe, [_doc("a.pdf")])
    assert outcome.status == "rejected"
    assert reason in (outcome.reason or "")
    assert Manifest.load(tmp_path).entries == []
    assert not (tmp_path / "example").exists()


def test_corrupted_xlsx_is_rejected(
    site: respx.MockRouter, make_fetcher: Any, recipe: Recipe, tmp_path: Path
) -> None:
    site.get(SITE + "/files/a.xlsx").respond(200, content=b"PK\x03\x04garbage-not-a-zip")
    [outcome] = Downloader(make_fetcher(), tmp_path).download_all(recipe, [_doc("a.xlsx")])
    assert outcome.status == "rejected"
    assert "bad zip" in (outcome.reason or "")


def test_same_content_at_two_urls_is_stored_once(
    site: respx.MockRouter, make_fetcher: Any, recipe: Recipe, tmp_path: Path
) -> None:
    site.get(SITE + "/files/a.xlsx").respond(200, content=xlsx_bytes("same"))
    site.get(SITE + "/files/b.xlsx").respond(200, content=xlsx_bytes("same"))
    outcomes = Downloader(make_fetcher(), tmp_path).download_all(
        recipe, [_doc("a.xlsx"), _doc("b.xlsx", period="2025-Q3")]
    )
    assert [o.status for o in outcomes] == ["downloaded", "duplicate_content"]
    [entry] = Manifest.load(tmp_path).entries
    assert entry.alternate_urls == [normalize_url(SITE + "/files/b.xlsx")]


def test_name_collision_gets_a_deterministic_suffix(
    site: respx.MockRouter, make_fetcher: Any, recipe: Recipe, tmp_path: Path
) -> None:
    site.get(SITE + "/files/a.pdf").respond(200, content=pdf_bytes("a"))
    site.get(SITE + "/files/b.pdf").respond(200, content=pdf_bytes("b"))
    outcomes = Downloader(make_fetcher(), tmp_path).download_all(
        recipe, [_doc("a.pdf"), _doc("b.pdf")]
    )
    paths = [o.entry.path for o in outcomes if o.entry]
    assert paths[0] == "example/2025/press_release_2025-Q4_fr.pdf"
    assert paths[1].startswith("example/2025/press_release_2025-Q4_fr_")
    assert len(set(paths)) == 2


def test_blocked_download_stops_the_loop(
    site: respx.MockRouter, make_fetcher: Any, recipe: Recipe, tmp_path: Path
) -> None:
    site.get(SITE + "/files/a.pdf").respond(429)
    outcomes = Downloader(make_fetcher(), tmp_path).download_all(
        recipe, [_doc("a.pdf"), _doc("b.pdf")]
    )
    assert [o.status for o in outcomes] == ["blocked"]


def test_settings_load_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "retrievr.toml"
    path.write_text(
        '[fetch]\nrate_limit_rps = 0.5\nrobots_unreachable = "deny"\n'
        '[output]\noutput_dir = "out"\n[llm]\nbase_url = "https://llm.example/v1"\n',
        encoding="utf-8",
    )
    settings = load_settings(path)
    assert settings.fetch.rate_limit_rps == 0.5
    assert settings.fetch.robots_unreachable == "deny"
    assert settings.output.output_dir == Path("out")
    assert load_settings().fetch.robots == "respect"


def test_settings_reject_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "retrievr.toml"
    path.write_text("[fetch]\nstealth = true\n", encoding="utf-8")
    with pytest.raises(pydantic.ValidationError):
        load_settings(path)
