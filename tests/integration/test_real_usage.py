"""What a user does in real use, on the Deutsche Bank snapshot (one static page listing
2007-2026), offline: document files are served by a fake transport."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx

from retrievr.api import Retrievr
from retrievr.config import (
    BrowserSettings,
    FetchSettings,
    LLMSettings,
    OutputSettings,
    Settings,
    StateSettings,
)
from retrievr.events import Event
from retrievr.fetch import Fetcher, NetworkError
from retrievr.snapshots import ReplayTransport, load_index

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "tests" / "fixtures" / "deutschebank"
RECIPE = ROOT / "recipes" / "deutschebank-ir@1.yaml"


class SiteWithFiles(httpx.BaseTransport):
    """Pages from the snapshot; every PDF is a small valid file, other files are absent."""

    def __init__(self) -> None:
        self.pages = ReplayTransport(SNAPSHOT, lenient_bodies=True)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.lower()
        if path.endswith(".pdf"):
            body = b"%PDF-1.4 " + str(request.url).encode()
            return httpx.Response(200, content=body, headers={"content-type": "application/pdf"})
        if path.endswith((".xls", ".xlsx", ".xlsb", ".mp3")):
            return httpx.Response(404)
        return self.pages.handle_request(request)


@pytest.fixture
def offline() -> Iterator[None]:
    with respx.mock(assert_all_called=False):
        yield


def _retrievr(tmp_path: Path, events: list[Event] | None = None) -> Retrievr:
    settings = Settings(
        llm=LLMSettings(api_key_env="RETRIEVR_TEST_NO_SUCH_KEY", cache="off"),
        state=StateSettings(state_dir=tmp_path / "state"),
        output=OutputSettings(output_dir=tmp_path / "out"),
        browser=BrowserSettings(enabled=False),
    )
    captured = load_index(SNAPSHOT).captured_on()
    return Retrievr(
        settings=settings,
        on_event=events.append if events is not None else None,
        fetch_transport=SiteWithFiles(),
        sleep=lambda _s: None,
        today=captured.date() if captured else None,
    )


def test_years_limit_what_is_returned_and_downloaded(tmp_path: Path, offline: None) -> None:
    r = _retrievr(tmp_path)
    r.install(RECIPE)
    outcome = r.run("deutschebank", years=[2025])
    assert outcome.status == "ok", outcome.errors
    assert {2024, 2025} <= {d.fiscal_year for d in outcome.all_documents}
    assert outcome.documents
    assert {d.fiscal_year for d in outcome.documents} == {2025}
    files = list((tmp_path / "out" / "deutschebank").iterdir())
    assert [f.name for f in files] == ["2025"]
    assert not (tmp_path / "out" / ".retrievr-tmp").exists()


def test_a_request_that_cannot_be_compiled_downloads_nothing(tmp_path: Path, offline: None) -> None:
    """Without an LLM key the stored request cannot be compiled: never fall back to
    downloading every document of the site."""
    r = _retrievr(tmp_path)
    r.install(RECIPE)
    r.store.set_site_request("deutschebank", "annual reports")
    outcome = r.run("deutschebank", years=[2025])
    assert outcome.status == "error"
    assert outcome.documents == [] and outcome.downloads == []
    assert "could not be compiled" in outcome.errors[0]
    assert "--env-file" in outcome.errors[0]
    assert not (tmp_path / "out").exists()


def test_a_deleted_file_is_downloaded_again(tmp_path: Path, offline: None) -> None:
    r = _retrievr(tmp_path)
    r.install(RECIPE)
    first = r.run("deutschebank", years=[2025])
    downloaded = [o for o in first.downloads if o.status == "downloaded"]
    assert downloaded
    gone = tmp_path / "out" / Path(downloaded[0].entry.path)  # type: ignore[union-attr]
    gone.unlink()
    second = r.run("deutschebank", years=[2025])
    statuses = [o.status for o in second.downloads]
    assert statuses.count("downloaded") == 1
    assert statuses.count("known_url") == len(downloaded) - 1
    assert gone.is_file()


def test_downloads_report_progress(tmp_path: Path, offline: None) -> None:
    events: list[Event] = []
    r = _retrievr(tmp_path, events)
    r.install(RECIPE)
    outcome = r.run("deutschebank", years=[2025])
    progress = [e for e in events if e.kind == "document_downloaded"]
    assert len(progress) == len(outcome.downloads)
    assert progress[-1].data["index"] == progress[-1].data["total"] == len(outcome.documents)


def test_state_works_from_another_working_directory(
    tmp_path: Path, offline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(state=StateSettings(state_dir=Path("state")))
    Retrievr(settings=settings).install(RECIPE)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    absolute = Settings(state=StateSettings(state_dir=tmp_path / "state"))
    recipe = Retrievr(settings=absolute).store.active_recipe("deutschebank")
    assert recipe is not None and recipe.issuer == "deutschebank"


def test_an_edited_recipe_needs_a_new_version(tmp_path: Path) -> None:
    r = Retrievr(settings=Settings(state=StateSettings(state_dir=tmp_path / "state")))
    r.install(RECIPE)
    r.install(RECIPE)  # the same recipe again is fine
    edited = tmp_path / "edited.yaml"
    text = RECIPE.read_text(encoding="utf-8")
    edited.write_text(text.replace("max_gap_days: 110", "max_gap_days: 120"), encoding="utf-8")
    assert edited.read_text(encoding="utf-8") != text
    with pytest.raises(ValueError, match="raise 'version'"):
        r.install(edited)


def test_unknown_site_says_which_sites_exist(tmp_path: Path) -> None:
    r = Retrievr(settings=Settings(state=StateSettings(state_dir=tmp_path / "state")))
    r.install(RECIPE)
    outcome = r.run("deutsche-bank")
    assert outcome.status == "no_recipe"
    assert "active sites: deutschebank" in outcome.errors[0]


def test_learn_rejects_a_start_url_without_scheme(tmp_path: Path) -> None:
    r = Retrievr(settings=Settings(state=StateSettings(state_dir=tmp_path / "state")))
    with pytest.raises(ValueError, match="invalid start URL"):
        r.learn("investor-relations.example-issuer.com/reports")


def test_a_get_is_retried_once_after_a_network_error_a_post_never() -> None:
    calls: dict[str, int] = {"GET": 0, "POST": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls[request.method] += 1
        if calls[request.method] == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200, text="<html><body>ok</body></html>")

    fetcher = Fetcher(FetchSettings(), transport=httpx.MockTransport(flaky), sleep=lambda _s: None)
    with fetcher:
        assert fetcher.get("https://ir.example-issuer.com/list").status == 200
        with pytest.raises(NetworkError):
            fetcher.request("POST", "https://ir.example-issuer.com/api", content=b"{}")
    assert calls == {"GET": 2, "POST": 1}


def test_a_run_where_no_file_can_be_downloaded_says_so(tmp_path: Path, offline: None) -> None:
    class NoFiles(SiteWithFiles):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.lower().endswith(".pdf"):
                return httpx.Response(404)
            return super().handle_request(request)

    r = _retrievr(tmp_path)
    r._fetch_transport = NoFiles()
    r.install(RECIPE)
    outcome = r.run("deutschebank", years=[2025])
    assert outcome.downloads and all(o.status == "error" for o in outcome.downloads)
    assert any("none of the" in w and "HTTP 404" in w for w in outcome.warnings)
