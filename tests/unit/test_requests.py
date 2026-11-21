"""Schema v3 and browser discovery: POST requests, snapshots v2, request sources, JSON
listings, opaque links, endpoint analysis, recorded discoveries, and learning a request
source with a scripted LLM. No network, no browser."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Sequence
from datetime import date
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import parse_qsl

import httpx
import pydantic
import pytest
from pydantic import BaseModel

from retrievr.api import Retrievr
from retrievr.browser import Discovery, Exchange, ReplayDiscovery, _target_options, save_discovery
from retrievr.candidates import extract_candidates, json_candidates, parse_html
from retrievr.classify import ClassifyRules, PeriodRule, classify
from retrievr.config import BrowserSettings, FetchSettings, Settings, StateSettings
from retrievr.endpoints import analyse, digest_endpoints, endpoints, templated_body
from retrievr.fetch import Fetcher
from retrievr.learn import (
    FieldChange,
    PeriodRuleDraft,
    RequestDraft,
    RulesReply,
    StructureReplyWithRequests,
    TypeRuleDraft,
)
from retrievr.llm import Message
from retrievr.recipe import RequestSource, recipe_from_mapping, run_recipe
from retrievr.snapshots import ReplayTransport, SnapshotRecorder, canonical_body

T = TypeVar("T", bound=BaseModel)
SITE = "https://ir.example-issuer.com"
TODAY = date(2026, 11, 21)


# -- a small site whose listing comes from POST endpoints --------------------------------

START = (
    "<html><head><title>Publications</title></head><body>"
    '<select id="year"><option value="9025">2025</option><option value="9024">2024</option>'
    '<option value="9023">2023</option></select><div id="list"></div></body></html>'
)


def _fragment(year: int) -> str:
    return (
        '<ul class="docs">'
        f'<li class="doc"><h3>Annual report {year}</h3>'
        f'<a href="/preview/{year}1" title="annual-report-{year}.pdf">Download</a></li>'
        f'<li class="doc"><h3>Half-year report {year}</h3>'
        f'<a href="/preview/{year}2" title="half-year-report-{year}.pdf">Download</a></li>'
        "</ul>"
    )


def _hits(page: int) -> list[dict[str, Any]]:
    data: dict[int, list[tuple[str, str | None]]] = {
        0: [("Q3 2025 sales", "q3-2025.pdf"), ("Policy", None)],
        1: [("Q1 2025 sales", "q1-2025.pdf")],
    }
    return [
        {"name": n, "file": {"url": f"https://cdn.example-issuer.com/{f}"} if f else None}
        for n, f in data.get(page, [])
    ]


def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/robots.txt":
        return httpx.Response(404)
    if path == "/publications" and request.method == "GET":
        return httpx.Response(200, text=START, headers={"content-type": "text/html"})
    if path == "/display" and request.method == "POST":
        form = dict(parse_qsl(request.content.decode()))
        year = {"9025": 2025, "9024": 2024, "9023": 2023}.get(form.get("loc", ""))
        if (
            year is None
            or form.get("page") != "1"
            or request.headers.get("x-requested-with") is None
        ):
            return httpx.Response(404)
        return httpx.Response(200, json={"more": False, "content": _fragment(year)})
    if path == "/api/search" and request.method == "POST":
        body = json.loads(request.content)
        return httpx.Response(200, json={"results": [{"hits": _hits(body["page"])}]})
    return httpx.Response(404)


@pytest.fixture
def fetcher() -> Fetcher:
    return Fetcher(FetchSettings(), transport=httpx.MockTransport(handler), sleep=lambda _s: None)


def _recipe(**sources: Any) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "recipe": "example-ir",
        "version": 1,
        "issuer": "example-issuer",
        "site": "ir.example-issuer.com",
        "allowed_domains": ["ir.example-issuer.com", "cdn.example-issuer.com"],
        "sources": list(sources.values()),
    }


FORM_SOURCE = {
    "id": "fragments",
    "kind": "request",
    "url": f"{SITE}/display",
    "headers": {"X-Requested-With": "XMLHttpRequest"},
    "form": {"loc": "{location}", "page": "{page}", "until": "{today:%d/%m/%Y}"},
    "params": {
        "location": {"kind": "options", "page": f"{SITE}/publications", "selector": "#year option"}
    },
    "pagination": {"kind": "page_number", "start": 1, "has_more": "more"},
    "response": {"format": "json", "html_path": "content"},
    "items": {"item": "li.doc", "title": "h3", "extensions": ["pdf"]},
}
JSON_SOURCE = {
    "id": "search",
    "kind": "request",
    "url": f"{SITE}/api/search",
    "json": {"query": "", "page": "{page}"},
    "pagination": {"kind": "page_number", "start": 0},
    "response": {
        "format": "json",
        "items": {"path": "results.0.hits", "url": ["file.url"], "title": "name"},
    },
    "items": {"extensions": ["pdf"]},
}


# -- fetch and snapshots ----------------------------------------------------------------


def test_post_keeps_only_allowed_headers() -> None:
    seen: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(200)

    f = Fetcher(FetchSettings(), transport=httpx.MockTransport(capture), sleep=lambda _s: None)
    f.request(
        "POST",
        f"{SITE}/x",
        content=b"a=1",
        headers={"X-Requested-With": "XMLHttpRequest", "Cookie": "session=1", "Authorization": "x"},
    )
    request = seen[-1]
    assert request.method == "POST" and request.content == b"a=1"
    assert request.headers["x-requested-with"] == "XMLHttpRequest"
    assert "cookie" not in request.headers and "authorization" not in request.headers


def test_canonical_bodies() -> None:
    assert canonical_body(b'{"b": 1, "a": [2]}') == '{"a": [2], "b": 1}'
    assert canonical_body("b=2&a=1&a%5Bx%5D=", "application/x-www-form-urlencoded") == (
        "a=1&a%5Bx%5D=&b=2"
    )
    assert canonical_body(b"") is None


def test_post_snapshot_round_trip(tmp_path: Path, fetcher: Fetcher) -> None:
    recorder = SnapshotRecorder(tmp_path, "example")
    live = Fetcher(
        FetchSettings(),
        transport=httpx.MockTransport(handler),
        event_hooks={"response": [recorder.hook]},
        sleep=lambda _s: None,
    )
    body = json.dumps({"page": 0, "query": ""}).encode()
    live.request("POST", f"{SITE}/api/search", content=body)
    recorder.save()
    replay = ReplayTransport(tmp_path)
    offline = Fetcher(FetchSettings(), transport=replay, sleep=lambda _s: None)
    same = json.dumps({"query": "", "page": 0}).encode()  # other key order: same request
    page = offline.request("POST", f"{SITE}/api/search", content=same)
    assert json.loads(page.content)["results"][0]["hits"][0]["name"] == "Q3 2025 sales"
    smaller = json.dumps({"page": 0}).encode()
    with pytest.raises(Exception, match="404"):
        offline.request("POST", f"{SITE}/api/search", content=smaller)
    lenient = Fetcher(
        FetchSettings(),
        transport=ReplayTransport(tmp_path, lenient_bodies=True),
        sleep=lambda _s: None,
    )
    assert lenient.request("POST", f"{SITE}/api/search", content=smaller).status == 200


# -- recipe v3 ---------------------------------------------------------------------------


def test_request_sources_run(fetcher: Fetcher) -> None:
    recipe = recipe_from_mapping(_recipe(a=FORM_SOURCE, b=JSON_SOURCE))
    run = run_recipe(recipe, fetcher, years=[2024], today=TODAY)
    assert run.status == "ok", run.errors
    names = {d.url.rsplit("/", 1)[1]: d for d in run.documents}
    # Options 2024 and 2025 (a bucket N+1 can hold fiscal year N), not 2023.
    assert {"20241", "20242", "20251", "20252"} <= set(names)
    assert "20231" not in names
    assert names["20241"].extension == "pdf"  # from the link title of an opaque link
    assert names["20241"].fiscal_year == 2024  # from the option label
    assert names["20241"].fields["file_name"] == "annual-report-2024.pdf"
    assert {"q3-2025.pdf", "q1-2025.pdf"} <= set(names)  # two pages, then an empty one
    search_pages = [p for p in run.pages if p.source_id == "search"]
    assert [p.values["page"] for p in search_pages] == [0, 1, 2]


def test_request_source_validation() -> None:
    bad_header = {**FORM_SOURCE, "headers": {"Cookie": "x"}}
    with pytest.raises(pydantic.ValidationError, match="headers not allowed"):
        recipe_from_mapping(_recipe(a=bad_header))
    unknown = {**JSON_SOURCE, "json": {"page": "{page}", "q": "{nope}"}}
    with pytest.raises(pydantic.ValidationError, match="unknown placeholders"):
        recipe_from_mapping(_recipe(a=unknown))
    with pytest.raises(pydantic.ValidationError, match="schema_version 3"):
        recipe_from_mapping({**_recipe(a=JSON_SOURCE), "schema_version": 2})
    outside = {**JSON_SOURCE, "url": "https://elsewhere.example/api"}
    with pytest.raises(pydantic.ValidationError, match="outside allowed_domains"):
        recipe_from_mapping(_recipe(a=outside))
    source = recipe_from_mapping(_recipe(a=JSON_SOURCE)).sources[0]
    assert isinstance(source, RequestSource)
    _url, body, _headers = source.build({"page": 3}, TODAY)
    assert json.loads(body or b"") == {"query": "", "page": 3}  # a whole placeholder stays a number


def test_opaque_links_and_json_candidates() -> None:
    root = parse_html(
        '<a href="/preview/1" title="report.pdf">x</a><a href="/preview/2">y</a>'
        '<a href="/files/a.pdf">z</a><a href="/doc?id=7">w</a>'
    )
    urls = [c.url for c in extract_candidates(root, SITE, extensions=["pdf"])]
    assert urls == [f"{SITE}/preview/1", f"{SITE}/files/a.pdf"]
    by_pattern = extract_candidates(root, SITE, extensions=["pdf"], document_urls=[r"/doc\?id=\d+"])
    assert f"{SITE}/doc?id=7" in [c.url for c in by_pattern]
    data = {"r": [{"t": "A", "u": "/a.pdf"}, {"t": "B", "alt": "/b.pdf"}, {"t": "C"}]}
    found = json_candidates(data, SITE, items="r", url=["u", "alt"], title="t", extensions=["pdf"])
    assert [(c.title, c.url) for c in found] == [("A", f"{SITE}/a.pdf"), ("B", f"{SITE}/b.pdf")]


def test_fixed_period_rule() -> None:
    rules = ClassifyRules(
        period=[PeriodRule(pattern=r"30 juin (?P<year>20\d{2})", target="title", period="H1")]
    )
    cls = classify(rules, url=f"{SITE}/x", fields={"title": "Capital au 30 juin 2025"})
    assert cls.period == "2025-H1"


# -- endpoints and recorded discovery -----------------------------------------------------


def _exchange(ex_id: str, action: str, form: dict[str, str], year: int) -> Exchange:
    body = json.dumps({"more": False, "content": _fragment(year)}).encode()
    return Exchange(
        id=ex_id,
        action=action,
        method="POST",
        url=f"{SITE}/display",
        request_body=canonical_body("&".join(f"{k}={v}" for k, v in form.items()), "form"),
        request_content_type="application/x-www-form-urlencoded",
        requested_with="XMLHttpRequest",
        status=200,
        content_type="application/json",
        body=body,
    )


EXCHANGES = [
    _exchange(
        "X1",
        'select #year = "2025" (value 9025)',
        {"loc": "9025", "page": "1", "until": "21/11/2026"},
        2025,
    ),
    _exchange(
        "X2",
        'select #year = "2024" (value 9024)',
        {"loc": "9024", "page": "1", "until": "21/11/2026"},
        2024,
    ),
]


def test_endpoint_analysis_and_digest() -> None:
    found = analyse(EXCHANGES[0])
    assert found.kind == "json_html" and found.html_path == "content"
    assert len(found.documents) == 2  # opaque links recognised by their title
    (ep,) = endpoints(EXCHANGES)
    assert ep.varying_fields() == {"loc": ["9025", "9024"]}
    text = digest_endpoints([ep], TODAY)
    assert "until=21/11/2026 (today's date)" in text
    assert "X2 after select #year" in text
    form, json_body = templated_body(EXCHANGES[0], {"loc": "{option}"}, TODAY)
    assert json_body is None
    assert form == {"loc": "{option}", "page": "1", "until": "{today:%d/%m/%Y}"}
    with pytest.raises(ValueError, match="not in the body"):
        templated_body(EXCHANGES[0], {"missing": "{page}"}, TODAY)


def test_target_options() -> None:
    options = [("9025", "2025"), ("9024", "2024"), ("9023", "2023"), ("", "Choose")]
    assert _target_options(options, [2024]) == [("9025", "2025"), ("9024", "2024")]
    assert _target_options(options, None) == [("9025", "2025"), ("9024", "2024")]
    assert _target_options([("a", "All")], [2024]) == []


def test_recorded_discovery_round_trip(tmp_path: Path) -> None:
    recorder = SnapshotRecorder(tmp_path, "example")
    for ex in EXCHANGES:
        recorder.record(
            method=ex.method,
            url=ex.url,
            request_body=ex.request_body,
            status=ex.status,
            headers={"content-type": "application/json"},
            body=ex.body,
        )
    recorder.save()
    discovery = Discovery(
        f"{SITE}/publications", ["load", *(e.action for e in EXCHANGES)], EXCHANGES
    )
    save_discovery(tmp_path, discovery, [2024])
    replay = ReplayDiscovery(tmp_path)
    assert replay.available(f"{SITE}/publications")
    again = replay.discover(f"{SITE}/publications", [2024])
    assert [e.id for e in again.exchanges] == ["X1", "X2"]
    assert again.exchanges[1].body == EXCHANGES[1].body
    assert not ReplayDiscovery(tmp_path).available(f"{SITE}/other")


# -- learning a request source ------------------------------------------------------------


class FakeDiscoverer:
    def __init__(self) -> None:
        self.calls = 0

    def discover(self, url: str, years: Collection[int] | None) -> Discovery:
        self.calls += 1
        return Discovery(url, ["load", *(e.action for e in EXCHANGES)], list(EXCHANGES))


class FakeLLM:
    def __init__(self, **answers: Callable[[list[Message]], BaseModel]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, list[Message]]] = []

    def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        step: str,
        tier: str = "small",
        model: str | None = None,
    ) -> T:
        self.calls.append((step, list(messages)))
        answer = self.answers[step](list(messages))
        assert isinstance(answer, schema)
        return answer


def structure(messages: list[Message]) -> StructureReplyWithRequests:
    assert "Endpoints the page called" in messages[1]["content"]
    assert "<untrusted_endpoints>" in messages[1]["content"]
    return StructureReplyWithRequests(
        reasoning="Years load by POST.",
        action="recipe",
        open_links=[],
        document_groups=[],
        sources=[],
        default_language="en",
        request_sources=[
            RequestDraft(
                exchange="X2",
                vary=[FieldChange(field="loc", value="options")],
                options_selector="#year option",
                options_year="fiscal",
                year_from=None,
                page_start=None,
                has_more_path=None,
                html_path="content",
                items_path=None,
                url_paths=[],
                title_path=None,
                file_name_path=None,
                date_path=None,
                context_paths=[],
                container=None,
                item="li.doc",
                title="h3",
            )
        ],
    )


def rules(messages: list[Message]) -> RulesReply:
    return RulesReply(
        reasoning="Titles.",
        type_rules=[
            TypeRuleDraft(pattern="half-year", target="title", type="half_year_report"),
            TypeRuleDraft(pattern="annual", target="title", type="annual_report"),
        ],
        period_rules=[PeriodRuleDraft(pattern=r"(?P<year>20\d{2})", target="title")],
        language_rules=[],
        labels=[],
    )


def test_learn_a_request_source_then_replay_without_browser(tmp_path: Path) -> None:
    llm = FakeLLM(structure=structure, rules=rules)
    discoverer = FakeDiscoverer()
    r = Retrievr(
        settings=Settings(
            state=StateSettings(state_dir=tmp_path / "state"),
            browser=BrowserSettings(enabled=False),
        ),
        llm=llm,
        discoverer=discoverer,
        fetch_transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
        today=TODAY,
    )
    result = r.learn(f"{SITE}/publications", years=[2024, 2025])
    assert result.status == "learned", (result.errors, result.notes)
    assert discoverer.calls == 1  # no document link and a year selector: discovery ran
    assert result.recipe is not None and result.recipe.schema_version == 3
    source = result.recipe.sources[0]
    assert isinstance(source, RequestSource) and source.form is not None
    assert source.form["loc"] == "{option}"
    assert source.form["until"] == "{today:%d/%m/%Y}"  # today's date templated by the engine
    assert source.headers == {"X-Requested-With": "XMLHttpRequest"}
    files = sorted(d.fields["file_name"] or "" for d in result.all_documents)
    assert files == [
        "annual-report-2024.pdf",
        "annual-report-2025.pdf",
        "half-year-report-2024.pdf",
        "half-year-report-2025.pdf",
    ]
    calls = len(llm.calls)
    warm = r.run(result.site, years=[2024, 2025], download=False)
    assert warm.status == "ok", warm.errors
    assert len(llm.calls) == calls and discoverer.calls == 1  # no LLM, no browser
    assert len(warm.all_documents) == 4


def test_no_discovery_on_a_static_listing(tmp_path: Path) -> None:
    page = "".join(f'<li><a href="/f/{i}.pdf">Report {i}</a></li>' for i in range(8))

    def static(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text=f"<ul>{page}</ul>", headers={"content-type": "text/html"})

    discoverer = FakeDiscoverer()
    llm = FakeLLM(structure=lambda _m: (_ for _ in ()).throw(RuntimeError("stop")))
    r = Retrievr(
        settings=Settings(
            state=StateSettings(state_dir=tmp_path / "state"),
            browser=BrowserSettings(enabled=False),
        ),
        llm=llm,
        discoverer=discoverer,
        fetch_transport=httpx.MockTransport(static),
        sleep=lambda _s: None,
        today=TODAY,
    )
    with pytest.raises(RuntimeError, match="stop"):
        r.learn(f"{SITE}/reports")
    assert discoverer.calls == 0
    assert re.search(r"Document link groups \(1\)", llm.calls[0][1][1]["content"])
