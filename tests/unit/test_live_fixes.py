"""Fixes found by live tests on new sites: URL query fields of a listing API, opaque
document links, listing APIs of another host, a request that selects nothing."""

from __future__ import annotations

import json
from datetime import date

from retrievr.api import selection_warnings
from retrievr.browser import Exchange
from retrievr.candidates import (
    extract_candidates,
    group_document_links,
    opaque_patterns,
    parse_html,
)
from retrievr.endpoints import (
    digest_endpoints,
    endpoints,
    query_fields,
    templated_body,
    templated_url,
)
from retrievr.learn import Filter, LearnContext, Learner
from retrievr.recipe import FoundDocument

SITE = "https://ir.example-issuer.com"


def _get(url: str, payload: object) -> Exchange:
    return Exchange(
        id="X1",
        action="load",
        method="GET",
        url=url,
        request_body=None,
        request_content_type=None,
        requested_with=None,
        status=200,
        content_type="application/json",
        body=json.dumps(payload).encode(),
    )


def _listing(host: str, files_on: str) -> Exchange:
    items = [
        {"title": f"Report Q{q} 2026", "file": f"{files_on}/doc/q{q}-2026.pdf"} for q in (1, 2, 3)
    ]
    return _get(
        f"{host}/api/documents-listing/2026+2025?_format=json&f%5B0%5D=type:3&page=1",
        {"rows": items},
    )


def test_query_fields_are_named_and_templated() -> None:
    ex = _listing(SITE, SITE)
    assert query_fields(ex) == {"?_format": "json", "?f[0]": "type:3", "?page": "1"}
    url = templated_url(ex, {"?page": "{page}"})
    assert (
        url == f"{SITE}/api/documents-listing/2026+2025?_format=json&f%5B0%5D=type:3&page={{page}}"
    )
    assert templated_body(ex, {"?page": "{page}"}, date(2026, 11, 21)) == (None, None)
    digest = digest_endpoints(endpoints([ex]), date(2026, 11, 21))
    assert "URL query of X1 (fields named ?name): ?_format=json; ?f[0]=type:3; ?page=1" in digest


def test_a_listing_api_of_another_host_is_kept_only_for_site_documents() -> None:
    learner = Learner(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        LearnContext("example-issuer", f"{SITE}/results", None, date(2026, 11, 21)),
    )
    own_files = endpoints([_listing("https://tools.content-provider.example", SITE)])
    other_files = endpoints(
        [_listing("https://tools.content-provider.example", "https://cdn.example")]
    )
    assert learner._own_listing(own_files[0])
    assert not learner._own_listing(other_files[0])


def test_opaque_document_links_are_seen_and_get_patterns() -> None:
    html = """<table class="results"><tr><td>Q1</td>
      <td><a href="/ww/en/assets/pdf/release-q1-revenues-2026">Release</a></td>
      <td><a href="https://download.example-issuer.com/files?p_Doc_Ref=presentation-q1-2026">Slides</a></td>
      <td><a href="/media/report.pdf">Report</a></td></tr></table>
      <a href="/en/documents">All documents</a><a href="/en/investors/results">Results</a>"""
    root = parse_html(html)
    groups = group_document_links(root, f"{SITE}/results")
    urls = {c.url for g in groups for c in g.candidates}
    assert urls == {
        f"{SITE}/ww/en/assets/pdf/release-q1-revenues-2026",
        "https://download.example-issuer.com/files?p_Doc_Ref=presentation-q1-2026",
        f"{SITE}/media/report.pdf",
    }
    patterns = opaque_patterns([c for g in groups for c in g.candidates])
    assert patterns == [
        "^https://ir\\.example\\-issuer\\.com/ww/en/assets/pdf/",
        "^https://download\\.example\\-issuer\\.com/files\\?(?:.*&)?p_Doc_Ref=",
    ]
    found = extract_candidates(
        root, f"{SITE}/results", container="table.results", document_urls=patterns
    )
    assert len(found) == 3


def _doc(type_: str) -> FoundDocument:
    return FoundDocument(
        url=f"{SITE}/{type_}.pdf",
        normalized_url=f"{SITE}/{type_}.pdf",
        title=type_,
        source_id="s",
        params={},
        page_url=SITE,
        extension="pdf",
        date=None,
        type=type_,
        period="2025-Q1",
        fiscal_year=2025,
        language="en",
        score=1.0,
        justification="",
    )


def test_a_request_that_selects_nothing_says_what_the_site_lists() -> None:
    docs = [_doc("press_release"), _doc("results_presentation")]
    compiled = Filter(
        types=["annual_report"], exclude_types=[], year_from=None, year_to=None, languages=None
    )
    (warning,) = selection_warnings(compiled, docs, compiled.select(docs))
    assert "selected none of the 2 documents" in warning
    assert "press_release 1" in warning and "results_presentation 1" in warning
    assert selection_warnings(compiled, docs, docs[:1]) == []


def test_field_names_are_matched_leniently() -> None:
    from retrievr.learn import _field_name, _has_more

    ex = _listing(SITE, SITE)
    assert _field_name(ex, "page") == "?page"
    assert _field_name(ex, "?page") == "?page"
    assert _field_name(ex, "unknown") == "unknown"
    assert _has_more(ex, "rows") == "rows"
    assert _has_more(ex, "has_more") is None  # absent: pagination must not stop early


def test_quarters_written_as_words() -> None:
    import re

    from retrievr.classify import normalize_period

    pattern = re.compile(r"(?P<q>\w+) (?:quarter|trimestre) (?P<year>20\d{2})", re.IGNORECASE)
    for text, period in [
        ("Interim report fourth quarter 2025", "2025-Q4"),
        ("Rapport du troisième trimestre 2025", "2025-Q3"),
        ("first quarter 2026", "2026-Q1"),
    ]:
        m = pattern.search(text)
        assert m is not None and normalize_period(m)[0] == period


def test_request_drafts_of_one_listing_are_merged() -> None:
    from retrievr.learn import FieldChange, RequestDraft

    learner = Learner(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        LearnContext("example-issuer", f"{SITE}/results", None, date(2026, 11, 21)),
    )
    learner.endpoints = endpoints([_listing(SITE, SITE)])
    base = {
        "exchange": "X1",
        "vary": [FieldChange(field="?page", value="page")],
        "options_selector": None,
        "options_year": None,
        "year_from": None,
        "page_start": 1,
        "has_more_path": None,
        "html_path": None,
        "items_path": "rows",
        "title_path": "title",
        "file_name_path": None,
        "date_path": None,
        "context_paths": [],
        "container": None,
        "item": None,
        "title": None,
    }
    drafts = [RequestDraft(**base, url_paths=["file"]), RequestDraft(**base, url_paths=["uri"])]
    (merged,) = learner._merge_drafts(drafts)
    assert merged.url_paths == ["file", "uri"]


def test_rules_giving_a_period_not_ended_yet_are_flagged() -> None:
    from retrievr.classify import ClassifyRules, DocumentLabel, PeriodRule
    from retrievr.learn import evaluate_rules

    doc = _doc("quarterly_report")
    doc = FoundDocument(
        **{**doc.__dict__, "url": f"{SITE}/files/2026-01/report-fourth-quarter-2025.pdf"}
    )
    rules = ClassifyRules(
        period=[PeriodRule(pattern=r"fourth.*?files/(?P<year>20\d{2})", target="any")]
    )
    rules = ClassifyRules(
        period=[
            PeriodRule(pattern=r"files/(?P<year>20\d{2})-01/report-(?P<q>fourth)", target="url")
        ]
    )
    label = DocumentLabel(id="D1", type="quarterly_report", period="2025-Q4", language=None)
    ev = evaluate_rules(rules, [label], {"D1": doc}, "en", date(2026, 11, 21))
    assert ev.future and "2026-Q4" in ev.future[0]
    assert not evaluate_rules(rules, [label], {"D1": doc}, "en", date(2027, 2, 1)).future


def test_a_short_name_matches_one_json_path() -> None:
    from retrievr.learn import _field_name

    body = {"queries": [{"params": {"page": 0, "hitsPerPage": 20}}]}
    ex = Exchange(
        id="X2",
        action="load",
        method="POST",
        url=f"{SITE}/api/search",
        request_body=json.dumps(body),
        request_content_type="application/json",
        requested_with=None,
        status=200,
        content_type="application/json",
        body=b"{}",
    )
    assert _field_name(ex, "page") == "queries.0.params.page"
    assert _field_name(ex, "params") == "params"  # not the end of a field: unchanged
