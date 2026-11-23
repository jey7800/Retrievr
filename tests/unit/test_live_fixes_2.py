"""Fixes found by cold tests on twelve new sites (banks, CAC 40, Nasdaq): requests mixing
kinds of documents and results, impossible fiscal years, health bounds, error and
challenge pages."""

from __future__ import annotations

import dataclasses
from datetime import date

from retrievr.fetch import detect_block
from retrievr.health import HealthReport, KnownDocument, _items_per_year
from retrievr.learn import Filter, VisitedPage, health_for, is_error_page, with_results_types
from retrievr.recipe import FoundDocument, HealthSpec, _published_year, recipe_from_mapping

SITE = "https://ir.example-issuer.com"
TODAY = date(2026, 11, 23)


def _doc(n: int, fiscal_year: int | None, *, type_: str = "annual_report") -> FoundDocument:
    return FoundDocument(
        url=f"{SITE}/{n}.pdf",
        normalized_url=f"{SITE}/{n}.pdf",
        title=str(n),
        source_id="s",
        params={},
        page_url=SITE,
        extension="pdf",
        date=None,
        type=type_,
        period=f"{fiscal_year}-FY" if fiscal_year else None,
        fiscal_year=fiscal_year,
        language="en",
        score=1.0,
        justification="",
    )


def test_kinds_and_results_keep_the_documents_of_a_results_release() -> None:
    named = Filter(types=["annual_report", "half_year_report"])
    request = "annual reports, half-year reports and quarterly results"
    compiled = with_results_types(request, named)
    assert compiled.types is not None
    assert {"annual_report", "half_year_report", "results_presentation"} <= set(compiled.types)
    assert "pillar3" not in compiled.types
    # Untouched: no results asked for, every type kept, or a results kind already named.
    assert with_results_types("annual reports", named) == named
    assert with_results_types("quarterly results", Filter(types=None)).types is None
    presentations = Filter(types=["results_presentation"])
    assert with_results_types("results presentations", presentations) == presentations
    # An excluded kind stays out.
    no_pr = Filter(types=["annual_report"], exclude_types=["press_release"])
    kept = with_results_types("annual reports and résultats, no press releases", no_pr)
    assert kept.types is not None and "press_release" not in kept.types


def test_a_fiscal_year_that_has_not_started_is_not_kept() -> None:
    bond = dataclasses.replace(_doc(1, 2030), date=date(2025, 3, 1))
    assert _published_year(bond, TODAY).fiscal_year == 2025  # from the publication date
    assert _published_year(bond, TODAY).period is None
    undated = _doc(2, 2031)
    assert _published_year(undated, TODAY).fiscal_year is None
    assert _published_year(_doc(3, 2026), TODAY).fiscal_year == 2026


def test_health_bounds_come_from_past_years() -> None:
    docs = [_doc(i, 2023) for i in range(30)] + [_doc(100 + i, 2024) for i in range(40)]
    docs += [_doc(200 + i, 2025) for i in range(38)] + [_doc(300, 2026)]
    spec = health_for(docs)
    assert spec.items_per_year == (15, 80)  # 2026 (in progress) is left out


def _recipe(bounds: tuple[int, int]) -> object:
    data = {
        "schema_version": 1,
        "recipe": "example-ir",
        "version": 1,
        "issuer": "example-issuer",
        "site": "ir.example-issuer.com",
        "allowed_domains": ["ir.example-issuer.com"],
        "sources": [{"id": "s", "url": f"{SITE}/reports"}],
    }
    recipe = recipe_from_mapping(data)
    return recipe.model_copy(update={"health": HealthSpec(items_per_year=bounds)})


def _known(doc: FoundDocument) -> KnownDocument:
    return KnownDocument.of(doc)


def test_a_year_the_site_never_listed_is_not_a_regression() -> None:
    recipe = _recipe((1, 24))
    docs = [_doc(i, 2026) for i in range(12)]
    known = [_known(d) for d in docs]
    report = HealthReport(site="s", recipe="example-ir@1", baseline=False)
    _items_per_year(report, recipe, docs, {2024, 2025, 2026}, TODAY, known)  # type: ignore[arg-type]
    assert report.checks[-1].verdict == "ok"
    # A year listed before that comes back empty is suspect.
    known_2024 = [*known, _known(_doc(99, 2024))]
    report = HealthReport(site="s", recipe="example-ir@1", baseline=False)
    _items_per_year(report, recipe, docs, {2024, 2025, 2026}, TODAY, known_2024)  # type: ignore[arg-type]
    assert report.checks[-1].verdict == "suspect"


def test_an_error_page_served_with_200_is_recognised() -> None:
    error = VisitedPage("P1", SITE, SITE, "Example Group - Error", groups=[], nav=[])
    assert is_error_page(error)
    normal = VisitedPage("P1", SITE, SITE, "Financial results", groups=[], nav=[])
    assert not is_error_page(normal)


def test_a_bot_protection_challenge_page_is_a_block() -> None:
    body = (
        "<title>Loading site please wait...</title> This website is using a security "
        "service to protect itself from online attacks. " + "x" * 30_000
    )
    assert detect_block(200, body) == "challenge page (bot protection)"
    other_wording = (
        "<h1>Please Enable JavaScript</h1> <p><strong>www.example-issuer.com</strong> is using "
        "a security service for protection against online attacks.<br />" + "x" * 30_000
    )
    assert detect_block(200, other_wording) == "challenge page (bot protection)"


def test_a_static_page_without_the_requested_years_is_opened_in_a_browser() -> None:
    from retrievr.candidates import Candidate, LinkGroup
    from retrievr.config import FetchSettings
    from retrievr.fetch import Fetcher
    from retrievr.learn import LearnContext, Learner

    def cand(n: int, year: int) -> Candidate:
        url = f"{SITE}/media/{year}/report-{n}-{year}.pdf"
        return Candidate(url, url, f"Report {n} {year}", f"Report {n} {year}", "", None, "ul>li>a")

    current = LinkGroup("ul>li>a", "ul>li>a", [cand(i, 2026) for i in range(12)])
    context = LearnContext(issuer="example-issuer", start_url=SITE, years=[2024, 2025], today=TODAY)
    learner = Learner(Fetcher(FetchSettings()), None, context)  # type: ignore[arg-type]
    page = VisitedPage("P1", SITE, SITE, "Reports", groups=[current], nav=[])
    assert learner._few_requested_years(page)  # the page shows 2026 only
    covered = LinkGroup("ul>li>a", "ul>li>a", [cand(i, 2024 + i % 2) for i in range(12)])
    page = VisitedPage("P1", SITE, SITE, "Reports", groups=[covered], nav=[])
    assert not learner._few_requested_years(page)


def test_clearly_named_kinds_are_never_dropped() -> None:
    from retrievr.learn import with_named_types

    def types(request: str, *llm_types: str) -> set[str]:
        compiled = with_named_types(request, Filter(types=list(llm_types)))
        return set(compiled.types or [])

    assert types(
        "Communiqués de résultats et tableaux financiers Excel de 2024", "press_release"
    ) == {
        "press_release",
        "financial_data_supplement",
    }
    assert "fixed_income_call_transcript" in types(
        "Fixed income presentations and call transcripts since 2024",
        "fixed_income_presentation",
        "analyst_call_transcript",
    )
    assert types("Pillar 3 reports for 2025", "other") == {"other", "pillar3"}
    # A negated phrase adds nothing; a filter keeping every type is left alone.
    assert types("annual reports, no press releases", "annual_report") == {"annual_report"}
    assert with_named_types("press releases", Filter(types=None)).types is None


def test_web_component_links_and_announced_formats_are_documents() -> None:
    from retrievr.candidates import group_document_links, parse_html, text_format

    page = parse_html(
        "<html><body><main>"
        '<div class="report"><h4>Interim report Q2 2026</h4>'
        '<gds-button-pw href="https://webapp.example-issuer.com/lib.nsf/dld/E37F?opendocument">'
        "<span>Interim Report Q2 2026 (pdf)</span></gds-button-pw></div>"
        '<div class="report"><a href="/contact">Contact us</a></div>'
        "</main></body></html>"
    )
    [group] = group_document_links(page, SITE)
    [cand] = group.candidates
    assert cand.url.endswith("dld/E37F?opendocument")
    assert cand.extension == "pdf"
    assert text_format("Fact Book figures Q3 2026 (excel)") == "xlsx"
    assert text_format("Download pdf 289 KB") == "pdf"
    assert text_format("PDF") == "pdf"
    assert text_format("Annual report 2025") == ""


def test_a_download_service_known_from_link_texts_gives_a_pattern() -> None:
    from retrievr.candidates import Candidate, opaque_patterns

    url = "https://webapp.example-issuer.com/lib/docs.nsf/dld/E37F?opendocument"
    announced = Candidate(url, url, "Interim Report (pdf)", "x", "", None, "a", format_hint="pdf")
    [pattern] = opaque_patterns([announced])
    import re

    assert re.search(
        pattern, "https://webapp.example-issuer.com/lib/docs.nsf/dld/A1B2?opendocument"
    )
    root = "https://ir.example-issuer.com/x?id=1"
    assert (
        opaque_patterns([Candidate(root, root, "(pdf)", "x", "", None, "a", format_hint="pdf")])
        == []
    )


def test_documents_robots_txt_disallows_are_not_downloadable() -> None:
    from retrievr.validate import Reachability

    assert not Reachability(sampled=3, reachable=0, gone=0, refused=3).ok
    assert Reachability(sampled=3, reachable=1, gone=0, refused=2).ok


def test_blank_class_and_id_attributes_do_not_break_the_dom_path() -> None:
    from retrievr.candidates import group_document_links, parse_html

    page = parse_html(
        '<html><body><div class=" " id=" "><ul class=""><li><a href="/r/annual-2025.pdf">'
        "Annual report 2025</a></li></ul></div></body></html>"
    )
    [group] = group_document_links(page, SITE)
    assert group.candidates[0].dom_path == "html>body>div>ul>li>a"
