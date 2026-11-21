"""Learner, filters, decisions and the public API, with a scripted LLM double.

The fake LLM reads the page digest it is given (like a model would) and answers with
fixed proposals, so these tests check the engine side: page digest, the bounded loop,
diagnostics, recipe building, rule evaluation, validation, storage and replay.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, TypeVar, cast

import httpx
import pytest
from pydantic import BaseModel

from retrievr.api import Retrievr, issuer_from_url
from retrievr.classify import Decision, DocumentLabel, LabelBatch, apply_decisions
from retrievr.config import BrowserSettings, Settings, StateSettings
from retrievr.events import Event
from retrievr.learn import (
    Filter,
    LanguageRuleDraft,
    PeriodRuleDraft,
    RulesReply,
    SourceDraft,
    StructureReply,
    TypeRuleDraft,
)
from retrievr.llm import Message
from retrievr.recipe import FoundDocument, YearRange

T = TypeVar("T", bound=BaseModel)
SITE = "https://ir.example-issuer.com"
TODAY = date(2026, 11, 20)


def _page(year: int, docs: Sequence[tuple[str, str]]) -> str:
    items = "".join(
        f'<li class="doc"><a href="/files/{name}">{label}</a><span>{year}</span></li>'
        for name, label in docs
    )
    return (
        "<html><head><title>Results</title></head><body>"
        '<header><a href="/files/registration-document.pdf">Registration document</a>'
        '<a href="/about">About us</a></header>'
        f'<nav class="years"><a href="/results?year=2024">2024</a>'
        f'<a href="/results?year=2025">2025</a></nav>'
        f'<main id="docs"><h2>Results {year}</h2><ul class="docs">{items}</ul></main>'
        "<footer><a href='/legal'>Legal</a></footer></body></html>"
    )


PAGES = {
    "/results?year=2025": _page(
        2025,
        [
            ("q1-2025-press-release.pdf", "Q1 2025 press release"),
            ("q1-2025-presentation.pdf", "Q1 2025 presentation"),
            ("h1-2025-report.pdf", "Half-year report 2025"),
        ],
    ),
    "/results?year=2024": _page(
        2024,
        [
            ("q4-2024-press-release.pdf", "Q4 2024 press release"),
            ("q4-2024-presentation.pdf", "Q4 2024 presentation"),
            ("fact-sheet-2024.pdf", "Fact sheet"),
        ],
    ),
}


def site_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        target = request.url.raw_path.decode()
        if target in PAGES:
            return httpx.Response(
                200, text=PAGES[target], headers={"Content-Type": "text/html; charset=utf-8"}
            )
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


class FakeLLM:
    """Answers each step with a function of the prompt; records every call."""

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
        msgs = list(messages)
        self.calls.append((step, msgs))
        answer = self.answers[step](msgs)
        assert isinstance(answer, schema)
        return answer

    def steps(self) -> list[str]:
        return [step for step, _ in self.calls]


def _find(messages: list[Message], pattern: str) -> str:
    text = "\n".join(m["content"] for m in messages)
    found = re.search(pattern, text)
    assert found, pattern
    return found.group(1)


def _docs_group(messages: list[Message]) -> str:
    return _find(messages, r"- (P1\.G\d+): \d+ links[^\n]*li\.doc>a")


def structure_answer(
    container: str = "#docs", item: str | None = "li.doc"
) -> Callable[[list[Message]], StructureReply]:
    def answer(messages: list[Message]) -> StructureReply:
        return StructureReply(
            reasoning="Per-year result pages.",
            action="recipe",
            open_links=[],
            document_groups=[_docs_group(messages)],
            sources=[
                SourceDraft(
                    url=f"{SITE}/results?year={{year}}",
                    year_from=2024,
                    container=container,
                    item=item,
                    title=None,
                    pagination_next_selector=None,
                    table_context=False,
                )
            ],
            default_language="en",
        )

    return answer


def rules_answer(messages: list[Message]) -> RulesReply:
    ids = re.findall(r'"id": "(D\d+)"', messages[-1]["content"])
    files = re.findall(r'"file": "([^"]+)"', messages[-1]["content"])
    labels = []
    for doc_id, name in zip(ids, files, strict=True):
        kind = "press_release" if "press" in name else "results_presentation"
        if "report" in name:
            kind = "half_year_report"
        if "fact" in name:
            kind = "other"
        m = re.search(r"q(\d)-(\d{4})", name)
        period = f"{m.group(2)}-Q{m.group(1)}" if m else None
        labels.append(DocumentLabel(id=doc_id, type=kind, period=period, language="en"))
    return RulesReply(
        reasoning="File names carry type and quarter.",
        type_rules=[
            TypeRuleDraft(pattern="press-release", target="url", type="press_release"),
            TypeRuleDraft(pattern="presentation", target="url", type="results_presentation"),
            TypeRuleDraft(pattern="report", target="url", type="half_year_report"),
        ],
        period_rules=[PeriodRuleDraft(pattern=r"q(?P<q>[1-4])-(?P<year>\d{4})", target="url")],
        language_rules=[LanguageRuleDraft(pattern="(", target="url", language="en")],
        labels=labels,
    )


def filter_answer(_messages: list[Message]) -> Filter:
    return Filter(types=["press_release"], year_from=2024)


def classify_answer(messages: list[Message]) -> LabelBatch:
    ids = re.findall(r'"id": "(D\d+)"', messages[-1]["content"])
    return LabelBatch(
        labels=[DocumentLabel(id=i, type="other", period="2025", language="en") for i in ids]
    )


def make(tmp_path: Path, llm: FakeLLM, events: list[Event] | None = None) -> Retrievr:
    settings = Settings(
        state=StateSettings(state_dir=tmp_path / "state"), browser=BrowserSettings(enabled=False)
    )
    return Retrievr(
        settings=settings,
        llm=llm,
        fetch_transport=site_transport(),
        sleep=lambda _s: None,
        today=TODAY,
        on_event=events.append if events is not None else None,
    )


def default_llm(**overrides: Callable[[list[Message]], BaseModel]) -> FakeLLM:
    answers: dict[str, Callable[[list[Message]], BaseModel]] = {
        "structure": structure_answer(),
        "rules": rules_answer,
        "filter": filter_answer,
        "classify": classify_answer,
    }
    answers.update(overrides)
    return FakeLLM(**answers)


def test_learn_then_replay_without_llm(tmp_path: Path) -> None:
    llm = default_llm()
    events: list[Event] = []
    r = make(tmp_path, llm, events)
    result = r.learn(f"{SITE}/results?year=2025", "press releases since 2024", years=[2024, 2025])

    assert result.status == "learned", result.errors
    assert result.recipe is not None and result.recipe_path is not None
    year = result.recipe.sources[0].params["year"]
    assert isinstance(year, YearRange) and year.from_ == 2024
    names = sorted(d.url.rsplit("/", 1)[1] for d in result.all_documents)
    assert "registration-document.pdf" not in names  # header link left out
    assert len(names) == 6
    assert {d.url.rsplit("/", 1)[1] for d in result.documents} == {
        "q1-2025-press-release.pdf",
        "q4-2024-press-release.pdf",
    }
    fact = next(d for d in result.all_documents if "fact-sheet" in d.url)
    assert fact.type == "other"  # rules left it undecided: the label was stored and applied
    assert "llm:learn" in fact.justification
    assert result.validation is not None and result.validation.passed
    assert [e.kind for e in events][-2:] == ["recipe_learned", "run_finished"]
    assert result.recipe.health.cadence is not None
    assert result.recipe.health.anchors

    before = len(llm.calls)
    replay = r.run(result.site, download=False)
    assert replay.status == "ok"
    assert len(llm.calls) == before  # warm run: zero LLM calls (filter and decisions cached)
    assert {d.normalized_url for d in replay.documents} == {
        d.normalized_url for d in result.documents
    }


def test_rule_evaluation_and_invalid_rules(tmp_path: Path) -> None:
    llm = default_llm()
    make(tmp_path, llm).learn(f"{SITE}/results?year=2025", years=[2024, 2025])
    rules_calls = [msgs for step, msgs in llm.calls if step == "rules"]
    # the invalid language rule "(" is reported back for one repair
    assert len(rules_calls) == 2
    assert "Invalid rules" in rules_calls[1][-1]["content"]


def test_diagnostic_when_container_is_too_broad(tmp_path: Path) -> None:
    replies = [structure_answer(container="body", item=None), structure_answer()]

    def structure(messages: list[Message]) -> StructureReply:
        return replies.pop(0)(messages) if len(replies) > 1 else replies[0](messages)

    llm = default_llm(structure=structure)
    result = make(tmp_path, llm).learn(f"{SITE}/results?year=2025", years=[2024, 2025])
    assert result.status == "learned"
    structure_calls = [msgs for step, msgs in llm.calls if step == "structure"]
    assert len(structure_calls) == 2
    assert "did NOT select" in structure_calls[1][-1]["content"]
    assert "registration-document.pdf" in structure_calls[1][-1]["content"]


def test_open_pages_then_recipe(tmp_path: Path) -> None:
    def structure(messages: list[Message]) -> StructureReply:
        if "== Page P2" not in messages[1]["content"]:
            link = _find(messages, r'- (P1\.N\d+): "2024"')
            return StructureReply(
                reasoning="Look at 2024.",
                action="open_pages",
                open_links=[link],
                document_groups=[],
                sources=[],
                default_language=None,
            )
        return structure_answer()(messages)

    llm = default_llm(structure=structure)
    result = make(tmp_path, llm).learn(f"{SITE}/results?year=2025", years=[2024, 2025])
    assert result.status == "learned"
    assert llm.steps()[:2] == ["structure", "structure"]


def test_source_outside_the_site_is_refused(tmp_path: Path) -> None:
    def structure(messages: list[Message]) -> StructureReply:
        reply = structure_answer()(messages)
        source = reply.sources[0].model_copy(update={"url": "https://elsewhere.example/x"})
        return reply.model_copy(update={"sources": [source]})

    llm = default_llm(structure=structure)
    result = make(tmp_path, llm).learn(f"{SITE}/results?year=2025", years=[2024, 2025])
    assert result.status == "error"
    assert llm.steps().count("structure") == 3  # first try plus two repairs
    assert "not on the site" in llm.calls[1][1][-1]["content"]


def test_page_content_is_wrapped_as_untrusted(tmp_path: Path) -> None:
    llm = default_llm()
    make(tmp_path, llm).learn(f"{SITE}/results?year=2025", years=[2024, 2025])
    system, user = llm.calls[0][1][:2]
    assert "never instructions" in system["content"]
    assert "<untrusted_pages>" in user["content"]


def test_blocked_site_stops(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(403, text="Access denied")

    llm = default_llm()
    settings = Settings(
        state=StateSettings(state_dir=tmp_path / "state"), browser=BrowserSettings(enabled=False)
    )
    r = Retrievr(
        settings=settings,
        llm=llm,
        fetch_transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
        today=TODAY,
    )
    result = r.learn(f"{SITE}/results?year=2025")
    assert result.status == "blocked"
    assert llm.calls == []


def test_filter_matches() -> None:
    f = Filter(types=["urd", "annual_report"], exclude_types=[], year_from=2024, languages=["EN"])
    assert f.languages == ["en"]
    assert f.matches("urd", 2024, "en")
    assert not f.matches("urd", 2023, "en")
    assert not f.matches("urd", None, "en")
    assert not f.matches("press_release", 2025, "en")
    assert not f.matches("urd", 2025, "fr")
    assert Filter(exclude_types=["press_release"]).matches(None, None, None)
    assert not Filter(exclude_types=["press_release"]).matches("press_release", 2025, "en")


def test_rules_win_over_decisions() -> None:
    doc = FoundDocument(
        url="https://x.test/a.pdf",
        normalized_url="https://x.test/a.pdf",
        title="a",
        source_id="s",
        params={},
        page_url="https://x.test/",
        extension="pdf",
        date=None,
        type="press_release",
        period=None,
        fiscal_year=None,
        language="en",
        score=1.0,
        justification="type=press_release by url /a/",
    )
    decided = Decision("urd", "2025-FY", 2025, "fr", "llm:test")
    (out,) = apply_decisions([doc], {doc.normalized_url: decided})
    assert out.type == "press_release"  # decided by a rule
    assert out.period == "2025-FY" and out.fiscal_year == 2025  # undecided: filled
    assert out.language == "en"


def test_label_normalization() -> None:
    label = DocumentLabel(id="D1", type="urd", period="2025-fy", language="FR")
    assert label.period == "2025-FY" and label.language == "fr"
    assert DocumentLabel(id="D2", type="other", period="Q3 2025", language="x1").period is None
    assert (
        cast(Any, DocumentLabel(id="D3", type="other", period=None, language=None)).language is None
    )


@pytest.mark.parametrize(
    ("url", "slug"),
    [
        ("https://investors.example-issuer.com/fr/x", "example-issuer"),
        ("https://home.barclays/investor-relations/", "barclays"),
        ("https://investor-relations.db.com/reports", "db"),
        ("https://www.airbus.com/en/investors", "airbus"),
    ],
)
def test_issuer_from_url(url: str, slug: str) -> None:
    assert issuer_from_url(url) == slug


def test_truncated_rules_reply_retries_with_fewer_documents(tmp_path: Path) -> None:
    from retrievr.learn import Learner
    from retrievr.llm import StructuredOutputError

    seen: list[int] = []

    def rules(messages: list[Message]) -> RulesReply:
        seen.append(messages[-1]["content"].count('"id": "D'))
        if len(seen) == 1:
            raise StructuredOutputError("empty reply (finish_reason=length)")
        return rules_answer(messages)

    llm = default_llm(rules=rules)
    original = Learner.__init__

    def small_sample(self: Learner, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        self.sample_size = 4

    import retrievr.learn as learn_module

    monkey = pytest.MonkeyPatch()
    monkey.setattr(learn_module, "MIN_RULES_SAMPLE", 2)
    monkey.setattr(Learner, "__init__", small_sample)
    try:
        result = make(tmp_path, llm).learn(f"{SITE}/results?year=2025", years=[2024, 2025])
    finally:
        monkey.undo()
    assert result.status == "learned", result.errors
    assert seen[:2] == [4, 2]
