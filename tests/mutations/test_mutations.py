"""Milestone 5 acceptance: health checks and repair on broken copies of real sites.

Each case serves the snapshot of a site through a mutation (``harness.py``), after a
healthy run of its hand-written recipe has recorded the site's history. Then:

- **zero false ok**: when the old recipe's output differs from the healthy one (documents
  or metadata), the health verdict is never ``ok``; when it does not differ, it is ``ok``;
- **no invalid recipe activated**: a repaired recipe, replayed on the broken site, finds
  every anchor and at least 95 % of the healthy documents with the same type and period,
  inside its allowed domains; when no repair passes, the old recipe stays active;
- **repair rate**: at least 80 % of the broken cases end repaired.

LLM answers (gpt-6-luna) are replayed from ``tests/fixtures/llm_cache/``; to record them
again (a few cents):

    RETRIEVR_LLM_RECORD=1 uv run --env-file .env pytest tests/mutations
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import respx

from retrievr.api import Retrievr, RunOutcome
from retrievr.classify import apply_decisions
from retrievr.config import BrowserSettings, FetchSettings, LLMSettings, Settings, StateSettings
from retrievr.events import Event
from retrievr.fetch import Fetcher
from retrievr.recipe import Recipe, load_recipe, run_recipe
from retrievr.snapshots import ReplayTransport, load_index
from retrievr.urls import is_allowed, normalize_url
from tests.conftest import FIXTURES, RECIPES
from tests.mutations.harness import ApiDiscovery, MutatedSite, Mutation, mutations_for

CACHE = FIXTURES / "llm_cache"
MODEL = "openai/gpt-6-luna"
RECORD = os.environ.get("RETRIEVR_LLM_RECORD") == "1"
YEARS = [2024, 2025]
SITES = {
    "societegenerale": "societegenerale-ir@1.yaml",
    "deutschebank": "deutschebank-ir@1.yaml",
    "ubs": "ubs-ir@1.yaml",
    "creditagricole": "creditagricole-ir@1.yaml",
}
MIN_REPAIR_RATE = 0.8


@dataclass
class Case:
    site: str
    mutation: Mutation
    healthy: dict[str, tuple[str | None, str | None]]
    """Healthy documents of the years checked: normalized URL -> (type, period)."""
    anchors: list[str]
    changed: bool
    """Whether the old recipe's output on the mutated site differs from the healthy one."""
    outcome: RunOutcome
    active: Recipe | None
    active_docs: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.site}-{self.mutation.name}"


def _settings(state: Path) -> Settings:
    return Settings(
        llm=LLMSettings(
            model_small=MODEL,
            model_large=MODEL,  # one model only: the large rung is skipped
            cache_dir=CACHE,
            cache="use" if RECORD else "replay",
        ),
        state=StateSettings(state_dir=state),
        browser=BrowserSettings(enabled=False),
    )


def _signature(docs: list) -> dict[str, tuple[str | None, str | None]]:  # type: ignore[type-arg]
    return {
        d.normalized_url: (d.type, d.period)
        for d in docs
        if d.fiscal_year in YEARS or d.params.get("year") in YEARS
    }


def _run_case(site: str, recipe_file: str, mutation: Mutation, state: Path) -> Case:
    snapshot = FIXTURES / site
    captured = load_index(snapshot).captured_on()
    assert captured is not None
    today = captured.date()
    replay = ReplayTransport(snapshot, lenient_bodies=True)
    healthy_r = Retrievr(
        settings=_settings(state), fetch_transport=replay, sleep=lambda _s: None, today=today
    )
    recipe = healthy_r.install(RECIPES / recipe_file, site=site)
    healthy = healthy_r.run(site, years=YEARS, download=False)
    assert healthy.status == "ok" and healthy.health is not None
    assert healthy.health.verdict == "ok", healthy.health.summary()

    mutated = MutatedSite(replay, recipe, mutation, today)
    fetcher = Fetcher(FetchSettings(), transport=mutated.transport(), sleep=lambda _s: None)
    with fetcher:
        old_run = run_recipe(recipe, fetcher, years=YEARS, today=today)
        decisions = healthy_r.store.decisions(recipe.recipe)
        old_docs = _signature(apply_decisions(old_run.documents, decisions))
    healthy_docs = _signature(healthy.all_documents)
    events: list[Event] = []
    broken_r = Retrievr(
        settings=_settings(state),
        fetch_transport=mutated.transport(),
        discoverer=ApiDiscovery(mutated),
        on_event=events.append,
        sleep=lambda _s: None,
        today=today,
    )
    outcome = broken_r.run(site, years=YEARS, download=False)
    active = broken_r.store.active_recipe(site)
    active_docs: dict[str, tuple[str | None, str | None]] = {}
    if active is not None:
        with Fetcher(FetchSettings(), transport=mutated.transport(), sleep=lambda _s: None) as f:
            run = run_recipe(active, f, years=YEARS, today=today)
            decided = apply_decisions(run.documents, broken_r.store.decisions(active.recipe))
            assert all(is_allowed(d.url, active.allowed_domains) for d in decided)
            active_docs = _signature(decided)
    return Case(
        site=site,
        mutation=mutation,
        healthy=healthy_docs,
        anchors=[normalize_url(a) for a in recipe.health.anchors],
        changed=old_docs != healthy_docs,
        outcome=outcome,
        active=active,
        active_docs=active_docs,
        events=events,
    )


def _cases() -> list[tuple[str, str, Mutation]]:
    out = []
    for site, recipe_file in SITES.items():
        recipe = load_recipe(RECIPES / recipe_file)
        out += [(site, recipe_file, m) for m in mutations_for(recipe)]
    return out


@pytest.fixture(scope="module")
def cases(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Case]]:
    out: dict[str, Case] = {}
    with respx.mock(assert_all_called=False) if not RECORD else nullcontext():
        for site, recipe_file, mutation in _cases():
            state = tmp_path_factory.mktemp(f"{site}-{mutation.name}")
            case = _run_case(site, recipe_file, mutation, state)
            out[case.id] = case
    yield out


CASE_IDS = [f"{site}-{m.name}" for site, _f, m in _cases()]


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_health_verdict_is_never_a_false_ok(cases: dict[str, Case], case_id: str) -> None:
    case = cases[case_id]
    health = case.outcome.health
    assert health is not None
    if case.changed:
        assert health.verdict != "ok", health.summary()
    else:
        assert health.verdict == "ok", health.summary()


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_no_invalid_recipe_is_activated(cases: dict[str, Case], case_id: str) -> None:
    case = cases[case_id]
    repair = case.outcome.repair
    assert case.active is not None
    if repair is None or repair.status != "repaired":
        assert case.active.version == 1, "a recipe was activated without passing validation"
        return
    assert case.active.provenance is not None and case.active.provenance.method == "repaired"
    found = case.active_docs
    missing_anchors = [a for a in case.anchors if a in case.healthy and a not in found]
    assert not missing_anchors
    kept = [u for u in case.healthy if u in found]
    assert len(kept) >= 0.95 * len(case.healthy), sorted(set(case.healthy) - set(found))[:5]
    same = [u for u in kept if found[u] == case.healthy[u]]
    assert len(same) >= 0.9 * len(kept)
    assert any(e.kind == "recipe_repaired" for e in case.events)


def test_repair_rate(cases: dict[str, Case]) -> None:
    broken = [c for c in cases.values() if c.outcome.health and c.outcome.health.repair_needed]
    assert broken, "no mutation broke a recipe"
    repairable = [c for c in broken if c.mutation.repairable]
    repaired = [c for c in repairable if c.outcome.repair and c.outcome.repair.status == "repaired"]
    report = {c.id: c.outcome.repair.reason if c.outcome.repair else "-" for c in broken}
    assert len(repaired) >= MIN_REPAIR_RATE * len(repairable), report


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_unrepairable_breakage_ends_degraded(cases: dict[str, Case], case_id: str) -> None:
    case = cases[case_id]
    if case.mutation.repairable:
        return
    assert case.outcome.health is not None and case.outcome.health.repair_needed
    assert case.outcome.status == "degraded"
    assert case.outcome.downloads == []
    assert any(e.kind == "site_degraded" for e in case.events)
