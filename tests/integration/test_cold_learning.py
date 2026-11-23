"""Cold learning replayed offline: the three milestone-1 sites (static pages) and
Crédit Agricole (listing loaded by JavaScript through POST requests).

Pages come from the snapshots in ``tests/fixtures/<site>/``, browser discoveries from
their ``discovery/`` records (no browser runs), LLM answers from the response cache in
``tests/fixtures/llm_cache/`` (a missing entry fails the test). To
record the cache again (calls the LLM over the network, a few cents):

    RETRIEVR_LLM_RECORD=1 uv run --env-file .env pytest tests/integration/test_cold_learning.py
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import respx
from bench.metrics import score_documents
from bench.model import BENCH, Site, load_ground_truth, load_sites

from retrievr.api import Retrievr
from retrievr.browser import ReplayDiscovery
from retrievr.config import BrowserSettings, LLMSettings, Settings, StateSettings
from retrievr.snapshots import ReplayTransport, load_index

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
CACHE = FIXTURES / "llm_cache"
MODEL = "openai/gpt-6-luna"
RECORD = os.environ.get("RETRIEVR_LLM_RECORD") == "1"
SITES = ["societegenerale", "deutschebank", "ubs"]
# Minimum recall and precision per site (default: all documents, precision 95 %).
THRESHOLDS: dict[str, tuple[float, float]] = {}


def _site(site_id: str) -> Site:
    return next(s for s in load_sites().sites if s.id == site_id)


@pytest.fixture
def offline() -> Iterator[None]:
    """Outside recording, any request through the default httpx transport fails."""
    if RECORD:
        yield
        return
    with respx.mock(assert_all_called=False):
        yield


@pytest.mark.parametrize("site_id", [*SITES, "creditagricole"])
def test_cold_learning_finds_every_document(site_id: str, tmp_path: Path, offline: None) -> None:
    site = _site(site_id)
    truth = load_ground_truth(BENCH / "ground_truth" / f"{site_id}.yaml")
    settings = Settings(
        llm=LLMSettings(model_small=MODEL, cache_dir=CACHE, cache="use" if RECORD else "replay"),
        state=StateSettings(state_dir=tmp_path / "state"),
        browser=BrowserSettings(enabled=False),
    )
    snapshot = FIXTURES / site_id
    discovery = ReplayDiscovery(snapshot)
    captured = load_index(snapshot).captured_on()
    r = Retrievr(
        settings=settings,
        fetch_transport=ReplayTransport(snapshot, lenient_bodies=True),
        discoverer=discovery if discovery.available(site.start_url) else None,
        sleep=lambda _s: None,
        today=captured.date() if captured else truth.captured_at,
    )
    learned = r.learn(site.start_url, issuer=site.id, years=truth.fiscal_years)
    assert learned.status == "learned", learned.errors
    assert learned.recipe is not None and learned.recipe.provenance is not None
    assert learned.recipe.provenance.method == "learned"
    score = score_documents(learned.all_documents, truth)
    min_recall, min_precision = THRESHOLDS.get(site_id, (1.0, 0.95))
    assert score.recall >= min_recall, score.missing[:5]
    assert score.precision >= min_precision, score.extra[:5]

    # Warm run on the unchanged site: the recipe alone, no LLM call.
    warm = r.run(site.id, years=truth.fiscal_years, download=False)
    assert warm.status == "ok", warm.errors
    assert warm.llm.calls == 0
    assert {d.normalized_url for d in warm.all_documents} == {
        d.normalized_url for d in learned.all_documents
    }


@pytest.mark.parametrize("site_id", SITES)
def test_requests_compile_to_the_expected_filter(
    site_id: str, tmp_path: Path, offline: None
) -> None:
    truth = load_ground_truth(BENCH / "ground_truth" / f"{site_id}.yaml")
    settings = Settings(
        llm=LLMSettings(model_small=MODEL, cache_dir=CACHE, cache="use" if RECORD else "replay"),
        state=StateSettings(state_dir=tmp_path / "state"),
        browser=BrowserSettings(enabled=False),
    )
    r = Retrievr(settings=settings, today=truth.captured_at)
    for request in truth.requests:
        compiled, _usage = r.compile_request(request.text)
        selected = {
            d.normalized_url
            for d in truth.documents
            if compiled.matches(d.type, d.fiscal_year, d.language)
        }
        assert selected == truth.expected_for(request), (request.text, compiled)
