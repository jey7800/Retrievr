"""Offline checks of the benchmark code and data (the live benchmark never runs here)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pydantic
import pytest
from bench.metrics import score_documents, snapshot_coverage
from bench.model import GroundTruth, load_ground_truth, load_sites
from bench.run import main

from retrievr.recipe import FoundDocument
from retrievr.urls import normalize_url

DOCS = [
    {
        "url": "https://x.example/a.pdf",
        "type": "press_release",
        "period": "2025-Q1",
        "fiscal_year": 2025,
        "language": "en",
    },
    {
        "url": "https://x.example/b.pdf",
        "type": "results_presentation",
        "period": "2025-Q1",
        "fiscal_year": 2025,
        "language": "en",
    },
]
REQUESTS = [
    {"text": "press releases", "expect": {"types": ["press_release"]}},
    {"text": "presentations", "expect": {"types": ["results_presentation"]}},
    {"text": "everything 2025", "expect": {"fiscal_years": [2025]}},
]


def _truth(**overrides: object) -> GroundTruth:
    data: dict[str, object] = {
        "ground_truth_version": 1,
        "site": "x",
        "scope": "Every results document of the page.",
        "fiscal_years": [2025],
        "captured_at": "2026-11-20",
        "documents": DOCS,
        "requests": REQUESTS,
    }
    data.update(overrides)
    return GroundTruth.model_validate(data)


def _found(url: str, fiscal_year: int | None = 2025, type_: str = "press_release") -> FoundDocument:
    return FoundDocument(
        url=url,
        normalized_url=normalize_url(url),
        title="t",
        source_id="s",
        params={},
        page_url="https://x.example/",
        extension="pdf",
        date=None,
        type=type_,
        period="2025-Q1",
        fiscal_year=fiscal_year,
        language="en",
        score=1.0,
        justification="",
    )


def test_score_counts_recall_precision_and_metadata() -> None:
    score = score_documents(
        [
            _found("https://x.example/a.pdf"),
            _found("https://x.example/b.pdf", type_="press_release"),  # wrong type
            _found("https://x.example/c.pdf"),  # extra, in window
            _found("https://x.example/d.pdf", fiscal_year=None),  # unknown year counts
            _found("https://x.example/old.pdf", fiscal_year=2019),  # outside window
        ],
        _truth(),
    )
    assert (score.recall, score.matched, score.found) == (1.0, 2, 4)
    assert score.precision == 0.5
    assert score.metadata_accuracy["type"] == 0.5
    assert score.metadata_accuracy["period"] == 1.0
    assert score.extra == ["https://x.example/c.pdf", "https://x.example/d.pdf"]


def test_score_of_an_empty_run() -> None:
    score = score_documents([], _truth())
    assert (score.recall, score.precision) == (0.0, 0.0)
    assert len(score.missing) == 2


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"documents": [{**DOCS[0], "type": "brochure"}]}, "unknown document type"),
        ({"documents": [{**DOCS[0], "period": "Q1 2025"}]}, "must look like"),
        ({"documents": [DOCS[0], DOCS[0]]}, "duplicate"),
        ({"fiscal_years": [2024]}, "outside fiscal_years"),
        ({"requests": REQUESTS[:2]}, "at least 3"),
        (
            {"requests": [*REQUESTS[:2], {"text": "pillar 3", "expect": {"types": ["pillar3"]}}]},
            "expects no document",
        ),
    ],
)
def test_invalid_ground_truth_is_rejected(override: dict[str, object], message: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=message):
        _truth(**override)


def test_bench_data_files_are_valid() -> None:
    sites = load_sites().sites
    assert len(sites) >= 22
    for site in sites:
        if site.ground_truth:
            truth = load_ground_truth(site.ground_truth_path())
            assert truth.site == site.id
        else:
            assert not site.ground_truth_path().exists(), site.id


def test_committed_snapshots_cover_their_ground_truth() -> None:
    for site in load_sites().sites:
        directory = site.snapshot_dir()
        if (
            site.ground_truth
            and directory is not None
            and directory.is_relative_to(Path(__file__).resolve().parents[1])
        ):
            truth = load_ground_truth(site.ground_truth_path())
            assert snapshot_coverage(directory, truth).ratio == 1.0, site.id


def test_runner_offline_on_committed_snapshots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--sites", "societegenerale,deutschebank,ubs", "--out", str(tmp_path)]) == 0
    report = next(tmp_path.glob("*.md")).read_text("utf-8")
    for site in ("societegenerale", "deutschebank", "ubs"):
        line = next(x for x in report.splitlines() if x.startswith(f"| {site} "))
        assert "| ok | 100% | 100% |" in line, line
    assert date.today().strftime("%Y") in report
