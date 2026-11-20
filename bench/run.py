"""Benchmark runner (opt-in, never part of pytest).

    uv run python -m bench.run                 # offline, from snapshots
    uv run python -m bench.run --mode live     # hits the network (monthly drift runs)
    uv run python -m bench.run --sites ubs,hsbc

For every site with a ground truth it replays the site's recipe (hand-written for now,
learned in a later milestone) and reports recall, precision, metadata accuracy, time,
LLM calls and cost, plus how much of the ground truth the snapshot covers.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from bench.metrics import DocumentScore, score_documents, snapshot_coverage
from bench.model import BENCH, ROOT, Site, load_ground_truth, load_sites
from retrievr.config import FetchSettings
from retrievr.fetch import Fetcher
from retrievr.recipe import load_recipe, run_recipe
from retrievr.snapshots import ReplayTransport

Mode = Literal["snapshot", "live"]
Status = Literal["ok", "blocked", "error", "no_recipe", "no_snapshot"]


@dataclass
class SiteResult:
    site: str
    profile: str
    status: Status
    recipe: str | None
    seconds: float = 0.0
    llm_calls: int = 0
    cost_usd: float = 0.0
    score: DocumentScore | None = None
    coverage: float | None = None
    coverage_missing: list[str] | None = None
    errors: list[str] | None = None


def evaluate_site(site: Site, mode: Mode) -> SiteResult:
    truth = load_ground_truth(site.ground_truth_path())
    snapshot = site.snapshot_dir()
    result = SiteResult(site.id, site.profile, "ok", site.recipe)
    if snapshot is not None and (snapshot / "index.yaml").exists():
        cov = snapshot_coverage(snapshot, truth)
        result.coverage, result.coverage_missing = cov.ratio, cov.missing
    elif mode == "snapshot":
        result.status = "no_snapshot"
        return result
    if site.recipe is None:
        result.status = "no_recipe"
        return result

    recipe = load_recipe(ROOT / "recipes" / site.recipe)
    if mode == "snapshot" and snapshot is not None:
        fetcher = Fetcher(
            FetchSettings(), transport=ReplayTransport(snapshot), sleep=lambda _s: None
        )
    else:
        fetcher = Fetcher(FetchSettings())
    started = time.monotonic()
    with fetcher:
        run = run_recipe(recipe, fetcher, years=truth.fiscal_years)
    result.seconds = round(time.monotonic() - started, 2)
    result.status = run.status
    result.errors = run.errors or None
    result.score = score_documents(run.documents, truth)
    return result


def to_markdown(results: list[SiteResult], mode: Mode, when: datetime) -> str:
    lines = [
        f"# Benchmark ({mode}, {when:%Y-%m-%d %H:%M} UTC)",
        "",
        "| Site | Profile | Status | Recall | Precision | Type | Period | Lang | "
        "Coverage | LLM calls | Cost $ | Time s |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value:.0%}"

    for r in results:
        s = r.score
        meta = s.metadata_accuracy if s else {}
        lines.append(
            f"| {r.site} | {r.profile} | {r.status} | {pct(s.recall if s else None)} | "
            f"{pct(s.precision if s else None)} | {pct(meta.get('type'))} | "
            f"{pct(meta.get('period'))} | {pct(meta.get('language'))} | {pct(r.coverage)} | "
            f"{r.llm_calls} | {r.cost_usd:.3f} | {r.seconds:.1f} |"
        )
    scored = [r.score for r in results if r.score]
    if scored:
        mean = sum(s.recall for s in scored) / len(scored)
        lines += ["", f"Mean recall over {len(scored)} sites with a recipe: {mean:.1%}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the retrievr benchmark.")
    parser.add_argument("--mode", choices=["snapshot", "live"], default="snapshot")
    parser.add_argument("--sites", help="comma-separated site ids (default: all with ground truth)")
    parser.add_argument("--out", type=Path, default=BENCH / "results")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    wanted = set(args.sites.split(",")) if args.sites else None
    sites = [s for s in load_sites().sites if s.ground_truth and (wanted is None or s.id in wanted)]
    if wanted and (unknown := wanted - {s.id for s in sites}):
        parser.error(f"unknown site ids or no ground truth: {sorted(unknown)}")
    results = [evaluate_site(site, args.mode) for site in sites]

    when = datetime.now(UTC)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{when:%Y%m%dT%H%M%S}-{args.mode}"
    stem.with_suffix(".json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = to_markdown(results, args.mode, when)
    stem.with_suffix(".md").write_text(report, encoding="utf-8")
    print(report)
    print(f"written {stem}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
