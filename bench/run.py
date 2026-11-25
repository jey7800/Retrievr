"""Benchmark runner (opt-in, never part of pytest).

    uv run python -m bench.run                 # offline: hand-written recipes on snapshots
    uv run python -m bench.run --mode live     # hits the network (monthly drift runs)
    uv run --env-file .env python -m bench.run --mode cold --model openai/gpt-6-luna
    uv run python -m bench.run --sites ubs,hsbc

``snapshot`` and ``live`` replay each site's hand-written recipe. ``cold`` learns a recipe
from scratch with the LLM (pages from the snapshot, the LLM over the network, responses
cached in ``bench/.llm_cache``), scores it against the ground truth, and also scores the
three requests of each ground truth (compiled filter applied to the ground truth).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from bench.metrics import DocumentScore, score_documents, snapshot_coverage
from bench.model import BENCH, ROOT, GroundTruth, Site, load_ground_truth, load_sites
from retrievr.api import Retrievr
from retrievr.browser import Discovery, ReplayDiscovery
from retrievr.config import FetchSettings, LLMSettings, Settings, StateSettings
from retrievr.fetch import Fetcher
from retrievr.recipe import load_recipe, run_recipe
from retrievr.snapshots import ReplayTransport, load_index

Mode = Literal["snapshot", "live", "cold"]
Status = Literal[
    "ok",
    "blocked",
    "unauthorized",
    "error",
    "no_recipe",
    "no_snapshot",
    "learned",
    "rejected",
    "budget_exceeded",
]
LLM_CACHE = BENCH / ".llm_cache"


@dataclass
class SiteResult:
    site: str
    profile: str
    status: Status
    recipe: str | None
    seconds: float = 0.0
    llm_calls: int = 0
    cached_calls: int = 0
    tokens: tuple[int, int] = (0, 0)
    cost_usd: float = 0.0
    score: DocumentScore | None = None
    coverage: float | None = None
    coverage_missing: list[str] | None = None
    requests_exact: float | None = None
    request_details: list[dict[str, object]] = field(default_factory=list)
    validation: str | None = None
    errors: list[str] | None = None
    warm: str | None = None
    """Cold mode: the warm run after learning (status, health, LLM calls, same documents)."""


def _snapshot_ready(site: Site) -> Path | None:
    snapshot = site.snapshot_dir()
    return snapshot if snapshot is not None and (snapshot / "index.yaml").exists() else None


def evaluate_site(site: Site, mode: Mode) -> SiteResult:
    truth = load_ground_truth(site.ground_truth_path())
    snapshot = _snapshot_ready(site)
    result = SiteResult(site.id, site.profile, "ok", site.recipe)
    if snapshot is not None:
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


def cold_settings(model: str, state_dir: Path) -> Settings:
    return Settings(
        llm=LLMSettings(model_small=model, cache_dir=LLM_CACHE),
        state=StateSettings(state_dir=state_dir),
    )


def evaluate_cold(site: Site, model: str, state_root: Path) -> SiteResult:
    """Learn the site from its snapshot with ``model`` and score the learned recipe."""
    truth = load_ground_truth(site.ground_truth_path())
    result = SiteResult(site.id, site.profile, "no_snapshot", None)
    snapshot = _snapshot_ready(site)
    if snapshot is None:
        return result
    cov = snapshot_coverage(snapshot, truth)
    result.coverage, result.coverage_missing = cov.ratio, cov.missing
    # Offline: pages from the snapshot, a recorded browser discovery if there is one
    # (never a live browser), and "today" set to the capture day (dates in bodies).
    discovery = ReplayDiscovery(snapshot)
    captured = load_index(snapshot).captured_on()
    retrievr = Retrievr(
        settings=cold_settings(model, state_root / site.id),
        fetch_transport=ReplayTransport(snapshot, lenient_bodies=True),
        discoverer=discovery if discovery.available(site.start_url) else _NoDiscovery(),
        sleep=lambda _s: None,
        today=captured.date() if captured else truth.captured_at,
    )
    learned = retrievr.learn(site.start_url, issuer=site.id, years=truth.fiscal_years)
    result.status = learned.status
    result.recipe = learned.recipe.key if learned.recipe else None
    result.seconds = learned.seconds
    result.errors = learned.errors or None
    result.validation = learned.validation.summary() if learned.validation else None
    if learned.recipe is not None:
        result.score = score_documents(learned.all_documents, truth)
    if learned.status == "learned":
        # Warm run: the recipe alone replays the site (no LLM call, same documents).
        warm = retrievr.run(site.id, years=truth.fiscal_years, download=False)
        same = {d.normalized_url for d in warm.all_documents} == {
            d.normalized_url for d in learned.all_documents
        }
        verdict = warm.health.verdict if warm.health else "-"
        result.warm = (
            f"{warm.status}, health {verdict}, {warm.llm.calls} LLM calls, "
            f"{'same documents' if same else 'DIFFERENT documents'}"
        )
    calls, cached = learned.llm.calls, learned.llm.cached_calls
    tokens_in, tokens_out = learned.llm.input_tokens, learned.llm.output_tokens
    cost = learned.llm.cost_usd
    exact = 0
    for req in truth.requests:
        compiled, usage = retrievr.compile_request(req.text)
        calls, cached, cost = (
            calls + usage.calls,
            cached + usage.cached_calls,
            cost + usage.cost_usd,
        )
        tokens_in, tokens_out = tokens_in + usage.input_tokens, tokens_out + usage.output_tokens
        selected = _select(truth, compiled.matches)
        expected = truth.expected_for(req)
        exact += selected == expected
        result.request_details.append(
            {
                "request": req.text,
                "filter": compiled.model_dump(mode="json"),
                "exact": selected == expected,
                "missing": len(expected - selected),
                "extra": len(selected - expected),
            }
        )
    result.requests_exact = exact / len(truth.requests)
    result.llm_calls, result.cached_calls, result.cost_usd = calls, cached, cost
    result.tokens = (tokens_in, tokens_out)
    return result


class _NoDiscovery:
    """Offline benchmark without a recorded discovery: no browser either."""

    def discover(self, url: str, years: object) -> Discovery:
        return Discovery(start_url=url, notes=["no recorded discovery"])


def _select(truth: GroundTruth, matches: object) -> set[str]:
    assert callable(matches)  # noqa: S101 - internal helper
    return {d.normalized_url for d in truth.documents if matches(d.type, d.fiscal_year, d.language)}


def to_markdown(results: list[SiteResult], mode: Mode, when: datetime, model: str | None) -> str:
    title = f"# Benchmark ({mode}{f', {model}' if model else ''}, {when:%Y-%m-%d %H:%M} UTC)"
    lines = [
        title,
        "",
        "| Site | Profile | Status | Recall | Precision | Type | Period | Lang | "
        "Coverage | Requests | LLM calls | Cost $ | Time s |"
        + (" Warm run |" if mode == "cold" else ""),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        + (" --- |" if mode == "cold" else ""),
    ]

    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value:.0%}"

    for r in results:
        s = r.score
        meta = s.metadata_accuracy if s else {}
        calls = f"{r.llm_calls}" + (f" (+{r.cached_calls} cached)" if r.cached_calls else "")
        lines.append(
            f"| {r.site} | {r.profile} | {r.status} | {pct(s.recall if s else None)} | "
            f"{pct(s.precision if s else None)} | {pct(meta.get('type'))} | "
            f"{pct(meta.get('period'))} | {pct(meta.get('language'))} | {pct(r.coverage)} | "
            f"{pct(r.requests_exact)} | {calls} | {r.cost_usd:.4f} | {r.seconds:.1f} |"
            + (f" {r.warm or '-'} |" if mode == "cold" else "")
        )
    scored = [r.score for r in results if r.score]
    if scored:
        mean = sum(s.recall for s in scored) / len(scored)
        lines += ["", f"Mean recall over {len(scored)} scored sites: {mean:.1%}."]
    if mode == "cold" and results:
        good = sum(1 for r in results if r.score and r.score.recall >= 0.9)
        lines.append(
            f"Cold recall >= 90 % on {good} of {len(results)} sites ({good / len(results):.0%})."
        )
        total = sum(r.cost_usd for r in results)
        lines.append(f"Total LLM cost: {total:.4f} USD.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the retrievr benchmark.")
    parser.add_argument("--mode", choices=["snapshot", "live", "cold"], default="snapshot")
    parser.add_argument("--model", help="LLM model id for --mode cold (default: config default)")
    parser.add_argument("--sites", help="comma-separated site ids (default: all with ground truth)")
    parser.add_argument("--out", type=Path, default=BENCH / "results")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    wanted = set(args.sites.split(",")) if args.sites else None
    sites = [s for s in load_sites().sites if s.ground_truth and (wanted is None or s.id in wanted)]
    if wanted and (unknown := wanted - {s.id for s in sites}):
        parser.error(f"unknown site ids or no ground truth: {sorted(unknown)}")

    when = datetime.now(UTC)
    model = (args.model or LLMSettings().model_small) if args.mode == "cold" else None
    if args.mode == "cold":
        assert model is not None  # noqa: S101
        slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
        state_root = args.out / "state" / f"{when:%Y%m%dT%H%M%S}-{slug}"
        results = [evaluate_cold(site, model, state_root) for site in sites]
    else:
        results = [evaluate_site(site, args.mode) for site in sites]

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = args.mode if model is None else f"cold-{re.sub(r'[^a-z0-9]+', '-', model.lower())}"
    stem = args.out / f"{when:%Y%m%dT%H%M%S}-{suffix}"
    stem.with_suffix(".json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = to_markdown(results, args.mode, when, model)
    stem.with_suffix(".md").write_text(report, encoding="utf-8")
    print(report)
    print(f"written {stem}.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
