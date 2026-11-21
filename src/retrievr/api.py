"""Public Python API.

    from retrievr import Retrievr
    r = Retrievr("retrievr.toml", on_event=print)
    result = r.learn("https://example-issuer.com/investors", "annual reports since 2022")
    result = r.run("example-issuer")   # replays the active recipe, no LLM call if healthy

``learn`` calls the LLM (cold path); ``run`` replays the recipe deterministically and only
calls the LLM for documents its rules cannot classify and no earlier decision covers.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import httpx

from retrievr.browser import BrowserDiscovery, Discoverer
from retrievr.classify import Decision, apply_decisions
from retrievr.config import Settings, load_settings
from retrievr.download import Downloader, DownloadOutcome
from retrievr.events import Event, EventHandler, emit
from retrievr.fetch import BlockedError, Fetcher
from retrievr.learn import (
    Filter,
    LearnContext,
    Learner,
    LearnError,
    classify_remaining,
    compile_request,
    filter_json,
    health_for,
)
from retrievr.llm import (
    BudgetExceededError,
    CostLedger,
    LLMClient,
    LLMError,
    LLMNotConfiguredError,
    ResponseCache,
    StructuredLLM,
)
from retrievr.recipe import FoundDocument, Recipe, load_recipe, run_recipe
from retrievr.store import StateStore
from retrievr.urls import host_of
from retrievr.validate import ValidationReport, validate_runs

log = logging.getLogger("retrievr")

LearnStatus = Literal["learned", "rejected", "blocked", "error", "budget_exceeded"]
RunStatus = Literal["ok", "blocked", "error", "no_recipe"]
_GENERIC_LABELS = {"www", "investors", "investor", "ir", "home", "en", "fr", "de", "corporate"}


@dataclass
class LLMUsage:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @classmethod
    def of(cls, ledger: CostLedger | None) -> LLMUsage:
        if ledger is None:
            return cls()
        tokens_in, tokens_out = ledger.tokens
        return cls(
            ledger.network_calls, ledger.cached_calls, tokens_in, tokens_out, ledger.spent_usd
        )


@dataclass
class LearnResult:
    site: str
    status: LearnStatus
    run_id: str
    recipe: Recipe | None = None
    recipe_path: Path | None = None
    documents: list[FoundDocument] = field(default_factory=list)
    """Documents matching the request."""
    all_documents: list[FoundDocument] = field(default_factory=list)
    """Every document the recipe lists (the recipe describes the site, not the request)."""
    filter: Filter | None = None
    validation: ValidationReport | None = None
    llm: LLMUsage = field(default_factory=LLMUsage)
    downloads: list[DownloadOutcome] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class RunOutcome:
    site: str
    status: RunStatus
    run_id: str
    recipe: str | None = None
    documents: list[FoundDocument] = field(default_factory=list)
    all_documents: list[FoundDocument] = field(default_factory=list)
    filter: Filter | None = None
    llm: LLMUsage = field(default_factory=LLMUsage)
    downloads: list[DownloadOutcome] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def issuer_from_url(url: str) -> str:
    """A slug for the site: ``investors.example-issuer.com`` -> ``example-issuer``."""
    labels = host_of(url).split(".")
    body = [label for label in labels[:-1] if label not in _GENERIC_LABELS]
    name = body[-1] if body else labels[-1]
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return slug or "site"


class Retrievr:
    def __init__(
        self,
        config_path: Path | str | None = None,
        *,
        settings: Settings | None = None,
        on_event: EventHandler | None = None,
        llm: StructuredLLM | None = None,
        discoverer: Discoverer | None = None,
        fetch_transport: httpx.BaseTransport | None = None,
        llm_transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        today: date | None = None,
    ) -> None:
        self.settings = settings or load_settings(config_path)
        self.on_event = on_event
        self.store = StateStore(self.settings.state.state_dir)
        self._llm_override = llm
        self._discoverer = discoverer
        self._fetch_transport = fetch_transport
        self._llm_transport = llm_transport
        self._sleep = sleep
        self._today = today

    @property
    def today(self) -> date:
        return self._today or date.today()

    # -- helpers -----------------------------------------------------------------------

    def _fetcher(self) -> Fetcher:
        return Fetcher(self.settings.fetch, transport=self._fetch_transport, sleep=self._sleep)

    def _llm(self, run_id: str) -> tuple[StructuredLLM, CostLedger | None]:
        if self._llm_override is not None:
            return self._llm_override, getattr(self._llm_override, "ledger", None)
        cfg = self.settings.llm
        cache_dir = cfg.cache_dir or self.settings.state.state_dir / "llm_cache"
        client = LLMClient(
            cfg,
            cache=ResponseCache(cache_dir),
            on_call=lambda record: self.store.record_llm_call(run_id, record),
            transport=self._llm_transport,
            sleep=self._sleep,
        )
        return client, client.ledger

    def compile(self, llm: StructuredLLM, request: str) -> Filter:
        """The filter of a request, from the state cache or compiled once by the LLM."""
        key = hashlib.sha256(f"{self.today.year}|{request.strip()}".encode()).hexdigest()
        if (cached := self.store.filter_json(key)) is not None:
            return Filter.model_validate_json(cached)
        compiled = compile_request(llm, request, today=self.today)
        self.store.save_filter(key, request, filter_json(compiled))
        return compiled

    def compile_request(self, request: str) -> tuple[Filter, LLMUsage]:
        """Compile ``request`` into a filter (cached), outside any learn or run."""
        run_id = self.store.start_run("compile", None)
        llm, ledger = self._llm(run_id)
        try:
            compiled = self.compile(llm, request)
        finally:
            if isinstance(llm, LLMClient):
                llm.close()
        usage = LLMUsage.of(ledger)
        self.store.finish_run(
            run_id, status="ok", documents=0, llm_calls=usage.calls, cost_usd=usage.cost_usd
        )
        return compiled, usage

    def _emit(self, kind: str, site: str, message: str, **data: object) -> None:
        emit(self.on_event, Event(kind, site, message, data))  # type: ignore[arg-type]

    # -- learn -------------------------------------------------------------------------

    def learn(
        self,
        url: str,
        request: str | None = None,
        *,
        issuer: str | None = None,
        years: Collection[int] | None = None,
        download: bool = False,
    ) -> LearnResult:
        """Learn a recipe for a new site (cold path), validate it, activate it if it
        passes, and return the documents matching ``request``.

        ``years`` limits the listing pages replayed while learning (the recipe itself
        covers every year)."""
        site = issuer or issuer_from_url(url)
        run_id = self.store.start_run("learn", site)
        result = LearnResult(site=site, status="error", run_id=run_id)
        started = time.monotonic()
        llm, ledger = self._llm(run_id)
        try:
            with self._fetcher() as fetcher:
                self._learn(url, request, site, years, download, fetcher, llm, result)
        except BlockedError as exc:
            result.status = "blocked"
            result.errors.append(str(exc))
            self._emit("blocked", site, str(exc))
        except BudgetExceededError as exc:
            result.status = "budget_exceeded"
            result.errors.append(str(exc))
            self._emit("budget_exceeded", site, str(exc))
        except (LearnError, LLMError) as exc:
            result.status = "error"
            result.errors.append(f"{type(exc).__name__}: {exc}")
            self._emit("llm_error", site, str(exc))
        finally:
            if isinstance(llm, LLMClient):
                llm.close()
            result.llm = LLMUsage.of(ledger)
            result.seconds = round(time.monotonic() - started, 2)
            self.store.finish_run(
                run_id,
                status=result.status,
                documents=len(result.all_documents),
                llm_calls=result.llm.calls,
                cost_usd=result.llm.cost_usd,
                detail={"errors": result.errors, "notes": result.notes[-20:]},
            )
            self._emit("run_finished", site, result.status, run="learn", run_id=run_id)
        return result

    def _learn(
        self,
        url: str,
        request: str | None,
        site: str,
        years: Collection[int] | None,
        download: bool,
        fetcher: Fetcher,
        llm: StructuredLLM,
        result: LearnResult,
    ) -> None:
        context = LearnContext(issuer=site, start_url=url, years=years, today=self.today)
        discoverer = self._discoverer
        if discoverer is None and self.settings.browser.enabled:
            discoverer = BrowserDiscovery(self.settings.browser, fetcher)
        learner = Learner(fetcher, llm, context, discoverer=discoverer)
        structure = learner.learn_structure()
        rules = learner.learn_rules(structure)
        result.notes = learner.notes
        version = self.store.next_version(context.recipe_name)
        recipe = learner.build_recipe(structure.reply, rules=rules.rules, version=version)
        check_years = learner.check_years(recipe)
        first = run_recipe(recipe, fetcher, years=check_years, today=self.today)
        decisions: dict[str, Decision] = dict(rules.decisions)
        decisions.update(
            classify_remaining(llm, first.documents, decisions, issuer=site, years=years)
        )
        documents = apply_decisions(first.documents, decisions)
        recipe = recipe.model_copy(update={"health": health_for(documents)})
        second = run_recipe(recipe, fetcher, years=check_years, today=self.today)
        report = validate_runs(recipe, first, second)
        result.recipe, result.validation = recipe, report
        result.all_documents = documents
        if report.passed:
            result.recipe_path = self.store.save_recipe(recipe)
            self.store.activate(site, recipe, result.recipe_path)
            self.store.save_decisions(recipe.recipe, decisions)
            result.status = "learned"
            self._emit("recipe_learned", site, recipe.key, documents=len(documents))
        else:
            result.status = "rejected"
            self._emit("recipe_rejected", site, report.summary())
        if request:
            self.store.set_site_request(site, request)
            result.filter = self.compile(llm, request)
            result.documents = result.filter.select(documents)
        else:
            result.documents = documents
        if download and report.passed:
            downloader = Downloader(fetcher, self.settings.output.output_dir)
            result.downloads = downloader.download_all(recipe, result.documents)

    # -- run ---------------------------------------------------------------------------

    def run(
        self,
        site: str,
        *,
        request: str | None = None,
        years: Collection[int] | None = None,
        download: bool = True,
    ) -> RunOutcome:
        """Replay the active recipe of ``site`` (or a recipe file path)."""
        run_id = self.store.start_run("run", site)
        outcome = RunOutcome(site=site, status="error", run_id=run_id)
        recipe = self._recipe(site)
        if recipe is None:
            outcome.status = "no_recipe"
            self.store.finish_run(run_id, status="no_recipe", documents=0, llm_calls=0, cost_usd=0)
            return outcome
        outcome.recipe = recipe.key
        ledger: CostLedger | None = None
        llm: StructuredLLM | None = None
        try:
            with self._fetcher() as fetcher:
                run = run_recipe(recipe, fetcher, years=years, today=self.today)
                outcome.errors = run.errors
                if run.status != "ok":
                    outcome.status = run.status
                    if run.status == "blocked":
                        self._emit("blocked", site, "; ".join(run.errors))
                    return outcome
                decisions = self.store.decisions(recipe.recipe)
                request = request or self.store.site_request(site)
                try:
                    llm, ledger = self._llm(run_id)
                    new = classify_remaining(
                        llm, run.documents, decisions, issuer=recipe.issuer, years=years
                    )
                    if new:
                        self.store.save_decisions(recipe.recipe, new)
                        decisions.update(new)
                    outcome.filter = self.compile(llm, request) if request else None
                except LLMNotConfiguredError as exc:
                    outcome.errors.append(f"LLM not configured: {exc}")
                outcome.all_documents = apply_decisions(run.documents, decisions)
                selected = (
                    outcome.filter.select(outcome.all_documents)
                    if outcome.filter
                    else outcome.all_documents
                )
                outcome.documents = selected
                if download:
                    downloader = Downloader(fetcher, self.settings.output.output_dir)
                    outcome.downloads = downloader.download_all(recipe, selected)
                outcome.status = "ok"
        except BlockedError as exc:
            outcome.status = "blocked"
            outcome.errors.append(str(exc))
            self._emit("blocked", site, str(exc))
        except LLMError as exc:
            outcome.errors.append(f"{type(exc).__name__}: {exc}")
            self._emit("llm_error", site, str(exc))
        finally:
            if isinstance(llm, LLMClient):
                llm.close()
            outcome.llm = LLMUsage.of(ledger)
            self.store.finish_run(
                run_id,
                status=outcome.status,
                documents=len(outcome.all_documents),
                llm_calls=outcome.llm.calls,
                cost_usd=outcome.llm.cost_usd,
                detail={"errors": outcome.errors},
            )
            self._emit("run_finished", site, outcome.status, run="replay", run_id=run_id)
        return outcome

    def _recipe(self, site: str) -> Recipe | None:
        if site.endswith((".yaml", ".yml")) and Path(site).exists():
            return load_recipe(site)
        return self.store.active_recipe(site)
