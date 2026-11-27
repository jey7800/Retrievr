"""Public Python API.

    from retrievr import Retrievr
    r = Retrievr("retrievr.toml", on_event=print)
    result = r.learn("https://example-issuer.com/investors", "annual reports since 2022")
    result = r.run("example-issuer")   # replays the active recipe, no LLM call if healthy

``learn`` calls the LLM (cold path); ``run`` replays the recipe deterministically and only
calls the LLM for documents its rules cannot classify and no earlier decision covers, or
when the health check finds the recipe broken (repair). ``check`` is the health check
alone (no LLM, no download).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import Counter
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
from retrievr.fetch import (
    BlockedError,
    Fetcher,
    FetchError,
    HostSession,
    HTTPStatusError,
    UnauthorizedError,
    detect_block,
)
from retrievr.health import HealthReport, Probe, assess
from retrievr.learn import (
    FILTER_SYSTEM,
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
    ResponseCache,
    StructuredLLM,
)
from retrievr.recipe import (
    FoundDocument,
    Recipe,
    RunResult,
    fetched_by_request,
    load_recipe,
    run_recipe,
    url_path_variants,
)
from retrievr.repair import RepairContext, Repairer, RepairResult
from retrievr.store import StateStore
from retrievr.urls import host_of
from retrievr.validate import Reachability, ValidationReport, validate_runs

log = logging.getLogger("retrievr")

LearnStatus = Literal["learned", "rejected", "blocked", "unauthorized", "error", "budget_exceeded"]
RunStatus = Literal[
    "ok", "degraded", "blocked", "unauthorized", "error", "no_recipe", "budget_exceeded"
]
MAX_LLM_REPAIRS_PER_DAY = 2
REACHABILITY_SAMPLE = 3
"""Documents fetched while learning to check that the listed files can be downloaded."""
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
    warnings: list[str] = field(default_factory=list)
    """Things the host should know about the result (e.g. a request that selects nothing)."""
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
    warnings: list[str] = field(default_factory=list)
    health: HealthReport | None = None
    repair: RepairResult | None = None
    """Set when the health check found the recipe broken."""


_PEEK_BYTES = 64 * 1024
"""How much of an HTML reply to a document probe is read to recognise a challenge."""


def in_years(documents: list[FoundDocument], years: Collection[int] | None) -> list[FoundDocument]:
    """The documents of the fiscal years asked for (all when ``years`` is None). A
    recipe may list more years than asked (a single page holding every year), and a
    document with no known fiscal year is left out when years are asked for."""
    if not years:
        return documents
    wanted = set(years)
    return [d for d in documents if d.fiscal_year in wanted]


def download_warnings(downloads: list[DownloadOutcome]) -> list[str]:
    """Say so when not one of the documents could be downloaded (they are listed, but
    the site answers 404 or serves an HTML page instead of the files)."""
    failed = [d for d in downloads if d.status in ("rejected", "error", "blocked", "unauthorized")]
    if not downloads or len(failed) < len(downloads):
        return []
    return [
        f"none of the {len(downloads)} documents could be downloaded; "
        f"first failure: {failed[0].reason}"
    ]


def blocked_downloads(downloads: list[DownloadOutcome]) -> str | None:
    """Why the site refused the downloads (a block status, a challenge or block page
    served instead of the file), or None. The downloader stops at the first refusal."""
    blocked = next((d for d in downloads if d.status == "blocked"), None)
    if blocked is None:
        return None
    return f"the site blocks downloads: {blocked.reason or 'blocked'}"


def selection_warnings(
    compiled: Filter, documents: list[FoundDocument], selected: list[FoundDocument]
) -> list[str]:
    """A request that selects nothing although the site lists documents is most often a
    request worded differently from the site's documents: say what the site has."""
    if selected or not documents:
        return []
    counts = Counter(d.type or "untyped" for d in documents)
    kinds = ", ".join(f"{t} {n}" for t, n in counts.most_common(8))
    years = sorted({d.fiscal_year for d in documents if d.fiscal_year})
    span = f"{years[0]}-{years[-1]}" if years else "unknown years"
    return [
        f"the request selected none of the {len(documents)} documents listed "
        f"(types {compiled.types or 'any'}, years {compiled.year_from}-{compiled.year_to}); "
        f"the site lists: {kinds} ({span})"
    ]


def issuer_from_url(url: str) -> str:
    """A slug for the site: ``investors.example-issuer.com`` -> ``example-issuer``."""
    labels = host_of(url).split(".")
    body = [label for label in labels[:-1] if label not in _GENERIC_LABELS]
    name = body[-1] if body else labels[-1]
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return slug or "site"


def _answer(fetcher: Fetcher, recipe: Recipe, url: str) -> str:
    """``ok``, ``gone`` (404/410), ``blocked`` (403 or a challenge page: the host refuses
    this client) or ``unknown`` (rate limited, unavailable, network error, a refused
    session, or a document fetched by a request that its identity URL cannot replay)."""
    if fetched_by_request(recipe, url):
        return "unknown"
    try:
        with fetcher.stream(url, allowed_domains=recipe.allowed_domains) as resp:
            if "html" in (resp.headers.get("content-type") or "").lower():
                body = fetcher.read(resp)[:_PEEK_BYTES].decode("utf-8", errors="replace")
                if detect_block(resp.status_code, body):
                    return "blocked"
            return "ok"
    except BlockedError as exc:
        return "blocked" if exc.status in (403, None) else "unknown"
    except HTTPStatusError as exc:
        return "gone" if exc.status in (404, 410) else "unknown"
    except FetchError:
        return "unknown"


class Retrievr:
    """Public entry point: learn, run, check, repair and install site recipes.

    ``llm_transport`` (an ``httpx.BaseTransport``) is the supported way to reach an LLM
    gateway that is not a plain OpenAI-compatible endpoint with a static key (OAuth
    tokens, per-deployment URLs, extra query parameters...). It receives every chat
    completions request (``POST {base_url}/chat/completions``, standard JSON body) and
    returns the endpoint's response. With a transport, the API key environment variable
    is optional: if it is unset, no ``Authorization`` header is sent and the transport
    authenticates. The transport is never closed by the library; its owner closes it.

    ``session`` (a ``HostSession``: an ``httpx.BaseTransport`` plus the values read at
    login) is how the host runs recipes on sites it has logged in to (ADR 0014). Every
    request of the run goes through its transport, which adds the cookies and tokens and
    answers 401 when the session is gone; recipes read its values as
    ``{session.<name>}``. A refused session ends the run with status ``unauthorized``:
    nothing is repaired, the host logs in again. Retrievr itself never logs in.
    """

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
        session: HostSession | None = None,
        sleep: Callable[[float], None] = time.sleep,
        today: date | None = None,
    ) -> None:
        if session is not None and fetch_transport is not None:
            raise ValueError("give fetch_transport or session, not both")
        self.settings = settings or load_settings(config_path)
        self.on_event = on_event
        self.store = StateStore(self.settings.state.state_dir)
        self._llm_override = llm
        self._discoverer = discoverer
        self._fetch_transport = fetch_transport
        self._session = session
        self._llm_transport = llm_transport
        self._sleep = sleep
        self._today = today

    @property
    def today(self) -> date:
        return self._today or date.today()

    # -- helpers -----------------------------------------------------------------------

    def _fetcher(self) -> Fetcher:
        return Fetcher(
            self.settings.fetch,
            transport=self._fetch_transport,
            session=self._session,
            sleep=self._sleep,
        )

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
            require_key=self._llm_transport is None,
            sleep=self._sleep,
        )
        return client, client.ledger

    def compile(self, llm: StructuredLLM, request: str) -> Filter:
        """The filter of a request, from the state cache or compiled once by the LLM."""
        prompt = hashlib.sha256(FILTER_SYSTEM.encode()).hexdigest()[:12]  # a new prompt recompiles
        key = hashlib.sha256(f"{self.today.year}|{prompt}|{request.strip()}".encode()).hexdigest()
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

    def _downloader(self, fetcher: Fetcher, site: str) -> Downloader:
        def progress(outcome: DownloadOutcome, index: int, total: int) -> None:
            where = outcome.entry.path if outcome.entry else outcome.document.url
            detail = f" ({outcome.reason})" if outcome.reason else ""
            self._emit(
                "document_downloaded",
                site,
                f"[{index}/{total}] {outcome.status}: {where}{detail}",
                status=outcome.status,
                index=index,
                total=total,
            )

        return Downloader(fetcher, self.settings.output.output_dir, on_outcome=progress)

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
        covers every year) and the documents returned and downloaded."""
        if not re.match(r"^https?://[^/\s]+", url.strip(), re.IGNORECASE):
            raise ValueError(f"invalid start URL {url!r}: expected http:// or https://...")
        url = url.strip()
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
        except UnauthorizedError as exc:
            result.status = "unauthorized"
            result.errors.append(str(exc))
            self._emit("unauthorized", site, str(exc))
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
        if first.status == "unauthorized":
            raise UnauthorizedError(url)
        recipe, first, reachability = self._downloadable(recipe, first, fetcher, check_years)
        decisions: dict[str, Decision] = dict(rules.decisions)
        decisions.update(
            classify_remaining(llm, first.documents, decisions, issuer=site, years=years)
        )
        documents = apply_decisions(first.documents, decisions)
        recipe = recipe.model_copy(update={"health": health_for(documents)})
        second = run_recipe(recipe, fetcher, years=check_years, today=self.today)
        report = validate_runs(recipe, first, second, reachability=reachability)
        result.recipe, result.validation = recipe, report
        result.all_documents = documents
        if report.passed:
            result.recipe_path = self.store.save_recipe(recipe)
            self.store.activate(site, recipe, result.recipe_path)
            self.store.save_decisions(recipe.recipe, decisions)
            self.store.set_site(site, start_url=url, status="ok")
            self.store.record_documents(site, recipe.key, documents)
            result.status = "learned"
            self._emit("recipe_learned", site, recipe.key, documents=len(documents))
        else:
            result.status = "rejected"
            self._emit("recipe_rejected", site, report.summary())
        if request:
            self.store.set_site_request(site, request)
            result.filter = self.compile(llm, request)
            result.documents = in_years(result.filter.select(documents), years)
            result.warnings += selection_warnings(result.filter, documents, result.documents)
        else:
            result.documents = in_years(documents, years)
        if download and report.passed:
            downloader = self._downloader(fetcher, site)
            result.downloads = downloader.download_all(recipe, result.documents)
            result.warnings += download_warnings(result.downloads)
            if reason := blocked_downloads(result.downloads):
                # The recipe is learned and active; the site refuses its files.
                result.status = "blocked"
                result.errors.append(reason)
                self._emit("blocked", site, reason)

    # -- run ---------------------------------------------------------------------------

    def run(
        self,
        site: str,
        *,
        request: str | None = None,
        years: Collection[int] | None = None,
        download: bool = True,
        repair: bool = True,
    ) -> RunOutcome:
        """Replay the active recipe of ``site`` (or a recipe file path), check its health
        and, when it is broken, repair it (``repair=True``) before returning documents.

        ``years`` limits the documents returned and downloaded to those fiscal years.
        A broken recipe that no repair fixes leaves the site ``degraded``: the documents
        it still lists are returned but not downloaded."""
        run_id = self.store.start_run("run", site)
        outcome = RunOutcome(site=site, status="error", run_id=run_id)
        recipe = self._recipe(site)
        if recipe is None:
            outcome.status = "no_recipe"
            outcome.errors.append(self._no_recipe_message(site))
            self.store.finish_run(run_id, status="no_recipe", documents=0, llm_calls=0, cost_usd=0)
            return outcome
        site = self._site_key(site, recipe)
        outcome.recipe = recipe.key
        ledger: CostLedger | None = None
        llm: StructuredLLM | None = None
        try:
            with self._fetcher() as fetcher:
                run = run_recipe(recipe, fetcher, years=years, today=self.today)
                outcome.errors = list(run.errors)
                outcome.warnings += run.warnings
                if run.status in ("blocked", "unauthorized"):
                    outcome.status = run.status
                    self._emit(run.status, site, "; ".join(run.errors))
                    return outcome
                decisions = self.store.decisions(recipe.recipe)
                listed = apply_decisions(run.documents, decisions)
                report = self._assess(site, recipe, run, listed, fetcher, years, run_id)
                outcome.health = report
                documents = run.documents
                if report.repair_needed:
                    outcome.all_documents = listed
                    outcome.status = "degraded"
                    if not repair:
                        return outcome
                    llm, ledger = self._llm(run_id)
                    fixed = self._repair(site, recipe, report, fetcher, years, run_id, llm, ledger)
                    outcome.repair = fixed
                    if fixed.status != "repaired" or fixed.recipe is None:
                        return outcome
                    recipe, documents, decisions = fixed.recipe, fixed.documents, fixed.decisions
                    outcome.recipe = recipe.key
                elif run.status != "ok":
                    outcome.status = run.status
                    return outcome
                request = request or self.store.site_request(site)
                if llm is None:
                    llm, ledger = self._llm(run_id)
                try:
                    new = classify_remaining(
                        llm, documents, decisions, issuer=recipe.issuer, years=years
                    )
                    if new:
                        self.store.save_decisions(recipe.recipe, new)
                        decisions.update(new)
                except BudgetExceededError:
                    raise
                except LLMError as exc:
                    # Documents the rules leave undecided stay untyped; the run goes on.
                    outcome.warnings.append(f"new documents left unclassified: {exc}")
                outcome.all_documents = apply_decisions(documents, decisions)
                if request:
                    try:
                        outcome.filter = self.compile(llm, request)
                    except BudgetExceededError:
                        raise
                    except LLMError as exc:
                        # Never fall back to every document: the request selects them.
                        outcome.errors.append(
                            f"the request {request!r} could not be compiled ({exc}); "
                            "nothing selected or downloaded"
                        )
                        return outcome
                if report.verdict in ("ok", "stale") or outcome.repair is not None:
                    self.store.record_documents(site, recipe.key, outcome.all_documents)
                selected = in_years(
                    outcome.filter.select(outcome.all_documents)
                    if outcome.filter
                    else outcome.all_documents,
                    years,
                )
                outcome.documents = selected
                if outcome.filter is not None:
                    outcome.warnings += selection_warnings(
                        outcome.filter, outcome.all_documents, selected
                    )
                if download:
                    downloader = self._downloader(fetcher, site)
                    outcome.downloads = downloader.download_all(recipe, selected)
                    outcome.warnings += download_warnings(outcome.downloads)
                    refused = [d for d in outcome.downloads if d.status == "unauthorized"]
                    if refused:
                        # The session ended during the downloads: the host logs in again
                        # and runs once more (the files already saved are known then).
                        outcome.status = "unauthorized"
                        outcome.errors.append(refused[0].reason or "unauthorized")
                        self._emit("unauthorized", site, refused[0].reason or "")
                        return outcome
                    if reason := blocked_downloads(outcome.downloads):
                        outcome.status = "blocked"
                        outcome.errors.append(reason)
                        self._emit("blocked", site, reason)
                        return outcome
                outcome.status = "ok"
        except BlockedError as exc:
            outcome.status = "blocked"
            outcome.errors.append(str(exc))
            self._emit("blocked", site, str(exc))
        except UnauthorizedError as exc:
            outcome.status = "unauthorized"
            outcome.errors.append(str(exc))
            self._emit("unauthorized", site, str(exc))
        except BudgetExceededError as exc:
            outcome.status = "budget_exceeded"
            outcome.errors.append(str(exc))
            self._emit("budget_exceeded", site, str(exc))
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

    # -- health and repair -------------------------------------------------------------

    def check(self, site: str, *, years: Collection[int] | None = None) -> HealthReport:
        """Health check only: replay the recipe and compare it with the site's history.
        No LLM call, no download."""
        recipe = self._recipe(site)
        if recipe is None:
            raise LookupError(self._no_recipe_message(site))
        site = self._site_key(site, recipe)
        run_id = self.store.start_run("check", site)
        report = HealthReport(site=site, recipe=recipe.key)
        try:
            with self._fetcher() as fetcher:
                run = run_recipe(recipe, fetcher, years=years, today=self.today)
                listed = apply_decisions(run.documents, self.store.decisions(recipe.recipe))
                report = self._assess(site, recipe, run, listed, fetcher, years, run_id)
                if report.verdict in ("ok", "stale"):
                    self.store.record_documents(site, recipe.key, listed)
        except BlockedError as exc:
            report.add("run", "blocked", str(exc))
            self._emit("blocked", site, str(exc))
        except UnauthorizedError as exc:
            report.add("run", "blocked", str(exc))
            self._emit("unauthorized", site, str(exc))
        finally:
            self.store.finish_run(
                run_id, status=report.verdict, documents=0, llm_calls=0, cost_usd=0.0
            )
        return report

    def repair(self, site: str, *, years: Collection[int] | None = None) -> RepairResult:
        """Check the health of ``site`` and run the repair ladder if it is broken."""
        recipe = self._recipe(site)
        if recipe is None:
            raise LookupError(self._no_recipe_message(site))
        site = self._site_key(site, recipe)
        run_id = self.store.start_run("repair", site)
        result = RepairResult(site=site, status="not_needed")
        llm: StructuredLLM | None = None
        ledger: CostLedger | None = None
        try:
            with self._fetcher() as fetcher:
                run = run_recipe(recipe, fetcher, years=years, today=self.today)
                listed = apply_decisions(run.documents, self.store.decisions(recipe.recipe))
                report = self._assess(site, recipe, run, listed, fetcher, years, run_id)
                if not report.repair_needed:
                    result.reason = f"health is {report.verdict}"
                    return result
                llm, ledger = self._llm(run_id)
                result = self._repair(site, recipe, report, fetcher, years, run_id, llm, ledger)
        finally:
            if isinstance(llm, LLMClient):
                llm.close()
            usage = LLMUsage.of(ledger)
            self.store.finish_run(
                run_id,
                status=result.status,
                documents=len(result.documents),
                llm_calls=usage.calls,
                cost_usd=usage.cost_usd,
                detail={"reason": result.reason},
            )
        return result

    def install(
        self, recipe_path: Path | str, *, site: str | None = None, reset_history: bool = False
    ) -> Recipe:
        """Store a hand-written recipe and make it the active recipe of its site.

        ``reset_history`` forgets the documents earlier runs listed for the site, for a
        recipe whose documents have other identities: health checks then start afresh
        instead of reading the change as documents gone. Downloaded files stay known."""
        recipe = load_recipe(recipe_path)
        site = site or recipe.issuer
        path = self.store.recipes_dir / f"{recipe.key}.yaml"
        if not path.exists():
            path = self.store.save_recipe(recipe)
        elif load_recipe(path) != recipe:
            # Stored versions are immutable: an edited recipe needs a new version.
            raise ValueError(
                f"{recipe.key} is already installed with other content; "
                f"raise 'version' in {recipe_path} to install the edited recipe"
            )
        self.store.activate(site, recipe, path)
        if reset_history:
            self.store.forget_documents(site)
        self.store.set_site(site, status="ok")
        return recipe

    def _no_recipe_message(self, site: str) -> str:
        known = self.store.sites()
        listed = f"active sites: {', '.join(known)}" if known else "no site has a recipe yet"
        return (
            f"no active recipe for {site!r} ({listed}); learn it with 'retrievr learn <url>' "
            "or activate a recipe file with 'retrievr install <file>'"
        )

    def _site_key(self, site: str, recipe: Recipe) -> str:
        """History is kept per site; a recipe run from a file belongs to its issuer."""
        return recipe.issuer if site.endswith((".yaml", ".yml")) else site

    def _assess(
        self,
        site: str,
        recipe: Recipe,
        run: RunResult,
        listed: list[FoundDocument],
        fetcher: Fetcher,
        years: Collection[int] | None,
        run_id: str,
    ) -> HealthReport:
        report = assess(
            recipe,
            run,
            listed,
            self.store.known_documents(site),
            site=site,
            today=self.today,
            years=years,
            probe=self._probe(fetcher, recipe),
        )
        self.store.record_health(run_id, report)
        if report.removed:
            self.store.mark_removed(site, report.removed)
        if report.verdict != "ok":
            self._emit("health_alert", site, report.summary(), verdict=report.verdict)
        return report

    def _downloadable(
        self,
        recipe: Recipe,
        run: RunResult,
        fetcher: Fetcher,
        years: Collection[int] | None,
    ) -> tuple[Recipe, RunResult, Reachability]:
        """Check that the listed documents can be fetched. A listing whose items carry
        several URLs (a storage path and a public link) may give one that answers 404:
        then the other paths are tried first, and the first variant whose documents are
        reachable is kept."""
        reach = self._reachability(run, fetcher, recipe)
        if reach.reachable >= reach.gone:
            return recipe, run, reach
        for variant in url_path_variants(recipe):
            rerun = run_recipe(variant, fetcher, years=years, today=self.today)
            if rerun.status != "ok" or not rerun.documents:
                continue
            other = self._reachability(rerun, fetcher, variant)
            if other.reachable > other.gone:
                log.info("documents unreachable with the first URL path; another path works")
                return variant, rerun, other
        return recipe, run, reach

    def _reachability(self, run: RunResult, fetcher: Fetcher, recipe: Recipe) -> Reachability:
        """Up to ``REACHABILITY_SAMPLE`` documents, the most recent first."""
        recent = sorted(run.documents, key=lambda d: d.fiscal_year or 0, reverse=True)
        sample = recent[:REACHABILITY_SAMPLE]
        allowed = [d for d in sample if fetcher.allows(d.url)]
        answers = [_answer(fetcher, recipe, d.url) for d in allowed]
        return Reachability(
            sampled=len(sample),
            reachable=answers.count("ok"),
            gone=answers.count("gone"),
            refused=len(sample) - len(allowed),
            blocked=answers.count("blocked"),
        )

    def _probe(self, fetcher: Fetcher, recipe: Recipe) -> Probe:
        """Whether a document is still online (a streamed GET, the body is not read)."""

        def probe(url: str) -> bool | None:
            answer = _answer(fetcher, recipe, url)
            return True if answer == "ok" else False if answer == "gone" else None

        return probe

    def _repair(
        self,
        site: str,
        recipe: Recipe,
        report: HealthReport,
        fetcher: Fetcher,
        years: Collection[int] | None,
        run_id: str,
        llm: StructuredLLM,
        ledger: CostLedger | None,
    ) -> RepairResult:
        status = self.store.site_status(site)
        used = self.store.llm_repairs_on(site, self.today)
        discoverer = self._discoverer
        if discoverer is None and self.settings.browser.enabled:
            discoverer = BrowserDiscovery(self.settings.browser, fetcher)
        ctx = RepairContext(
            site=site,
            today=self.today,
            years=years,
            version=self.store.next_version(recipe.recipe),
            start_url=status[1] if status else None,
            decisions=self.store.decisions(recipe.recipe),
            llm=llm,
            llm_attempts_left=max(0, MAX_LLM_REPAIRS_PER_DAY - used),
            ledger=ledger,
            discoverer=discoverer,
            probe=self._probe(fetcher, recipe),
        )
        result = Repairer(recipe, report, self.store.known_documents(site), fetcher, ctx).repair()
        for attempt in result.attempts:
            self.store.record_repair(
                site,
                run_id,
                self.today,
                rung=attempt.rung,
                outcome="passed" if attempt.passed else "failed",
                recipe_key=attempt.recipe,
                cost_usd=attempt.cost_usd,
                detail=attempt.detail[:1000],
            )
            if not attempt.passed and attempt.recipe is not None:
                self._emit("recipe_rejected", site, attempt.detail, rung=attempt.rung)
        if result.status == "repaired" and result.recipe is not None:
            path = self.store.save_recipe(result.recipe)
            self.store.activate(site, result.recipe, path)
            self.store.save_decisions(result.recipe.recipe, result.decisions)
            self.store.set_site(site, status="ok", detail=result.reason)
            self._emit(
                "recipe_repaired",
                site,
                f"{recipe.key} -> {result.recipe.key} ({result.reason})",
                previous=recipe.key,
                recipe=result.recipe.key,
            )
        else:
            self.store.set_site(site, status="degraded", detail=result.reason)
            self._emit("site_degraded", site, f"{recipe.key} stays active: {result.reason}")
        return result

    def _recipe(self, site: str) -> Recipe | None:
        if site.endswith((".yaml", ".yml")) and Path(site).exists():
            return load_recipe(site)
        return self.store.active_recipe(site)
