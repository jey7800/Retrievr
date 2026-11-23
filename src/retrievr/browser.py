"""Optional browser (extra ``retrievr[browser]``) for pages that load their listing with
JavaScript.

Used on cold paths only: the engine opens the page, performs a few deterministic
actions (pick years in year selectors, press "load more" once) and passively records
the XHR/fetch exchanges they trigger. The learner then turns one of those endpoints into
a ``request`` source that later runs replay with plain httpx, without a browser.

Respectful by construction: the same identifiable User-Agent, robots.txt checked for
every page and request of the site (through the ``Fetcher``), images, fonts and media
never loaded, requests of other sites limited to scripts and stylesheets. No stealth,
no evasion, no CAPTCHA solving: a challenge page stops the discovery.

A discovery can be recorded into a snapshot (``discovery/<page>.yaml`` plus the response
bodies) and replayed offline by :class:`ReplayDiscovery`, without any browser.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

from retrievr.candidates import in_site, label_year, site_domain
from retrievr.config import BrowserSettings
from retrievr.fetch import Fetcher, detect_block
from retrievr.snapshots import ReplayTransport, SnapshotRecorder, canonical_body, is_text_type
from retrievr.urls import host_of, normalize_url

log = logging.getLogger("retrievr.browser")

DISCOVERY_DIR = "discovery"
MAX_BODY_BYTES = 5 * 1024 * 1024
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_LOAD_MORE = re.compile(
    r"^\s*(load|show|view|see|find) more|^\s*more (results|files|documents|reports)|"
    r"voir plus|afficher plus|plus de (r[ée]sultats|documents)|charger plus|"
    r"mehr (anzeigen|laden)",
    re.IGNORECASE,
)
_SKIPPED_TYPES = {"image", "media", "font"}
_OTHER_SITE_TYPES = {"script", "stylesheet"}


class BrowserUnavailableError(RuntimeError):
    """Playwright (extra ``retrievr[browser]``) or the configured browser is missing."""


class BrowserBlockedError(RuntimeError):
    """The page answered with a block or challenge."""


@dataclass(frozen=True)
class Exchange:
    """One XHR/fetch request of the page and its response."""

    id: str
    action: str
    """What triggered it, e.g. ``load`` or ``select #year = "2024" (value 171628)``."""
    method: str
    url: str
    request_body: str | None
    """Canonical body (``snapshots.canonical_body``)."""
    request_content_type: str | None
    requested_with: str | None
    """``X-Requested-With`` header, which some endpoints require."""
    status: int
    content_type: str | None
    body: bytes = field(repr=False)

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


@dataclass
class Discovery:
    start_url: str
    actions: list[str] = field(default_factory=list)
    exchanges: list[Exchange] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    """Document links of the page as rendered (after the actions): they tell how the
    page turns a file name of a JSON listing into a download URL."""


class Discoverer(Protocol):
    def discover(self, url: str, years: Collection[int] | None) -> Discovery: ...


def discovery_file(directory: Path, url: str) -> Path:
    digest = hashlib.sha256(normalize_url(url).encode()).hexdigest()[:12]
    return directory / DISCOVERY_DIR / f"{digest}.yaml"


def _target_options(
    options: list[tuple[str, str]], years: Collection[int] | None, limit: int = 3
) -> list[tuple[str, str]]:
    """Options to pick in a year selector: the requested years and the next one (a
    bucket of publication year N+1 can hold fiscal year N), else the first two."""
    with_year = [(v, t, y) for v, t in options if (y := label_year(t)) and v]
    if len(with_year) < 2:
        return []
    if years:
        # The requested years first, then the next one.
        chosen = [(v, t) for v, t, y in with_year if y in set(years)]
        chosen += [(v, t) for v, t, y in with_year if y == max(years) + 1]
        if chosen:
            return chosen[:limit]
    return [(v, t) for v, t, _y in with_year[:2]]


class BrowserDiscovery:
    """Live discovery with Playwright."""

    def __init__(
        self,
        settings: BrowserSettings,
        fetcher: Fetcher,
        *,
        recorder: SnapshotRecorder | None = None,
    ) -> None:
        self.settings = settings
        self.fetcher = fetcher
        self.recorder = recorder

    def discover(self, url: str, years: Collection[int] | None) -> Discovery:
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise BrowserUnavailableError(
                "the browser extra is not installed: pip install 'retrievr[browser]'"
            ) from exc
        if not self.fetcher.allows(url):
            raise BrowserBlockedError(f"robots.txt disallows {url}")
        domain = site_domain(host_of(url))
        discovery = Discovery(start_url=url)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    channel=self.settings.channel or None, headless=self.settings.headless
                )
                try:
                    context = browser.new_context(user_agent=self.fetcher.settings.user_agent)
                    page = context.new_page()
                    page.set_default_timeout(self.settings.timeout_s * 1000)
                    self._run(page, url, domain, years, discovery)
                finally:
                    browser.close()
        except PlaywrightError as exc:
            raise BrowserUnavailableError(f"browser failed: {exc}") from exc
        if self.recorder is not None:
            self._record(discovery, years)
        return discovery

    # -- page driving --------------------------------------------------------------

    def _run(
        self,
        page: Any,
        url: str,
        domain: str,
        years: Collection[int] | None,
        discovery: Discovery,
    ) -> None:
        state = {"action": "load"}
        refused: list[str] = []
        try:
            self._drive(page, url, domain, years, discovery, state, refused)
        finally:
            if refused:
                discovery.notes.append(
                    "calls of the page refused by robots.txt: " + ", ".join(refused)
                )

    def _drive(
        self,
        page: Any,
        url: str,
        domain: str,
        years: Collection[int] | None,
        discovery: Discovery,
        state: dict[str, str],
        refused: list[str],
    ) -> None:
        def route(route: Any) -> None:
            request = route.request
            kind = request.resource_type
            host = host_of(request.url)
            if kind in _SKIPPED_TYPES:
                route.abort()
            elif in_site(host, domain):
                if self.fetcher.allows(request.url):
                    route.continue_()
                else:
                    route.abort()
                    if kind in ("xhr", "fetch") and len(refused) < 3:
                        refused.append(request.url.split("?", 1)[0])
            elif kind in _OTHER_SITE_TYPES:
                route.continue_()
            elif kind in ("xhr", "fetch") and request.method == "GET":
                # A listing API of another host (an IR content provider): read-only
                # calls, robots.txt checked. Other-site POSTs (beacons, telemetry)
                # stay blocked.
                route.continue_() if self.fetcher.allows(request.url) else route.abort()
            else:
                route.abort()

        def on_response(response: Any) -> None:
            request = response.request
            if request.resource_type not in ("xhr", "fetch"):
                return
            if not in_site(host_of(request.url), domain) and request.method != "GET":
                return
            try:
                body = response.body()[:MAX_BODY_BYTES]
            except Exception:  # a body that cannot be read is skipped
                return
            headers = request.headers
            content_type = headers.get("content-type")
            exchange = Exchange(
                id=f"X{len(discovery.exchanges) + 1}",
                action=state["action"],
                method=request.method,
                url=request.url,
                request_body=canonical_body(request.post_data_buffer, content_type),
                request_content_type=content_type,
                requested_with=headers.get("x-requested-with"),
                status=response.status,
                content_type=response.headers.get("content-type"),
                body=body,
            )
            discovery.exchanges.append(exchange)

        page.route("**/*", route)
        page.on("response", on_response)
        response = page.goto(url, wait_until="load")
        page.wait_for_timeout(self.settings.settle_ms)
        status = response.status if response is not None else 0
        if reason := detect_block(status, page.content()[:200_000]):
            raise BrowserBlockedError(f"{url}: {reason}")
        discovery.actions.append("load")
        # Lists rendered lazily (when scrolled into view) need a scroll to exist at all.
        state["action"] = "scroll"
        for _ in range(2):
            page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(self.settings.settle_ms)
        discovery.actions.append("scroll")

        budget = self.settings.max_actions
        for select in self._year_selects(page):
            for value, label in _target_options(select["options"], years):
                if budget <= 0:
                    break
                budget -= 1
                state["action"] = f'select {select["css"]} = "{label}" (value {value})'
                page.evaluate(_CHOOSE_JS, [select["index"], value])
                page.wait_for_timeout(self.settings.settle_ms)
                discovery.actions.append(state["action"])
        for index, label in self._load_more(page)[:2]:
            if budget <= 0:
                break
            budget -= 1
            state["action"] = f'click "{label}"'
            page.evaluate(_CLICK_JS, index)
            page.wait_for_timeout(self.settings.settle_ms)
            discovery.actions.append(state["action"])
        state["action"] = "done"
        discovery.links = [str(h) for h in page.evaluate(_LINKS_JS)][:MAX_RENDERED_LINKS]

    def _year_selects(self, page: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = page.evaluate(_SELECTS_JS)
        return [s for s in found if _target_options([tuple(o) for o in s["options"]], None)]

    def _load_more(self, page: Any) -> list[tuple[int, str]]:
        found: list[list[Any]] = page.evaluate(_BUTTONS_JS)
        return [(int(i), str(t)) for i, t in found if _LOAD_MORE.search(str(t))]

    def _record(self, discovery: Discovery, years: Collection[int] | None) -> None:
        from retrievr.endpoints import analyse  # local: endpoints imports this module

        recorder = self.recorder
        assert recorder is not None  # noqa: S101
        # Keep only the calls that list documents: the others (analytics, anti-bot
        # telemetry about the capturing machine, quotes) are useless and must not be
        # written into a snapshot that may be shared.
        kept = [ex for ex in discovery.exchanges if analyse(ex).kind != "none"]
        discovery = Discovery(
            discovery.start_url, discovery.actions, kept, discovery.notes, discovery.links
        )
        for ex in discovery.exchanges:
            recorder.record(
                method=ex.method,
                url=ex.url,
                request_body=ex.request_body,
                status=ex.status,
                headers={"content-type": ex.content_type or ""},
                body=ex.body if is_text_type(ex.content_type) else None,
            )
        recorder.save()
        save_discovery(recorder.directory, discovery, years)


_SELECTS_JS = """() => Array.from(document.querySelectorAll('select')).map((s, index) => ({
  index,
  css: s.id ? '#' + s.id : (s.name ? `select[name="${s.name}"]` : 'select'),
  options: Array.from(s.options).map(o => [o.value, (o.textContent || '').trim()]),
}))"""

_CHOOSE_JS = """([index, value]) => {
  const s = document.querySelectorAll('select')[index];
  s.value = value;
  s.dispatchEvent(new Event('input', {bubbles: true}));
  s.dispatchEvent(new Event('change', {bubbles: true}));
  if (window.jQuery) { window.jQuery(s).trigger('change'); }
}"""

_BUTTONS_JS = """() => Array.from(document.querySelectorAll('button, a[href="#"], a:not([href])'))
  .map((b, i) => [i, (b.innerText || b.textContent || '').trim().slice(0, 60)])"""

MAX_RENDERED_LINKS = 500
_LINKS_JS = """() => Array.from(document.querySelectorAll('a[href]'))
  .map(a => a.href)
  .filter(h => /\\.(pdf|xlsx?|xlsb|xlsm|docx?|pptx?|zip|xhtml|csv)\\b/i.test(h))"""

_CLICK_JS = """(index) => {
  const b = document.querySelectorAll('button, a[href="#"], a:not([href])')[index];
  if (b) { b.click(); }
}"""


def save_discovery(directory: Path, discovery: Discovery, years: Collection[int] | None) -> Path:
    path = discovery_file(directory, discovery.start_url)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "discovery_version": 1,
        "start_url": discovery.start_url,
        "years": sorted(years) if years else None,
        "captured_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "actions": discovery.actions,
        "links": discovery.links,
        "exchanges": [
            {
                "id": ex.id,
                "action": ex.action,
                "method": ex.method,
                "url": ex.url,
                "request_body": ex.request_body,
                "request_content_type": ex.request_content_type,
                "requested_with": ex.requested_with,
                "status": ex.status,
                "content_type": ex.content_type,
            }
            for ex in discovery.exchanges
        ],
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8"
    )
    return path


class ReplayDiscovery:
    """A recorded discovery, served from a snapshot (no browser, no network)."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.transport = ReplayTransport(self.directory)

    def available(self, url: str) -> bool:
        return discovery_file(self.directory, url).exists()

    def discover(self, url: str, years: Collection[int] | None) -> Discovery:
        path = discovery_file(self.directory, url)
        if not path.exists():
            return Discovery(start_url=url, notes=["no recorded discovery for this page"])
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        discovery = Discovery(
            start_url=url,
            actions=list(data.get("actions") or []),
            links=[str(h) for h in data.get("links") or []],
        )
        for raw in data.get("exchanges") or []:
            entry = self.transport.find(raw["method"], raw["url"], raw.get("request_body"))
            body = self.transport.body_of(entry) if entry is not None else b""
            discovery.exchanges.append(
                Exchange(
                    id=raw["id"],
                    action=raw["action"],
                    method=raw["method"],
                    url=raw["url"],
                    request_body=raw.get("request_body"),
                    request_content_type=raw.get("request_content_type"),
                    requested_with=raw.get("requested_with"),
                    status=int(raw["status"]),
                    content_type=raw.get("content_type"),
                    body=body,
                )
            )
        return discovery
