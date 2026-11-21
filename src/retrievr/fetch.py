"""HTTP fetching: proxy from environment, OS trust store, robots.txt, rate limit,
manual redirects (each hop checked), conditional GET and block detection.

Never calls an LLM.
"""

from __future__ import annotations

import logging
import re
import ssl
import time
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, Self

import httpx
import truststore
from protego import Protego

from retrievr.config import FetchSettings
from retrievr.urls import host_of, is_allowed, origin_of, resolve_url

log = logging.getLogger("retrievr.fetch")

BLOCK_STATUSES = frozenset({403, 429, 503})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
REQUEST_HEADERS = frozenset({"accept", "content-type", "x-requested-with"})
"""Headers a recipe may set on a request (nothing that carries identity or session)."""
Method = Literal["GET", "POST"]
ROBOTS_MAX_REDIRECTS = 5
_SMALL_PAGE_BYTES = 20_000

# Strong markers only appear on challenge or block pages.
_STRONG_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("challenge page (Cloudflare)", re.compile(r"<title>\s*Just a moment\.\.\.\s*</title>", re.I)),
    ("block page (Cloudflare)", re.compile(r"Attention Required! \| Cloudflare", re.I)),
    (
        "access denied page (Akamai)",
        re.compile(
            r"<title>\s*Access Denied\s*</title>[\s\S]{0,2000}Reference\s*(?:&#32;)?#", re.I
        ),
    ),
    ("block page (Imperva)", re.compile(r"Request unsuccessful\. Incapsula incident ID", re.I)),
)
# Weak markers are also injected into normal pages by CDNs and bot managers (for
# example every page behind Imperva loads "/_Incapsula_Resource"), so they only count
# on a small page, where they are the whole content.
_WEAK_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("challenge page (Cloudflare)", re.compile(r"/cdn-cgi/challenge-platform/|\bcf-chl-", re.I)),
    ("captcha page (DataDome)", re.compile(r"captcha-delivery\.com", re.I)),
    ("captcha page (PerimeterX)", re.compile(r"\bpx-captcha\b", re.I)),
    ("challenge page (Imperva)", re.compile(r"_Incapsula_Resource", re.I)),
    ("captcha page", re.compile(r"captcha", re.I)),
)


class FetchError(Exception):
    """Base class for fetch failures."""

    def __init__(self, message: str, *, url: str) -> None:
        super().__init__(message)
        self.url = url


class BlockedError(FetchError):
    """The site is blocking us (status 403/429/503, challenge or CAPTCHA page).

    Never retried, never worked around: the run stops with status ``blocked``.
    """

    def __init__(self, url: str, reason: str, status: int | None = None) -> None:
        super().__init__(f"blocked: {reason} at {url}", url=url)
        self.reason = reason
        self.status = status


class RobotsDisallowedError(FetchError):
    def __init__(self, url: str) -> None:
        super().__init__(f"robots.txt disallows {url}", url=url)


class DomainNotAllowedError(FetchError):
    def __init__(self, url: str) -> None:
        super().__init__(f"URL outside allowed domains: {url}", url=url)


class NetworkError(FetchError):
    """Connection, TLS, timeout or protocol failure (no usable HTTP response)."""

    def __init__(self, url: str, exc: Exception) -> None:
        super().__init__(f"network error at {url}: {type(exc).__name__}: {exc}", url=url)


class HTTPStatusError(FetchError):
    def __init__(self, url: str, status: int) -> None:
        super().__init__(f"HTTP {status} at {url}", url=url)
        self.status = status


@dataclass(frozen=True)
class Page:
    url: str
    """URL that was requested."""
    final_url: str
    """URL after redirects."""
    status: int
    content: bytes
    content_type: str | None
    encoding: str
    etag: str | None = None
    last_modified: str | None = None
    not_modified: bool = False

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding, errors="replace")


def detect_block(status: int, body: str | None = None) -> str | None:
    """Return a human-readable block reason, or None if the response looks normal."""
    if status in BLOCK_STATUSES:
        return f"HTTP {status}"
    if body is None:
        return None
    for reason, pattern in _STRONG_MARKERS:
        if pattern.search(body):
            return reason
    if len(body) < _SMALL_PAGE_BYTES:
        for reason, pattern in _WEAK_MARKERS:
            if pattern.search(body):
                return reason
    return None


class RateLimiter:
    """Minimum interval between two requests to the same host."""

    def __init__(
        self,
        rps: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 1.0 / rps
        self._clock = clock
        self._sleep = sleep
        self._next: dict[str, float] = {}

    def wait(self, host: str) -> None:
        now = self._clock()
        ready_at = self._next.get(host, now)
        if ready_at > now:
            self._sleep(ready_at - now)
            now = ready_at
        self._next[host] = now + self._interval


RobotsOutcome = Literal["parsed", "missing", "server_error", "unreachable", "ignored"]


@dataclass
class _RobotsEntry:
    outcome: RobotsOutcome
    parser: Protego | None = None
    allow_all: bool = True


@dataclass
class _RobotsCache:
    entries: dict[str, _RobotsEntry] = field(default_factory=dict)


class Fetcher:
    """Synchronous HTTP fetcher shared by the interpreter and the downloader."""

    def __init__(
        self,
        settings: FetchSettings | None = None,
        *,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        event_hooks: Mapping[str, list[Callable[..., Any]]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """``transport`` replaces the network (e.g. a snapshot replay); environment
        proxies are then ignored. ``event_hooks`` are httpx hooks (e.g. a recorder)."""
        self.settings = settings or FetchSettings()
        self._client = client or _default_client(
            self.settings, transport=transport, event_hooks=event_hooks
        )
        self._owns_client = client is None
        self._limiter = RateLimiter(self.settings.rate_limit_rps, clock=clock, sleep=sleep)
        self._robots = _RobotsCache()
        self._ua_token = self.settings.user_agent.split("/", 1)[0].strip() or "retrievr"
        if self.settings.robots == "ignore":
            log.warning("robots.txt checks are DISABLED by configuration (robots = 'ignore')")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- public API ---------------------------------------------------------

    def get(
        self,
        url: str,
        *,
        allowed_domains: Collection[str] | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> Page:
        """GET a page (HTML/JSON) into memory. Raises on block, robots, domain or HTTP error."""
        return self.request(
            "GET", url, allowed_domains=allowed_domains, etag=etag, last_modified=last_modified
        )

    def request(
        self,
        method: Method,
        url: str,
        *,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        allowed_domains: Collection[str] | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> Page:
        """GET or POST a page (HTML/JSON) into memory, with the same checks as ``get``:
        robots.txt, allowed domains, rate limit, block detection. Only the headers in
        ``REQUEST_HEADERS`` can be set by the caller."""
        extra = {k: v for k, v in (headers or {}).items() if k.lower() in REQUEST_HEADERS}
        extra.update(_conditional_headers(etag, last_modified))
        with self._open(
            url, allowed_domains=allowed_domains, headers=extra, method=method, content=content
        ) as resp:
            if resp.status_code == 304:
                return Page(
                    url=url,
                    final_url=str(resp.url),
                    status=304,
                    content=b"",
                    content_type=None,
                    encoding="utf-8",
                    etag=etag,
                    last_modified=last_modified,
                    not_modified=True,
                )
            content = self._read_capped(resp)
            encoding = resp.charset_encoding or "utf-8"
            content_type = resp.headers.get("content-type")
            if content_type is None or "html" in content_type or "text" in content_type:
                reason = detect_block(resp.status_code, content.decode(encoding, errors="replace"))
                if reason:
                    raise BlockedError(str(resp.url), reason, resp.status_code)
            return Page(
                url=url,
                final_url=str(resp.url),
                status=resp.status_code,
                content=content,
                content_type=content_type,
                encoding=encoding,
                etag=resp.headers.get("etag"),
                last_modified=resp.headers.get("last-modified"),
            )

    @contextmanager
    def stream(
        self, url: str, *, allowed_domains: Collection[str] | None = None
    ) -> Iterator[httpx.Response]:
        """Open a streamed GET (for downloads). The body is not read yet."""
        with self._open(url, allowed_domains=allowed_domains, headers={}) as resp:
            yield resp

    def allows(self, url: str, allowed_domains: Collection[str] | None = None) -> bool:
        """Whether ``url`` may be fetched (robots.txt and allowed domains), without
        fetching it. Used to vet the requests a browser page makes."""
        if allowed_domains is not None and not is_allowed(url, set(allowed_domains)):
            return False
        return self._robots_allows(url)

    def robots_outcome(self, url: str) -> RobotsOutcome:
        """How robots.txt was resolved for the URL's origin (for logs and tests)."""
        return self._robots_entry(origin_of(url)).outcome

    # -- internals ----------------------------------------------------------

    @contextmanager
    def _open(
        self,
        url: str,
        *,
        allowed_domains: Collection[str] | None,
        headers: dict[str, str],
        method: Method = "GET",
        content: bytes | None = None,
    ) -> Iterator[httpx.Response]:
        current = url
        for _hop in range(self.settings.max_redirects + 1):
            self._check_url(current, allowed_domains)
            self._limiter.wait(host_of(current))
            request = self._client.build_request(
                method,
                current,
                headers={"User-Agent": self.settings.user_agent, **headers},
                content=content,
            )
            try:
                resp = self._client.send(request, stream=True)
            except httpx.RequestError as exc:
                raise NetworkError(current, exc) from exc
            try:
                if resp.status_code in REDIRECT_STATUSES and "location" in resp.headers:
                    current = resolve_url(resp.headers["location"], current)
                    if resp.status_code not in (307, 308):
                        method, content = "GET", None  # 301/302/303 turn a POST into a GET
                    continue
                if resp.status_code in BLOCK_STATUSES:
                    raise BlockedError(current, f"HTTP {resp.status_code}", resp.status_code)
                if resp.status_code >= 400:
                    raise HTTPStatusError(current, resp.status_code)
                yield resp
                return
            finally:
                resp.close()
        raise FetchError(f"too many redirects (>{self.settings.max_redirects})", url=url)

    def _check_url(self, url: str, allowed_domains: Collection[str] | None) -> None:
        if allowed_domains is not None and not is_allowed(url, set(allowed_domains)):
            raise DomainNotAllowedError(url)
        if not self._robots_allows(url):
            raise RobotsDisallowedError(url)

    def _read_capped(self, resp: httpx.Response) -> bytes:
        limit = self.settings.max_download_bytes
        chunks: list[bytes] = []
        size = 0
        try:
            for chunk in resp.iter_bytes():
                size += len(chunk)
                if size > limit:
                    raise FetchError(f"response larger than {limit} bytes", url=str(resp.url))
                chunks.append(chunk)
        except httpx.RequestError as exc:
            raise NetworkError(str(resp.url), exc) from exc
        return b"".join(chunks)

    def _robots_allows(self, url: str) -> bool:
        if self.settings.robots == "ignore":
            return True
        entry = self._robots_entry(origin_of(url))
        if entry.parser is not None:
            return bool(entry.parser.can_fetch(url, self._ua_token))
        return entry.allow_all

    def _robots_entry(self, origin: str) -> _RobotsEntry:
        if self.settings.robots == "ignore":
            return _RobotsEntry("ignored")
        cached = self._robots.entries.get(origin)
        if cached is None:
            cached = self._load_robots(origin)
            self._robots.entries[origin] = cached
        return cached

    def _load_robots(self, origin: str) -> _RobotsEntry:
        url = origin + "/robots.txt"
        try:
            for _hop in range(ROBOTS_MAX_REDIRECTS + 1):
                self._limiter.wait(host_of(url))
                resp = self._client.get(url, headers={"User-Agent": self.settings.user_agent})
                if resp.status_code in REDIRECT_STATUSES and "location" in resp.headers:
                    url = resolve_url(resp.headers["location"], url)
                    continue
                return self._robots_from_response(origin, resp)
            reason = f"more than {ROBOTS_MAX_REDIRECTS} redirects"
        except httpx.HTTPError as exc:
            reason = f"{type(exc).__name__}: {exc}"
        allow = self.settings.robots_unreachable == "allow"
        log.warning(
            "robots.txt unreachable for %s (%s); policy robots_unreachable=%s -> %s",
            origin,
            reason,
            self.settings.robots_unreachable,
            "allow all" if allow else "deny all",
        )
        return _RobotsEntry("unreachable", allow_all=allow)

    def _robots_from_response(self, origin: str, resp: httpx.Response) -> _RobotsEntry:
        status = resp.status_code
        if 200 <= status < 300:
            return _RobotsEntry("parsed", parser=Protego.parse(resp.text))
        if 400 <= status < 500:
            log.info("no robots.txt for %s (HTTP %d): allow all", origin, status)
            return _RobotsEntry("missing", allow_all=True)
        log.warning("robots.txt server error for %s (HTTP %d): deny all", origin, status)
        return _RobotsEntry("server_error", allow_all=False)


def _default_client(
    settings: FetchSettings,
    *,
    transport: httpx.BaseTransport | None = None,
    event_hooks: Mapping[str, list[Callable[..., Any]]] | None = None,
) -> httpx.Client:
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return httpx.Client(
        verify=ctx,
        # HTTP(S)_PROXY, ALL_PROXY, NO_PROXY; never with a custom transport, where an
        # environment proxy would silently route around it.
        trust_env=transport is None,
        transport=transport,
        event_hooks=dict(event_hooks or {}),
        timeout=settings.timeout_s,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=False,
    )


def _conditional_headers(etag: str | None, last_modified: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    return headers
