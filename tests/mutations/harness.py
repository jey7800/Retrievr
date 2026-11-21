"""Breakages applied to the snapshots of real sites, to test health checks and repair.

A :class:`MutatingTransport` sits in front of the snapshot replay and changes the site the
way sites change: renamed classes and ids, an extra wrapper, reordered table columns, a
renamed URL parameter, pagination turned into a "load more" link, the listing moved into
a JSON API called by the page's JavaScript. For the last one, :class:`ApiDiscovery` plays
the browser: it reports the calls the page would make.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from urllib.parse import urlsplit

import httpx
from lxml import html as lxml_html
from lxml.html import HtmlElement

from retrievr.browser import Discovery, Exchange
from retrievr.candidates import extract_candidates, parse_html, select
from retrievr.recipe import Recipe, RequestSource, UrlTemplateSource
from retrievr.snapshots import ReplayTransport, canonical_body
from retrievr.urls import normalize_url, resolve_url

API_PATH = "/api/documents"
_TOKEN = re.compile(r"[.#]([A-Za-z_][A-Za-z0-9_-]*)")


class Mutation:
    """Identity: subclasses change requests, responses or add endpoints."""

    name = "identity"
    repairable = True
    """False when no recipe can list the documents any more: the site must end degraded."""

    def __init__(self, recipe: Recipe | None = None) -> None:
        self.recipe_ = recipe

    def applies(self, recipe: Recipe) -> bool:
        return True

    def map_request(self, request: httpx.Request, site: MutatedSite) -> httpx.Request:
        return request

    def serve(self, request: httpx.Request, site: MutatedSite) -> httpx.Response | None:
        return None

    def html(self, url: str, root: HtmlElement) -> None:
        """Change a parsed HTML page in place."""

    def text(self, url: str, text: str) -> str:
        """Change the raw text of an HTML or JSON response."""
        return text

    def discover(self, url: str, years: Collection[int] | None, site: MutatedSite) -> Discovery:
        return Discovery(start_url=url, notes=["no call recorded"])


@dataclass
class MutatedSite:
    """The snapshot of a site seen through a mutation."""

    replay: ReplayTransport
    recipe: Recipe
    mutation: Mutation
    today: date

    def transport(self) -> MutatingTransport:
        return MutatingTransport(self)

    def original(self, url: str) -> bytes | None:
        """A page of the unmutated snapshot."""
        response = self.replay.handle_request(httpx.Request("GET", url))
        return response.content if response.status_code == 200 else None


class MutatingTransport(httpx.BaseTransport):
    def __init__(self, site: MutatedSite) -> None:
        self.site = site
        self.requests: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        mutation = self.site.mutation
        self.requests.append(f"{request.method} {request.url}")
        served = mutation.serve(request, self.site)
        if served is not None:
            return served
        response = self.site.replay.handle_request(mutation.map_request(request, self.site))
        kind = response.headers.get("content-type", "")
        if response.status_code != 200 or not ("html" in kind or "json" in kind):
            return response
        text = response.content.decode("utf-8", errors="replace")
        url = str(request.url)
        if "html" in kind and mutation.html.__func__ is not Mutation.html:  # type: ignore[attr-defined]
            root = parse_html(text)
            mutation.html(url, root)
            text = lxml_html.tostring(root, encoding="unicode")
        elif "json" in kind and mutation.html.__func__ is not Mutation.html:  # type: ignore[attr-defined]
            text = json.dumps(
                _map_html_strings(json.loads(text), lambda f: _apply(mutation, url, f))
            )
        text = mutation.text(url, text)
        headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
        return httpx.Response(200, headers=headers, content=text.encode("utf-8"), request=request)


def _apply(mutation: Mutation, url: str, fragment: str) -> str:
    root = lxml_html.fragment_fromstring(fragment, create_parent="div")
    mutation.html(url, root)
    return "".join(lxml_html.tostring(child, encoding="unicode") for child in root) or (
        root.text or ""
    )


def _map_html_strings(node: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(node, dict):
        return {k: _map_html_strings(v, fn) for k, v in node.items()}
    if isinstance(node, list):
        return [_map_html_strings(v, fn) for v in node]
    if isinstance(node, str) and "<" in node and ">" in node:
        return fn(node)
    return node


def recipe_tokens(recipe: Recipe) -> list[str]:
    """Class and id names the recipe's selectors rely on."""
    selectors: list[str] = []
    for source in recipe.sources:
        items = source.items
        selectors += [s for s in (items.container, items.item, items.title, items.link) if s]
        if isinstance(source, UrlTemplateSource) and source.pagination:
            selectors.append(source.pagination.selector)
        if isinstance(source, RequestSource):
            selectors += [p.selector for p in source.params.values() if hasattr(p, "selector")]
    tokens: list[str] = []
    for selector in selectors:
        for token in _TOKEN.findall(selector):
            if token not in tokens:
                tokens.append(token)
    return tokens


def renamed(token: str) -> str:
    digest = hashlib.sha256(token.encode()).hexdigest()
    letters = "".join(chr(ord("a") + int(c, 16) % 26) for c in digest[:8])
    return f"{letters[:4]}-{letters[4:]}"


class RenameClasses(Mutation):
    """Every class and id the recipe relies on gets a new name (a site redesign)."""

    name = "rename_classes"

    def __init__(self, recipe: Recipe) -> None:
        self.mapping = {t: renamed(t) for t in recipe_tokens(recipe)}

    def applies(self, recipe: Recipe) -> bool:
        return bool(self.mapping)

    def text(self, url: str, text: str) -> str:
        for old, new in self.mapping.items():
            text = re.sub(rf"(?<![\w-]){re.escape(old)}(?![\w-])", new, text)
        return text


class ExtraWrapper(Mutation):
    """Each listing entry and each listing container gets an extra ``<div>`` around its
    content."""

    name = "extra_wrapper"

    def __init__(self, recipe: Recipe) -> None:
        self.selectors = [
            s for src in recipe.sources for s in (src.items.container, src.items.item) if s
        ]
        if not self.selectors:
            self.selectors = ["table", "main"]

    def html(self, url: str, root: HtmlElement) -> None:
        for selector in self.selectors:
            for el in select(root, selector):
                wrapper = lxml_html.Element("div", {"class": "inner-wrap"})
                wrapper.text, el.text = el.text, None
                for child in list(el):
                    wrapper.append(child)
                el.append(wrapper)


class ReorderColumns(Mutation):
    """Table columns in another order (the first column, the row header, stays first)."""

    name = "reorder_columns"

    def applies(self, recipe: Recipe) -> bool:
        return any(s.items.table_context for s in recipe.sources)

    def html(self, url: str, root: HtmlElement) -> None:
        for row in root.iter("tr"):
            cells = [c for c in row if c.tag in ("td", "th")]
            if len(cells) < 3:
                continue
            tail = cells[1:]
            for cell in tail:
                row.remove(cell)
            for cell in reversed(tail):
                row.append(cell)


class RenameUrlParameter(Mutation):
    """The year filter of the listing URLs is renamed; the old name is ignored by the
    site, which then shows its default (latest) listing."""

    name = "rename_url_param"

    def __init__(self, recipe: Recipe, old: str = "fiscal", new: str = "annee") -> None:
        self.old, self.new = old, new
        self.source = next(
            (
                s
                for s in recipe.sources
                if isinstance(s, UrlTemplateSource) and f"{old}[{{year}}]" in s.url
            ),
            None,
        )

    def applies(self, recipe: Recipe) -> bool:
        return self.source is not None

    def map_request(self, request: httpx.Request, site: MutatedSite) -> httpx.Request:
        url = str(request.url)
        if re.search(rf"{self.old}(%5B|\[)\d{{4}}", url):
            # An unknown parameter: the site shows its default listing (the last year).
            default = re.sub(r"(%5B|\[)\d{4}", rf"\g<1>{site.today.year - 1}", url)
            return httpx.Request("GET", default)
        return httpx.Request("GET", re.sub(rf"{self.new}(%5B|\[)", rf"{self.old}\1", url))

    def text(self, url: str, text: str) -> str:
        return re.sub(rf"{self.old}(?=%5B|\[|-\d{{4}})", self.new, text)


class LoadMore(Mutation):
    """The pager becomes a single "load more" link to the next page."""

    name = "load_more"

    def __init__(self, recipe: Recipe) -> None:
        self.selector = next(
            (
                s.pagination.selector
                for s in recipe.sources
                if isinstance(s, UrlTemplateSource) and s.pagination
            ),
            None,
        )

    def applies(self, recipe: Recipe) -> bool:
        return self.selector is not None

    def html(self, url: str, root: HtmlElement) -> None:
        assert self.selector is not None
        links = select(root, self.selector)
        href = links[0].get("href") if links else None
        pager = links[0] if links else None
        for el in [*(links[0].iterancestors() if links else [])]:
            if el.tag == "nav":
                pager = el
                break
        if pager is None:
            for nav in root.iter("nav"):
                if "pager" in (nav.get("class") or "") or "pagination" in (
                    nav.get("aria-label") or ""
                ):
                    pager = nav
                    break
        if pager is None:
            return
        replacement = lxml_html.Element("div", {"class": "load-more"})
        if href:
            link = lxml_html.Element("a", {"class": "button-more", "href": href})
            link.text = "Voir plus de documents"
            replacement.append(link)
        parent = pager.getparent()
        if parent is not None:
            parent.replace(pager, replacement)


@dataclass
class ApiListing:
    """The JSON items of one listing page of the original site."""

    items: list[dict[str, Any]] = field(default_factory=list)
    has_more: bool = False


class JsonApi(Mutation):
    """The listing leaves the HTML: the page shows an empty shell and its JavaScript
    POSTs ``{"year": ..., "page": ...}`` to a JSON API that returns the documents."""

    name = "json_list"

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.source = next((s for s in recipe.sources if isinstance(s, UrlTemplateSource)), None)

    def applies(self, recipe: Recipe) -> bool:
        return self.source is not None

    def endpoint(self) -> str:
        return f"https://{self.recipe.site}{API_PATH}"

    # -- the page ----------------------------------------------------------------------

    def html(self, url: str, root: HtmlElement) -> None:
        assert self.source is not None
        items = self.source.items
        targets = select(root, items.container) if items.container else []
        if not targets:
            targets = [el for el in select(root, "table") if select(el, items.link)]
        if not targets and select(root, "main"):
            targets = select(root, "main")
        shell = lxml_html.Element(
            "div", {"id": "documents-app", "data-api": API_PATH, "class": "js-documents"}
        )
        shell.text = "Chargement..."
        if self.source.params:
            # The year selector the page's JavaScript reads.
            choice = lxml_html.Element("select", {"id": "year", "name": "year"})
            for year in sorted(self.source.params["year"].values(date.today()), reverse=True):
                option = lxml_html.Element("option", {"value": str(year)})
                option.text = str(year)
                choice.append(option)
            shell.insert(0, choice)
        for i, el in enumerate(targets):
            parent = el.getparent()
            if parent is None:
                continue
            if i == 0:
                parent.replace(el, shell)
            else:
                parent.remove(el)
        pagination = self.source.pagination
        if pagination is not None:
            for link in select(root, pagination.selector):
                for anc in link.iterancestors():
                    if anc.tag == "nav":
                        if anc.getparent() is not None:
                            anc.getparent().remove(anc)
                        break

    # -- the API -----------------------------------------------------------------------

    def serve(self, request: httpx.Request, site: MutatedSite) -> httpx.Response | None:
        if request.method != "POST" or urlsplit(str(request.url)).path != API_PATH:
            return None
        try:
            body = json.loads(request.content or b"{}")
        except ValueError:
            return httpx.Response(400, request=request)
        listing = self.listing(site, body.get("year"), int(body.get("page") or 0))
        payload = {"data": {"documents": listing.items}, "hasMore": listing.has_more}
        return httpx.Response(
            200,
            headers={"content-type": "application/json; charset=utf-8"},
            content=json.dumps(payload).encode(),
            request=request,
        )

    def listing(self, site: MutatedSite, year: Any, page: int) -> ApiListing:
        source = self.source
        assert source is not None
        params = {"year": int(year)} if source.params and str(year).isdigit() else {}
        if source.params and not params:
            return ApiListing()
        url: str | None = next((u for p, u in source.expand(site.today) if p == params), None)
        for _ in range(page + 1):
            if url is None:
                return ApiListing()
            content = site.original(url)
            if content is None:
                return ApiListing()
            root = parse_html(content)
            current, url = url, self._next(root, url)
        out = ApiListing(has_more=url is not None)
        items = source.items
        for cand in extract_candidates(
            root,
            current,
            container=items.container,
            item=items.item,
            link=items.link,
            title=items.title,
            extensions=items.extensions,
            table_context=items.table_context,
        ):
            when = items.date.parse(cand.context_text) if items.date else None
            out.items.append(
                {
                    "title": cand.title or cand.link_text,
                    "file": {"url": cand.url, "label": cand.link_text},
                    "publishedAt": when.isoformat() if when else None,
                    "section": cand.heading,
                    "row": cand.row_header,
                    "column": cand.column_header,
                }
            )
        return out

    def _next(self, root: HtmlElement, url: str) -> str | None:
        assert self.source is not None
        if self.source.pagination is None:
            return None
        for el in select(root, self.source.pagination.selector):
            if el.get("href"):
                return resolve_url(el.get("href"), url)
        return None

    # -- what a browser would see ------------------------------------------------------

    def discover(self, url: str, years: Collection[int] | None, site: MutatedSite) -> Discovery:
        assert self.source is not None
        year = None
        if self.source.params:
            for params, page_url in self.source.expand(site.today):
                if normalize_url(page_url) == normalize_url(url):
                    year = params["year"]
        calls: list[tuple[str, dict[str, Any]]] = [("load", {"year": year, "page": 0})]
        if self.listing(site, year, 0).has_more:
            calls.append(('click "Voir plus"', {"year": year, "page": 1}))
        for other in sorted(set(years or ()) - {year} if year else ()):
            calls.append((f'select #year = "{other}"', {"year": other, "page": 0}))
        discovery = Discovery(start_url=url, actions=[a for a, _ in calls])
        for i, (action, body) in enumerate(calls, start=1):
            content = json.dumps(body).encode()
            request = httpx.Request(
                "POST",
                self.endpoint(),
                content=content,
                headers={"content-type": "application/json"},
            )
            response = self.serve(request, site)
            assert response is not None
            discovery.exchanges.append(
                Exchange(
                    id=f"X{i}",
                    action=action,
                    method="POST",
                    url=self.endpoint(),
                    request_body=canonical_body(content, "application/json"),
                    request_content_type="application/json",
                    requested_with=None,
                    status=response.status_code,
                    content_type="application/json",
                    body=response.content,
                )
            )
        return discovery


class EmptyListing(Mutation):
    """The listing pages no longer list the documents (the files are still online): no
    recipe can find them, so a repair must not be activated."""

    name = "empty_listing"
    repairable = False

    def __init__(self, recipe: Recipe) -> None:
        self.containers = [
            s.items.container
            for s in recipe.sources
            if isinstance(s, UrlTemplateSource) and s.items.container
        ]

    def applies(self, recipe: Recipe) -> bool:
        return bool(self.containers)

    def html(self, url: str, root: HtmlElement) -> None:
        for selector in self.containers:
            for el in select(root, selector):
                for child in list(el):
                    el.remove(child)
                el.text = "Aucun document."


class ApiDiscovery:
    """Plays the browser on a mutated site: reports the calls its pages make."""

    def __init__(self, site: MutatedSite) -> None:
        self.site = site
        self.calls: list[str] = []

    def discover(self, url: str, years: Collection[int] | None) -> Discovery:
        self.calls.append(url)
        return self.site.mutation.discover(url, years, self.site)


MUTATIONS: list[Callable[[Recipe], Mutation]] = [
    RenameClasses,
    ExtraWrapper,
    ReorderColumns,
    RenameUrlParameter,
    LoadMore,
    JsonApi,
    EmptyListing,
]


def mutations_for(recipe: Recipe) -> Iterator[Mutation]:
    for factory in MUTATIONS:
        mutation = factory(recipe)
        if mutation.applies(recipe):
            yield mutation
