"""Candidate extraction: document links plus their context, from an HTML page.

Deterministic, never calls an LLM. HTML pruning for prompts comes with the LLM milestone.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from cssselect import SelectorError
from lxml import etree
from lxml import html as lxml_html
from lxml.cssselect import CSSSelector

from retrievr.urls import normalize_url, path_extension, resolve_url

HtmlElement = Any  # lxml.html.HtmlElement (lxml ships no type stubs)

_HEADINGS_XPATH = (
    "preceding::*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6][1]"
)


@dataclass(frozen=True)
class Candidate:
    url: str
    normalized_url: str
    link_text: str
    title: str
    """Item title when a title selector is given, else the link text or the row text."""
    context_text: str
    """Text of the item (or of the closest row-like ancestor) the link belongs to."""
    heading: str | None
    dom_path: str


@lru_cache(maxsize=256)
def compile_selector(selector: str) -> CSSSelector:
    """Compile a CSS selector, raising ``ValueError`` if it is invalid."""
    try:
        return CSSSelector(selector)
    except SelectorError as exc:
        raise ValueError(f"invalid CSS selector {selector!r}: {exc}") from exc


def select(root: HtmlElement, selector: str) -> list[HtmlElement]:
    return list(compile_selector(selector)(root))


def parse_html(content: bytes | str) -> HtmlElement:
    """Parse an HTML document (bytes let lxml honour the page's declared charset)."""
    if isinstance(content, bytes) and not content.strip():
        content = b"<html></html>"
    return lxml_html.document_fromstring(content)


def text_of(el: HtmlElement) -> str:
    """Whitespace-normalized text content."""
    return " ".join(str(el.text_content()).split())


def dom_path(el: HtmlElement) -> str:
    """Stable structural path, e.g. ``html>body>div#main>ul.list>li.doc>a`` (no indices)."""
    parts: list[str] = []
    node: HtmlElement | None = el
    while node is not None and isinstance(node.tag, str):
        label = node.tag
        if node.get("id"):
            label += "#" + node.get("id")
        elif node.get("class"):
            label += "." + node.get("class").split()[0]
        parts.append(label)
        node = node.getparent()
    return ">".join(reversed(parts))


def nearest_heading(el: HtmlElement) -> str | None:
    found = el.xpath(_HEADINGS_XPATH)
    return text_of(found[0]) or None if found else None


def extract_candidates(
    root: HtmlElement,
    base_url: str,
    *,
    container: str | None = None,
    item: str | None = None,
    link: str = "a[href]",
    title: str | None = None,
    extensions: Collection[str] | None = None,
) -> list[Candidate]:
    """Extract document links from ``root``.

    - ``container``: CSS selector(s) restricting the search (all matches are used).
    - ``item``: CSS selector of one document entry (card, row) inside the container.
      Links, title and context text are then read inside each item.
    - ``link``: CSS selector of the link element (relative to item or container).
    - ``title``: CSS selector of the title, relative to the item.
    - ``extensions``: keep only links whose path ends with one of these extensions.
    """
    scopes = select(root, container) if container else [root]
    exts = {e.lower().lstrip(".") for e in extensions} if extensions else None
    out: list[Candidate] = []
    for scope in scopes:
        if item:
            for item_el in select(scope, item):
                out.extend(_from_item(item_el, base_url, link, title, exts))
        else:
            out.extend(_from_scope(scope, base_url, link, exts))
    return out


def _from_item(
    item_el: HtmlElement,
    base_url: str,
    link: str,
    title: str | None,
    exts: set[str] | None,
) -> Iterable[Candidate]:
    context = text_of(item_el)
    title_text = None
    if title:
        matches = select(item_el, title)
        title_text = text_of(matches[0]) if matches else None
    for a in _links(item_el, link, base_url, exts):
        url = resolve_url(a.get("href"), base_url)
        link_text = text_of(a)
        yield Candidate(
            url=url,
            normalized_url=normalize_url(url),
            link_text=link_text,
            title=title_text or a.get("data-title") or link_text or context,
            context_text=context,
            heading=nearest_heading(a),
            dom_path=dom_path(a),
        )


def _from_scope(
    scope: HtmlElement, base_url: str, link: str, exts: set[str] | None
) -> Iterable[Candidate]:
    for a in _links(scope, link, base_url, exts):
        url = resolve_url(a.get("href"), base_url)
        normalized = normalize_url(url)
        link_text = text_of(a)
        context = text_of(_row_element(a, normalized, base_url))
        yield Candidate(
            url=url,
            normalized_url=normalized,
            link_text=link_text,
            title=link_text or a.get("data-title") or context,
            context_text=context,
            heading=nearest_heading(a),
            dom_path=dom_path(a),
        )


def _links(
    scope: HtmlElement, link: str, base_url: str, exts: set[str] | None
) -> list[HtmlElement]:
    result = []
    for a in select(scope, link):
        href = a.get("href")
        if not href or href.strip().lower().startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        if exts is not None and path_extension(resolve_url(href, base_url)) not in exts:
            continue
        result.append(a)
    return result


def _row_element(a: HtmlElement, own_url: str, base_url: str) -> HtmlElement:
    """Largest ancestor of ``a`` that contains no link to another URL (the "row")."""
    best = a
    for anc in a.iterancestors():
        if not isinstance(anc.tag, str) or anc.tag in ("body", "html"):
            break
        for other in anc.iter("a"):
            href = other.get("href")
            if not href or href.strip().lower().startswith(("javascript:", "mailto:", "#")):
                continue
            if normalize_url(resolve_url(href, base_url)) != own_url:
                return best
        best = anc
    return best


def to_string(el: HtmlElement) -> str:
    """Serialize an element (debug helper)."""
    return str(etree.tostring(el, encoding="unicode", method="html"))
