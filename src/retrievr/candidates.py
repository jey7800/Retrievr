"""Candidate extraction: document links plus their context, from an HTML page.

Deterministic, never calls an LLM. Also builds the compact page digest (document link
groups, navigation links, pruned HTML) that the learner shows to the LLM.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from cssselect import SelectorError
from lxml import etree
from lxml import html as lxml_html
from lxml.cssselect import CSSSelector

from retrievr.urls import host_of, normalize_url, path_extension, resolve_url

HtmlElement = Any  # lxml.html.HtmlElement (lxml ships no type stubs)

DOCUMENT_EXTENSIONS = frozenset(
    {"pdf", "xlsx", "xls", "xlsb", "xlsm", "csv", "zip", "xbri", "doc", "docx", "ppt", "pptx"}
)
"""File extensions treated as documents when a recipe declares none."""

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
    link_title: str | None = None
    """The link's ``title`` attribute."""
    row_header: str | None = None
    """Table context: first cell of the link's row, carried down over blank cells."""
    column_header: str | None = None
    """Table context: header cell above the link's column (colspan/rowspan aware)."""

    def fields(self) -> dict[str, str | None]:
        """Named texts that classification rules can target."""
        return {
            "title": self.title,
            "link_text": self.link_text,
            "link_title": self.link_title,
            "context": self.context_text,
            "heading": self.heading,
            "row_header": self.row_header,
            "column_header": self.column_header,
            "text": f"{self.title} {self.context_text}",
        }


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
    return (text_of(found[0]) or None) if found else None


def extract_candidates(
    root: HtmlElement,
    base_url: str,
    *,
    container: str | None = None,
    item: str | None = None,
    link: str = "a[href]",
    title: str | None = None,
    extensions: Collection[str] | None = None,
    table_context: bool = False,
) -> list[Candidate]:
    """Extract document links from ``root``.

    - ``container``: CSS selector(s) restricting the search (all matches are used).
    - ``item``: CSS selector of one document entry (card, row) inside the container.
      Links, title and context text are then read inside each item.
    - ``link``: CSS selector of the link element (relative to item or container).
    - ``title``: CSS selector of the title, relative to the item.
    - ``extensions``: keep only links whose path ends with one of these extensions.
    - ``table_context``: for links inside a table cell, also read the row header and
      the column header.
    """
    scopes = select(root, container) if container else [root]
    exts = {e.lower().lstrip(".") for e in extensions} if extensions else None
    grids: dict[int, _Grid] | None = {} if table_context else None
    out: list[Candidate] = []
    for scope in scopes:
        if item:
            for item_el in select(scope, item):
                context = text_of(item_el)
                title_text = None
                if title:
                    matches = select(item_el, title)
                    title_text = text_of(matches[0]) if matches else None
                for a in _links(item_el, link, base_url, exts):
                    out.append(_candidate(a, base_url, context, title_text, grids))
        else:
            for a in _links(scope, link, base_url, exts):
                own = normalize_url(resolve_url(a.get("href"), base_url))
                context = text_of(_row_element(a, own, base_url))
                out.append(_candidate(a, base_url, context, None, grids))
    return out


def _candidate(
    a: HtmlElement,
    base_url: str,
    context: str,
    title_text: str | None,
    grids: dict[int, _Grid] | None,
) -> Candidate:
    url = resolve_url(a.get("href"), base_url)
    link_text = text_of(a)
    row_header = column_header = None
    if grids is not None:
        row_header, column_header = _table_headers(a, grids)
    return Candidate(
        url=url,
        normalized_url=normalize_url(url),
        link_text=link_text,
        title=title_text or a.get("data-title") or link_text or context,
        context_text=context,
        heading=nearest_heading(a),
        dom_path=dom_path(a),
        link_title=" ".join((a.get("title") or "").split()) or None,
        row_header=row_header,
        column_header=column_header,
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


@dataclass
class _Grid:
    """Cell positions of a table, with row and column spans expanded.

    Dictionaries are keyed by ``id()`` of lxml elements. lxml creates element proxies
    on demand and an ``id`` can be reused once its proxy is gone, so the grid keeps a
    reference to the table and to every cell for as long as it is in use.
    """

    table: HtmlElement
    cells: dict[tuple[int, int], HtmlElement]
    positions: dict[int, tuple[int, int]]
    header_row: int | None


def _own_rows(table: HtmlElement) -> list[HtmlElement]:
    """Rows of ``table`` itself, excluding rows of nested tables."""
    return [tr for tr in table.iter("tr") if next(tr.iterancestors("table"), None) is table]


def _span(cell: HtmlElement, attr: str) -> int:
    try:
        return max(1, min(int(cell.get(attr) or 1), 100))
    except ValueError:
        return 1


def _build_grid(table: HtmlElement) -> _Grid:
    cells: dict[tuple[int, int], HtmlElement] = {}
    positions: dict[int, tuple[int, int]] = {}
    header_row = None
    for r, tr in enumerate(_own_rows(table)):
        row_cells = [c for c in tr if isinstance(c.tag, str) and c.tag in ("td", "th")]
        if (
            header_row is None
            and row_cells
            and (tr.getparent().tag == "thead" or all(c.tag == "th" for c in row_cells))
        ):
            header_row = r
        col = 0
        for cell in row_cells:
            while (r, col) in cells:
                col += 1
            positions[id(cell)] = (r, col)
            for dr in range(_span(cell, "rowspan")):
                for dc in range(_span(cell, "colspan")):
                    cells[(r + dr, col + dc)] = cell
            col += _span(cell, "colspan")
    return _Grid(table, cells, positions, header_row)


def _table_headers(a: HtmlElement, grids: dict[int, _Grid]) -> tuple[str | None, str | None]:
    cell = next((e for e in a.iterancestors() if e.tag in ("td", "th")), None)
    table = next(a.iterancestors("table"), None)
    if cell is None or table is None:
        return None, None
    grid = grids.get(id(table))
    if grid is None:
        grid = grids[id(table)] = _build_grid(table)
    if id(cell) not in grid.positions:
        return None, None
    row, col = grid.positions[id(cell)]
    column_header = None
    if grid.header_row is not None and grid.header_row != row:
        header = grid.cells.get((grid.header_row, col))
        column_header = (text_of(header) or None) if header is not None else None
    row_header = None
    first_data_row = -1 if grid.header_row is None else grid.header_row
    for r in range(row, first_data_row, -1):
        first = grid.cells.get((r, 0))
        if first is None or first is cell:
            break
        if text := text_of(first):
            row_header = text
            break
    return row_header, column_header


def to_string(el: HtmlElement) -> str:
    """Serialize an element (debug helper)."""
    return str(etree.tostring(el, encoding="unicode", method="html"))


# -- page digest for prompts ---------------------------------------------------------

_DIGITS = re.compile(r"\d+")
_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
_NAV_WORDS = re.compile(
    r"archive|result|report|financ|publication|document|quarter|annual|interim|investor|"
    r"rapport|r[ée]sultat|semest|trimest|exercice|year|ann[ée]e|page|next|suivant|more|all|"
    r"previous|older|take=|view",
    re.IGNORECASE,
)
_PRUNE_TAGS = ("script", "style", "noscript", "svg", "iframe", "template", "link", "meta")
_KEEP_ATTRS = ("id", "class", "href", "title", "rel", "role", "data-title", "aria-label")


@dataclass
class LinkGroup:
    """Document links sharing a DOM path (digits folded, so per-year panels merge)."""

    key: str
    example_path: str
    candidates: list[Candidate] = field(default_factory=list)
    elements: list[HtmlElement] = field(default_factory=list)

    @property
    def extensions(self) -> Counter[str]:
        return Counter(path_extension(c.url) for c in self.candidates)

    @property
    def hosts(self) -> set[str]:
        return {host_of(c.url) for c in self.candidates}


@dataclass(frozen=True)
class NavLink:
    url: str
    text: str
    score: int


def group_document_links(
    root: HtmlElement, base_url: str, extensions: Collection[str] = DOCUMENT_EXTENSIONS
) -> list[LinkGroup]:
    """Every document link of the page, grouped by folded DOM path, in page order."""
    groups: dict[str, LinkGroup] = {}
    seen: set[str] = set()
    for a in _links(root, "a[href]", base_url, {e.lower() for e in extensions}):
        cand = _candidate(
            a,
            base_url,
            text_of(_row_element(a, normalize_url(resolve_url(a.get("href"), base_url)), base_url)),
            None,
            None,
        )
        if cand.normalized_url in seen:
            continue
        seen.add(cand.normalized_url)
        key = _DIGITS.sub("N", cand.dom_path)
        group = groups.setdefault(key, LinkGroup(key, cand.dom_path))
        group.candidates.append(cand)
        group.elements.append(a)
    return list(groups.values())


def site_domain(host: str) -> str:
    """Registrable-ish domain used to tell "same site" links: the last two labels."""
    labels = host.lower().split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else host.lower()


def in_site(host: str, domain: str) -> bool:
    host = host.lower()
    return host == domain or host.endswith("." + domain)


def entry_element(a: HtmlElement, base_url: str) -> HtmlElement:
    """The "row" of a link: its largest ancestor that links to nothing else."""
    return _row_element(a, normalize_url(resolve_url(a.get("href"), base_url)), base_url)


def page_title(root: HtmlElement) -> str:
    found = root.xpath("//title")
    return text_of(found[0])[:150] if found else ""


def navigation_links(
    root: HtmlElement,
    base_url: str,
    domain: str,
    *,
    limit: int = 80,
    extensions: Collection[str] = DOCUMENT_EXTENSIONS,
) -> list[NavLink]:
    """Links to other pages of the site (hosts under ``domain``), most promising first
    (years, archives, paging)."""
    exts = {e.lower() for e in extensions}
    found: dict[str, NavLink] = {}
    for order, a in enumerate(select(root, "a[href]")):
        href = (a.get("href") or "").strip()
        if not href or href.lower().startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        url = resolve_url(href, base_url)
        if not in_site(host_of(url), domain) or path_extension(url) in exts:
            continue
        key = normalize_url(url)
        if key == normalize_url(base_url) or key in found:
            continue
        text = " ".join((text_of(a) or a.get("title") or a.get("aria-label") or "").split())[:80]
        blob = f"{text} {url}"
        score = 2 * bool(_YEAR.search(blob)) + bool(_NAV_WORDS.search(blob))
        score += 2 * (a.get("rel") == "next")
        found[key] = NavLink(url, text, score * 10_000 - order)
    ranked = sorted(found.values(), key=lambda n: -n.score)
    return ranked[:limit]


def pruned_html(el: HtmlElement, max_chars: int = 1200) -> str:
    """Compact HTML of ``el`` for a prompt: no scripts or styles, few attributes, repeated
    siblings reduced to two examples plus a count."""
    clone = lxml_html.fromstring(etree.tostring(el))
    for bad in clone.xpath("|".join(f".//{t}" for t in _PRUNE_TAGS)):
        bad.getparent().remove(bad)
    for node in list(clone.iter()):
        if not isinstance(node.tag, str):
            continue
        for attr in list(node.attrib):
            if attr not in _KEEP_ATTRS:
                del node.attrib[attr]
        if node.get("class"):
            node.set("class", " ".join(node.get("class").split()[:3]))
        _fold_siblings(node)
    text = str(etree.tostring(clone, encoding="unicode", method="html"))
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= max_chars else text[:max_chars] + " ...(truncated)"


def _fold_siblings(node: HtmlElement) -> None:
    children = [c for c in node if isinstance(c.tag, str)]
    if len(children) <= 3:
        return
    signature = Counter((c.tag, c.get("class")) for c in children)
    for (tag, cls), count in signature.items():
        if count <= 3:
            continue
        same = [c for c in children if (c.tag, c.get("class")) == (tag, cls)]
        for extra in same[2:]:
            node.remove(extra)
        marker = etree.Comment(f" {count - 2} more <{tag}> siblings ")
        same[1].addnext(marker)
