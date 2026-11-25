"""Candidate extraction: document links plus their context, from an HTML page.

Deterministic, never calls an LLM. Also builds the compact page digest (document link
groups, navigation links, pruned HTML) that the learner shows to the LLM.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from urllib.parse import quote, urlsplit

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

_WEB_PAGE = r"[^/?#]*\.(?:html?|aspx?|php|jsp)(?:[?#]|$)"
"""A path segment that names a web page (``default.htm``), never a document."""

OPAQUE_DOCUMENT = re.compile(
    r"/pdfs?/[^/?#]+"  # /assets/pdf/release-q1-2026
    r"|/pdf[a-z]+/[^/?#]+"  # /pdfPreview/123
    r"|/(?:downloads?|files?|attachments?)/(?!" + _WEB_PAGE + r")[^/?#]+(?:\?[^#]*)?$"  # /file/9
    r"|/downloads?/(?:[^/?#]+/)+[^/?#.]+(?:\?[^#]*)?$"  # /downloads/documents/us-en/15db52
    r"|[?&](?:p_doc_ref|docref|doc_?id|document_?id|file_?id|attachment_?id|file_?name)="
    r"|[?&]disposition=(?:inline|attachment)",
    re.IGNORECASE,
)
"""Links without a file extension that are documents all the same (download services,
asset paths). Used to show them to the learner; a recipe lists them through
``items.document_urls`` patterns."""

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
    file_name: str | None = None
    """File name when the page gives one (``download`` or ``title`` attribute, JSON field):
    the only readable name of an opaque link such as ``/pdfPreview/123``."""
    format_hint: str | None = None
    """File format the link text announces ("Interim report (pdf)", "PDF 2.1 MB"): the
    extension of an opaque link that gives no file name."""
    extra: Mapping[str, str] = field(default_factory=dict, compare=False)
    """Values of a JSON item kept with the document (recipe schema v6)."""

    @property
    def extension(self) -> str:
        """Extension of the URL path, else of ``file_name``, else the announced format."""
        return (
            path_extension(self.url) or file_extension(self.file_name) or (self.format_hint or "")
        )

    def fields(self) -> dict[str, str | None]:
        """Named texts that classification rules can target."""
        return {
            "title": self.title,
            "link_text": self.link_text,
            "link_title": self.link_title,
            "file_name": self.file_name,
            "context": self.context_text,
            "heading": self.heading,
            "row_header": self.row_header,
            "column_header": self.column_header,
            "text": f"{self.title} {self.context_text}",
        }


_FILE_NAME = re.compile(r"[^/\\]*\.([A-Za-z0-9]{2,5})$")
_FORMAT_WORDS = r"(pdf|xlsx?|xlsb|excel|csv|zip|docx?|pptx?)"
_TEXT_FORMAT = re.compile(
    r"[(\[]\s*" + _FORMAT_WORDS + r"\s*[)\]]"  # (pdf), [PDF]
    r"|\b" + _FORMAT_WORDS + r"\s*[(,]?\s*\d+(?:[.,]\d+)?\s*[kmg]i?[bo]\b"  # pdf 289 KB, PDF (2 Mo)
    r"|^\s*" + _FORMAT_WORDS + r"\s*$",  # a link text that is only the format
    re.I,
)
_ROW_FORMAT = re.compile(
    r"[(\[]\s*" + _FORMAT_WORDS + r"\s*[)\]]"
    r"|\b" + _FORMAT_WORDS + r"\b[^\d\n]{0,12}?\d+(?:[.,]\d+)?\s*[kmg]i?[bo]\b",  # pdf, EN, 6 Mo
    re.I,
)
"""A format announced next to a link (its row has no other link): needs a size or
brackets, as a row also holds the document title."""
_LINK_XPATH = ".//a[@href] | .//*[contains(local-name(), '-') and @href]"
"""Links: ``<a href>`` and custom elements (Web Components) that carry an ``href``."""


def text_format(text: str | None, pattern: re.Pattern[str] = _TEXT_FORMAT) -> str:
    """The file format a link text announces ('' when it names none)."""
    m = pattern.search(text or "")
    if not m:
        return ""
    word = next(g for g in m.groups() if g).lower()
    return "xlsx" if word == "excel" else word


def announced_format(a: HtmlElement, base_url: str) -> str:
    """The format the link text announces, else the one its row announces ("View" next
    to "pdf, EN, 6 Mo")."""
    if found := text_format(text_of(a)):
        return found
    href = a.get("href") or ""
    row = _row_element(a, normalize_url(resolve_url(href, base_url)), base_url)
    if row is a:
        return ""
    words = " ".join(" ".join(t for t in row.itertext() if isinstance(t, str)).split())
    return text_format(words, _ROW_FORMAT)


def _link_elements(scope: HtmlElement, link: str) -> list[HtmlElement]:
    if link == "a[href]":
        return list(scope.xpath(_LINK_XPATH))
    return select(scope, link)


def file_extension(name: str | None) -> str:
    """Lowercased extension of a file name ('' if it does not look like one)."""
    m = _FILE_NAME.search((name or "").strip())
    return m.group(1).lower() if m else ""


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
        node_id = (node.get("id") or "").strip()
        classes = (node.get("class") or "").split()  # may be blank: class=" "
        if node_id:
            label += "#" + node_id
        elif classes:
            label += "." + classes[0]
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
    document_urls: Collection[str] | None = None,
) -> list[Candidate]:
    """Extract document links from ``root``.

    - ``container``: CSS selector(s) restricting the search (all matches are used).
    - ``item``: CSS selector of one document entry (card, row) inside the container.
      Links, title and context text are then read inside each item.
    - ``link``: CSS selector of the link element (relative to item or container).
    - ``title``: CSS selector of the title, relative to the item.
    - ``extensions``: keep only links whose path, or whose file name (``download`` or
      ``title`` attribute), ends with one of these extensions.
    - ``table_context``: for links inside a table cell, also read the row header and
      the column header.
    - ``document_urls``: regular expressions; a link whose URL matches one is a document
      even without a known extension (opaque links such as ``/pdfPreview/123``).
    """
    scopes = select(root, container) if container else [root]
    exts = {e.lower().lstrip(".") for e in extensions} if extensions else None
    patterns = [re.compile(p) for p in document_urls or ()]
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
                for a in _links(item_el, link, base_url, exts, patterns):
                    out.append(_candidate(a, base_url, context, title_text, grids))
        else:
            for a in _links(scope, link, base_url, exts, patterns):
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
        file_name=link_file_name(a),
        format_hint=announced_format(a, base_url) or None,
    )


def link_file_name(a: HtmlElement) -> str | None:
    """A file name given by the link itself: ``download`` or ``title`` attribute."""
    for attr in ("download", "title", "data-filename"):
        value = " ".join((a.get(attr) or "").split())
        if value and file_extension(value):
            return value[-200:]
    return None


def _links(
    scope: HtmlElement,
    link: str,
    base_url: str,
    exts: set[str] | None,
    patterns: Collection[re.Pattern[str]] = (),
) -> list[HtmlElement]:
    result = []
    for a in _link_elements(scope, link):
        href = a.get("href")
        if not href or href.strip().lower().startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        url = resolve_url(href, base_url)
        if exts is not None and not (
            path_extension(url) in exts
            or file_extension(link_file_name(a)) in exts
            or announced_format(a, base_url) in exts
            or any(p.search(url) for p in patterns)
        ):
            continue
        result.append(a)
    return result


def _row_element(a: HtmlElement, own_url: str, base_url: str) -> HtmlElement:
    """Largest ancestor of ``a`` that contains no link to another URL (the "row")."""
    best = a
    for anc in a.iterancestors():
        if not isinstance(anc.tag, str) or anc.tag in ("body", "html"):
            break
        for other in anc.xpath(_LINK_XPATH):
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
        return Counter(c.extension for c in self.candidates)

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
    exts = {e.lower() for e in extensions}
    for a in _links(root, "a[href]", base_url, exts, [OPAQUE_DOCUMENT]):
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


def opaque_patterns(candidates: Collection[Candidate], limit: int = 6) -> list[str]:
    """``items.document_urls`` patterns for the opaque links among ``candidates``: the
    folder of a path (``/assets/pdf/``) or the URL and query key of a download service
    (``/files?p_Doc_Ref=``)."""
    out: list[str] = []
    for cand in candidates:
        if file_extension(cand.url.split("?", 1)[0]) or cand.file_name:
            continue
        m = OPAQUE_DOCUMENT.search(cand.url)
        if m is None and not cand.format_hint:
            continue
        parts = urlsplit(cand.url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if m is None:
            # A download service known only from link texts ("Interim report (pdf)"):
            # every link into the same folder is a document, announced or not.
            folder = parts.path[: parts.path.rfind("/") + 1]
            if folder.count("/") < 2:
                continue  # the site root: too broad
            pattern = "^" + re.escape(f"{origin}{folder}")
        elif m.group(0)[0] in "?&":
            key = m.group(0)[1:].split("=", 1)[0]
            pattern = "^" + re.escape(f"{origin}{parts.path}?") + "(?:.*&)?" + re.escape(key) + "="
        else:
            folder = parts.path[: parts.path.rfind("/") + 1]
            pattern = "^" + re.escape(f"{origin}{folder}")
        if pattern not in out:
            out.append(pattern)
        if len(out) >= limit:
            break
    return out


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


# -- JSON responses --------------------------------------------------------------------

_MISSING = object()


def json_path(data: Any, path: str) -> Any:
    """Value at a dotted path (``results.0.hits``; list indexes are numbers), else None."""
    node = data
    for part in path.split(".") if path else ():
        if isinstance(node, dict):
            node = node.get(part, _MISSING)
        elif isinstance(node, list) and part.lstrip("-").isdigit():
            index = int(part)
            node = node[index] if -len(node) <= index < len(node) else _MISSING
        else:
            return None
        if node is _MISSING:
            return None
    return node


def json_items(data: Any, path: str) -> list[tuple[list[str], Any]]:
    """The items of the list at ``path``, each with the keys its ``*`` segments went
    through (schema v5: ``*.*`` reads ``{"2025": {"Q1": [...]}}``, keys 2025 and Q1)."""
    found: list[tuple[list[str], Any]] = []

    def walk(node: Any, parts: list[str], keys: list[str]) -> None:
        if not parts:
            if isinstance(node, list):
                found.extend((keys, entry) for entry in node)
            return
        head, rest = parts[0], parts[1:]
        if head == "*":
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, rest, [*keys, str(key)])
            elif isinstance(node, list):
                for value in node:
                    walk(value, rest, keys)
            return
        value = json_path(node, head)
        if value is not None:
            walk(value, rest, keys)

    walk(data, path.split(".") if path else [], [])
    return found


def _text(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        return ""
    return " ".join(str(value).split())


def json_candidates(
    data: Any,
    base_url: str,
    *,
    items: str,
    url: str | Collection[str],
    title: str | None = None,
    file_name: str | None = None,
    context: Collection[str] = (),
    extensions: Collection[str] | None = None,
    document_urls: Collection[str] | None = None,
    url_template: str | None = None,
    extra: Mapping[str, str] | None = None,
    identity: Callable[[Mapping[str, str]], str | None] | None = None,
) -> list[Candidate]:
    """Documents listed in a JSON response: ``items`` is the path to the list, the other
    arguments are paths inside one item (``url`` may list several paths: the first one
    present wins). Items without a URL are skipped. With ``url_template`` the item holds
    a file name, placed (URL-encoded) at ``{value}`` in the template. ``extra`` names
    values of the item kept with the document; ``identity`` gives the URL of an item
    that has none (a document fetched by a request built from those values)."""
    entries = json_items(data, items)
    url_paths = [url] if isinstance(url, str) else list(url)
    exts = {e.lower().lstrip(".") for e in extensions} if extensions else None
    patterns = [re.compile(p) for p in document_urls or ()]
    out: list[Candidate] = []
    for keys, entry in entries:
        kept = {
            name: text
            for name, path in (extra or {}).items()
            if (text := _text(json_path(entry, path)))
        }
        raw = next((t for p in url_paths if (t := _text(json_path(entry, p)))), "")
        if not raw and identity is not None:
            raw = identity(kept) or ""
        if not raw:
            continue
        if url_template:
            absolute = resolve_url(url_template.replace("{value}", quote(raw, safe="")), base_url)
        else:
            absolute = resolve_url(raw, base_url)
        name = _text(json_path(entry, file_name)) if file_name else ""
        if url_template and not name:
            name = raw  # the file name is the only readable name of the link
        ext = path_extension(absolute) or file_extension(name)
        if exts is not None and ext not in exts and not any(p.search(absolute) for p in patterns):
            continue
        heading = _text(json_path(entry, title)) if title else ""
        # Keys a ``*`` went through (a year, a quarter) describe the item too.
        texts = [*keys, *(_text(json_path(entry, p)) for p in context)]
        ctx = " | ".join(t for t in texts if t)
        out.append(
            Candidate(
                url=absolute,
                normalized_url=normalize_url(absolute),
                link_text=heading,
                title=heading or name or ctx,
                context_text=ctx,
                heading=None,
                dom_path=f"json:{items}",
                file_name=name or None,
                extra=kept,
            )
        )
    return out


# -- interactive controls --------------------------------------------------------------

_LOAD_MORE_TEXT = re.compile(
    r"^\s*(load|show|view|see|find) more|^\s*more (results|files|documents|reports)|"
    r"voir plus|afficher plus|plus de (r[ée]sultats|documents)|charger plus|"
    r"mehr (anzeigen|laden)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PageControls:
    """Controls that suggest a listing loaded by JavaScript."""

    year_selects: list[str]
    """CSS of ``<select>`` elements with at least two year options."""
    load_more: list[str]
    """Labels of "load more" buttons."""


_SHORT_PERIOD = re.compile(
    r"(?<![a-z0-9])(?:q[1-4]|[1-4]q|h[12]|[12]h|fy|9m)\s?'?(\d{2})(?!\d)", re.I
)


def label_year(text: str) -> int | None:
    """The year an option or tab label names: ``2025``, or a short period such as
    ``Q325``, ``Q3 25``, ``1H25`` or ``FY24`` (years 2000-2099)."""
    if m := _YEAR.search(text):
        return int(m.group(0))
    if m := _SHORT_PERIOD.search(text):
        return 2000 + int(m.group(1))
    return None


def page_controls(root: HtmlElement) -> PageControls:
    selects = []
    for s in select(root, "select"):
        years = [o for o in select(s, "option") if label_year(text_of(o)) and o.get("value")]
        if len(years) >= 2:
            selects.append(f"#{s.get('id')}" if s.get("id") else "select")
    buttons = [
        text_of(b)[:60]
        for b in select(root, "button, a[href='#'], a:not([href])")
        if _LOAD_MORE_TEXT.search(text_of(b))
    ]
    return PageControls(selects, buttons)
