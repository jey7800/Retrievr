"""Deterministic analysis of the exchanges a browser discovery captured (no LLM).

- which exchanges list documents, where (a JSON list, an HTML fragment inside JSON, an
  HTML response) and with which fields;
- a compact digest of the endpoints for the learner's prompt;
- building a ``request`` source from the exchange the LLM picked and the fields it says
  vary (page number, year selector option, year). Request bodies are copied from the
  captured exchange; the LLM never writes them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal
from urllib.parse import parse_qsl, quote, unquote_plus

from retrievr.browser import Exchange
from retrievr.candidates import (
    DOCUMENT_EXTENSIONS,
    LinkGroup,
    entry_element,
    file_extension,
    group_document_links,
    parse_html,
    pruned_html,
)
from retrievr.urls import host_of, normalize_url, path_extension, resolve_url

DATE_FORMATS = ("%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y", "%d-%m-%Y")
MAX_SAMPLE_CHARS = 1500
_HTML_HINT = re.compile(r"<a\s[^>]*href=", re.IGNORECASE)

Vary = Literal["page", "options", "year"]


# -- documents inside a response -------------------------------------------------------


@dataclass
class JsonList:
    """A list of objects in a JSON response whose items hold document URLs."""

    path: str
    size: int
    url_paths: dict[str, int]
    """Item paths holding a document URL, with how many items have one there."""
    sample: Any
    url_templates: dict[str, str] = field(default_factory=dict)
    """Item paths holding a bare file name, with the URL the page builds from it
    (``{value}`` stands for the name)."""


@dataclass
class ResponseDocuments:
    kind: Literal["json_list", "json_html", "html", "none"]
    documents: set[str] = field(default_factory=set)
    """Normalized URLs of the documents found."""
    lists: list[JsonList] = field(default_factory=list)
    html_path: str | None = None
    groups: list[LinkGroup] = field(default_factory=list)
    keys: dict[str, str] = field(default_factory=dict)
    """Top-level JSON keys and the type of their value."""


def _is_document_url(value: str) -> bool:
    if len(value) > 2000 or " " in value.strip() or "/" not in value:
        return False
    return path_extension(value) in DOCUMENT_EXTENSIONS


def _is_file_name(value: str) -> bool:
    """A bare file name (``Annual_report_2025.pdf``): the page builds its URL."""
    return (
        len(value) <= 300
        and "/" not in value
        and value.strip() == value
        and " " not in value
        and path_extension("/" + value) in DOCUMENT_EXTENSIONS
    )


def file_url_template(name: str, links: Sequence[str]) -> str | None:
    """The URL template a page uses for a file name: the rendered link that holds the
    name (as is or URL-encoded), with ``{value}`` in its place."""
    for link in links:
        for form in (quote(name, safe=""), quote(name), name):
            at = link.find(form)
            if at >= 0:
                return link[:at] + "{value}" + link[at + len(form) :]
    return None


def _leaves(node: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, f"{prefix}.{i}" if prefix else str(i))
    else:
        yield prefix, node


def _lists(node: Any, prefix: str = "") -> Iterator[tuple[str, list[Any]]]:
    if isinstance(node, list):
        if node and all(isinstance(x, dict) for x in node):
            yield prefix, node
        for i, v in enumerate(node[:3]):
            yield from _lists(v, f"{prefix}.{i}" if prefix else str(i))
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _lists(v, f"{prefix}.{k}" if prefix else str(k))


def _type_name(value: Any) -> str:
    if isinstance(value, str):
        return "HTML" if _HTML_HINT.search(value) else "text"
    return {bool: "bool", int: "number", float: "number", list: "list", dict: "object"}.get(
        type(value), "null"
    )


def analyse(exchange: Exchange, links: Sequence[str] = ()) -> ResponseDocuments:
    """Where the documents of a response are (if any). ``links``: document links of the
    rendered page, to resolve bare file names of a JSON listing."""
    if exchange.status != 200 or not exchange.body:
        return ResponseDocuments("none")
    content_type = (exchange.content_type or "").lower()
    text = exchange.text
    if "json" in content_type or text.lstrip().startswith(("{", "[")):
        try:
            data = json.loads(text)
        except ValueError:
            return ResponseDocuments("none")
        return _analyse_json(data, exchange.url, links)
    if "html" in content_type or _HTML_HINT.search(text[:20000]):
        groups = group_document_links(parse_html(exchange.body), exchange.url)
        docs = {c.normalized_url for g in groups for c in g.candidates}
        return ResponseDocuments("html" if docs else "none", docs, groups=groups)
    return ResponseDocuments("none")


def _analyse_json(data: Any, base_url: str, links: Sequence[str] = ()) -> ResponseDocuments:
    keys = {str(k): _type_name(v) for k, v in data.items()} if isinstance(data, dict) else {}
    found = ResponseDocuments("none", keys=keys)
    holding: dict[str, int] = {}
    for path, items in _lists(data):
        counts: dict[str, int] = {}
        templates: dict[str, str] = {}
        docs: set[str] = set()
        with_url = 0
        for item in items:
            had = False
            for leaf, value in _leaves(item):
                if not isinstance(value, str):
                    continue
                if _is_document_url(value):
                    url = resolve_url(value, base_url)
                elif links and _is_file_name(value):
                    template = templates.get(leaf) or file_url_template(value, links)
                    if template is None:
                        continue
                    templates[leaf] = template
                    url = resolve_url(template.replace("{value}", quote(value, safe="")), base_url)
                else:
                    continue
                counts[leaf] = counts.get(leaf, 0) + 1
                docs.add(normalize_url(url))
                had = True
            with_url += had
        if counts:
            sample = next((i for i in items if any(_path_value(i, p) for p in counts)), items[0])
            found.lists.append(JsonList(path, len(items), counts, sample, templates))
            holding[path] = with_url
            found.documents |= docs
    if found.lists:
        found.kind = "json_list"
        # The documents' own list: where the most items carry a document URL.
        found.lists.sort(key=lambda lst: (-holding[lst.path], len(lst.url_paths)))
        return found
    for leaf, value in _leaves(data):
        if isinstance(value, str) and _HTML_HINT.search(value):
            groups = group_document_links(
                parse_html(f"<html><body>{value}</body></html>"), base_url
            )
            docs = {c.normalized_url for g in groups for c in g.candidates}
            if docs:
                found.kind, found.html_path, found.groups = "json_html", leaf, groups
                found.documents = docs
                return found
    return found


def _path_value(node: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


# -- request bodies --------------------------------------------------------------------


def body_fields(exchange: Exchange) -> dict[str, str]:
    """Flat view of the request body: form fields, or JSON leaves by dotted path."""
    body = exchange.request_body
    if not body:
        return {}
    if body.lstrip().startswith(("{", "[")):
        try:
            return {
                p: json.dumps(v) if not isinstance(v, str) else v
                for p, v in _leaves(json.loads(body))
            }
        except ValueError:
            return {}
    return dict(parse_qsl(body, keep_blank_values=True))


QUERY_PREFIX = "?"
"""Fields of the URL query string are named ``?name`` (body fields have no prefix)."""


def query_fields(exchange: Exchange) -> dict[str, str]:
    """The URL query string of a call, as ``?name`` fields."""
    query = exchange.url.partition("?")[2]
    return {QUERY_PREFIX + k: v for k, v in parse_qsl(query, keep_blank_values=True)}


def request_fields(exchange: Exchange) -> dict[str, str]:
    """Body fields, then URL query fields (``?name``)."""
    return {**body_fields(exchange), **query_fields(exchange)}


def templated_url(exchange: Exchange, vary: Mapping[str, str]) -> str:
    """The call's URL with the ``?name`` fields of ``vary`` replaced by placeholders (the
    other parameters are kept exactly as captured)."""
    base, sep, query = exchange.url.partition("?")
    wanted = {k[len(QUERY_PREFIX) :]: v for k, v in vary.items() if k.startswith(QUERY_PREFIX)}
    if not wanted:
        return exchange.url
    segments = query.split("&") if sep else []
    found: set[str] = set()
    for i, segment in enumerate(segments):
        key = unquote_plus(segment.partition("=")[0])
        if key in wanted:
            segments[i] = f"{segment.partition('=')[0]}={wanted[key]}"
            found.add(key)
    if missing := set(wanted) - found:
        raise ValueError(f"fields {sorted(missing)} are not in the URL of {exchange.id}")
    return f"{base}?{'&'.join(segments)}"


def is_json_body(exchange: Exchange) -> bool:
    body = exchange.request_body or ""
    if not body.lstrip().startswith(("{", "[")):
        return False
    try:
        json.loads(body)
    except ValueError:
        return False
    return True


def today_formats(value: str, today: date) -> str | None:
    """The strftime format if ``value`` is today's date in a common format."""
    return next((f for f in DATE_FORMATS if value == today.strftime(f)), None)


# -- endpoints and digest --------------------------------------------------------------


@dataclass
class Endpoint:
    id: str
    method: str
    url: str
    exchanges: list[Exchange]
    documents: dict[str, ResponseDocuments]
    """Analysis per exchange id (only exchanges that list documents)."""

    def varying_fields(self) -> dict[str, list[str]]:
        values: dict[str, list[str]] = {}
        for ex in self.exchanges:
            for name, value in request_fields(ex).items():
                values.setdefault(name, [])
                if value not in values[name]:
                    values[name].append(value)
        return {k: v for k, v in values.items() if len(v) > 1}


def endpoints(exchanges: Sequence[Exchange], links: Sequence[str] = ()) -> list[Endpoint]:
    """Endpoints whose responses list documents, in order of first call. ``links``:
    document links of the rendered page (see ``analyse``)."""
    grouped: dict[tuple[str, str], list[Exchange]] = {}
    for ex in exchanges:
        base = normalize_url(ex.url).split("?", 1)[0]
        grouped.setdefault((ex.method, base), []).append(ex)
    out: list[Endpoint] = []
    for (method, _base), group in grouped.items():
        analysed = {ex.id: a for ex in group if (a := analyse(ex, links)).kind != "none"}
        if not analysed:
            continue
        useful = [ex for ex in group if ex.id in analysed]
        out.append(Endpoint(f"E{len(out) + 1}", method, useful[0].url, useful, analysed))
    return out


def digest_endpoints(found: Sequence[Endpoint], today: date) -> str:
    lines = ["== Endpoints the page called (captured in a browser)"]
    for ep in found:
        first = ep.exchanges[0]
        kind = "JSON" if is_json_body(first) else ("form" if first.request_body else "no")
        extra = f", X-Requested-With: {first.requested_with}" if first.requested_with else ""
        lines.append(
            f"{ep.id}: {ep.method} {ep.url} ({kind} body{extra}); {len(ep.exchanges)} call(s)"
        )
        fields = body_fields(first)
        if fields:
            shown = []
            for name, value in fields.items():
                note = " (today's date)" if today_formats(value, today) else ""
                shown.append(f"{name}={value[:80]}{note}")
            lines.append(f"  Body of {first.id}: " + "; ".join(shown)[:MAX_SAMPLE_CHARS])
        query = query_fields(first)
        if query:
            shown = [f"{name}={value[:80]}" for name, value in query.items()]
            lines.append(
                f"  URL query of {first.id} (fields named ?name): "
                + "; ".join(shown)[:MAX_SAMPLE_CHARS]
            )
        varying = ep.varying_fields()
        if varying:
            lines.append(
                "  Fields that change between calls: "
                + "; ".join(f"{k} ({', '.join(v[:6])})" for k, v in varying.items())[:800]
            )
        lines.append("  Calls:")
        for ex in ep.exchanges:
            lines.append(f"  - {ex.id} after {ex.action}: {_describe(ep.documents[ex.id])}")
        lines.extend(_sample(ep))
    return "\n".join(lines)


def _describe(found: ResponseDocuments) -> str:
    if found.kind == "json_list":
        lst = found.lists[0]
        urls = ", ".join(f"{p} ({n} items)" for p, n in lst.url_paths.items())
        return f"JSON list at {lst.path} ({lst.size} items), document URLs at {urls}"
    if found.kind == "json_html":
        keys = ", ".join(f"{k}: {t}" for k, t in found.keys.items())
        return (
            f"JSON {{{keys}}}; HTML at {found.html_path} with {len(found.documents)} document links"
        )
    return f"HTML with {len(found.documents)} document links"


def _sample(ep: Endpoint) -> list[str]:
    ex = max(ep.exchanges, key=lambda e: len(ep.documents[e.id].documents))
    found = ep.documents[ex.id]
    if found.kind == "json_list":
        text = json.dumps(_trim(found.lists[0].sample), ensure_ascii=False)
        return [f"  One item of {ex.id}: {text[:MAX_SAMPLE_CHARS]}"]
    # One sample per distinct link structure: sections of one endpoint can differ.
    lines = []
    shown: set[str] = set()
    for exchange in sorted(ep.exchanges, key=lambda e: -len(ep.documents[e.id].documents)):
        for group in sorted(ep.documents[exchange.id].groups, key=lambda g: -len(g.candidates)):
            if group.key in shown or len(shown) >= 4:
                continue
            shown.add(group.key)
            cand = group.candidates[0]
            name = f" [file name: {cand.file_name}]" if cand.file_name else ""
            lines.append(
                f"  Links of {exchange.id} ({len(group.candidates)}, DOM path "
                f'{group.example_path}): e.g. "{cand.link_text[:60]}"{name} -> {cand.url[-80:]}'
            )
            if group.elements:
                entry = entry_element(group.elements[0], exchange.url)
                parent = entry.getparent()
                html = pruned_html(parent if parent is not None else entry, 900)
                lines.append(f"    HTML: {html}")
    return lines


def _trim(node: Any, depth: int = 0) -> Any:
    """A short version of a JSON item: long strings cut, deep or long parts elided."""
    if isinstance(node, dict):
        if depth > 4:
            return "{...}"
        return {k: _trim(v, depth + 1) for k, v in list(node.items())[:25]}
    if isinstance(node, list):
        return [_trim(v, depth + 1) for v in node[:2]] + (["..."] if len(node) > 2 else [])
    if isinstance(node, str) and len(node) > 160:
        return node[:160] + "..."
    return node


# -- building a request source ---------------------------------------------------------


def templated_body(
    exchange: Exchange, vary: Mapping[str, str], today: date
) -> tuple[dict[str, str] | None, Any]:
    """(form, json) of a request source: the captured body with ``vary`` fields replaced
    by placeholders and today's date by ``{today:<format>}``."""
    body = exchange.request_body
    vary = {k: v for k, v in vary.items() if not k.startswith(QUERY_PREFIX)}
    if not body:
        if vary:
            raise ValueError(f"{exchange.id} has no body: fields {sorted(vary)} are not in it")
        return None, None
    if is_json_body(exchange):
        data = json.loads(body)
        for path, placeholder in vary.items():
            _set_path(data, path, placeholder)
        return None, _today_in_json(data, today)
    form = dict(parse_qsl(body, keep_blank_values=True))
    for name in vary:
        if name not in form:
            raise ValueError(f"field {name!r} is not in the body of {exchange.id}")
    out = {}
    for name, value in form.items():
        if name in vary:
            out[name] = vary[name]
        elif fmt := today_formats(value, today):
            out[name] = f"{{today:{fmt}}}"
        else:
            out[name] = value
    return out, None


def _set_path(data: Any, path: str, value: str) -> None:
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node.get(part)
        if node is None:
            raise ValueError(f"path {path!r} is not in the JSON body")
    last = parts[-1]
    if isinstance(node, list) and last.isdigit() and int(last) < len(node):
        node[int(last)] = value
    elif isinstance(node, dict) and last in node:
        node[last] = value
    else:
        raise ValueError(f"path {path!r} is not in the JSON body")


def _today_in_json(node: Any, today: date) -> Any:
    if isinstance(node, dict):
        return {k: _today_in_json(v, today) for k, v in node.items()}
    if isinstance(node, list):
        return [_today_in_json(v, today) for v in node]
    if isinstance(node, str) and (fmt := today_formats(node, today)):
        return f"{{today:{fmt}}}"
    return node


def field_value(exchange: Exchange, name: str) -> str | None:
    return request_fields(exchange).get(name)


def document_hosts(found: Collection[ResponseDocuments]) -> set[str]:
    return {host_of(u) for f in found for u in f.documents}


def document_extensions(found: Collection[ResponseDocuments]) -> set[str]:
    exts: set[str] = set()
    for f in found:
        for group in f.groups:
            exts |= {c.extension for c in group.candidates}
        for url in f.documents:
            exts.add(path_extension(url))
    return {e for e in exts if e in DOCUMENT_EXTENSIONS or file_extension(f"x.{e}")} & (
        DOCUMENT_EXTENSIONS
    )
