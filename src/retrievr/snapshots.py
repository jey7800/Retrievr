"""Snapshots: record the HTTP responses of a live run, replay them offline.

A snapshot is a directory with an ``index.yaml`` and the bodies of text responses
(HTML, JSON, XML, robots.txt) under ``pages/``. Document bodies (PDF, XLSX…) are never
stored: only their status, headers, size and SHA-256. On replay, documents are served
as small synthetic files with the right magic bytes, so downloads can be exercised.

Recording is an httpx response hook, so it sees exactly what the ``Fetcher`` sees
(robots.txt, each redirect hop, listing pages) and works behind a proxy. The browser
capture (``retrievr.browser``) records into the same format.

Format v2 adds POST requests: an entry is identified by its method, normalized URL and
canonical request body (JSON with sorted keys, or sorted form fields).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal
from urllib.parse import parse_qsl, urlencode

import httpx
import yaml
from pydantic import BaseModel, ConfigDict

from retrievr.urls import normalize_url, path_extension

SNAPSHOT_VERSION: Final = 2
INDEX_NAME = "index.yaml"
PAGES_DIR = "pages"
DOCUMENT_EXTENSIONS = frozenset(
    {"pdf", "xlsx", "xlsb", "xlsm", "xls", "docx", "doc", "pptx", "ppt", "zip", "csv", "xbri"}
)
_TEXT_TYPES = ("text/", "json", "xml", "javascript")
_BODY_EXT = (("json", "json"), ("xml", "xml"), ("text/plain", "txt"), ("javascript", "js"))
_DOCUMENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.ms-excel.sheet.binary.macroenabled.12": "xlsb",
    "application/zip": "zip",
    "text/csv": "csv",
}

Method = Literal["GET", "POST"]
EntryKey = tuple[str, str, str | None]


class SnapshotEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    normalized_url: str
    method: Method = "GET"
    request_body: str | None = None
    """Canonical request body (see ``canonical_body``), for POST requests."""
    status: int
    content_type: str | None = None
    location: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    file: str | None = None
    """Body file relative to the snapshot directory; None when the body is not stored."""
    sha256: str | None = None
    size: int | None = None
    captured_at: datetime

    @property
    def key(self) -> EntryKey:
        return entry_key(self.method, self.normalized_url, self.request_body)


class SnapshotIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_version: Literal[1, 2] = SNAPSHOT_VERSION
    site: str
    user_agent: str | None = None
    responses: list[SnapshotEntry] = []

    def by_url(self) -> dict[str, SnapshotEntry]:
        """GET entries by normalized URL."""
        return {e.normalized_url: e for e in self.responses if e.method == "GET"}

    def by_key(self) -> dict[EntryKey, SnapshotEntry]:
        return {e.key: e for e in self.responses}

    def captured_on(self) -> datetime | None:
        """When the snapshot was taken (latest response)."""
        return max((e.captured_at for e in self.responses), default=None)


def entry_key(method: str, normalized_url: str, body: str | None) -> EntryKey:
    return method.upper(), normalized_url, body or None


def canonical_body(content: bytes | str | None, content_type: str | None = None) -> str | None:
    """A stable text form of a request body: JSON with sorted keys, form fields sorted,
    anything else as text. None for an empty body."""
    if not content:
        return None
    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    lowered = (content_type or "").lower()
    if "json" in lowered or text.lstrip().startswith(("{", "[")):
        try:
            return json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False)
        except ValueError:
            pass
    if "form" in lowered or ("=" in text and not text.lstrip().startswith("<")):
        pairs = parse_qsl(text, keep_blank_values=True)
        if pairs:
            return urlencode(sorted(pairs))
    return text


def document_extension(content_type: str | None) -> str | None:
    """File extension of a document content type (``application/pdf`` -> ``pdf``)."""
    lowered = (content_type or "").split(";")[0].strip().lower()
    return _DOCUMENT_TYPES.get(lowered)


def load_index(directory: Path) -> SnapshotIndex:
    data = yaml.safe_load((directory / INDEX_NAME).read_text(encoding="utf-8"))
    return SnapshotIndex.model_validate(data)


def save_index(directory: Path, index: SnapshotIndex) -> None:
    index.responses.sort(key=lambda e: (e.normalized_url, e.method, e.request_body or ""))
    index.snapshot_version = SNAPSHOT_VERSION
    payload = index.model_dump(mode="json", exclude_none=True, exclude_defaults=False)
    for entry in payload["responses"]:
        if entry.get("method") == "GET":
            del entry["method"]  # keep v1-style entries for GET
    (directory / INDEX_NAME).write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8"
    )


def is_text_type(content_type: str | None) -> bool:
    return content_type is not None and any(t in content_type.lower() for t in _TEXT_TYPES)


class SnapshotRecorder:
    """Write responses into a snapshot directory.

    As an httpx response hook: ``Fetcher(event_hooks={"response": [recorder.hook]})``,
    run, then ``save()``. The browser capture calls ``record`` directly.
    """

    def __init__(self, directory: Path, site: str, user_agent: str | None = None) -> None:
        self.directory = directory
        (directory / PAGES_DIR).mkdir(parents=True, exist_ok=True)
        if (directory / INDEX_NAME).exists():
            self.index = load_index(directory)
        else:
            self.index = SnapshotIndex(site=site, user_agent=user_agent)
        self._entries = self.index.by_key()

    def hook(self, response: httpx.Response) -> None:
        request = response.request
        if request.method not in ("GET", "POST"):
            return
        content_type = response.headers.get("content-type")
        body = None
        if not response.is_redirect and is_text_type(content_type):
            body = response.read()
        self.record(
            method=request.method,
            url=str(request.url),
            request_body=canonical_body(request.content, request.headers.get("content-type")),
            status=response.status_code,
            headers=response.headers,
            body=body,
        )

    def record(
        self,
        *,
        method: str,
        url: str,
        request_body: str | None,
        status: int,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> SnapshotEntry:
        """Add one exchange. The body is stored only for text responses, never for
        documents."""
        normalized = normalize_url(url)
        content_type = headers.get("content-type")
        entry = SnapshotEntry(
            url=url,
            normalized_url=normalized,
            method="POST" if method.upper() == "POST" else "GET",
            request_body=request_body,
            status=status,
            content_type=content_type,
            location=headers.get("location"),
            etag=headers.get("etag"),
            last_modified=headers.get("last-modified"),
            captured_at=datetime.now(UTC).replace(microsecond=0),
        )
        is_document = path_extension(url) in DOCUMENT_EXTENSIONS or bool(
            document_extension(content_type)
        )
        if body is not None and not is_document and is_text_type(content_type):
            name = f"{PAGES_DIR}/{_slug(entry.key)}.{_body_extension(content_type)}"
            (self.directory / name).write_bytes(body)
            entry.file, entry.size = name, len(body)
            entry.sha256 = hashlib.sha256(body).hexdigest()
        self._entries[entry.key] = entry
        return entry

    def save(self) -> SnapshotIndex:
        self.index.responses = list(self._entries.values())
        save_index(self.directory, self.index)
        return self.index


class ReplayTransport(httpx.BaseTransport):
    """Serve a snapshot as if it were the network. Unknown requests answer 404 and are
    listed in ``misses``; every request is listed in ``requests``."""

    def __init__(
        self,
        directory: Path,
        *,
        documents: Literal["synthetic", "missing"] = "synthetic",
        lenient_bodies: bool = False,
    ) -> None:
        """``lenient_bodies``: a POST whose exact body was not captured is answered by a
        captured POST to the same URL whose body holds every field of the request with
        the same value, plus others (the closest one). For offline benchmarks, where a
        learned recipe may copy a slightly smaller body than the captured follow-up pages.
        """
        self.directory = directory
        self.index = load_index(directory)
        self._entries = self.index.by_key()
        self.documents = documents
        self.lenient_bodies = lenient_bodies
        self.requests: list[str] = []
        self.misses: list[str] = []

    def find(self, method: str, url: str, body: str | None) -> SnapshotEntry | None:
        exact = self._entries.get(entry_key(method, normalize_url(url), body))
        if exact is not None or not self.lenient_bodies or not body:
            return exact
        wanted = _flat_body(body)
        best: SnapshotEntry | None = None
        for entry in self.index.responses:
            if entry.method != method.upper() or entry.normalized_url != normalize_url(url):
                continue
            have = _flat_body(entry.request_body or "")
            if all(have.get(k) == v for k, v in wanted.items()) and (
                best is None or len(have) < len(_flat_body(best.request_body or ""))
            ):
                best = entry
        return best

    def body_of(self, entry: SnapshotEntry) -> bytes:
        if entry.file is not None:
            return (self.directory / entry.file).read_bytes()
        if self.documents == "synthetic" and 200 <= entry.status < 300:
            ext = path_extension(entry.url) or document_extension(entry.content_type) or ""
            return synthetic_document(ext, entry.url) or b""
        return b""

    def headers_of(self, entry: SnapshotEntry) -> dict[str, str]:
        return {
            k: v
            for k, v in (
                ("content-type", entry.content_type),
                ("location", entry.location),
                ("etag", entry.etag),
                ("last-modified", entry.last_modified),
            )
            if v
        }

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url if request.method == "GET" else f"{request.method} {url}")
        body = canonical_body(request.content, request.headers.get("content-type"))
        entry = self.find(request.method, url, body)
        if entry is not None:
            return self._from_entry(entry, request)
        synthetic = synthetic_document(path_extension(url), url)
        if self.documents == "synthetic" and request.method == "GET" and synthetic:
            return httpx.Response(200, content=synthetic, request=request)
        self.misses.append(url if request.method == "GET" else f"{request.method} {url} {body}")
        return httpx.Response(404, content=b"", request=request)

    def _from_entry(self, entry: SnapshotEntry, request: httpx.Request) -> httpx.Response:
        if entry.etag and request.headers.get("if-none-match") == entry.etag:
            return httpx.Response(304, headers={"etag": entry.etag}, request=request)
        return httpx.Response(
            entry.status,
            headers=self.headers_of(entry),
            content=self.body_of(entry),
            request=request,
        )


def _flat_body(body: str) -> dict[str, str]:
    """Fields of a canonical body: JSON leaves by dotted path, or form fields."""
    if body.lstrip().startswith(("{", "[")):
        try:
            data = json.loads(body)
        except ValueError:
            return {"": body}
        out: dict[str, str] = {}
        stack: list[tuple[str, object]] = [("", data)]
        while stack:
            prefix, node = stack.pop()
            if isinstance(node, dict):
                stack.extend((f"{prefix}.{k}" if prefix else str(k), v) for k, v in node.items())
            elif isinstance(node, list):
                stack.extend((f"{prefix}.{i}" if prefix else str(i), v) for i, v in enumerate(node))
            else:
                out[prefix] = json.dumps(node)
        return out
    return dict(parse_qsl(body, keep_blank_values=True))


def synthetic_document(extension: str, marker: str) -> bytes | None:
    """A tiny valid-looking file for ``extension`` (unique per ``marker``), or None."""
    if extension == "pdf":
        return b"%PDF-1.7\n% retrievr synthetic document\n" + marker.encode() + b"\n%%EOF\n"
    if extension in ("xlsx", "xlsb", "xlsm", "docx", "pptx", "zip", "xbri"):
        main = {"docx": "word/document.xml", "pptx": "ppt/presentation.xml"}.get(
            extension, "xl/workbook.xml"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr(main, f"<doc><!-- {marker} --></doc>")
        return buf.getvalue()
    if extension == "csv":
        return f"marker\n{marker}\n".encode()
    return None


def _slug(key: EntryKey) -> str:
    method, normalized_url, body = key
    rest = normalized_url.split("//", 1)[-1]
    slug = re.sub(r"%5B|%5D|[^A-Za-z0-9]+", "-", rest).strip("-").lower()[:90]
    # GET pages keep their v1 file names; a POST adds its method and body to the hash.
    seed = normalized_url if method == "GET" else f"{method}|{normalized_url}|{body or ''}"
    digest = hashlib.sha256(seed.encode()).hexdigest()[:8]
    return f"{slug}-{digest}"


def _body_extension(content_type: str | None) -> str:
    lowered = (content_type or "").lower()
    if "html" in lowered:
        return "html"
    return next((ext for marker, ext in _BODY_EXT if marker in lowered), "bin")
