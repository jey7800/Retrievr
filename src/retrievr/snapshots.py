"""Snapshots: record the HTTP responses of a live run, replay them offline.

A snapshot is a directory with an ``index.yaml`` and the bodies of text responses
(HTML, JSON, XML, robots.txt) under ``pages/``. Document bodies (PDF, XLSX…) are never
stored: only their status, headers, size and SHA-256. On replay, documents are served
as small synthetic files with the right magic bytes, so downloads can be exercised.

Recording is an httpx response hook, so it sees exactly what the ``Fetcher`` sees
(robots.txt, each redirect hop, listing pages) and works behind a proxy.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import httpx
import yaml
from pydantic import BaseModel, ConfigDict

from retrievr.urls import normalize_url, path_extension

SNAPSHOT_VERSION: Final = 1
INDEX_NAME = "index.yaml"
PAGES_DIR = "pages"
DOCUMENT_EXTENSIONS = frozenset(
    {"pdf", "xlsx", "xlsb", "xlsm", "xls", "docx", "doc", "pptx", "ppt", "zip", "csv"}
)
_TEXT_TYPES = ("text/", "json", "xml", "javascript")
_BODY_EXT = (("json", "json"), ("xml", "xml"), ("text/plain", "txt"), ("javascript", "js"))


class SnapshotEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    normalized_url: str
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


class SnapshotIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_version: Literal[1] = SNAPSHOT_VERSION
    site: str
    user_agent: str | None = None
    responses: list[SnapshotEntry] = []

    def by_url(self) -> dict[str, SnapshotEntry]:
        return {e.normalized_url: e for e in self.responses}


def load_index(directory: Path) -> SnapshotIndex:
    data = yaml.safe_load((directory / INDEX_NAME).read_text(encoding="utf-8"))
    return SnapshotIndex.model_validate(data)


def save_index(directory: Path, index: SnapshotIndex) -> None:
    index.responses.sort(key=lambda e: e.normalized_url)
    payload = index.model_dump(mode="json", exclude_none=True)
    (directory / INDEX_NAME).write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8"
    )


def is_text_type(content_type: str | None) -> bool:
    return content_type is not None and any(t in content_type.lower() for t in _TEXT_TYPES)


class SnapshotRecorder:
    """httpx response hook writing every GET response into a snapshot directory.

    Usage: ``Fetcher(event_hooks={"response": [recorder.hook]})``, run, then ``save()``.
    """

    def __init__(self, directory: Path, site: str, user_agent: str | None = None) -> None:
        self.directory = directory
        (directory / PAGES_DIR).mkdir(parents=True, exist_ok=True)
        if (directory / INDEX_NAME).exists():
            self.index = load_index(directory)
        else:
            self.index = SnapshotIndex(site=site, user_agent=user_agent)
        self._entries = self.index.by_url()

    def hook(self, response: httpx.Response) -> None:
        request = response.request
        if request.method != "GET":
            return
        url = str(request.url)
        key = normalize_url(url)
        content_type = response.headers.get("content-type")
        entry = SnapshotEntry(
            url=url,
            normalized_url=key,
            status=response.status_code,
            content_type=content_type,
            location=response.headers.get("location"),
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
            captured_at=datetime.now(UTC).replace(microsecond=0),
        )
        is_document = path_extension(url) in DOCUMENT_EXTENSIONS and not is_text_type(content_type)
        if not response.is_redirect and not is_document and is_text_type(content_type):
            body = response.read()
            name = f"{PAGES_DIR}/{_slug(key)}.{_body_extension(content_type)}"
            (self.directory / name).write_bytes(body)
            entry.file, entry.size = name, len(body)
            entry.sha256 = hashlib.sha256(body).hexdigest()
        self._entries[key] = entry

    def save(self) -> SnapshotIndex:
        self.index.responses = list(self._entries.values())
        save_index(self.directory, self.index)
        return self.index


class ReplayTransport(httpx.BaseTransport):
    """Serve a snapshot as if it were the network. Unknown URLs answer 404 and are
    listed in ``misses``; every requested URL is listed in ``requests``."""

    def __init__(
        self,
        directory: Path,
        *,
        documents: Literal["synthetic", "missing"] = "synthetic",
    ) -> None:
        self.directory = directory
        self.index = load_index(directory)
        self._entries = self.index.by_url()
        self.documents = documents
        self.requests: list[str] = []
        self.misses: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        entry = self._entries.get(normalize_url(url))
        if entry is not None:
            return self._from_entry(entry, request)
        if self.documents == "synthetic" and (body := synthetic_document(path_extension(url), url)):
            return httpx.Response(200, content=body, request=request)
        self.misses.append(url)
        return httpx.Response(404, content=b"", request=request)

    def _from_entry(self, entry: SnapshotEntry, request: httpx.Request) -> httpx.Response:
        if entry.etag and request.headers.get("if-none-match") == entry.etag:
            return httpx.Response(304, headers={"etag": entry.etag}, request=request)
        headers = {
            k: v
            for k, v in (
                ("content-type", entry.content_type),
                ("location", entry.location),
                ("etag", entry.etag),
                ("last-modified", entry.last_modified),
            )
            if v
        }
        if entry.file is not None:
            body = (self.directory / entry.file).read_bytes()
        elif self.documents == "synthetic" and 200 <= entry.status < 300:
            body = synthetic_document(path_extension(entry.url), entry.url) or b""
        else:
            body = b""
        return httpx.Response(entry.status, headers=headers, content=body, request=request)


def synthetic_document(extension: str, marker: str) -> bytes | None:
    """A tiny valid-looking file for ``extension`` (unique per ``marker``), or None."""
    if extension == "pdf":
        return b"%PDF-1.7\n% retrievr synthetic document\n" + marker.encode() + b"\n%%EOF\n"
    if extension in ("xlsx", "xlsb", "xlsm", "docx", "pptx"):
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


def _slug(normalized_url: str) -> str:
    rest = normalized_url.split("//", 1)[-1]
    slug = re.sub(r"%5B|%5D|[^A-Za-z0-9]+", "-", rest).strip("-").lower()[:90]
    digest = hashlib.sha256(normalized_url.encode()).hexdigest()[:8]
    return f"{slug}-{digest}"


def _body_extension(content_type: str | None) -> str:
    lowered = (content_type or "").lower()
    if "html" in lowered:
        return "html"
    return next((ext for marker, ext in _BODY_EXT if marker in lowered), "bin")
