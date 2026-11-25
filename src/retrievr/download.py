"""Streaming download with integrity checks, deduplication, file tree and manifest.json.

Never calls an LLM.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import dataclasses
import hashlib
import json
import logging
import re
import time
import uuid
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from retrievr.candidates import json_path
from retrievr.fetch import BlockedError, Fetcher, FetchError, UnauthorizedError
from retrievr.recipe import DownloadRequest, FoundDocument, Recipe, download_request
from retrievr.snapshots import document_extension
from retrievr.urls import resolve_url

log = logging.getLogger("retrievr.download")

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
TMP_DIR = ".retrievr-tmp"

_ZIP = (b"PK\x03\x04",)
_OLE = (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",)
MAGIC: dict[str, tuple[bytes, ...]] = {
    "pdf": (b"%PDF-",),
    "xlsx": _ZIP,
    "xlsm": _ZIP,
    "xlsb": _ZIP,
    "docx": _ZIP,
    "pptx": _ZIP,
    "zip": _ZIP,
    "xls": _OLE,
    "doc": _OLE,
    "ppt": _OLE,
}
_HTML_TYPES = ("text/html", "application/xhtml+xml")
_MARKUP_EXTENSIONS = ("xhtml", "html", "htm")
"""Documents that are markup themselves (ESEF annual reports are inline XBRL in XHTML)."""
_MARKUP_STARTS = (b"<?xml", b"<html", b"<!doctype html")
_BOM_AND_SPACE = b"\xef\xbb\xbf \t\r\n"
_OOXML_MAIN_PART = {
    "xlsx": "xl/",
    "xlsm": "xl/",
    "xlsb": "xl/",
    "docx": "word/",
    "pptx": "ppt/",
}
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    normalized_url: str
    alternate_urls: list[str] = []
    title: str
    issuer: str
    type: str | None
    date: date | None
    period: str | None
    fiscal_year: int | None
    language: str | None
    sha256: str
    size: int
    content_type: str | None
    path: str
    """Relative to the output directory, POSIX separators."""
    recipe: str
    recipe_version: int
    score: float
    justification: str
    downloaded_at: datetime
    extra: dict[str, str] = {}
    """Values of the listed item the recipe keeps (recipe schema v6, ``items.extra``)."""


class Manifest:
    """``<output_dir>/manifest.json``: one entry per distinct document (by content)."""

    def __init__(self, path: Path, entries: Iterable[ManifestEntry] = ()) -> None:
        self.path = path
        self.entries: list[ManifestEntry] = list(entries)

    @classmethod
    def load(cls, output_dir: Path) -> Manifest:
        path = output_dir / MANIFEST_NAME
        if not path.exists():
            return cls(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("manifest_version") != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version in {path}")
        return cls(path, (ManifestEntry.model_validate(e) for e in data["documents"]))

    def save(self) -> None:
        self.entries.sort(key=lambda e: e.path)
        payload = {
            "manifest_version": MANIFEST_VERSION,
            "documents": [e.model_dump(mode="json") for e in self.entries],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(text, encoding="utf-8")
        _replace_with_retry(tmp, self.path)

    def by_url(self, normalized_url: str) -> ManifestEntry | None:
        for e in self.entries:
            if e.normalized_url == normalized_url or normalized_url in e.alternate_urls:
                return e
        return None

    def by_sha(self, sha256: str) -> ManifestEntry | None:
        return next((e for e in self.entries if e.sha256 == sha256), None)

    def path_owner(self, rel_path: str) -> ManifestEntry | None:
        key = rel_path.lower()  # Windows and macOS file systems are case-insensitive
        return next((e for e in self.entries if e.path.lower() == key), None)


DownloadStatus = Literal[
    "downloaded", "known_url", "duplicate_content", "rejected", "blocked", "unauthorized", "error"
]
STOPPING_STATUSES = ("blocked", "unauthorized")
"""Outcomes after which no other document is tried (the site or the session refuses us)."""


@dataclass(frozen=True)
class DownloadOutcome:
    document: FoundDocument
    status: DownloadStatus
    entry: ManifestEntry | None = None
    reason: str | None = None


class Downloader:
    def __init__(
        self,
        fetcher: Fetcher,
        output_dir: Path | str,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        on_outcome: Callable[[DownloadOutcome, int, int], None] | None = None,
    ) -> None:
        """``on_outcome(outcome, index, total)`` is called after each document (progress)."""
        self.fetcher = fetcher
        self.output_dir = Path(output_dir)
        self._now = now
        self._on_outcome = on_outcome

    def download_all(
        self, recipe: Recipe, documents: Iterable[FoundDocument]
    ) -> list[DownloadOutcome]:
        """Download every document not already known; the manifest is saved at the end
        (also when a block or an unexpected error interrupts the loop)."""
        manifest = Manifest.load(self.output_dir)
        self._forget_missing_files(manifest)
        outcomes: list[DownloadOutcome] = []
        todo = list(documents)
        try:
            for index, doc in enumerate(todo, start=1):
                outcome = self.download(recipe, doc, manifest)
                outcomes.append(outcome)
                if self._on_outcome is not None:
                    self._on_outcome(outcome, index, len(todo))
                if outcome.status in STOPPING_STATUSES:
                    break
        finally:
            manifest.save()
            with contextlib.suppress(OSError):  # left only while a download runs
                (self.output_dir / TMP_DIR).rmdir()
        return outcomes

    def _forget_missing_files(self, manifest: Manifest) -> None:
        """A file deleted from the output directory is downloaded again: its manifest
        entry would otherwise keep it out as already known."""
        kept = []
        for entry in manifest.entries:
            if (self.output_dir / Path(*PurePosixPath(entry.path).parts)).is_file():
                kept.append(entry)
            else:
                log.info("%s is no longer on disk: it will be downloaded again", entry.path)
        manifest.entries = kept

    def download(self, recipe: Recipe, doc: FoundDocument, manifest: Manifest) -> DownloadOutcome:
        known = manifest.by_url(doc.normalized_url)
        if known is not None:
            return DownloadOutcome(doc, "known_url", known)
        tmp_dir = self.output_dir / TMP_DIR
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"{uuid.uuid4().hex}.part"
        try:
            sha, size, content_type, head = self._fetch_to(tmp, doc, recipe)
            if not doc.extension:  # opaque link: name the file from what was served
                ext = document_extension(content_type) or sniff_extension(head)
                doc = dataclasses.replace(doc, extension=ext or "bin")
                if reason := check_magic(doc.extension, head):
                    return DownloadOutcome(doc, "rejected", reason=reason)
            reason = check_container(doc.extension, tmp)
            if reason:
                return DownloadOutcome(doc, "rejected", reason=reason)
            duplicate = manifest.by_sha(sha)
            if duplicate is not None:
                if doc.normalized_url not in duplicate.alternate_urls:
                    duplicate.alternate_urls.append(doc.normalized_url)
                return DownloadOutcome(doc, "duplicate_content", duplicate)
            rel = self._free_path(recipe.issuer, doc, manifest)
            final = self.output_dir / Path(*PurePosixPath(rel).parts)
            final.parent.mkdir(parents=True, exist_ok=True)
            _replace_with_retry(tmp, final)
            entry = ManifestEntry(
                url=doc.url,
                normalized_url=doc.normalized_url,
                title=doc.title,
                issuer=recipe.issuer,
                type=doc.type,
                date=doc.date,
                period=doc.period,
                fiscal_year=doc.fiscal_year,
                language=doc.language,
                sha256=sha,
                size=size,
                content_type=content_type,
                path=rel,
                recipe=recipe.recipe,
                recipe_version=recipe.version,
                score=doc.score,
                justification=doc.justification,
                downloaded_at=self._now(),
                extra=dict(doc.extra),
            )
            manifest.entries.append(entry)
            return DownloadOutcome(doc, "downloaded", entry)
        except _RejectedError as exc:
            return DownloadOutcome(doc, "rejected", reason=str(exc))
        except BlockedError as exc:
            return DownloadOutcome(doc, "blocked", reason=str(exc))
        except UnauthorizedError as exc:
            return DownloadOutcome(doc, "unauthorized", reason=str(exc))
        except FetchError as exc:
            return DownloadOutcome(doc, "error", reason=str(exc))
        except httpx.RequestError as exc:  # failure while streaming the body
            return DownloadOutcome(doc, "error", reason=f"network error: {exc!r}")
        finally:
            tmp.unlink(missing_ok=True)

    def _fetch_to(
        self, tmp: Path, doc: FoundDocument, recipe: Recipe
    ) -> tuple[str, int, str | None, bytes]:
        spec = download_request(recipe, doc)
        if spec is not None:
            return self._fetch_by_request(tmp, doc, recipe, spec)
        with self.fetcher.stream(doc.url, allowed_domains=recipe.allowed_domains) as resp:
            content_type = resp.headers.get("content-type")
            return self._save(tmp, doc, resp.iter_bytes(), content_type)

    def _fetch_by_request(
        self, tmp: Path, doc: FoundDocument, recipe: Recipe, spec: DownloadRequest
    ) -> tuple[str, int, str | None, bytes]:
        """Schema v6: the request built from the document's item; its reply is the file,
        or a JSON reply holding the file (base64) or the URL of the file."""
        url, body, headers = spec.build(doc.extra, self.fetcher.session_values, date.today())
        domains = recipe.allowed_domains
        with self.fetcher.stream(
            url, allowed_domains=domains, method=spec.method, content=body, headers=headers
        ) as resp:
            content_type = resp.headers.get("content-type")
            if spec.reply == "file" or "json" not in (content_type or "").lower():
                return self._save(tmp, doc, resp.iter_bytes(), content_type)
            data = _json_reply(self.fetcher.read(resp), url)
        value = json_path(data, spec.path) if spec.path else data
        if not isinstance(value, str) or not value.strip():
            raise _RejectedError(f"no {spec.reply} value at {spec.path or 'the reply'}")
        if spec.reply == "base64":
            try:
                content = base64.b64decode(value.strip(), validate=True)
            except (binascii.Error, ValueError) as exc:
                raise _RejectedError(f"invalid base64 content at {spec.path}") from exc
            return self._save(tmp, doc, iter([content]), None)
        file_url = resolve_url(value.strip(), url)
        with self.fetcher.stream(file_url, allowed_domains=domains) as resp:
            return self._save(tmp, doc, resp.iter_bytes(), resp.headers.get("content-type"))

    def _save(
        self,
        tmp: Path,
        doc: FoundDocument,
        chunks: Iterator[bytes],
        content_type: str | None,
    ) -> tuple[str, int, str | None, bytes]:
        """Write ``chunks`` to ``tmp``, checking type, magic bytes and size on the way."""
        limit = self.fetcher.settings.max_download_bytes
        digest = hashlib.sha256()
        size = 0
        head = b""
        media = (content_type or "").split(";")[0].strip().lower()
        if media in _HTML_TYPES and doc.extension not in _MARKUP_EXTENSIONS:
            raise _RejectedError(f"HTML page instead of a .{doc.extension or '?'} file")
        with tmp.open("wb") as fh:
            for chunk in chunks:
                if len(head) < 16:
                    head = (head + chunk)[:16]
                    if len(head) >= 8 and (reason := check_magic(doc.extension, head)):
                        raise _RejectedError(reason)
                size += len(chunk)
                if size > limit:
                    raise _RejectedError(f"file larger than {limit} bytes")
                digest.update(chunk)
                fh.write(chunk)
        if size == 0:
            raise _RejectedError("empty file")
        if reason := check_magic(doc.extension, head):
            raise _RejectedError(reason)
        return digest.hexdigest(), size, content_type, head

    def _free_path(self, issuer: str, doc: FoundDocument, manifest: Manifest) -> str:
        base = target_path(issuer, doc)
        stem, dot, ext = base.rpartition(".")
        candidates = [base]
        suffix = hashlib.sha256(doc.normalized_url.encode()).hexdigest()[:8]
        candidates.append(f"{stem}_{suffix}{dot}{ext}")
        candidates.extend(f"{stem}_{suffix}-{n}{dot}{ext}" for n in range(2, 100))
        for rel in candidates:
            on_disk = self.output_dir / Path(*PurePosixPath(rel).parts)
            if manifest.path_owner(rel) is None and not on_disk.exists():
                return rel
        raise FetchError("could not find a free file name", url=doc.url)


class _RejectedError(Exception):
    pass


def _json_reply(content: bytes, url: str) -> object:
    try:
        return json.loads(content)
    except ValueError as exc:
        raise _RejectedError(f"the reply of {url} is not JSON: {exc}") from exc


def target_path(issuer: str, doc: FoundDocument) -> str:
    """``<issuer>/<year>/<type>_<period>_<lang>.<ext>`` with safe components."""
    year = str(doc.fiscal_year) if doc.fiscal_year else "unknown-year"
    name = "_".join(
        _safe(part)
        for part in (doc.type or "document", doc.period or "noperiod", doc.language or "und")
    )
    ext = _safe(doc.extension or "bin").lower()
    return f"{_safe(issuer)}/{year}/{name}.{ext}"


def sniff_extension(head: bytes) -> str | None:
    """Extension guessed from the first bytes (PDF, or a zip container)."""
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(_ZIP):
        return "zip"
    return None


def check_magic(extension: str, head: bytes) -> str | None:
    """Reason to reject the file from its first bytes, or None."""
    stripped = head.lstrip(_BOM_AND_SPACE)[:15].lower()
    if extension in _MARKUP_EXTENSIONS:
        if len(stripped) >= 5 and not stripped.startswith(_MARKUP_STARTS):
            return f"not a .{extension} document: {head[:8]!r}"
        return None
    expected = MAGIC.get(extension)
    if stripped.startswith((b"<!doctype", b"<html", b"<?xml", b"<head", b"<body")) and (
        expected is not None
    ):
        return "HTML content instead of a binary document"
    if expected is not None and not head.startswith(expected):
        return f"bad magic bytes for .{extension}: {head[:8]!r}"
    return None


def check_container(extension: str, path: Path) -> str | None:
    """Deeper check for Office Open XML files: a valid zip with the expected main part."""
    main_part = _OOXML_MAIN_PART.get(extension)
    if main_part is None:
        return None
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        return f"corrupted .{extension} (bad zip)"
    if "[Content_Types].xml" not in names or not any(n.startswith(main_part) for n in names):
        return f"not a .{extension} document"
    return None


def _safe(value: str) -> str:
    """One path component: letters, digits, ``_`` and ``-`` only (no dots, no separators)."""
    cleaned = re.sub(r"-{2,}", "-", _UNSAFE.sub("-", value)).strip("-")
    return cleaned or "x"


def _replace_with_retry(src: Path, dst: Path, *, attempts: int = 5, delay_s: float = 0.2) -> None:
    """Atomic rename with retries: antivirus and indexers can briefly lock files on Windows."""
    for attempt in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            log.debug("file locked, retrying replace %s -> %s", src, dst)
            time.sleep(delay_s * (attempt + 1))
