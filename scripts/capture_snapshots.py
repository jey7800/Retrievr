"""Capture HTML snapshots of listing pages into tests/fixtures/<site>/.

Developer tool, not part of the library. It is deliberately standalone (plain httpx)
so that fixtures do not depend on the code under test. It behaves like the library:
identifiable User-Agent, 1 request per second per host, robots.txt checked, and it
stops on block signals.

Example (one --url per listing start page; quote URLs that contain brackets):
    uv run python scripts/capture_snapshots.py --site societegenerale \
        --follow-next "a[rel=next]" --url "<listing URL, 2024>" --url "<listing URL, 2025>"
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from lxml import html as lxml_html
from protego import Protego

from retrievr import __version__
from retrievr.urls import normalize_url, origin_of, resolve_url

USER_AGENT = f"retrievr/{__version__} (+https://pypi.org/project/retrievr/) snapshot-capture"
MIN_INTERVAL_S = 1.0
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


class Capture:
    def __init__(self, site: str) -> None:
        self.dir = FIXTURES / site
        self.pages_dir = self.dir / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=False
        )
        self.last_request: dict[str, float] = {}
        self.robots: dict[str, Protego | None] = {}
        self.index_path = self.dir / "index.yaml"
        loaded = (
            yaml.safe_load(self.index_path.read_text("utf-8")) if self.index_path.exists() else None
        )
        self.index: dict[str, Any] = loaded or {"pages": [], "robots": {}}

    def _wait(self, url: str) -> None:
        origin = origin_of(url)
        elapsed = time.monotonic() - self.last_request.get(origin, 0.0)
        if elapsed < MIN_INTERVAL_S:
            time.sleep(MIN_INTERVAL_S - elapsed)
        self.last_request[origin] = time.monotonic()

    def _robots_allows(self, url: str) -> bool:
        origin = origin_of(url)
        if origin not in self.robots:
            robots_url = origin + "/robots.txt"
            self._wait(robots_url)
            record: dict[str, Any]
            try:
                with httpx.Client(
                    headers={"User-Agent": USER_AGENT},
                    timeout=30,
                    follow_redirects=True,
                    max_redirects=50,
                ) as client:
                    resp = client.get(robots_url)
                record = {"status": resp.status_code, "final_url": str(resp.url)}
                if resp.is_success:
                    self.robots[origin] = Protego.parse(resp.text)
                    robots_file = self.dir / f"robots-{origin.split('//')[1]}.txt"
                    robots_file.write_bytes(resp.content)
                    record["file"] = robots_file.name
                else:
                    self.robots[origin] = None
            except httpx.TooManyRedirects:
                record = {"error": "too_many_redirects"}
                self.robots[origin] = None
                print(
                    f"WARNING robots.txt unreachable (redirect loop) at {robots_url}; policy=allow",
                    file=sys.stderr,
                )
            self.index["robots"][origin] = record
        parser = self.robots[origin]
        return True if parser is None else bool(parser.can_fetch(url, USER_AGENT))

    def fetch(self, url: str) -> httpx.Response:
        if not self._robots_allows(url):
            raise SystemExit(f"robots.txt disallows {url}")
        self._wait(url)
        resp = self.client.get(url)
        if resp.status_code in (403, 429, 503):
            raise SystemExit(f"blocked: HTTP {resp.status_code} on {url}")
        resp.raise_for_status()
        return resp

    def save(self, url: str, resp: httpx.Response) -> Path:
        name = _slug(url) + ".html"
        path = self.pages_dir / name
        path.write_bytes(resp.content)
        entry = {
            "url": url,
            "normalized_url": normalize_url(url),
            "file": f"pages/{name}",
            "status": resp.status_code,
            "content_type": resp.headers.get("content-type"),
            "etag": resp.headers.get("etag"),
            "last_modified": resp.headers.get("last-modified"),
            "sha256": hashlib.sha256(resp.content).hexdigest(),
            "captured_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
        pages = [p for p in self.index["pages"] if p["normalized_url"] != entry["normalized_url"]]
        pages.append(entry)
        self.index["pages"] = sorted(pages, key=lambda p: p["file"])
        return path

    def write_index(self) -> None:
        self.index_path.write_text(
            yaml.safe_dump(self.index, sort_keys=False, allow_unicode=True), "utf-8"
        )


def _slug(url: str) -> str:
    parts = normalize_url(url).split("//", 1)[1]
    slug = re.sub(r"%5B|%5D|[^A-Za-z0-9]+", "-", parts.split("/", 1)[1]).strip("-").lower()
    return slug[:120]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--site", required=True)
    parser.add_argument("--url", action="append", required=True)
    parser.add_argument("--follow-next", default=None, help="CSS selector of the next-page link")
    parser.add_argument("--max-pages", type=int, default=10)
    args = parser.parse_args(argv)

    capture = Capture(args.site)
    try:
        for start in args.url:
            url: str | None = start
            seen: set[str] = set()
            while url and len(seen) < args.max_pages and normalize_url(url) not in seen:
                seen.add(normalize_url(url))
                resp = capture.fetch(url)
                path = capture.save(url, resp)
                print(f"{resp.status_code} {len(resp.content):>8} {path.name}")
                url = None
                if args.follow_next:
                    tree = lxml_html.fromstring(resp.content)
                    links = tree.cssselect(args.follow_next)
                    if links and links[0].get("href"):
                        url = resolve_url(links[0].get("href"), str(resp.url))
    finally:
        capture.write_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
