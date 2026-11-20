"""Shared fixtures. Tests never hit the network: every request goes through respx."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml

from retrievr.config import FetchSettings
from retrievr.fetch import Fetcher
from retrievr.urls import normalize_url, path_extension

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
RECIPES = ROOT / "recipes"


class FakeClock:
    """Monotonic clock whose ``sleep`` advances time instantly."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_fetcher(clock: FakeClock) -> Iterator[Any]:
    created: list[Fetcher] = []

    def factory(**overrides: Any) -> Fetcher:
        fetcher = Fetcher(FetchSettings(**overrides), clock=clock, sleep=clock.sleep)
        created.append(fetcher)
        return fetcher

    yield factory
    for f in created:
        f.close()


def pdf_bytes(marker: str) -> bytes:
    return b"%PDF-1.7\n% retrievr test document\n" + marker.encode() + b"\n%%EOF\n"


def xlsx_bytes(marker: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("xl/workbook.xml", f"<workbook><!-- {marker} --></workbook>")
    return buf.getvalue()


@dataclass
class SiteSnapshot:
    """Serves a captured site from tests/fixtures/<site>/index.yaml."""

    name: str
    pages: dict[str, Path] = field(default_factory=dict)
    robots: dict[str, bytes] = field(default_factory=dict)
    robots_redirects: dict[str, str] = field(default_factory=dict)
    requested: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, name: str) -> SiteSnapshot:
        base = FIXTURES / name
        index = yaml.safe_load((base / "index.yaml").read_text("utf-8"))
        snap = cls(name)
        for page in index["pages"]:
            snap.pages[page["normalized_url"]] = base / page["file"]
        for origin, record in index.get("robots", {}).items():
            if "file" not in record:
                continue
            final = record.get("final_url", origin + "/robots.txt")
            final_origin = final.rsplit("/robots.txt", 1)[0]
            snap.robots[final_origin] = (base / record["file"]).read_bytes()
            if final_origin != origin:
                snap.robots_redirects[origin] = final
        return snap

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requested.append(url)
        if request.url.path == "/robots.txt":
            origin = f"{request.url.scheme}://{request.url.host}"
            if origin in self.robots_redirects:
                return httpx.Response(301, headers={"location": self.robots_redirects[origin]})
            body = self.robots.get(origin)
            return httpx.Response(200, content=body) if body else httpx.Response(404)
        key = normalize_url(url)
        if key in self.pages:
            return httpx.Response(
                200,
                content=self.pages[key].read_bytes(),
                headers={"content-type": "text/html; charset=utf-8"},
            )
        ext = path_extension(url)
        if ext == "pdf":
            return httpx.Response(
                200, content=pdf_bytes(key), headers={"content-type": "application/pdf"}
            )
        if ext == "xlsx":
            return httpx.Response(
                200,
                content=xlsx_bytes(key),
                headers={
                    "content-type": "application/vnd.openxmlformats-officedocument"
                    ".spreadsheetml.sheet"
                },
            )
        self.unexpected.append(url)
        return httpx.Response(404)


@pytest.fixture
def sg_site() -> Iterator[SiteSnapshot]:
    snap = SiteSnapshot.load("societegenerale")
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=snap.handler)
        yield snap


@pytest.fixture
def no_network() -> Iterator[respx.MockRouter]:
    """A respx router with no routes: any unmocked request fails the test."""
    with respx.mock(assert_all_called=False) as router:
        yield router
