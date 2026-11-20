"""Shared fixtures. Tests never hit the network: every request goes through respx."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import respx

from retrievr.config import FetchSettings
from retrievr.fetch import Fetcher
from retrievr.snapshots import ReplayTransport, synthetic_document

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
    body = synthetic_document("pdf", marker)
    assert body is not None
    return body


def xlsx_bytes(marker: str) -> bytes:
    body = synthetic_document("xlsx", marker)
    assert body is not None
    return body


@contextmanager
def serve_site(name: str) -> Iterator[ReplayTransport]:
    """Serve the snapshot tests/fixtures/<name>/ (via respx) for the duration of the block.

    Every request of the default ``Fetcher`` client is answered by a ``ReplayTransport``;
    ``.requests`` and ``.misses`` record what was asked and what the snapshot lacked.
    """
    replay = ReplayTransport(FIXTURES / name)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay.handle_request)
        yield replay


@pytest.fixture
def sg_site() -> Iterator[ReplayTransport]:
    with serve_site("societegenerale") as replay:
        yield replay


@pytest.fixture
def no_network() -> Iterator[respx.MockRouter]:
    """A respx router with no routes: any unmocked request fails the test."""
    with respx.mock(assert_all_called=False) as router:
        yield router
