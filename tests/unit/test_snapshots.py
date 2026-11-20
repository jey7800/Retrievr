from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from retrievr.config import FetchSettings
from retrievr.fetch import Fetcher, HTTPStatusError
from retrievr.snapshots import ReplayTransport, SnapshotRecorder, load_index
from tests.conftest import FakeClock

SITE = "https://ir.example.com"
PAGE = "<html><body><a href='/files/q1-2025.pdf'>Q1</a></body></html>"


def _record(tmp_path: Path, no_network: respx.MockRouter, clock: FakeClock) -> SnapshotRecorder:
    no_network.get(f"{SITE}/robots.txt").respond(200, text="User-agent: *\nAllow: /\n")
    no_network.get(f"{SITE}/old").respond(301, headers={"location": f"{SITE}/docs"})
    no_network.get(f"{SITE}/docs").respond(200, html=PAGE, headers={"etag": '"v1"'})
    no_network.get(f"{SITE}/files/q1-2025.pdf").respond(
        200, content=b"%PDF-1.7 real bytes", headers={"content-type": "application/pdf"}
    )
    recorder = SnapshotRecorder(tmp_path, "example", user_agent="ua")
    with Fetcher(
        FetchSettings(), event_hooks={"response": [recorder.hook]}, clock=clock, sleep=clock.sleep
    ) as fetcher:
        page = fetcher.get(f"{SITE}/old")
        assert page.text == PAGE  # the hook reads the body without consuming it
        with fetcher.stream(f"{SITE}/files/q1-2025.pdf") as resp:
            assert resp.read() == b"%PDF-1.7 real bytes"
    recorder.save()
    return recorder


def test_recorder_stores_text_bodies_only(
    tmp_path: Path, no_network: respx.MockRouter, clock: FakeClock
) -> None:
    _record(tmp_path, no_network, clock)
    entries = load_index(tmp_path).by_url()
    assert entries[f"{SITE}/old"].status == 301
    assert entries[f"{SITE}/old"].location == f"{SITE}/docs"
    docs = entries[f"{SITE}/docs"]
    assert docs.file and (tmp_path / docs.file).read_text("utf-8") == PAGE
    assert docs.etag == '"v1"'
    pdf = entries[f"{SITE}/files/q1-2025.pdf"]
    assert pdf.file is None and pdf.status == 200  # document bodies are never stored
    assert entries[f"{SITE}/robots.txt"].file is not None


def test_replay_serves_snapshot_without_network(
    tmp_path: Path, no_network: respx.MockRouter, clock: FakeClock, make_fetcher: Any
) -> None:
    _record(tmp_path, no_network, clock)
    replay = ReplayTransport(tmp_path)
    with Fetcher(FetchSettings(), transport=replay, clock=clock, sleep=clock.sleep) as fetcher:
        page = fetcher.get(f"{SITE}/old")
        assert page.final_url == f"{SITE}/docs"
        assert page.text == PAGE
        assert fetcher.get(f"{SITE}/docs", etag='"v1"').not_modified
        with fetcher.stream(f"{SITE}/files/q1-2025.pdf") as resp:
            assert resp.read().startswith(b"%PDF-")  # synthetic stand-in
        with pytest.raises(HTTPStatusError, match="404"):
            fetcher.get(f"{SITE}/unknown")
    assert replay.misses == [f"{SITE}/unknown"]
    assert f"{SITE}/robots.txt" in replay.requests


def test_replay_ignores_environment_proxy(
    tmp_path: Path, no_network: respx.MockRouter, clock: FakeClock, monkeypatch: Any
) -> None:
    _record(tmp_path, no_network, clock)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    replay = ReplayTransport(tmp_path)
    with Fetcher(FetchSettings(), transport=replay, clock=clock, sleep=clock.sleep) as fetcher:
        assert fetcher.get(f"{SITE}/docs").status == 200
        assert fetcher._client._trust_env is False


def test_replay_can_refuse_documents(
    tmp_path: Path, no_network: respx.MockRouter, clock: FakeClock
) -> None:
    _record(tmp_path, no_network, clock)
    replay = ReplayTransport(tmp_path, documents="missing")
    response = replay.handle_request(httpx.Request("GET", f"{SITE}/files/other.pdf"))
    assert response.status_code == 404
