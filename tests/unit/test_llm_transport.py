"""Custom LLM transport (``Retrievr(llm_transport=...)``): the supported extension point
for gateways that need their own authentication or URL scheme. No network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from retrievr import Retrievr
from retrievr.config import LLMSettings, Settings, StateSettings

FILTER_REPLY = {
    "types": ["annual_report"],
    "exclude_types": [],
    "year_from": 2025,
    "year_to": 2025,
    "languages": None,
}


class Gateway(httpx.BaseTransport):
    """A gateway adapter: adds its own token and rewrites the URL, like a host would."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.closed = False

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        assert not self.closed, "the library closed a transport it does not own"
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        reply = {
            "choices": [{"message": {"role": "assistant", "content": json.dumps(FILTER_REPLY)}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }
        return httpx.Response(200, json=reply)

    def close(self) -> None:
        self.closed = True


def _retrievr(tmp_path: Path, gateway: Gateway) -> Retrievr:
    settings = Settings(
        llm=LLMSettings(
            base_url="https://gateway.test/v1",
            model_small="deployment-a",
            extra_body={},
            prices={"deployment-a": (1.0, 2.0)},
        ),
        state=StateSettings(state_dir=tmp_path / "state"),
    )
    return Retrievr(settings=settings, llm_transport=gateway)


def test_transport_receives_standard_requests_without_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RETRIEVR_LLM_API_KEY", raising=False)
    gateway = Gateway()
    compiled, usage = _retrievr(tmp_path, gateway).compile_request("annual reports 2025")

    assert compiled.types == ["annual_report"]
    assert usage.calls == 1
    assert usage.cost_usd > 0  # price table used: the budget cap still applies
    request = gateway.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://gateway.test/v1/chat/completions"
    assert "authorization" not in request.headers  # the transport authenticates
    body = gateway.bodies[0]
    assert body["model"] == "deployment-a"
    assert body["messages"][0]["role"] == "system"
    assert "provider" not in body  # extra_body = {} replaces the default routing hints


def test_key_still_sent_when_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVR_LLM_API_KEY", "static-key")
    gateway = Gateway()
    _retrievr(tmp_path, gateway).compile_request("annual reports 2025")
    assert gateway.requests[0].headers["Authorization"] == "Bearer static-key"


def test_transport_outlives_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RETRIEVR_LLM_API_KEY", raising=False)
    gateway = Gateway()
    r = _retrievr(tmp_path, gateway)
    r.compile_request("annual reports 2025")
    r.compile_request("half-year reports 2025")  # a new run, same transport
    assert len(gateway.requests) == 2
    assert not gateway.closed


def test_command_line_passes_the_host_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host script keeps the whole command line: ``main(llm_transport=...)``."""
    import io

    import retrievr.api
    from retrievr.cli import main

    seen: dict[str, Any] = {}

    class Recorder:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        def check(self, site: str, **_kwargs: Any) -> Any:
            raise LookupError(f"no recipe for {site}")

    monkeypatch.setattr(retrievr.api, "Retrievr", Recorder)
    gateway = Gateway()
    out = io.StringIO()
    code = main(
        ["--state-dir", str(tmp_path), "--json", "check", "example"], out=out, llm_transport=gateway
    )
    assert code == 1
    assert seen["llm_transport"] is gateway
