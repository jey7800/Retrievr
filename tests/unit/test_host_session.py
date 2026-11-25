"""Recipe schema v6 and host sessions (ADR 0014): a site the host application logged in
to. The host gives a transport that carries the session and the values read at login;
recipes read accounts from one call, list documents per account and fetch each document
by a request built from its item. A refused session ends the run as ``unauthorized``,
never as a broken recipe. No network: a synthetic portal answers."""

from __future__ import annotations

import base64
import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pydantic
import pytest
import yaml

from retrievr import HostSession
from retrievr.api import Retrievr
from retrievr.cli import EXIT_UNAUTHORIZED, main
from retrievr.config import BrowserSettings, LLMSettings, OutputSettings, Settings, StateSettings
from retrievr.events import Event
from retrievr.fetch import Fetcher, UnauthorizedError
from retrievr.recipe import fetched_by_request, recipe_from_mapping, run_recipe
from retrievr.snapshots import synthetic_document

PORTAL = "https://portal.example-manager.com"
FILES = "https://files.example-manager.com"
RELATIONSHIPS = f"{PORTAL}/api/v1/helper/relationships?personId={{session.person_id}}"
TODAY = date(2026, 11, 25)
DOCS = {  # account id -> documents (id, title, date)
    11: [
        (1, "Capital call notice", "2025-03-31T00:00:00"),
        (2, "Quarterly report Q1 2025", "2025-04-15T00:00:00"),
    ],
    12: [(3, "Distribution notice", "2025-06-30T00:00:00")],
}


def _pdf(name: str) -> bytes:
    return synthetic_document("pdf", name) or b"%PDF-1.4"


class Portal:
    """A synthetic investor portal: every API call needs the session cookie; download
    keys change on every login (they are not an identity)."""

    def __init__(self) -> None:
        self.login = 1
        self.valid = True
        self.calls: list[tuple[str, str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, path, body))
        if path == "/robots.txt":
            return httpx.Response(404)
        if request.url.host == "files.example-manager.com":
            return httpx.Response(
                200, content=_pdf(path), headers={"content-type": "application/pdf"}
            )
        if not self.valid or "session=s3cr3t" not in request.headers.get("cookie", ""):
            return httpx.Response(401)
        if path == "/api/v1/helper/relationships":
            if request.url.params.get("personId") != "P-7":
                return httpx.Response(400)
            return httpx.Response(200, json=[{"relId": 11}, {"relId": 12}, {"relId": 11}])
        if path == "/api/v1/reports/docs":
            assert body is not None and isinstance(
                body["relId"], int
            )  # a number, as sent by the site
            assert body["personId"] == "P-7" and body["endDate"] == TODAY.isoformat()
            return httpx.Response(
                200, json=[self._item(body["relId"], *d) for d in DOCS[body["relId"]]]
            )
        if path == "/api/v1/download/report":
            assert body is not None
            doc_id = int(body["Key"].split("-")[1])
            if doc_id == 1:  # the file itself
                return httpx.Response(
                    200, content=_pdf("1"), headers={"content-type": "application/pdf"}
                )
            if doc_id == 2:  # base64 in a JSON reply
                encoded = base64.b64encode(_pdf("2")).decode()
                return httpx.Response(200, json={"content": encoded})
            return httpx.Response(200, json={"url": f"{FILES}/signed/{doc_id}.pdf?sig=abc"})
        return httpx.Response(404)

    def _item(self, rel: int, doc_id: int, title: str, as_of: str) -> dict[str, Any]:
        return {
            "title": title,
            "asOf": as_of,
            "docId": doc_id,
            "downloadKey": f"L{self.login}-{doc_id}",
            "accountNumber": f"ACC-{rel}",
        }


class SessionTransport(httpx.BaseTransport):
    """What a host gives: it adds the session cookie and owns its own lifetime."""

    def __init__(self, portal: Portal) -> None:
        self.portal = portal
        self.closed = False

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.headers["Cookie"] = "session=s3cr3t"
        return self.portal.handle(request)

    def close(self) -> None:
        self.closed = True


def _recipe(reply: str = "base64", path: str | None = "content", **download: Any) -> dict[str, Any]:
    return {
        "schema_version": 6,
        "recipe": "example-portal",
        "version": 1,
        "issuer": "example-manager",
        "site": "portal.example-manager.com",
        "allowed_domains": ["portal.example-manager.com", "files.example-manager.com"],
        "sources": [
            {
                "id": "reports",
                "kind": "request",
                "method": "POST",
                "url": f"{PORTAL}/api/v1/reports/docs",
                "json": {
                    "personId": "{session.person_id}",
                    "relId": "{account}",
                    "startDate": "2024-01-01",
                    "endDate": "{today:%Y-%m-%d}",
                },
                "params": {
                    "account": {
                        "kind": "response",
                        "url": RELATIONSHIPS,
                        "items": "",
                        "value": "relId",
                    }
                },
                "response": {
                    "format": "json",
                    "items": {
                        "path": "",
                        "title": "title",
                        "date": "asOf",
                        "extra": {
                            "key": "downloadKey",
                            "doc_id": "docId",
                            "account": "accountNumber",
                        },
                    },
                },
                "download": {
                    "url": f"{PORTAL}/api/v1/download/report",
                    "json": {"Extension": "pdf", "Key": "{item.key}"},
                    "reply": reply,
                    "path": path,
                    "identity": ["doc_id"],
                    **download,
                },
            }
        ],
    }


def _session(portal: Portal) -> HostSession:
    return HostSession(transport=SessionTransport(portal), values={"person_id": "P-7"})


def _retrievr(tmp_path: Path, session: HostSession, events: list[Event] | None = None) -> Retrievr:
    settings = Settings(
        llm=LLMSettings(api_key_env="RETRIEVR_TEST_NO_SUCH_KEY", cache="off"),
        state=StateSettings(state_dir=tmp_path / "state"),
        output=OutputSettings(output_dir=tmp_path / "out"),
        browser=BrowserSettings(enabled=False),
    )
    return Retrievr(
        settings=settings,
        on_event=events.append if events is not None else None,
        session=session,
        sleep=lambda _s: None,
        today=TODAY,
    )


def _install(r: Retrievr, tmp_path: Path, data: dict[str, Any]) -> None:
    path = tmp_path / "example-portal@1.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    r.install(path)


def test_accounts_then_documents_per_account_with_session_values() -> None:
    portal = Portal()
    recipe = recipe_from_mapping(_recipe())
    with Fetcher(session=_session(portal), sleep=lambda _s: None) as fetcher:
        result = run_recipe(recipe, fetcher, today=TODAY)
    assert result.status == "ok", result.errors
    assert sorted(d.extra["doc_id"] for d in result.documents) == ["1", "2", "3"]
    first = min(result.documents, key=lambda d: d.url)
    assert first.url == f"{PORTAL}/api/v1/download/report?doc_id=1"
    assert first.extra == {"key": "L1-1", "doc_id": "1", "account": "ACC-11"}
    assert first.date == date(2025, 3, 31)
    # One listing call per account (repeats dropped); session values stay out of params.
    listings = [c for c in portal.calls if c[1] == "/api/v1/reports/docs"]
    assert [c[2]["relId"] for c in listings] == [11, 12]
    assert all("session.person_id" not in d.params for d in result.documents)
    assert recipe.session_values() == {"person_id"}


def test_the_host_transport_is_used_never_closed_and_keeps_the_cookies() -> None:
    portal = Portal()
    session = _session(portal)
    with Fetcher(session=session, sleep=lambda _s: None) as fetcher:
        fetcher.get(f"{PORTAL}/api/v1/helper/relationships?personId=P-7")
    assert not session.transport.closed  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="not both"):
        Fetcher(transport=httpx.MockTransport(portal.handle), session=session)


def test_a_refused_session_is_unauthorized_not_a_broken_recipe(tmp_path: Path) -> None:
    portal = Portal()
    events: list[Event] = []
    r = _retrievr(tmp_path, _session(portal), events)
    _install(r, tmp_path, _recipe())
    assert r.run("example-manager").status == "ok"
    portal.valid = False  # the session expired
    outcome = r.run("example-manager")
    assert outcome.status == "unauthorized"
    assert "HTTP 401" in outcome.errors[0]
    assert outcome.repair is None and outcome.health is None
    assert any(e.kind == "unauthorized" for e in events)
    assert r.store.site_status("example-manager")[0] == "ok"  # type: ignore[index]
    report = r.check("example-manager")
    assert report.verdict == "blocked" and not report.repair_needed


def test_without_session_values_the_run_says_what_is_missing(tmp_path: Path) -> None:
    portal = Portal()
    r = _retrievr(tmp_path, HostSession(transport=SessionTransport(portal)))
    _install(r, tmp_path, _recipe())
    outcome = r.run("example-manager")
    assert outcome.status == "unauthorized"
    assert "person_id" in outcome.errors[0]
    assert portal.calls == []


def test_documents_are_fetched_by_request_and_deduplicated_by_identity(tmp_path: Path) -> None:
    portal = Portal()
    r = _retrievr(tmp_path, _session(portal))
    _install(r, tmp_path, _recipe(reply="url", path="url"))
    first = r.run("example-manager")
    assert first.status == "ok", first.errors
    statuses = {o.document.extra["doc_id"]: o.status for o in first.downloads}
    # 1: the reply is the file; 3: a pre-signed URL on the files host; 2: a JSON reply
    # without a URL at the path is refused, never saved as a document.
    assert statuses == {"1": "downloaded", "2": "rejected", "3": "downloaded"}
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    entries = sorted(manifest["documents"], key=lambda e: e["extra"]["doc_id"])
    assert [e["extra"]["account"] for e in entries] == ["ACC-11", "ACC-12"]
    assert all(e["path"].endswith(".pdf") for e in entries)
    assert any(c[1] == "/signed/3.pdf" for c in portal.calls)
    # A new login gives new download keys: the documents are the same ones.
    portal.login = 2
    again = r.run("example-manager")
    assert again.status == "ok", again.errors
    statuses = {o.document.extra["doc_id"]: o.status for o in again.downloads}
    assert statuses == {"1": "known_url", "2": "rejected", "3": "known_url"}
    assert again.health is not None and again.health.verdict == "ok"


def test_a_base64_reply_and_a_file_reply_are_both_saved(tmp_path: Path) -> None:
    portal = Portal()
    r = _retrievr(tmp_path, _session(portal))
    _install(r, tmp_path, _recipe(reply="base64", path="content"))
    outcome = r.run("example-manager")
    by_id = {o.document.extra["doc_id"]: o for o in outcome.downloads}
    assert by_id["1"].status == "downloaded"  # application/pdf: the file itself
    assert by_id["2"].status == "downloaded"  # JSON with the base64 content
    assert by_id["3"].status == "rejected"  # JSON without content at the path
    saved = (tmp_path / "out" / by_id["2"].entry.path).read_bytes()  # type: ignore[union-attr]
    assert saved == _pdf("2")
    posts = [c for c in portal.calls if c[1] == "/api/v1/download/report"]
    assert posts[0][0] == "POST" and posts[0][2] == {"Extension": "pdf", "Key": "L1-1"}


def test_identity_urls_are_never_probed(tmp_path: Path) -> None:
    recipe = recipe_from_mapping(_recipe())
    assert fetched_by_request(recipe, f"{PORTAL}/api/v1/download/report?doc_id=9")
    assert not fetched_by_request(recipe, f"{PORTAL}/files/report.pdf")
    portal = Portal()
    r = _retrievr(tmp_path, _session(portal))
    _install(r, tmp_path, _recipe())
    r.run("example-manager", download=False)
    DOCS[11].pop()  # a document leaves the listing
    try:
        r.check("example-manager")
    finally:
        DOCS[11].append((2, "Quarterly report Q1 2025", "2025-04-15T00:00:00"))
    assert not [c for c in portal.calls if c[0] == "GET" and c[1].startswith("/api/v1/download")]


def test_the_command_line_takes_the_host_session(tmp_path: Path) -> None:
    portal = Portal()
    portal.valid = False
    path = tmp_path / "example-portal@1.yaml"
    path.write_text(yaml.safe_dump(_recipe(), sort_keys=False), encoding="utf-8")
    argv = ["--state-dir", str(tmp_path / "state"), "--output-dir", str(tmp_path / "out"), "--json"]
    out = io.StringIO()
    assert main([*argv, "install", str(path)], out=out) == 0
    code = main([*argv, "run", "example-manager"], out=out, session=_session(portal))
    assert code == EXIT_UNAUTHORIZED
    assert '"status": "unauthorized"' in out.getvalue()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 5}, "requires schema_version 6"),
        ({"download_url": "https://elsewhere.example.org/dl"}, "download URL outside domains"),
        ({"download_key": "{item.missing}"}, "not in items.extra"),
        ({"listing_item": True}, "belongs to download only"),
        ({"param_placeholder": True}, "only {session.<name>} placeholders"),
        ({"identity_url": True}, "may only use {item.<name>} of identity"),
    ],
)
def test_invalid_v6_recipes_are_rejected(change: dict[str, Any], message: str) -> None:
    data = _recipe()
    source = data["sources"][0]
    if "schema_version" in change:
        data["schema_version"] = change["schema_version"]
    if "download_url" in change:
        source["download"]["url"] = change["download_url"]
    if "download_key" in change:
        source["download"]["json"]["Key"] = change["download_key"]
    if "listing_item" in change:
        source["json"]["key"] = "{item.key}"
    if "param_placeholder" in change:
        source["params"]["account"]["url"] += "&y={year}"
    if "identity_url" in change:
        source["download"]["url"] = f"{PORTAL}/api/v1/download/{{item.key}}"
    with pytest.raises(pydantic.ValidationError, match=message):
        recipe_from_mapping(data)


def test_401_is_its_own_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404 if request.url.path == "/robots.txt" else 401)

    with (
        Fetcher(transport=httpx.MockTransport(handler), sleep=lambda _s: None) as fetcher,
        pytest.raises(UnauthorizedError),
    ):
        fetcher.get(f"{PORTAL}/api/v1/helper/relationships")
