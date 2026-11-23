"""Recipe store and internal state.

- Recipes: ``<state_dir>/recipes/<recipe>@<version>.yaml``, written once (immutable).
- State: ``<state_dir>/state.sqlite3`` (stdlib ``sqlite3``): active recipe per site, runs,
  the LLM ledger, classification decisions and compiled request filters. SQLite is an
  internal detail, not a public interface.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from retrievr.classify import Decision
from retrievr.health import HealthReport, KnownDocument
from retrievr.llm import CallRecord
from retrievr.recipe import FoundDocument, Recipe, load_recipe

SCHEMA = """
CREATE TABLE IF NOT EXISTS active (
    site TEXT PRIMARY KEY, recipe_key TEXT NOT NULL, path TEXT NOT NULL,
    activated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, site TEXT, started_at TEXT NOT NULL,
    finished_at TEXT, status TEXT, documents INTEGER, llm_calls INTEGER, cost_usd REAL,
    detail TEXT);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, step TEXT NOT NULL,
    model TEXT NOT NULL, key TEXT NOT NULL, messages TEXT NOT NULL, response TEXT NOT NULL,
    input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, latency_s REAL,
    cached INTEGER NOT NULL, strategy TEXT, attempts INTEGER, at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decisions (
    recipe TEXT NOT NULL, normalized_url TEXT NOT NULL, type TEXT, period TEXT,
    fiscal_year INTEGER, language TEXT, source TEXT NOT NULL, at TEXT NOT NULL,
    PRIMARY KEY (recipe, normalized_url));
CREATE TABLE IF NOT EXISTS filters (
    request_hash TEXT PRIMARY KEY, request TEXT NOT NULL, filter TEXT NOT NULL,
    at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS site_requests (
    site TEXT PRIMARY KEY, request TEXT NOT NULL, at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sites (
    site TEXT PRIMARY KEY, start_url TEXT, status TEXT NOT NULL, detail TEXT,
    at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    site TEXT NOT NULL, normalized_url TEXT NOT NULL, url TEXT NOT NULL, title TEXT,
    source_id TEXT, params TEXT NOT NULL, type TEXT, period TEXT, fiscal_year INTEGER,
    language TEXT, recipe_key TEXT NOT NULL, first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL, removed_at TEXT, PRIMARY KEY (site, normalized_url));
CREATE TABLE IF NOT EXISTS health (
    id INTEGER PRIMARY KEY AUTOINCREMENT, site TEXT NOT NULL, run_id TEXT,
    recipe_key TEXT NOT NULL, verdict TEXT NOT NULL, checks TEXT NOT NULL, at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS repairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, site TEXT NOT NULL, run_id TEXT, day TEXT NOT NULL,
    rung TEXT NOT NULL, outcome TEXT NOT NULL, recipe_key TEXT, cost_usd REAL,
    detail TEXT, at TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class StateStore:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.recipes_dir = self.state_dir / "recipes"
        self.recipes_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "state.sqlite3"
        with self._db() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            yield conn

    # -- recipes -------------------------------------------------------------------

    def next_version(self, name: str) -> int:
        versions = [
            int(p.stem.rsplit("@", 1)[1])
            for p in self.recipes_dir.glob(f"{name}@*.yaml")
            if p.stem.rsplit("@", 1)[1].isdigit()
        ]
        return max(versions, default=0) + 1

    def save_recipe(self, recipe: Recipe) -> Path:
        """Write a new recipe version; never overwrites an existing one."""
        path = self.recipes_dir / f"{recipe.key}.yaml"
        data = recipe.model_dump(mode="json", by_alias=True, exclude_none=True)
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
        with path.open("x", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def activate(self, site: str, recipe: Recipe, path: Path) -> None:
        # A recipe of the store is kept by file name, so the state directory can be
        # used from any working directory (or moved).
        inside = path.resolve().parent == self.recipes_dir.resolve()
        stored = path.name if inside else str(path.resolve())
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO active VALUES (?, ?, ?, ?)",
                (site, recipe.key, stored, _now()),
            )

    def active_recipe(self, site: str) -> Recipe | None:
        with self._db() as db:
            row = db.execute("SELECT path FROM active WHERE site = ?", (site,)).fetchone()
        if row is None:
            return None
        stored = Path(row[0])
        path = stored if stored.is_absolute() else self.recipes_dir / stored.name
        return load_recipe(path)

    def sites(self) -> list[str]:
        """Sites that have an active recipe."""
        with self._db() as db:
            rows = db.execute("SELECT site FROM active ORDER BY site").fetchall()
        return [r[0] for r in rows]

    # -- runs and ledger -----------------------------------------------------------

    def start_run(self, kind: str, site: str | None) -> str:
        run_id = uuid.uuid4().hex
        with self._db() as db:
            db.execute(
                "INSERT INTO runs (id, kind, site, started_at) VALUES (?, ?, ?, ?)",
                (run_id, kind, site, _now()),
            )
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        documents: int,
        llm_calls: int,
        cost_usd: float,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        with self._db() as db:
            db.execute(
                "UPDATE runs SET finished_at = ?, status = ?, documents = ?, llm_calls = ?, "
                "cost_usd = ?, detail = ? WHERE id = ?",
                (
                    _now(),
                    status,
                    documents,
                    llm_calls,
                    cost_usd,
                    json.dumps(detail or {}, ensure_ascii=False),
                    run_id,
                ),
            )

    def record_llm_call(self, run_id: str | None, record: CallRecord) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO llm_calls (run_id, step, model, key, messages, response, "
                "input_tokens, output_tokens, cost_usd, latency_s, cached, strategy, attempts, "
                "at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    record.step,
                    record.model,
                    record.key,
                    json.dumps(record.messages, ensure_ascii=False),
                    record.response,
                    record.input_tokens,
                    record.output_tokens,
                    record.cost_usd,
                    record.latency_s,
                    int(record.cached),
                    record.strategy,
                    record.attempts,
                    record.at,
                ),
            )

    def llm_calls(self, run_id: str) -> list[dict[str, Any]]:
        with self._db() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT * FROM llm_calls WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- decisions, filters, requests ----------------------------------------------

    def save_decisions(self, recipe: str, decisions: Mapping[str, Decision]) -> None:
        with self._db() as db:
            db.executemany(
                "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (recipe, url, d.type, d.period, d.fiscal_year, d.language, d.source, _now())
                    for url, d in decisions.items()
                ],
            )

    def decisions(self, recipe: str) -> dict[str, Decision]:
        with self._db() as db:
            rows = db.execute(
                "SELECT normalized_url, type, period, fiscal_year, language, source "
                "FROM decisions WHERE recipe = ?",
                (recipe,),
            ).fetchall()
        return {r[0]: Decision(r[1], r[2], r[3], r[4], r[5]) for r in rows}

    def save_filter(self, request_hash: str, request: str, filter_json: str) -> None:
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO filters VALUES (?, ?, ?, ?)",
                (request_hash, request, filter_json, _now()),
            )

    def filter_json(self, request_hash: str) -> str | None:
        with self._db() as db:
            row = db.execute(
                "SELECT filter FROM filters WHERE request_hash = ?", (request_hash,)
            ).fetchone()
        return str(row[0]) if row else None

    def set_site_request(self, site: str, request: str) -> None:
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO site_requests VALUES (?, ?, ?)", (site, request, _now())
            )

    def site_request(self, site: str) -> str | None:
        with self._db() as db:
            row = db.execute("SELECT request FROM site_requests WHERE site = ?", (site,)).fetchone()
        return str(row[0]) if row else None

    # -- sites, history, health, repairs -------------------------------------------

    def set_site(
        self,
        site: str,
        *,
        start_url: str | None = None,
        status: str = "ok",
        detail: str | None = None,
    ) -> None:
        """Record the site's status (``ok`` or ``degraded``); keeps a known start URL."""
        with self._db() as db:
            row = db.execute("SELECT start_url FROM sites WHERE site = ?", (site,)).fetchone()
            url = start_url or (row[0] if row else None)
            db.execute(
                "INSERT OR REPLACE INTO sites VALUES (?, ?, ?, ?, ?)",
                (site, url, status, detail, _now()),
            )

    def site_status(self, site: str) -> tuple[str, str | None, str | None] | None:
        """(status, start_url, detail) of a site, if recorded."""
        with self._db() as db:
            row = db.execute(
                "SELECT status, start_url, detail FROM sites WHERE site = ?", (site,)
            ).fetchone()
        return (str(row[0]), row[1], row[2]) if row else None

    def record_documents(self, site: str, recipe_key: str, docs: Iterable[FoundDocument]) -> None:
        """Remember the documents a healthy run listed (with their metadata)."""
        now = _now()
        rows = [
            (
                site,
                d.normalized_url,
                d.url,
                d.title,
                d.source_id,
                json.dumps(dict(d.params), sort_keys=True),
                d.type,
                d.period,
                d.fiscal_year,
                d.language,
                recipe_key,
                now,
                now,
            )
            for d in docs
        ]
        with self._db() as db:
            db.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL) "
                "ON CONFLICT (site, normalized_url) DO UPDATE SET url = excluded.url, "
                "title = excluded.title, source_id = excluded.source_id, "
                "params = excluded.params, type = excluded.type, period = excluded.period, "
                "fiscal_year = excluded.fiscal_year, language = excluded.language, "
                "recipe_key = excluded.recipe_key, last_seen = excluded.last_seen, "
                "removed_at = NULL",
                rows,
            )

    def mark_removed(self, site: str, normalized_urls: Iterable[str]) -> None:
        """Documents gone from the site: no longer expected in listings."""
        with self._db() as db:
            db.executemany(
                "UPDATE documents SET removed_at = ? WHERE site = ? AND normalized_url = ?",
                [(_now(), site, u) for u in normalized_urls],
            )

    def known_documents(self, site: str) -> list[KnownDocument]:
        with self._db() as db:
            rows = db.execute(
                "SELECT normalized_url, url, title, source_id, params, type, period, "
                "fiscal_year, language FROM documents WHERE site = ? AND removed_at IS NULL "
                "ORDER BY normalized_url",
                (site,),
            ).fetchall()
        return [
            KnownDocument(r[0], r[1], r[2] or "", r[3] or "", json.loads(r[4]), *r[5:9])
            for r in rows
        ]

    def record_health(self, run_id: str | None, report: HealthReport) -> None:
        checks = [{"name": c.name, "verdict": c.verdict, "detail": c.detail} for c in report.checks]
        with self._db() as db:
            db.execute(
                "INSERT INTO health (site, run_id, recipe_key, verdict, checks, at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (report.site, run_id, report.recipe, report.verdict, json.dumps(checks), _now()),
            )

    def health_history(self, site: str, limit: int = 10) -> list[tuple[str, str]]:
        """(verdict, recipe key) of the latest health checks, newest first."""
        with self._db() as db:
            rows = db.execute(
                "SELECT verdict, recipe_key FROM health WHERE site = ? ORDER BY id DESC LIMIT ?",
                (site, limit),
            ).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    def record_repair(
        self,
        site: str,
        run_id: str | None,
        day: date,
        *,
        rung: str,
        outcome: str,
        recipe_key: str | None,
        cost_usd: float,
        detail: str,
    ) -> None:
        with self._db() as db:
            db.execute(
                "INSERT INTO repairs (site, run_id, day, rung, outcome, recipe_key, cost_usd, "
                "detail, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    site,
                    run_id,
                    day.isoformat(),
                    rung,
                    outcome,
                    recipe_key,
                    cost_usd,
                    detail,
                    _now(),
                ),
            )

    def llm_repairs_on(self, site: str, day: date) -> int:
        """LLM repair attempts of a site on a day (deterministic relocation is free)."""
        with self._db() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM repairs WHERE site = ? AND day = ? AND rung LIKE 'llm%'",
                (site, day.isoformat()),
            ).fetchone()
        return int(row[0])
