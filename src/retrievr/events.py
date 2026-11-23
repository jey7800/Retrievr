"""Typed events emitted to the host application (Python callback + log entry)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

log = logging.getLogger("retrievr.events")

EventKind = Literal[
    "recipe_learned",  # a learned recipe passed validation and is active
    "recipe_rejected",  # a candidate recipe failed validation: nothing activated
    "blocked",  # the site refused us (403/429/503, challenge, CAPTCHA)
    "budget_exceeded",  # the LLM cost cap of the run was reached
    "llm_error",  # the LLM endpoint failed or never returned valid output
    "run_finished",  # a run (learn or replay) ended, with its status
    "health_alert",  # a health check gave suspect, stale or broken
    "recipe_repaired",  # a repaired recipe passed validation and replaced the broken one
    "site_degraded",  # the recipe is broken and no repair passed: the old one stays active
    "document_downloaded",  # one document handled by the downloader (progress, with status)
]


@dataclass(frozen=True)
class Event:
    kind: EventKind
    site: str
    message: str
    data: Mapping[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


EventHandler = Callable[[Event], None]


def emit(handler: EventHandler | None, event: Event) -> None:
    warn = ("recipe_rejected", "blocked", "health_alert", "site_degraded")
    level = logging.WARNING if event.kind in warn else logging.INFO
    log.log(level, "%s %s: %s", event.kind, event.site, event.message)
    if handler is not None:
        try:
            handler(event)
        except Exception:  # the host's callback must not break a run
            log.exception("event handler failed on %s", event.kind)
