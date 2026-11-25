"""Command line: ``retrievr learn | run | check | repair | install``.

    retrievr learn https://example-issuer.com/investors "annual reports since 2024"
    retrievr run example-issuer
    retrievr check example-issuer
    retrievr repair example-issuer
    retrievr install recipes/example-issuer-ir@1.yaml

Settings come from ``--config`` (TOML), else ``./retrievr.toml`` when it exists, else the
defaults. ``--json`` prints one JSON object (for scripts); events go to stderr with
``--verbose``.

Exit codes: 0 success (learned, ok, repaired, healthy), 1 failure (rejected, error,
degraded, broken or suspect health, no recipe), 3 blocked by the site, 4 LLM budget
exceeded, 5 host session refused or missing (log in again), 2 bad usage.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO
from urllib.parse import unquote

from retrievr import __version__
from retrievr.config import Settings, load_settings
from retrievr.events import Event
from retrievr.health import HealthReport
from retrievr.recipe import FoundDocument

if TYPE_CHECKING:
    import httpx

    from retrievr.fetch import HostSession

DEFAULT_CONFIG = "retrievr.toml"
EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_BLOCKED, EXIT_BUDGET = 0, 1, 2, 3, 4
EXIT_UNAUTHORIZED = 5
_YEARS = re.compile(r"^\s*(\d{4})\s*(?:-\s*(\d{4}))?\s*$")


def parse_years(text: str | None) -> list[int] | None:
    """``2024,2025`` or ``2022-2025`` (or a mix) -> sorted years."""
    if text is None or not text.strip():
        return None
    years: set[int] = set()
    for part in text.split(","):
        m = _YEARS.match(part)
        if not m:
            raise argparse.ArgumentTypeError(f"invalid year or range: {part!r}")
        first, last = int(m.group(1)), int(m.group(2) or m.group(1))
        if last < first or last - first > 50:
            raise argparse.ArgumentTypeError(f"invalid year range: {part!r}")
        years.update(range(first, last + 1))
    return sorted(years)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrievr",
        description="Find and download documents from websites (investor relations...).",
    )
    parser.add_argument("--version", action="version", version=f"retrievr {__version__}")
    _global_options(parser, top=True)
    # The same options are accepted after the command (``retrievr learn ... -v``).
    common = argparse.ArgumentParser(add_help=False)
    _global_options(common, top=False)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    learn = sub.add_parser(
        "learn", parents=[common], help="learn a recipe for a new site (calls the LLM)"
    )
    learn.add_argument("url", help="start page, e.g. the investor-relations page")
    learn.add_argument("request", nargs="?", help='e.g. "annual and half-year reports since 2024"')
    learn.add_argument("--issuer", help="site name (slug); default from the URL")
    _years_option(learn, "only these fiscal years (the recipe covers all)")
    learn.add_argument("--download", action="store_true", help="download the matching documents")
    learn.add_argument("--no-browser", action="store_true", help="never open a browser")

    run = sub.add_parser(
        "run", parents=[common], help="replay the active recipe and download new documents"
    )
    run.add_argument("site", help="site name, or the path of a recipe file")
    run.add_argument("--request", help="filter (default: the request given to learn)")
    _years_option(run, "only these fiscal years")
    run.add_argument("--no-download", action="store_true", help="list documents only")
    run.add_argument("--no-repair", action="store_true", help="never repair a broken recipe")

    check = sub.add_parser(
        "check", parents=[common], help="health check only (no LLM, no download)"
    )
    check.add_argument("site")
    _years_option(check, "only these fiscal years")

    repair = sub.add_parser(
        "repair", parents=[common], help="check health and repair a broken recipe"
    )
    repair.add_argument("site")
    _years_option(repair, "only these fiscal years")
    repair.add_argument("--no-browser", action="store_true", help="never open a browser")

    install = sub.add_parser("install", parents=[common], help="activate a hand-written recipe")
    install.add_argument("recipe", type=Path)
    install.add_argument("--site", help="site name (default: the recipe's issuer)")
    return parser


def _global_options(parser: argparse.ArgumentParser, *, top: bool) -> None:
    """Options of every command. After the command they default to "not given", so
    that they never reset a value given before the command."""

    def default(value: object) -> object:
        return value if top else argparse.SUPPRESS

    parser.add_argument(
        "--config",
        type=Path,
        default=default(None),
        help=f"TOML settings (default ./{DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--state-dir", type=Path, default=default(None), help="recipes, SQLite state, LLM cache"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default(None), help="where documents are downloaded"
    )
    parser.add_argument(
        "--json", action="store_true", default=default(False), help="print one JSON object"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=default(False),
        help="events and logs on stderr",
    )


def _years_option(parser: argparse.ArgumentParser, text: str) -> None:
    parser.add_argument("--years", type=parse_years, help=f"{text}: 2024,2025 or 2022-2025")


def settings_from(args: argparse.Namespace) -> Settings:
    path = args.config
    if path is None and Path(DEFAULT_CONFIG).exists():
        path = Path(DEFAULT_CONFIG)
    settings = load_settings(path)
    if args.state_dir is not None:
        settings = settings.model_copy(
            update={"state": settings.state.model_copy(update={"state_dir": args.state_dir})}
        )
    if args.output_dir is not None:
        settings = settings.model_copy(
            update={"output": settings.output.model_copy(update={"output_dir": args.output_dir})}
        )
    if getattr(args, "no_browser", False):
        settings = settings.model_copy(
            update={"browser": settings.browser.model_copy(update={"enabled": False})}
        )
    return settings


def _utf8(stream: TextIO) -> None:
    """Windows consoles default to a legacy code page: page titles (accents, other
    scripts) would fail to print or end up unreadable in a redirected file."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    llm_transport: httpx.BaseTransport | None = None,
    session: HostSession | None = None,
) -> int:
    """Run the command line. A host with an LLM gateway keeps the whole command line by
    calling ``main(llm_transport=MyGateway())`` from its own script (see the README); a
    host that logged in to a site passes its ``session`` the same way."""
    if out is None:
        _utf8(sys.stdout)
        _utf8(sys.stderr)
        out = sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = settings_from(args)
    except (OSError, ValueError) as exc:
        print(f"retrievr: invalid settings: {exc}", file=sys.stderr)
        return EXIT_USAGE
    from retrievr.api import Retrievr

    on_event = _print_event if args.verbose else (None if args.json else _print_progress)
    try:
        r = Retrievr(
            settings=settings, on_event=on_event, llm_transport=llm_transport, session=session
        )
        payload, code = COMMANDS[args.command](r, args)
    except KeyboardInterrupt:
        print("retrievr: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # report, never a bare traceback for a host script
        if _expected(exc):  # a usage error (unknown site, missing file, bad value)
            payload = {"status": "error", "errors": [_one_line(exc)]}
        else:  # a bug: say its type, and log the traceback with --verbose
            if args.verbose:
                logging.getLogger("retrievr").exception("unexpected error")
            payload = {"status": "error", "errors": [f"{type(exc).__name__}: {_one_line(exc)}"]}
        code = EXIT_FAILED
    site = getattr(args, "site", None) or getattr(args, "issuer", None)
    payload = {"command": args.command, **({"site": site} if site else {}), **payload}
    saved = ("downloaded", "known_url", "duplicate_content")
    if code == EXIT_OK and any(k in saved for k in payload.get("downloads") or {}):
        payload["output_dir"] = str(settings.output.output_dir.resolve())
    if args.json:
        json.dump(payload, out, ensure_ascii=False, indent=2, default=str)
        out.write("\n")
    else:
        _print_human(payload, out)
    return code


# -- commands ------------------------------------------------------------------------


def _learn(r: Any, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    result = r.learn(
        args.url, args.request, issuer=args.issuer, years=args.years, download=args.download
    )
    payload = {
        "site": result.site,
        "status": result.status,
        "recipe": result.recipe.key if result.recipe else None,
        "recipe_path": str(result.recipe_path) if result.recipe_path else None,
        "validation": result.validation.summary() if result.validation else None,
        "listed": len(result.all_documents),
        "documents": [_document(d) for d in result.documents],
        "downloads": _downloads(result.downloads),
        "download_problems": _download_problems(result.downloads),
        "llm": vars(result.llm),
        "seconds": result.seconds,
        "warnings": result.warnings,
        "errors": result.errors,
        "notes": result.notes,
    }
    codes = {
        "learned": EXIT_OK,
        "blocked": EXIT_BLOCKED,
        "unauthorized": EXIT_UNAUTHORIZED,
        "budget_exceeded": EXIT_BUDGET,
    }
    return payload, codes.get(result.status, EXIT_FAILED)


def _run(r: Any, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    outcome = r.run(
        args.site,
        request=args.request,
        years=args.years,
        download=not args.no_download,
        repair=not args.no_repair,
    )
    payload = {
        "site": outcome.site,
        "status": outcome.status,
        "recipe": outcome.recipe,
        "health": _health(outcome.health),
        "repair": _repair(outcome.repair),
        "listed": len(outcome.all_documents),
        "documents": [_document(d) for d in outcome.documents],
        "downloads": _downloads(outcome.downloads),
        "download_problems": _download_problems(outcome.downloads),
        "llm": vars(outcome.llm),
        "warnings": outcome.warnings,
        "errors": outcome.errors,
    }
    codes = {
        "ok": EXIT_OK,
        "blocked": EXIT_BLOCKED,
        "unauthorized": EXIT_UNAUTHORIZED,
        "budget_exceeded": EXIT_BUDGET,
    }
    return payload, codes.get(outcome.status, EXIT_FAILED)


def _check(r: Any, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    report: HealthReport = r.check(args.site, years=args.years)
    payload = {"site": report.site, "status": report.verdict, "health": _health(report)}
    codes = {"ok": EXIT_OK, "stale": EXIT_OK, "blocked": EXIT_BLOCKED}
    return payload, codes.get(report.verdict, EXIT_FAILED)


def _repair_command(r: Any, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    result = r.repair(args.site, years=args.years)
    payload = {"site": result.site, "status": result.status, "repair": _repair(result)}
    return payload, EXIT_OK if result.status in ("repaired", "not_needed") else EXIT_FAILED


def _install(r: Any, args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    recipe = r.install(args.recipe, site=args.site)
    site = args.site or recipe.issuer
    return {"site": site, "status": "installed", "recipe": recipe.key}, EXIT_OK


COMMANDS = {
    "learn": _learn,
    "run": _run,
    "check": _check,
    "repair": _repair_command,
    "install": _install,
}


# -- output --------------------------------------------------------------------------


def _document(d: FoundDocument) -> dict[str, Any]:
    return {
        "url": d.url,
        "title": d.title,
        "type": d.type,
        "period": d.period,
        "fiscal_year": d.fiscal_year,
        "language": d.language,
        "date": d.date.isoformat() if d.date else None,
    }


def _downloads(downloads: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in downloads:
        status = str(getattr(item, "status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _download_problems(downloads: Sequence[Any]) -> list[str]:
    """``<status>: <url> (<reason>)`` for each document not downloaded nor already known."""
    return [
        f"{d.status}: {d.document.url} ({d.reason})"
        for d in downloads
        if d.status in ("rejected", "blocked", "unauthorized", "error")
    ]


def _health(report: HealthReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "verdict": report.verdict,
        "summary": report.summary(),
        "checks": [
            {"name": c.name, "verdict": c.verdict, "detail": c.detail} for c in report.checks
        ],
        "missing": report.missing[:20],
        "removed": report.removed[:20],
    }


def _repair(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "status": result.status,
        "reason": result.reason,
        "recipe": result.recipe.key if result.recipe else None,
        "attempts": [
            {"rung": a.rung, "recipe": a.recipe, "passed": a.passed, "detail": a.detail}
            for a in result.attempts
        ],
    }


_GENERIC_TITLES = {"pdf", "xls", "xlsx", "xlsb", "doc", "docx", "download", "télécharger", "here"}


def _display_title(doc: dict[str, Any]) -> str:
    """A link text such as "PDF" says nothing: show the file name with it."""
    title = str(doc.get("title") or "").strip()
    if title.lower() in _GENERIC_TITLES or len(title) < 4:
        name = unquote(str(doc["url"]).split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1])
        return f"{title} {name}".strip() if title else name
    return title


def _expected(exc: BaseException) -> bool:
    """Errors a user can cause (unknown site, missing or invalid file, bad value). An
    IndexError or KeyError is a LookupError too, but always a bug."""
    if isinstance(exc, (IndexError, KeyError)):
        return False
    return isinstance(exc, (LookupError, FileNotFoundError, ValueError))


def _one_line(exc: BaseException) -> str:
    """An error message on one line (YAML and pydantic errors span several)."""
    return " ".join(str(exc).split()) or type(exc).__name__


def _print_progress(event: Event) -> None:
    """Without --verbose: one line on stderr per document fetched (long downloads)."""
    if event.kind == "document_downloaded" and event.data.get("status") != "known_url":
        print(f"  {event.message}", file=sys.stderr, flush=True)


def _print_event(event: Event) -> None:
    print(f"[{event.kind}] {event.site}: {event.message}", file=sys.stderr)


def _print_human(payload: dict[str, Any], out: TextIO) -> None:
    head = f"{payload['command']} {payload.get('site', '')}: {payload.get('status')}"
    if payload.get("recipe"):
        head += f" ({payload['recipe']})"
    print(head.strip(), file=out)
    health = payload.get("health")
    if health:
        print(f"  health: {health['summary']}", file=out)
    repair = payload.get("repair")
    if repair:
        print(f"  repair: {repair['status']}, {repair['reason']}", file=out)
    if payload.get("validation"):
        print(f"  validation: {payload['validation']}", file=out)
    if "listed" in payload and (payload["listed"] or payload.get("status") == "ok"):
        selected = len(payload.get("documents") or [])
        print(f"  documents: {payload['listed']} listed, {selected} selected", file=out)
    for doc in (payload.get("documents") or [])[:15]:
        meta = " ".join(str(doc[k]) for k in ("type", "period", "language") if doc.get(k))
        print(f"    - {_display_title(doc)[:70]} [{meta}]", file=out)
    if len(payload.get("documents") or []) > 15:
        print(f"    ... ({len(payload['documents']) - 15} more; --json lists them all)", file=out)
    if payload.get("downloads"):
        parts = ", ".join(f"{n} {k}" for k, n in payload["downloads"].items())
        print(f"  downloads: {parts}", file=out)
    if payload.get("output_dir"):
        print(f"  files in: {payload['output_dir']} (see manifest.json)", file=out)
    for problem in (payload.get("download_problems") or [])[:10]:
        print(f"  download {problem}", file=out)
    llm = payload.get("llm")
    if llm and llm.get("calls"):
        print(f"  llm: {llm['calls']} calls, {llm['cost_usd']:.4f} USD", file=out)
    for warning in payload.get("warnings") or []:
        print(f"  warning: {warning}", file=out)
    for error in payload.get("errors") or []:
        print(f"  error: {error}", file=out)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
