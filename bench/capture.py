"""Capture a snapshot of live pages (opt-in: this hits the network).

It goes through the library's ``Fetcher`` (identifiable User-Agent, robots.txt, rate
limit, block detection) and records every response with ``SnapshotRecorder``.

Examples:
    # record what a recipe run fetches
    uv run python -m bench.capture --site societegenerale \
        --recipe recipes/societegenerale-ir@1.yaml --years 2024 2025 --out tests/fixtures/x
    # record start pages, their rel=next pages and the year pages they link to
    uv run python -m bench.capture --site hsbc --url "<start URL>" \
        --follow "a[href*='/results/']" --follow-next "a[rel=next]"
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from pathlib import Path

from retrievr.candidates import parse_html, select
from retrievr.fetch import BlockedError, Fetcher, FetchError
from retrievr.recipe import load_recipe, run_recipe
from retrievr.snapshots import SnapshotRecorder
from retrievr.urls import host_of, normalize_url, resolve_url

BENCH = Path(__file__).resolve().parent


def capture_urls(
    fetcher: Fetcher,
    starts: list[str],
    *,
    follow: str | None,
    follow_next: str | None,
    max_pages: int,
) -> list[str]:
    """Fetch start pages, the pages they link to via ``follow`` (one hop, same host)
    and every ``follow_next`` page. Returns the URLs fetched."""
    hosts = {host_of(u) for u in starts}
    queue: deque[tuple[str, int]] = deque((u, 0) for u in starts)
    seen: set[str] = set()
    fetched: list[str] = []
    while queue and len(fetched) < max_pages:
        url, depth = queue.popleft()
        key = normalize_url(url)
        if key in seen:
            continue
        seen.add(key)
        try:
            page = fetcher.get(url)
        except BlockedError:
            raise
        except FetchError as exc:
            print(f"skip {url}: {exc}", file=sys.stderr)
            continue
        fetched.append(url)
        print(f"{page.status} {len(page.content):>9} {url}")
        root = parse_html(page.content)
        if follow_next:
            for el in select(root, follow_next)[:1]:
                if el.get("href"):
                    queue.append((resolve_url(el.get("href"), page.final_url), depth))
        if follow and depth == 0:
            for el in select(root, follow):
                href = el.get("href")
                if href and host_of(nxt := resolve_url(href, page.final_url)) in hosts:
                    queue.append((nxt, 1))
    return fetched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture a snapshot of live pages.")
    parser.add_argument("--site", required=True, help="site id (see bench/sites.yaml)")
    parser.add_argument("--out", type=Path, help="snapshot directory")
    parser.add_argument("--url", action="append", default=[], help="start URL (repeatable)")
    parser.add_argument("--follow", help="CSS selector of links to capture one hop away")
    parser.add_argument("--follow-next", help="CSS selector of the next-page link")
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument("--recipe", type=Path, help="record a run of this recipe instead")
    parser.add_argument("--years", type=int, nargs="*", help="years for --recipe")
    args = parser.parse_args(argv)
    if not args.url and not args.recipe:
        parser.error("give --url or --recipe")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    out: Path = args.out or BENCH / "snapshots" / args.site
    recorder = SnapshotRecorder(out, args.site)
    with Fetcher(event_hooks={"response": [recorder.hook]}) as fetcher:
        recorder.index.user_agent = fetcher.settings.user_agent
        try:
            if args.recipe:
                result = run_recipe(load_recipe(args.recipe), fetcher, years=args.years)
                print(f"{result.status}: {len(result.documents)} documents, {result.errors}")
            else:
                capture_urls(
                    fetcher,
                    args.url,
                    follow=args.follow,
                    follow_next=args.follow_next,
                    max_pages=args.max_pages,
                )
        except BlockedError as exc:
            print(f"blocked: {exc}", file=sys.stderr)
            return 2
        finally:
            index = recorder.save()
    print(f"{len(index.responses)} responses in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
