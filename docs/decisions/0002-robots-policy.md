# 0002 — robots.txt policy

Date: 2026-11-20 · Status: accepted

## Context

robots.txt is respected by default. Some sites serve it badly: one site in the benchmark
was observed redirect-looping on robots.txt (more than 50 redirects), and later
redirecting it to another host. The behaviour when robots.txt cannot be read must be
explicit and logged, not implicit.

## Decision

Per origin, resolved once per `Fetcher` and cached:

| robots.txt response | Outcome | Effect |
| --- | --- | --- |
| 2xx | `parsed` | rules applied (protego), matched on the User-Agent product token |
| 4xx | `missing` | allow all (RFC 9309 "unavailable") |
| 5xx | `server_error` | deny all (RFC 9309 "unreachable"), WARNING |
| more than 5 redirects, network error | `unreachable` | `fetch.robots_unreachable` (default `allow`), WARNING |

Redirects are followed manually, up to 5 hops, including to another host.
`fetch.robots = "ignore"` is an explicit opt-out, logged at WARNING when the fetcher is
created. A recipe cannot change either setting.

## Consequences

The default `allow` on an unreachable robots.txt favours availability. Users who prefer
the strict reading set `robots_unreachable = "deny"`.
