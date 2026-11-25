# 0014 — Host sessions and recipe schema v6

Date: 2026-11-25 · Status: accepted · Extends 0008, 0012

## Context

A host application wants to collect documents from sites that need a login (investor
portals). Logging in is not Retrievr's job: credentials, vaults, multi-factor codes and
browser sign-in pages belong to the host, and Retrievr stays a library that never sees a
password. What these sites have in common once the host is logged in is ordinary
Retrievr work: JSON listing APIs, pagination, downloads, integrity, dedup and drift
detection.

Three things were missing to describe such sites:

1. a way to run with the host's session (cookies and tokens it obtained) and to tell a
   refused session from a broken recipe; today a 401 ends as `error`, which the health
   check may read as a site change and repair;
2. listings asked once per account: the accounts come from another call
   (`/accounts` -> `relId`), and the calls need a value read at login (an account
   holder id);
3. documents with no URL of their own: each is fetched by a request built from its item
   (a POST with a document key), whose reply is the file, a JSON reply holding the file
   in base64, or a JSON reply giving a pre-signed URL.

## Decision

**Host session.** `HostSession(transport, values)` (`retrievr.HostSession`), passed as
`Retrievr(session=...)` or `retrievr.cli.main(session=...)`:

- every request of the run goes through `transport` (an `httpx.BaseTransport`), which
  adds the session's cookies and tokens, keeps them up to date from the responses, and
  answers **401** when the session is gone (a redirect to the login page included);
- the fetcher's client keeps no cookies of its own (they would shadow the session's);
  robots.txt, allowed domains, the rate limit and block detection apply unchanged;
- Retrievr never closes the transport and never logs in;
- `values` (read by the host at login) are used by recipes as `{session.<name>}`.

**Status `unauthorized`.** HTTP 401 raises `UnauthorizedError`. A run, a learn or a
download that meets it stops with status `unauthorized` (event `unauthorized`, CLI exit
code 5); a recipe that needs session values the host did not give stops the same way,
naming them. It is never a reason to repair: the health check reports `blocked` and the
host logs in again.

**Recipe schema v6** (v1–v5 recipes stay valid; learned recipes never use v6):

- `{session.<name>}` in the URL and body of request sources, parameter calls and
  downloads;
- `params.<name>: {kind: response, url, method, json, items, value}`: the values a JSON
  call lists (repeats dropped; a JSON number stays a number in a JSON body);
- `response.items.extra: {name: path}`: values of the item kept with the document
  (`FoundDocument.extra`, manifest `extra`), for the host (an account number) and for
  the download request; `response.items.url` may then be omitted;
- `download: {method, url, json | form, headers, reply: file | base64 | url, path,
  identity}` on a request source: the request that fetches each document, with
  `{item.<name>}` of `extra`. With `base64` or `url`, a reply that is not JSON is the file
  itself; a pre-signed URL must stay within the allowed domains.

A document fetched by a request has an **identity URL**: the download URL plus its
identity values as query pairs (`.../download/report?doc_id=42`). It is the key of dedup
and history, so `identity` must name values that do not change between sessions (a
document id, not a per-session download key). Identity URLs are never probed with a GET.

## Consequences

The host keeps everything about logging in and about its own filing (fund names, folders);
Retrievr keeps one job. The same recipe runs warm with zero LLM calls, deduplicates across
sessions and still detects a listing that changed. Not covered: recipes learned by the
LLM on such sites (they are hand-written for now), a listing whose request needs values
from the previous page of results, and replies that need a second request to unlock the
file.
