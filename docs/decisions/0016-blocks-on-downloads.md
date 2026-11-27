# 0016 — Bot protections answering in place of a document

Date: 2026-11-27 · Status: accepted · Extends 0002

## Context

A site behind a bot protection may let a client read its pages and still refuse its
files. Seen live: every PDF of an issuer answered HTTP 200 with a 212-byte HTML page
whose only content was the challenge script of the protection (Imperva), served to one
network and not to another. Retrievr caught it only through its file checks: each
document was `rejected` as "HTML page instead of a .pdf file", 28 times in a row, the
run ended `ok` with a warning, and nothing said the site was refusing us. The probe that
decides whether a learned recipe's documents are downloadable also read such a reply as
"ok".

Block detection already knew this page, but only for pages, not for the replies to
downloads, and it knew few protections.

## Decision

- **Downloads.** When the reply to a document is a page (HTML content type, or HTML
  where the magic bytes of the file were expected), its first 64 KB go through the same
  block detection as pages. A challenge or block page raises `BlockedError`: the outcome
  is `blocked`, with the protection named, and the downloader stops (no other file is
  requested). Any other page stays `rejected`.
- **Run and learn status.** A run (or a learn that downloads) whose downloads meet a
  block ends with status `blocked`, the reason in `errors` ("the site blocks downloads:
  blocked: challenge page (Imperva) at <url>"), and a `blocked` event. A learned recipe
  stays active: the site is fine, the files are refused.
- **Downloadable probe.** An HTML reply to the probe of a document is checked the same
  way and answers `blocked`.
- **More protections.** Strong markers (anywhere on a page): "Pardon Our Interruption"
  (Imperva), "Checking your browser before accessing" (Cloudflare), the F5 rejection
  page, Sucuri's firewall page, PerimeterX's "Access to this page has been denied". Weak
  markers (only on a page under 20 KB, where they are the whole content): AWS WAF,
  Radware, Kasada, "verify you are human" / "are you a robot".

Never solved, never worked around: no browser to pass the challenge, no other headers,
no retries. Detect, name, stop.

## Consequences

The host gets one clear `blocked` status and reason instead of a list of rejected files,
and a blocked site costs one request instead of one per document. Normal pages that load
a protection's script (every page of some sites does) are still not blocks: weak markers
only count on small pages, and the snapshot test of such a site guards it.
