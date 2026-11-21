# 0008 — Recipe schema v3: request sources, and browser discovery

Date: 2026-11-21 · Status: accepted · Extends 0001, 0004, 0006, 0007

## Context

Some investor-relations pages fill their listing with JavaScript: a year selector posts
a form and inserts the HTML it gets back, or a search component calls a JSON API page by
page. The static HTML then holds a default view or nothing. Links may also be opaque
(`/pdfPreview/<id>`, the file name only in the `title` attribute). The engine must learn
such sites once, then replay them without a browser and without the LLM.

## Decision

- **Fetcher**: `request("GET" | "POST", url, content=, headers=)` with the checks of
  `get` (robots.txt, allowed domains, rate limit, block detection). A recipe can only set
  `Accept`, `Content-Type` and `X-Requested-With`; no cookie, no authorization.
- **Recipe schema v3** (`SCHEMA_VERSION = 3`; v1 and v2 recipes are unchanged and still
  valid, a source without `kind` is a `url_template`):
  - `kind: request` sources: method, URL, `form` or `json` body, placeholders `{param}`,
    `{year}`, `{page}` and `{today:<strftime>}` (a JSON string that is only `{page}` or
    `{year}` becomes a number);
  - `params` of kind `options`: values read from elements of a page (the `<option>` of a
    year selector, whose values are opaque ids), the year read in their text, with
    `year: fiscal | publication`. When years are requested, the bucket of the next year
    is read too (annual documents are often listed under the year they are published);
  - `pagination: {kind: page_number, start, max_pages, has_more}`: stops at the first page
    with no new document, a 404, `has_more` false, or `max_pages`;
  - `response: {format: html | json, html_path | items}`: an HTML response, a JSON
    response holding an HTML fragment, or a JSON list (`items.path`, `url` as one path or
    several, `title`, `file_name`, `date`, `context`);
  - `items.document_urls`: patterns for links that are documents without an extension;
    links whose `download` or `title` attribute is a file name with a document extension
    are documents too, and the `file_name` rule target reads that name;
  - period rules may give a fixed sub-period (`period: H1`) for patterns such as
    "au 30 juin (?P<year>20\d{2})".
- **Downloads**: a document whose URL has no extension takes it from the file name, else
  from the `Content-Type`, else from its first bytes.
- **Snapshot format v2**: entries carry the method and a canonical request body (JSON with
  sorted keys, or sorted form fields); v1 indexes still load. For offline benchmarks only,
  `ReplayTransport(lenient_bodies=True)` answers a POST with the closest captured body
  that holds every field of the request.
- **Browser** (`retrievr.browser`, optional extra `retrievr[browser]` = Playwright,
  Apache-2.0, driving the installed Edge by default, `channel = "msedge"`): used on cold
  paths only. The engine, not the LLM, drives it: load the page, pick the requested years
  (and the next one) in each year selector, press "load more" once, at most 12 actions,
  and passively record the XHR/fetch exchanges of the site. Same User-Agent; robots.txt
  checked for every request of the site; images, fonts and media not loaded; requests to
  other sites limited to scripts and stylesheets; a block or challenge stops discovery.
- **Discovery records**: a discovery is saved in a snapshot as `discovery/<page>.yaml`
  (actions and exchange keys) with the response bodies, and replayed by
  `ReplayDiscovery` without a browser: tests and the offline benchmark never store the
  sites' JavaScript and never launch a browser.
- **Learning** (`retrievr.endpoints`, `retrievr.learn`): discovery runs only when the
  static page has fewer than 5 document links or a year selector, so static sites keep
  their prompts. The LLM then sees the endpoints (bodies, fields that change, where the
  documents are in each response, samples) and names the captured call to copy and the
  fields that vary (`page`, `options`, `year`); the engine copies the request exactly,
  templates today's date itself, builds the source and checks that it finds the
  documents of that call. Warm runs replay the source with httpx only.

## Consequences

Crédit Agricole (form POST, opaque links, year selectors) and LVMH (JSON search API) are
learned cold with gpt-6-luna at 100 % and 95 % recall from their snapshots, and replayed
without a browser. A site whose API requires a session token or a cookie is not
supported (no session handling by design); a page that only renders documents with
JavaScript and calls no listing endpoint would need a browser at run time, which this
milestone does not add.
