# 0011 — Recipe schema v4: detail pages and file names; fixes from twelve cold tests

Date: 2026-11-23 · Status: accepted · Extends 0008, 0010

## Context

Cold learning was run live on twelve investor-relations sites the library had never
seen (six banks, four CAC 40 companies, two Nasdaq companies), each with the same
request ("annual reports, half-year reports and quarterly results", fiscal years
2024–2025), download included, then a warm run. Two listing patterns could not be
described by a recipe, and several generic defects showed up.

## Decision

Recipe schema **v4** (v1–v3 recipes stay valid and unchanged):

- **`detail` on a URL-template source**: the listing links to one detail page per event
  or period ("H1 2025 results", "View details") and the documents are on those pages.
  `detail.link` is the CSS selector of those links on the listing page, `detail.items`
  describes the documents on a detail page, `detail.max_pages` caps the pages opened per
  listing page (60). Links outside the allowed domains are not opened. The detail page
  heading (`<h1>`, else `<title>`) becomes the heading of its documents, for the rules
  and the fiscal-year fallback. The learner offers this to the LLM only when the start
  page lists no document and no call of the page lists any.
- **`items.url_template` on a JSON listing**: the items give a bare file name
  (`First_Half_Report_2026_EN_d6129bcfad.pdf`) that the page turns into a download URL
  (`/api/download-file/?path={value}`). The engine derives the template itself, without
  the LLM: browser discovery records the document links of the rendered page, and a
  file name found (as is or URL-encoded) inside one of them gives the template.

Other changes:

- Requests naming kinds of documents **and** results ("annual reports and quarterly
  results") keep the documents of a results release as well (prompt rule plus a
  deterministic check after compilation). A short lexicon of unambiguous phrases (Pillar
  3, URD, Excel / financial tables, press releases, fixed income presentations and
  transcripts) adds a kind the compiled filter dropped, unless the phrase is negated
  ("no press releases"): the compiled filter no longer depends on the LLM for those.
- Browser discovery runs when the static page lists few documents, few of the requested
  years (it shows the current period only) or has a year selector; it scrolls the page
  first (lists rendered when scrolled into view); calls refused by robots.txt are named
  in the learn notes.
- A fiscal year after the current one is impossible for a published document (it comes
  from a bond maturity or a target year in the title): the publication date gives the
  year, else it stays unknown. Health bounds of documents per year are computed from the
  years before the latest one; a year the site never listed is not reported as missing.
- Bot-protection challenge pages served with HTTP 200 ("using a security service ...
  online attacks") are a block (`blocked`, no LLM call); an error page served with 200
  (error title, no document, almost no link) stops learning with a clear message.
- Year selectors whose labels are short periods (`Q325`, `1H24`, `FY24`) are read as
  years; the browser picks the requested years first. Opaque download links may carry a
  query (`/file/29269?download=1`). Browser discovery also runs (once) on an archive page
  the LLM opens when it has a year selector or a "load more"/"find more files" button.

## Consequences

Final live run with gpt-6-luna, all twelve sites from scratch (0.10 USD of LLM calls):

| Site | Listed | Downloaded (2024–2025) | Warm run |
|---|---:|---:|---|
| Danske Bank (archive API found by browser discovery) | 2654 | 95 (+2 duplicates) | `ok`, 0 LLM calls |
| Standard Chartered | 401 | 74 | `ok`, 0 LLM calls |
| Intel | 377 | 48 | `ok`, 0 LLM calls |
| Capgemini (detail pages, v4) | 50 | 18 | `ok`, 0 LLM calls |
| ING | 32 | 8 | `ok`, 0 LLM calls |
| Intesa Sanpaolo | 18 | 8 | `ok`, 0 LLM calls |
| PepsiCo | 19 | 2 | `ok`, 0 LLM calls |
| Michelin (results page: recent periods only) | 17 | 1 | `ok`, 0 LLM calls |
| Michelin, publications page `?year=2025` as start | 26 | 17 | `ok` |
| Kering | — | — | `rejected`: robots.txt disallows its listing API (`/*?*page=`), said in the notes |
| Cisco | — | — | `blocked` (HTTP 403), no LLM call |
| Orange | — | — | `blocked` (bot-protection challenge), no LLM call |
| Lloyds Banking Group | — | — | stopped: the start page is an error page for this client, no LLM call |

Eight of twelve sites are learned and downloaded without help; the four others end with
a clear status and cost (almost) nothing. The file-name template (Kering's pattern) is
covered by unit tests; Kering itself cannot be used, by its robots.txt.

Not supported, by design: pages that refuse an identifiable client (403, challenge,
error page) — the library stops with a clear status and never tries to get around them.

## Addendum (2026-11-23): eleven more sites, same code

A second round on eleven other sites (KBC, NatWest, Siemens, Novartis, Vinci, Adobe, SEB,
Erste Group, Hermès, UniCredit, ABN AMRO), run with the code above before any change:

| Site | Outcome |
|---|---|
| NatWest (928 listed, 104 downloaded), Erste (623, 50), Hermès (1230, 44), Adobe (131, 36), Siemens (365, 27), KBC (243, 20), Novartis (68, 18) | learned first time, warm run `ok` with 0 LLM calls |
| UniCredit | `blocked` (HTTP 403) |
| ABN AMRO, Vinci | stopped: robots.txt disallows the start page (Vinci: the URL it redirects to) |
| SEB | the only surprise, two generic causes, fixed below |

SEB showed two patterns now handled for every site:

- **Links on custom elements**: a design system's Web Components carry the link
  (`<gds-button-pw href="...">`). Any element whose tag has a dash and that carries an
  `href` is read as a link, like `<a href>`.
- **Formats announced in link texts**: "Interim Report Q2 2026 (pdf)", "PDF 2.1 MB",
  "Fact Book (excel)" make an opaque link (`.../dld/<id>?opendocument`) a document with
  that extension; the folder of such links becomes an `items.document_urls` pattern so
  that links of the same service without the announcement are documents too.
- The downloadable gate also counts sampled documents that robots.txt disallows: SEB's
  files are on a host whose robots.txt disallows them, so its recipe is now `rejected`
  with that reason instead of being activated for files that cannot be downloaded.

Trend: the first round (twelve sites) needed nine generic fixes; this round needed two,
on one site out of eleven.

## Addendum (2026-11-23): third round, US and Asia-Pacific sites

Twelve more sites (Coca-Cola, IBM, Walmart, Berkshire Hathaway, Microsoft, ExxonMobil,
Samsung, Toyota, DBS, Hitachi, SoftBank, Commonwealth Bank), run first with the code above:

| Site | Outcome |
|---|---|
| Hitachi (834 listed, 78 downloaded), DBS (584, 53), Coca-Cola (323, 48), Walmart (411, 40), Toyota (498, 36), SoftBank (170, 10), Samsung (63, 8), Berkshire Hathaway (32, 8) | learned first time, warm run `ok` with 0 LLM calls |
| Commonwealth Bank | learned; the start URL redirects to a sustainability page, whose documents are rightly `other` |
| ExxonMobil, IBM, Microsoft | three generic causes, fixed below; then ExxonMobil 474 listed / 66 downloaded, IBM 253 / 22, Microsoft 15 listed |

Many US investor sites on a common hosted IR platform answer 403 to an identifiable
client (Oracle, Pfizer, Visa, Caterpillar, J&J): they were left out of the round, as were
Sony, TSMC, Infosys, HDFC Bank, BHP and Honda (403 too).

Fixes:

- A blank `class=" "` attribute crashed the DOM path (`IndexError`): fixed; and the CLI no
  longer reports an `IndexError`/`KeyError` as a usage error (they are bugs: type shown,
  traceback logged with `-v`).
- Opaque links in nested download folders (`/downloads/documents/us-en/15db52…`) are
  documents; a web page in a download folder (`/downloads/default.htm`, `.aspx`, `.php`)
  never is.
- `{yy}` (schema v4): a two-digit year in a listing URL (`/ar25/download-center/`). The
  learner turns the LLM's `{year}` into `{yy}` when the pages it saw use two digits.
- A detail page whose container matches nothing (an older layout) is read from its
  `<main>` content (a warning says so) instead of giving nothing.

Known limit: detail pages that fill their document list with JavaScript (Microsoft's
2017–2024 download centers) are not opened in a browser; their documents are missed.

Trend, counting the sites that answer an identifiable client: round 1 worked first time
on 3 of 8 and needed nine generic fixes; round 2 on 7 of 8, two fixes; round 3 on 9 of 12,
four fixes (one of them a crash on malformed markup).
