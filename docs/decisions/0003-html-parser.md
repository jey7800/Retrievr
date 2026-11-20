# 0003 — HTML parsing with lxml + cssselect

Date: 2026-11-20 · Status: accepted

## Context

The initial dependency target listed selectolax (fast CSS) plus lxml (XPath). selectolax
wheels bundle the Modest engine, distributed under LGPL-2.1, next to Lexbor
(Apache-2.0). The project only accepts permissive runtime licenses.

## Decision

Use lxml (BSD-3, bundled libxml2/libxslt under MIT) for parsing and XPath, and
cssselect (BSD-3) to compile CSS selectors to XPath. One parser, one tree type.

## Consequences

One extra small dependency (cssselect). Parsing speed is ample for listing pages. If
selectolax is reconsidered for speed, re-check the license of the exact wheels first.
