from __future__ import annotations

import pytest

from retrievr.candidates import compile_selector, dom_path, extract_candidates, parse_html
from retrievr.urls import is_allowed, normalize_url, path_extension, resolve_url


def test_normalize_url_is_order_and_encoding_insensitive() -> None:
    a = "https://IR.Example.com:443/fr/base?theme=finance&category[x]&fiscal[2025]#top"
    b = "https://ir.example.com/fr/base?category%5Bx%5D=&fiscal%5B2025%5D=&theme=finance"
    assert normalize_url(a) == normalize_url(b)


def test_normalize_url_drops_tracking_and_dot_segments() -> None:
    assert normalize_url("http://x.example/a/./b/../c.pdf?utm_source=n&id=1") == (
        "http://x.example/a/c.pdf?id=1"
    )


def test_resolve_and_extension() -> None:
    url = resolve_url("?page=1", "https://x.example/fr/base?theme=a")
    assert url == "https://x.example/fr/base?page=1"
    assert path_extension("https://x.example/files/T4-2025_FR.PDF?dl=1") == "pdf"
    assert path_extension("https://x.example/files/") == ""


def test_is_allowed_requires_exact_host_and_http() -> None:
    allowed = ["ir.example.com"]
    assert is_allowed("https://ir.example.com/x", allowed)
    assert not is_allowed("https://evil-ir.example.com/x", allowed)
    assert not is_allowed("https://ir.example.com.evil.test/x", allowed)
    assert not is_allowed("file:///etc/passwd", allowed)


PAGE = b"""<html><body>
<nav><a href="/files/banner.pdf">Banner</a></nav>
<div id="results">
  <h2>2025</h2>
  <ul>
    <li class="doc"><h3>Q4 report</h3><span>Published 06.02.2026</span>
        <a href="/files/q4-2025.pdf">Download</a></li>
    <li class="doc"><h3>Q4 data</h3><a href="https://cdn.example.com/q4-2025.xlsx">XLSX</a></li>
    <li class="doc"><a href="javascript:void(0)">noop</a><a href="/about">About</a></li>
  </ul>
</div>
</body></html>"""


def test_extract_with_item_selector_reads_title_and_context() -> None:
    root = parse_html(PAGE)
    cands = extract_candidates(
        root,
        "https://ir.example.com/docs",
        container="#results",
        item="li.doc",
        title="h3",
        extensions=["pdf", "xlsx"],
    )
    assert [c.url for c in cands] == [
        "https://ir.example.com/files/q4-2025.pdf",
        "https://cdn.example.com/q4-2025.xlsx",
    ]
    assert cands[0].title == "Q4 report"
    assert "Published 06.02.2026" in cands[0].context_text
    assert cands[0].heading == "Q4 report"
    assert dom_path(root.cssselect("li.doc a")[0]).endswith("ul>li.doc>a")


def test_extract_without_item_uses_row_heuristic() -> None:
    root = parse_html(PAGE)
    cands = extract_candidates(root, "https://ir.example.com/docs", extensions=["pdf"])
    by_url = {c.url.rsplit("/", 1)[-1]: c for c in cands}
    assert set(by_url) == {"banner.pdf", "q4-2025.pdf"}
    assert by_url["q4-2025.pdf"].context_text.startswith("Q4 report")


def test_invalid_selector_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid CSS selector"):
        compile_selector("li[[")
