"""Command line, replayed from the Société Générale snapshot (no network, no LLM)."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import pytest

from retrievr.cli import EXIT_FAILED, EXIT_OK, main, parse_years
from retrievr.snapshots import ReplayTransport
from tests.conftest import RECIPES


@pytest.fixture
def config(tmp_path: Path) -> list[str]:
    path = tmp_path / "retrievr.toml"
    path.write_text("[fetch]\nrate_limit_rps = 10\n", encoding="utf-8")
    return ["--config", str(path)]


def _cli(*argv: str) -> tuple[int, dict[str, Any]]:
    out = io.StringIO()
    code = main(["--json", *argv], out=out)
    return code, json.loads(out.getvalue())


def test_parse_years() -> None:
    assert parse_years("2024,2025") == [2024, 2025]
    assert parse_years("2022-2024, 2026") == [2022, 2023, 2024, 2026]
    assert parse_years(None) is None
    with pytest.raises(argparse.ArgumentTypeError):
        parse_years("24")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_years("2025-2020")


def test_bad_years_are_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["check", "x", "--years", "soon"], out=io.StringIO())
    assert exc.value.code == 2


def test_install_run_and_check(tmp_path: Path, sg_site: ReplayTransport, config: list[str]) -> None:
    state = [*config, "--state-dir", str(tmp_path / "state"), "--output-dir", str(tmp_path / "o")]
    code, payload = _cli(*state, "install", str(RECIPES / "societegenerale-ir@1.yaml"))
    assert code == EXIT_OK and payload["recipe"] == "societegenerale-ir@1"

    code, payload = _cli(*state, "run", "societegenerale", "--years", "2024-2025", "--no-download")
    assert code == EXIT_OK, payload
    assert payload["status"] == "ok" and payload["listed"] == 30
    assert payload["health"]["verdict"] == "ok"
    assert payload["llm"]["calls"] == 0
    assert {d["type"] for d in payload["documents"]} >= {"press_release", "results_presentation"}

    code, payload = _cli(*state, "check", "societegenerale", "--years", "2024,2025")
    assert code == EXIT_OK and payload["status"] == "ok"
    assert any(c["name"] == "known_documents" for c in payload["health"]["checks"])


def test_human_output(tmp_path: Path, sg_site: ReplayTransport, config: list[str]) -> None:
    state = [*config, "--state-dir", str(tmp_path / "state")]
    main([*state, "install", str(RECIPES / "societegenerale-ir@1.yaml")], out=io.StringIO())
    out = io.StringIO()
    code = main([*state, "run", "societegenerale", "--years", "2025", "--no-download"], out=out)
    text = out.getvalue()
    assert code == EXIT_OK
    assert text.startswith("run societegenerale: ok (societegenerale-ir@1)")
    assert "documents: 14 listed, 14 selected" in text


def test_unknown_site_fails(tmp_path: Path, no_network: Any) -> None:
    code, payload = _cli("--state-dir", str(tmp_path), "check", "unknown-site")
    assert code == EXIT_FAILED
    assert payload["status"] == "error" and "no active recipe" in payload["errors"][0]
    code, payload = _cli("--state-dir", str(tmp_path), "run", "unknown-site")
    assert code == EXIT_FAILED and payload["status"] == "no_recipe"


def test_generic_link_texts_show_the_file_name() -> None:
    from retrievr.cli import _display_title

    doc = {"title": "PDF", "url": "https://ir.example-issuer.com/files/Q1%202025%20report.pdf"}
    assert _display_title(doc) == "PDF Q1 2025 report.pdf"
    assert _display_title({"title": "Annual report 2025", "url": "x"}) == "Annual report 2025"


def test_unknown_site_names_the_active_sites(config: list[str], tmp_path: Path) -> None:
    state = ["--state-dir", str(tmp_path / "state")]
    assert (
        main([*config, *state, "install", str(RECIPES / "ubs-ir@1.yaml")], out=io.StringIO()) == 0
    )
    code, payload = _cli(*config, *state, "check", "usb")
    assert code == EXIT_FAILED
    assert payload["site"] == "usb"
    assert "active sites: ubs" in payload["errors"][0]


@pytest.mark.parametrize(
    "argv",
    [
        ["-v", "--json", "--state-dir", "S", "run", "ubs"],
        ["run", "ubs", "-v", "--json", "--state-dir", "S"],
        ["--state-dir", "S", "run", "-v", "ubs", "--json"],
    ],
)
def test_global_options_before_or_after_the_command(argv: list[str]) -> None:
    from retrievr.cli import build_parser

    args = build_parser().parse_args(argv)
    assert (args.verbose, args.json, args.state_dir) == (True, True, Path("S"))
    assert args.command == "run" and args.site == "ubs"


def test_options_absent_after_the_command_keep_their_defaults() -> None:
    from retrievr.cli import build_parser

    args = build_parser().parse_args(["run", "ubs"])
    assert (args.verbose, args.json, args.config, args.output_dir) == (False, False, None, None)


def test_a_bug_is_never_reported_as_a_usage_error() -> None:
    from retrievr.cli import _expected

    assert _expected(LookupError("no active recipe")) and _expected(ValueError("bad"))
    assert not _expected(IndexError("list index out of range"))
    assert not _expected(KeyError("x"))
