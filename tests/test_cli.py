"""Smoke tests for the CLI wiring (Milestone 8)."""

from __future__ import annotations

from typer.testing import CliRunner

from segmentation.cli import app

runner = CliRunner()


def test_all_subcommands_registered() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("etl", "features", "train", "viz", "analysis", "all"):
        assert command in result.output
