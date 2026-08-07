"""Tests for the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kronikas.cli import _parse_date, _parse_shares, build_parser, main

FAST_ARGS = [
    "--draws",
    "40",
    "--tune",
    "40",
    "--chains",
    "1",
    "--cores",
    "1",
    "--time-step-days",
    "30",
    "--quiet",
]


class TestArgumentParsing:
    def test_date_converter(self):
        from datetime import date

        assert _parse_date("2024-11-05") == date(2024, 11, 5)

    def test_bad_date_rejected(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="Not a valid date"):
            _parse_date("05/11/2024")

    def test_shares_converter(self):
        assert _parse_shares("A=48.2, B=47.1") == {"A": 48.2, "B": 47.1}

    def test_shares_rejects_missing_equals(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="NAME=VALUE"):
            _parse_shares("A:48.2")

    def test_shares_rejects_non_numeric(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="not a number"):
            _parse_shares("A=high")

    def test_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_election_date_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["forecast", "polls.csv"])

    def test_thresholds_accumulate(self):
        args = build_parser().parse_args(
            [
                "forecast",
                "polls.csv",
                "--election-date",
                "2024-06-01",
                "--threshold",
                "5",
                "--threshold",
                "10",
            ]
        )
        assert args.threshold == [5.0, 10.0]


@pytest.mark.slow
class TestForecastCommand:
    def test_text_output(self, polls_csv: Path, capsys):
        code = main(
            [
                "forecast",
                str(polls_csv),
                "--election-date",
                "2024-06-01",
                "--today",
                "2024-03-20",
                *FAST_ARGS,
            ]
        )
        out = capsys.readouterr().out
        assert "Election Forecast Summary" in out
        assert "Plurality probabilities" in out
        assert code in (0, 1)  # 1 signals a convergence warning, not a crash

    def test_threshold_output(self, polls_csv: Path, capsys):
        main(
            [
                "forecast",
                str(polls_csv),
                "--election-date",
                "2024-06-01",
                "--threshold",
                "5",
                *FAST_ARGS,
            ]
        )
        assert "P(share >= 5%)" in capsys.readouterr().out

    def test_json_to_stdout(self, polls_csv: Path, capsys):
        main(
            [
                "forecast",
                str(polls_csv),
                "--election-date",
                "2024-06-01",
                "--today",
                "2024-03-20",
                "--json",
                "-",
                "--threshold",
                "5",
                *FAST_ARGS,
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["election_date"] == "2024-06-01"
        assert payload["today"] == "2024-03-20"
        assert len(payload["election_day_estimates"]) == 3
        assert set(payload["threshold_probabilities"]) == {"5"}

    def test_shared_bias_scenarios(self, polls_csv: Path, capsys):
        main(
            [
                "forecast",
                str(polls_csv),
                "--election-date",
                "2024-06-01",
                "--shared-bias",
                "2",
                "--shared-bias",
                "4",
                *FAST_ARGS,
            ]
        )
        out = capsys.readouterr().out
        assert "industry-wide error" in out
        assert "2pp" in out and "4pp" in out

    def test_shared_bias_in_json(self, polls_csv: Path, capsys):
        main(
            [
                "forecast",
                str(polls_csv),
                "--election-date",
                "2024-06-01",
                "--shared-bias",
                "3",
                "--json",
                "-",
                *FAST_ARGS,
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert set(payload["shared_bias_scenarios"]) == {"3"}
        assert "shared_bias_breakeven_pp" in payload

    def test_json_to_file(self, polls_csv: Path, tmp_path: Path):
        out = tmp_path / "forecast.json"
        main(
            [
                "forecast",
                str(polls_csv),
                "--election-date",
                "2024-06-01",
                "--json",
                str(out),
                *FAST_ARGS,
            ]
        )
        assert json.loads(out.read_text())["candidates"] == [
            "Candidate_A",
            "Candidate_B",
            "Candidate_C",
        ]

    def test_save_trace(self, polls_csv: Path, tmp_path: Path):
        from kronikas.model import ForecastResult

        trace_path = tmp_path / "trace.nc"
        main(
            [
                "forecast",
                str(polls_csv),
                "--election-date",
                "2024-06-01",
                "--json",
                str(tmp_path / "f.json"),
                "--save-trace",
                str(trace_path),
                *FAST_ARGS,
            ]
        )
        assert trace_path.is_file()
        assert ForecastResult.load(trace_path).candidates[0] == "Candidate_A"

    def test_missing_file_exits_cleanly(self, tmp_path: Path):
        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "forecast",
                    str(tmp_path / "nope.csv"),
                    "--election-date",
                    "2024-06-01",
                    *FAST_ARGS,
                ]
            )
        assert excinfo.value.code == 2


@pytest.mark.slow
class TestBacktestCommand:
    def test_text_report(self, polls_csv: Path, capsys):
        code = main(
            [
                "backtest",
                str(polls_csv),
                "--election-date",
                "2024-06-01",
                "--as-of",
                "2024-03-01",
                "--actual",
                "Candidate_A=47,Candidate_B=41,Candidate_C=12",
                *FAST_ARGS,
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "Backtest report" in out
        assert "90% coverage" in out

    def test_json_report(self, polls_csv: Path, capsys):
        main(
            [
                "backtest",
                str(polls_csv),
                "--election-date",
                "2024-06-01",
                "--as-of",
                "2024-03-01",
                "--as-of",
                "2024-03-15",
                "--json",
                "-",
                *FAST_ARGS,
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["election_date"] == "2024-06-01"
        assert payload["metrics"]["n_forecasts"] == 2
        assert len(payload["points"]) == 6
