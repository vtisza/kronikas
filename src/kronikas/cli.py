"""Command-line interface.

Exposes the forecast and backtest workflows as a ``kronikas`` executable so a
scheduled job can produce machine-readable output without a Python wrapper::

    kronikas forecast polls.csv --election-date 2026-04-12 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .backtest import backtest
from .config import ModelConfig
from .forecast import ElectionForecast


def _parse_date(value: str) -> date:
    """argparse type converter for ISO-8601 dates."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Not a valid date: {value!r}. Use YYYY-MM-DD."
        ) from None


def _parse_shares(value: str) -> dict[str, float]:
    """Parse ``"Alice=48.2,Bob=47.1"`` into a mapping."""
    shares: dict[str, float] = {}
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, raw = chunk.partition("=")
        if not _:
            raise argparse.ArgumentTypeError(
                f"Expected NAME=VALUE pairs separated by commas, got {chunk!r}."
            )
        try:
            shares[name.strip()] = float(raw)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Value for {name.strip()!r} is not a number: {raw!r}."
            ) from None
    if not shares:
        raise argparse.ArgumentTypeError("No NAME=VALUE pairs found.")
    return shares


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CSV schema options shared by all subcommands."""
    group = parser.add_argument_group("input data")
    group.add_argument("polls_csv", type=Path, help="Path to the poll CSV.")
    group.add_argument(
        "--election-date",
        type=_parse_date,
        required=True,
        help="Election date (YYYY-MM-DD).",
    )
    group.add_argument("--date-column", default="date")
    group.add_argument("--pollster-column", default="pollster")
    group.add_argument("--sample-size-column", default="sample_size")
    group.add_argument(
        "--candidate-column",
        action="append",
        dest="candidate_columns",
        metavar="NAME",
        help="Restrict to this candidate column. Repeat for each candidate.",
    )
    group.add_argument(
        "--date-format", default=None, help="strftime format for parsing dates."
    )
    group.add_argument(
        "--decimal",
        default=".",
        help="Decimal separator in the CSV (use ',' for European-style files).",
    )


def _add_sampler_arguments(parser: argparse.ArgumentParser) -> None:
    """Register sampler and prior options shared by all subcommands."""
    group = parser.add_argument_group("sampler")
    group.add_argument("--draws", type=int, default=None, help="Draws per chain.")
    group.add_argument("--tune", type=int, default=None, help="Warmup per chain.")
    group.add_argument("--chains", type=int, default=None, help="Number of chains.")
    group.add_argument("--cores", type=int, default=None, help="Cores for sampling.")
    group.add_argument("--target-accept", type=float, default=None)
    group.add_argument("--seed", type=int, default=None, help="Random seed.")
    group.add_argument("--time-step-days", type=int, default=None)
    group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the sampler progress bar.",
    )


def _build_config(args: argparse.Namespace) -> ModelConfig:
    """Assemble a ModelConfig from parsed arguments, keeping library defaults."""
    overrides = {
        "num_draws": args.draws,
        "num_tune": args.tune,
        "num_chains": args.chains,
        "cores": args.cores,
        "target_accept": args.target_accept,
        "random_seed": args.seed,
        "time_step_days": args.time_step_days,
    }
    supplied = {k: v for k, v in overrides.items() if v is not None}
    if args.quiet:
        supplied["progressbar"] = False
    return ModelConfig(**supplied)


def _data_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the CSV schema options into loader keyword arguments."""
    return {
        "date_column": args.date_column,
        "pollster_column": args.pollster_column,
        "sample_size_column": args.sample_size_column,
        "candidate_columns": args.candidate_columns,
        "date_format": args.date_format,
        "decimal": args.decimal,
    }


def _write_json(payload: dict, destination: str) -> None:
    """Write JSON to a file, or to stdout when *destination* is ``-``."""
    text = json.dumps(payload, indent=2, sort_keys=False)
    if destination == "-":
        print(text)
    else:
        Path(destination).write_text(text, encoding="utf-8")


def _run_forecast(args: argparse.Namespace) -> int:
    """Handle the ``forecast`` subcommand."""
    config = _build_config(args)
    forecast = ElectionForecast(
        polls_csv=args.polls_csv,
        election_date=args.election_date,
        today=args.today,
        config=config,
        **_data_kwargs(args),
    )
    result = forecast.run()

    if not args.json:
        print(result.summary())
        for threshold in args.threshold or []:
            print(f"\nP(share >= {threshold:g}%) on election day")
            print("-" * 34)
            probs = result.threshold_probabilities(threshold)
            for name, prob in sorted(probs.items(), key=lambda kv: -kv[1]):
                print(f"  {name:<20s} {prob:6.1%}")
        if result.diagnostics is not None and result.diagnostics.converged:
            print()
            print(result.diagnostics.summary())
    else:
        _write_json(result.to_dict(thresholds=args.threshold), args.json)

    if args.save_trace:
        result.save(args.save_trace)
        if not args.json:
            print(f"\nTrace written to {args.save_trace}")

    diagnostics = result.diagnostics
    return 0 if diagnostics is None or diagnostics.converged else 1


def _run_backtest(args: argparse.Namespace) -> int:
    """Handle the ``backtest`` subcommand."""
    config = _build_config(args)
    report = backtest(
        args.polls_csv,
        election_date=args.election_date,
        as_of_dates=args.as_of,
        actual=args.actual,
        config=config,
        **_data_kwargs(args),
    )
    if args.json:
        payload = {
            "election_date": report.election_date.isoformat(),
            "actual": report.actual,
            "metrics": report.metrics(),
            "points": [
                {**vars(p), "as_of": p.as_of.isoformat()} for p in report.points
            ],
        }
        _write_json(payload, args.json)
    else:
        print(report.summary())
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="kronikas",
        description="Hierarchical Bayesian election forecasting from opinion polls.",
    )
    parser.add_argument(
        "--version", action="version", version=f"kronikas {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    forecast_parser = subparsers.add_parser(
        "forecast", help="Fit the model and report a forecast."
    )
    _add_data_arguments(forecast_parser)
    _add_sampler_arguments(forecast_parser)
    forecast_parser.add_argument(
        "--today",
        type=_parse_date,
        default=None,
        help="Reference date for current estimates (default: today).",
    )
    forecast_parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        metavar="PP",
        help=(
            "Report P(vote share >= PP) on election day. Repeat for several "
            "thresholds, e.g. --threshold 5 --threshold 10."
        ),
    )
    forecast_parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write a JSON summary to PATH ('-' for stdout) instead of text.",
    )
    forecast_parser.add_argument(
        "--save-trace",
        metavar="PATH",
        help="Persist the full posterior to a netCDF file at PATH.",
    )
    forecast_parser.set_defaults(func=_run_forecast)

    backtest_parser = subparsers.add_parser(
        "backtest", help="Refit at past dates and score against the true result."
    )
    _add_data_arguments(backtest_parser)
    _add_sampler_arguments(backtest_parser)
    backtest_parser.add_argument(
        "--as-of",
        type=_parse_date,
        action="append",
        required=True,
        metavar="YYYY-MM-DD",
        help="Refit using only polls up to this date. Repeat for each date.",
    )
    backtest_parser.add_argument(
        "--actual",
        type=_parse_shares,
        default=None,
        metavar="NAME=VALUE,...",
        help="True result, e.g. 'Alice=48.2,Bob=47.1,Carol=4.7'.",
    )
    backtest_parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write a JSON report to PATH ('-' for stdout) instead of text.",
    )
    backtest_parser.set_defaults(func=_run_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
