"""Command-line interface.

Exposes the forecast and backtest workflows as a ``kronikas`` executable so a
scheduled job can produce machine-readable output without a Python wrapper::

    kronikas forecast polls.csv --election-date 2026-04-12 --json out.json

and the guided workflow for people who would rather answer questions than
assemble command lines::

    kronikas skill install          # hand the workflow to an AI assistant
    kronikas guided forecast.yaml   # or drive it yourself
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .backtesting import backtest
from .config import ModelConfig
from .forecast import ElectionForecast
from .guided.settings import SettingsError
from .model import ForecastResult


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
        "--undecided-column",
        default=None,
        help=(
            "Optional undecided/respondent-residual column. It is excluded "
            "from candidates and reduces each poll's effective sample size."
        ),
    )
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
        "undecided_column": args.undecided_column,
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


def _uniform_offsets(result: ForecastResult, pp: float) -> dict[str, float]:
    """Offsets moving *pp* points from the front-runner to the runner-up."""
    means = {e.name: e.mean for e in result.election_day_estimates}
    ranked = sorted(means, key=lambda name: -means[name])
    return {ranked[0]: pp, ranked[1]: -pp}


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

    scenarios = {
        f"{pp:g}": result.assume_shared_bias(
            _uniform_offsets(result, pp)
        ).win_probabilities
        for pp in (args.shared_bias or [])
    }

    if not args.json:
        print(result.summary())
        for threshold in args.threshold or []:
            print(f"\nP(share >= {threshold:g}%) on election day")
            print("-" * 34)
            probs = result.threshold_probabilities(threshold)
            for name, prob in sorted(probs.items(), key=lambda kv: -kv[1]):
                print(f"  {name:<20s} {prob:6.1%}")
        if scenarios:
            print("\nIf the polls carry an industry-wide error")
            print("(shifted from the front-runner to the runner-up)")
            print("-" * 48)
            for pp, probs in scenarios.items():
                leading = max(probs.items(), key=lambda kv: kv[1])
                print(f"  {pp:>5}pp  P({leading[0]} leads) = {leading[1]:6.1%}")
        if result.diagnostics is not None and result.diagnostics.converged:
            print()
            print(result.diagnostics.summary())
    else:
        payload = result.to_dict(thresholds=args.threshold)
        if scenarios:
            payload["shared_bias_scenarios"] = scenarios
        payload["shared_bias_breakeven_pp"] = result.shared_bias_breakeven()
        _write_json(payload, args.json)

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


def _run_guided(args: argparse.Namespace) -> int:
    """Handle the ``guided`` subcommand."""
    from .guided.runner import run
    from .guided.settings import load_plan

    plan = load_plan(args.settings)
    if args.output is not None:
        plan.output_dir = args.output
    return run(
        plan,
        check_only=args.check,
        save_trace=args.save_trace,
        build_report=not args.no_report,
    )


def _run_form(args: argparse.Namespace) -> int:
    """Handle the ``form`` subcommand."""
    from .data import load_polls
    from .guided.form import build

    poll_data = load_polls(
        args.polls_csv,
        date_column=args.date_column,
        pollster_column=args.pollster_column,
        sample_size_column=args.sample_size_column,
        undecided_column=args.undecided_column,
        candidate_columns=args.candidate_columns,
        date_format=args.date_format,
        decimal=args.decimal,
    )
    destination = args.output or args.polls_csv.parent / "settings-builder.html"
    written = build(
        poll_data,
        destination,
        election_date=args.election_date,
        polls_filename=args.polls_csv.name,
        decimal=args.decimal,
    )
    print(f"Settings form: {written.resolve()}")
    print(
        "Open it in a browser, fill it in, and save the file it gives you as "
        "forecast.yaml next to your polls."
    )
    return 0


def _run_report(args: argparse.Namespace) -> int:
    """Handle the ``report`` subcommand."""
    from .guided.report import build

    written = build(args.data, args.output)
    print(f"Report: {written.resolve()}")
    return 0


def _run_skill(args: argparse.Namespace) -> int:
    """Handle the ``skill`` subcommand."""
    from .guided import skill

    if args.skill_command == "path":
        print(skill.packaged_path())
        return 0
    target = args.dir or skill.default_target()
    try:
        installed = skill.copy_to(target, force=args.force)
    except FileExistsError as exc:
        raise ValueError(str(exc)) from None
    print(f"Skill installed to {installed}")
    print(
        "Start your assistant in any directory and ask it to forecast an "
        "election from your polls."
    )
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
        "--shared-bias",
        type=float,
        action="append",
        metavar="PP",
        help=(
            "Report the forecast under an assumed industry-wide polling error "
            "of PP points, shifted from the front-runner to the runner-up. "
            "Repeat for several sizes. A bias shared by every pollster cannot "
            "be measured from polls alone, so this shows what it would cost."
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

    guided_parser = subparsers.add_parser(
        "guided",
        help="Run a forecast from a plain-language settings file.",
        description=(
            "Fit a forecast described by a settings file written in words "
            "rather than statistics, and write a self-contained HTML report. "
            "Start from the template printed by 'kronikas skill path'."
        ),
    )
    guided_parser.add_argument(
        "settings", type=Path, help="Path to the settings file (forecast.yaml)."
    )
    guided_parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the settings and the poll file, then stop. No sampling.",
    )
    guided_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override the settings file's report.output_dir.",
    )
    guided_parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Also write the full posterior to posterior.nc in the output directory.",
    )
    guided_parser.add_argument(
        "--no-report", action="store_true", help="Skip building report.html."
    )
    guided_parser.set_defaults(func=_run_guided)

    form_parser = subparsers.add_parser(
        "form",
        help="Build a browser form for writing the settings file.",
        description=(
            "Read a poll file and write a self-contained HTML page with a "
            "control for every party and pollster in it. Fill it in, download "
            "the settings file, and run 'kronikas guided' on it."
        ),
    )
    form_parser.add_argument("polls_csv", type=Path, help="Path to the poll CSV.")
    form_parser.add_argument(
        "--election-date",
        type=_parse_date,
        default=None,
        help="Pre-fill the election date (YYYY-MM-DD).",
    )
    form_parser.add_argument("--date-column", default="date")
    form_parser.add_argument("--pollster-column", default="pollster")
    form_parser.add_argument("--sample-size-column", default="sample_size")
    form_parser.add_argument("--undecided-column", default=None)
    form_parser.add_argument(
        "--candidate-column",
        action="append",
        dest="candidate_columns",
        metavar="NAME",
        help="Restrict to this candidate column. Repeat for each candidate.",
    )
    form_parser.add_argument("--date-format", default=None)
    form_parser.add_argument(
        "--decimal",
        default=".",
        help="Decimal separator in the CSV (use ',' for European-style files).",
    )
    form_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Where to write the page (default: settings-builder.html "
        "beside the poll file).",
    )
    form_parser.set_defaults(func=_run_form)

    report_parser = subparsers.add_parser(
        "report",
        help="Rebuild report.html from a finished run, without refitting.",
    )
    report_parser.add_argument(
        "data", type=Path, metavar="REPORT_DATA_JSON", help="Path to report_data.json."
    )
    report_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Where to write the HTML (default: report.html beside the JSON).",
    )
    report_parser.set_defaults(func=_run_report)

    skill_parser = subparsers.add_parser(
        "skill",
        help="Install the assistant skill for the guided workflow.",
    )
    skill_subparsers = skill_parser.add_subparsers(dest="skill_command", required=True)
    install_parser = skill_subparsers.add_parser(
        "install", help="Copy the skill where an AI assistant will find it."
    )
    install_parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Skills directory to install into (default: ~/.claude/skills).",
    )
    install_parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing installation."
    )
    path_parser = skill_subparsers.add_parser(
        "path", help="Print where the skill lives inside the installed package."
    )
    path_parser.set_defaults(func=_run_skill)
    install_parser.set_defaults(func=_run_skill)
    skill_parser.set_defaults(func=_run_skill)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, SettingsError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
