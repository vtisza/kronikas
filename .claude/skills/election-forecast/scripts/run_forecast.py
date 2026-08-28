#!/usr/bin/env python3
"""Run a kronikas forecast from a plain-language settings file.

    python run_forecast.py forecast.yaml --check    # validate, no sampling
    python run_forecast.py forecast.yaml            # fit, then write a report

Everything the report needs is written to ``report_data.json`` in the output
directory, so the report can be rebuilt or restyled without refitting.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import json
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _settings import (  # noqa: E402 - after the sys.path fix above
    SettingsError,
    build_model_config,
    describe,
    load_plan,
)

# Okabe-Ito, which stays distinguishable under the common forms of colour
# blindness. Used when the settings file does not name a colour for a party;
# real parties usually have their own, set via report.party_colors.
DEFAULT_COLORS = [
    "#0072b2",
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#e69f00",
    "#56b4e9",
    "#8b5a2b",
    "#666666",
    "#7570b3",
    "#a6761d",
]


# Warnings that say nothing about the forecast's trustworthiness. They would
# only frighten a reader of the report's model-health section.
NOISY_WARNINGS = ("BLAS", "PyTensor could not link")


def _is_noise(message: str) -> bool:
    return any(fragment in message for fragment in NOISY_WARNINGS)


def _fail(message: str) -> None:
    """Stop with a message a non-technical user can act on."""
    print(f"\nProblem: {message}\n", file=sys.stderr)
    raise SystemExit(2)


def _suggest(name: str, known: list[str]) -> str:
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    return f" Did you mean {close[0]!r}?" if close else ""


def _check_names(plan: Any, poll_data: Any) -> None:
    """Refuse party or pollster names that do not appear in the poll file."""
    parties = list(poll_data.candidates)
    for name in sorted(plan.named_parties):
        if name not in parties:
            _fail(
                f"Your settings mention the party {name!r}, but the poll file "
                f"has {', '.join(parties)}.{_suggest(name, parties)}"
            )
    pollsters = list(poll_data.pollsters)
    for name in sorted(plan.pollsters):
        if name not in pollsters:
            _fail(
                f"Your settings mention the pollster {name!r}, but the poll "
                f"file has {', '.join(pollsters)}.{_suggest(name, pollsters)}"
            )


def _data_overview(plan: Any, poll_data: Any) -> list[str]:
    """Plain-language description of the poll file that was read."""
    counts: dict[str, int] = {}
    for index in poll_data.pollster_ids:
        name = poll_data.pollsters[int(index)]
        counts[name] = counts.get(name, 0) + 1
    lines = [
        f"{len(poll_data.dates)} polls, "
        f"{poll_data.first_poll_date} to {poll_data.last_poll_date}",
        f"Parties: {', '.join(poll_data.candidates)}",
        "Pollsters: "
        + ", ".join(f"{name} ({n})" for name, n in sorted(counts.items())),
        f"Typical sample size: {int(np.median(poll_data.sample_sizes))}",
    ]
    cutoff = plan.as_of or date.today()
    if poll_data.last_poll_date > min(cutoff, plan.election_date):
        lines.append(
            "Note: some polls are dated after the forecast date and will be "
            "left out of the fit."
        )
    if len(poll_data.pollsters) == 1:
        lines.append(
            "Note: only one pollster, so house effects cannot be separated "
            "from real movement and will not be estimated."
        )
    if len(poll_data.dates) < 10:
        lines.append(
            "Note: fewer than 10 polls. The forecast will lean heavily on the "
            "priors, so treat the intervals as wide guidance, not precision."
        )
    return lines


def _uniform_offsets(estimates: list, pp: float) -> dict[str, float]:
    """Move *pp* points from the front-runner to the runner-up."""
    ranked = sorted(estimates, key=lambda e: -e.mean)
    return {ranked[0].name: pp, ranked[1].name: -pp}


def _estimate_rows(estimates: list) -> list[dict[str, Any]]:
    return [
        {
            "name": e.name,
            "mean": float(e.mean),
            "median": float(e.median),
            "ci_lower": float(e.ci_lower),
            "ci_upper": float(e.ci_upper),
        }
        for e in estimates
    ]


def _assign_colors(parties: list[str], overrides: dict[str, str]) -> dict[str, str]:
    return {
        party: overrides.get(party, DEFAULT_COLORS[index % len(DEFAULT_COLORS)])
        for index, party in enumerate(parties)
    }


def _trend_payload(result: Any) -> dict[str, Any]:
    frame = result.latent_trend_dataframe()
    dates = [d.isoformat() for d in result.time_grid] or [
        str(i) for i in range(len(frame))
    ]
    series = {
        party: {
            "mean": [round(float(v), 3) for v in frame[f"{party}_mean"]],
            "lo": [round(float(v), 3) for v in frame[f"{party}_p_5"]],
            "hi": [round(float(v), 3) for v in frame[f"{party}_p_95"]],
        }
        for party in result.candidates
    }
    return {"dates": dates, "series": series}


def _poll_payload(poll_data: Any) -> list[dict[str, Any]]:
    rows = []
    for index, poll_date in enumerate(poll_data.poll_dates):
        rows.append(
            {
                "date": poll_date.isoformat(),
                "pollster": poll_data.pollsters[int(poll_data.pollster_ids[index])],
                "sample_size": int(poll_data.sample_sizes[index]),
                "shares": {
                    party: round(float(poll_data.poll_values[index, k]), 2)
                    for k, party in enumerate(poll_data.candidates)
                },
            }
        )
    return rows


def _house_effect_payload(result: Any) -> dict[str, Any] | None:
    try:
        frame = result.house_effects_dataframe()
    except RuntimeError:
        return None
    return {
        "pollsters": list(result.pollsters),
        "parties": list(result.candidates),
        "mean": [
            [round(float(frame[(p, c)].mean()), 2) for c in result.candidates]
            for p in result.pollsters
        ],
        "lo": [
            [round(float(frame[(p, c)].quantile(0.05)), 2) for c in result.candidates]
            for p in result.pollsters
        ],
        "hi": [
            [round(float(frame[(p, c)].quantile(0.95)), 2) for c in result.candidates]
            for p in result.pollsters
        ],
    }


def _write_tables(result: Any, plan: Any, out: Path) -> None:
    """Write the CSVs a user might want to open in a spreadsheet."""
    import pandas as pd

    rows = []
    for scope, estimates in (
        ("today", result.today_estimates),
        ("election_day", result.election_day_estimates),
    ):
        for estimate in estimates:
            rows.append(
                {
                    "when": scope,
                    "party": estimate.name,
                    "mean_pp": round(estimate.mean, 2),
                    "median_pp": round(estimate.median, 2),
                    "low_90_pp": round(estimate.ci_lower, 2),
                    "high_90_pp": round(estimate.ci_upper, 2),
                    "probability_of_finishing_first": round(
                        result.win_probabilities.get(estimate.name, float("nan")), 4
                    )
                    if scope == "election_day"
                    else "",
                }
            )
    pd.DataFrame(rows).to_csv(out / "estimates.csv", index=False)
    result.latent_trend_dataframe().round(3).to_csv(out / "trend.csv")
    result.party_forecast_dataframe(day="election_day").round(3).to_csv(
        out / "draws_election_day.csv", index=False
    )
    if plan.thresholds_pp:
        pd.DataFrame(
            {
                f"P(share >= {t:g}pp)": result.threshold_probabilities(t)
                for t in plan.thresholds_pp
            }
        ).to_csv(out / "threshold_probabilities.csv")
    with contextlib.suppress(RuntimeError):
        result.house_effects_dataframe().describe().round(3).to_csv(
            out / "house_effects.csv"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an election forecast from a plain-language settings file."
    )
    parser.add_argument("settings", type=Path, help="Path to forecast.yaml")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the settings and the poll file, then stop. No sampling.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Override report.output_dir."
    )
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Also write the full posterior to posterior.nc (needs h5netcdf).",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Skip building report.html."
    )
    args = parser.parse_args(argv)

    try:
        plan = load_plan(args.settings)
    except SettingsError as exc:
        _fail(str(exc))
        return 2
    if args.output is not None:
        plan.output_dir = args.output

    try:
        from kronikas import __version__ as kronikas_version
        from kronikas import load_polls
    except ImportError:
        _fail(
            "kronikas is not installed in this Python environment. Run the "
            "setup script (scripts/setup_kronikas.sh) first."
        )
        return 2

    if not plan.polls_path.exists():
        _fail(f"No poll file at {plan.polls_path}.")

    try:
        poll_data = load_polls(plan.polls_path, **plan.loader_kwargs)
    except (ValueError, KeyError) as exc:
        _fail(f"The poll file could not be read: {exc}")
        return 2

    _check_names(plan, poll_data)
    readback = describe(plan)
    overview = _data_overview(plan, poll_data)

    print("\nWhat will be forecast")
    print("-" * 60)
    for line in readback:
        print(line)
    print("\nWhat is in the poll file")
    print("-" * 60)
    for line in overview:
        print(line)

    try:
        config = build_model_config(plan)
    except (SettingsError, ValueError) as exc:
        _fail(str(exc))
        return 2

    if args.check:
        print("\nSettings and data look usable. Remove --check to run the forecast.")
        return 0

    from kronikas import ElectionForecast

    forecast = ElectionForecast(
        polls_csv=plan.polls_path,
        election_date=plan.election_date,
        today=plan.as_of,
        config=config,
        **plan.loader_kwargs,
    )

    print("\nSampling. This takes a few minutes on a laptop.\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = forecast.run()
    run_warnings = sorted(
        {
            f"{w.category.__name__}: {w.message}"
            for w in caught
            if not _is_noise(str(w.message))
        }
    )

    out = plan.output_dir
    out.mkdir(parents=True, exist_ok=True)

    scenarios = []
    for pp in plan.scenarios_pp:
        shifted = result.assume_shared_bias(
            _uniform_offsets(result.election_day_estimates, pp)
        )
        scenarios.append(
            {
                "pp": pp,
                "win_probabilities": {
                    k: float(v) for k, v in shifted.win_probabilities.items()
                },
                "estimates": _estimate_rows(shifted.election_day_estimates),
            }
        )

    diagnostics = result.diagnostics
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "kronikas_version": kronikas_version,
        "election": {
            "name": plan.election_name,
            "date": plan.election_date.isoformat(),
            "as_of": (result.today or plan.as_of or date.today()).isoformat(),
            "days_to_go": (
                plan.election_date - (result.today or plan.as_of or date.today())
            ).days,
        },
        "parties": list(result.candidates),
        "pollsters": list(result.pollsters),
        "colors": _assign_colors(list(result.candidates), plan.party_colors),
        "today_estimates": _estimate_rows(result.today_estimates),
        "election_day_estimates": _estimate_rows(result.election_day_estimates),
        "win_probabilities": {k: float(v) for k, v in result.win_probabilities.items()},
        "threshold_probabilities": {
            f"{t:g}": {
                k: float(v) for k, v in result.threshold_probabilities(t).items()
            }
            for t in plan.thresholds_pp
        },
        "trend": _trend_payload(result),
        "polls": _poll_payload(poll_data),
        "house_effects": _house_effect_payload(result),
        "scenarios": scenarios,
        "breakeven_pp": result.shared_bias_breakeven(),
        "diagnostics": {
            **(diagnostics.to_dict() if diagnostics else {}),
            "converged": bool(diagnostics.converged) if diagnostics else None,
            "issues": list(diagnostics.issues) if diagnostics else [],
            "notes": list(diagnostics.notes) if diagnostics else [],
            "text": diagnostics.summary() if diagnostics else "",
        },
        "settings_readback": readback,
        "data_overview": overview,
        "warnings": run_warnings,
    }
    (out / "report_data.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (out / "forecast.json").write_text(
        json.dumps(result.to_dict(thresholds=plan.thresholds_pp), indent=2),
        encoding="utf-8",
    )
    _write_tables(result, plan, out)

    if args.save_trace:
        try:
            result.save(out / "posterior.nc")
        except Exception as exc:  # noqa: BLE001 - netCDF backends vary
            print(f"Could not save the posterior: {exc}", file=sys.stderr)

    print(result.summary())
    if diagnostics is not None and diagnostics.converged:
        # summary() stays silent when the fit is healthy; say so anyway, so a
        # clean run is visibly clean rather than merely un-complained-about.
        health = [
            f"R-hat {diagnostics.max_r_hat:.3f}"
            if diagnostics.max_r_hat is not None
            else None,
            f"effective sample size {diagnostics.min_ess_bulk:.0f}"
            if diagnostics.min_ess_bulk is not None
            else None,
            f"{diagnostics.n_divergences} divergences",
        ]
        print(
            "\nModel health: the sampler converged ("
            + ", ".join(item for item in health if item)
            + ")."
        )
        for note in diagnostics.notes:
            print(f"  Note: {note}")

    if not args.no_report:
        import make_report

        make_report.build(out / "report_data.json", out / "report.html")
        print(f"\nReport: {(out / 'report.html').resolve()}")
    print(f"Files:  {out.resolve()}")

    if run_warnings:
        print("\nWarnings raised during the run:")
        for message in run_warnings:
            print(f"  - {message}")

    if diagnostics is not None and not diagnostics.converged:
        print(
            "\nThe sampler reported a convergence problem, so these numbers "
            "are not trustworthy yet. See the report's model health section.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
