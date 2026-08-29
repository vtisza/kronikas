"""Fit a forecast from a plain-language settings file.

The guided workflow's engine: takes a validated :class:`~kronikas.guided.settings.Plan`,
runs the model, writes the data files and the report, and reports back in
words rather than statistics.  Driven by ``kronikas guided``.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__ as kronikas_version
from ..data import PollData, load_polls
from ..forecast import ElectionForecast
from ..model import CandidateEstimate, ForecastResult
from . import report as report_builder
from .settings import Plan, SettingsError, build_model_config, describe

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
    """Abort with a message a non-technical user can act on."""
    raise SettingsError(message)


def _suggest(name: str, known: list[str]) -> str:
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    return f" Did you mean {close[0]!r}?" if close else ""


def _check_names(plan: Plan, poll_data: PollData) -> None:
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


def _data_overview(plan: Plan, poll_data: PollData) -> list[str]:
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


def _uniform_offsets(estimates: list[CandidateEstimate], pp: float) -> dict[str, float]:
    """Move *pp* points from the front-runner to the runner-up."""
    ranked = sorted(estimates, key=lambda e: -e.mean)
    return {ranked[0].name: pp, ranked[1].name: -pp}


def _estimate_rows(estimates: list[CandidateEstimate]) -> list[dict[str, Any]]:
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


def _trend_payload(result: ForecastResult) -> dict[str, Any]:
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


def _poll_payload(poll_data: PollData) -> list[dict[str, Any]]:
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


def _house_effect_payload(result: ForecastResult) -> dict[str, Any] | None:
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


def _write_tables(result: ForecastResult, plan: Plan, out: Path) -> None:
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


def run(
    plan: Plan,
    *,
    check_only: bool = False,
    save_trace: bool = False,
    build_report: bool = True,
) -> int:
    """Validate, fit, and write out a forecast.

    Parameters
    ----------
    plan:
        Validated settings, from :func:`kronikas.guided.settings.load_plan`.
    check_only:
        Read the poll file and report what would be run, then stop.  Nothing
        is sampled and nothing is written.
    save_trace:
        Also persist the full posterior to ``posterior.nc``.
    build_report:
        Write ``report.html`` alongside the data files.

    Returns
    -------
    int
        ``0`` normally, ``1`` when the sampler reported a convergence problem
        — so a scheduled run fails loudly rather than publishing bad numbers.

    Raises
    ------
    SettingsError
        When the settings and the poll file disagree, or the file cannot be
        read.  The message is written for the person who wrote the settings.
    """
    if not plan.polls_path.exists():
        _fail(f"No poll file at {plan.polls_path}.")
    try:
        poll_data = load_polls(plan.polls_path, **plan.loader_kwargs)
    except (ValueError, KeyError) as exc:
        _fail(f"The poll file could not be read: {exc}")

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

    config = build_model_config(plan)

    if check_only:
        print("\nSettings and data look usable. Drop --check to run the forecast.")
        return 0

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
    reference_day = result.today or plan.as_of or date.today()
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "kronikas_version": kronikas_version,
        "election": {
            "name": plan.election_name,
            "country": plan.country,
            "system": plan.system,
            "date": plan.election_date.isoformat(),
            "as_of": reference_day.isoformat(),
            "days_to_go": (plan.election_date - reference_day).days,
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

    if save_trace:
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

    if build_report:
        report_builder.build(out / "report_data.json", out / "report.html")
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
