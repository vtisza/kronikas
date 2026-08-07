"""Backtesting: refit the model as of past dates and score it against the result.

A forecast that has never been scored is an assertion, not a measurement.  This
module replays a campaign: for each *as-of* date it discards every poll
published later, refits the model, and records what the model would have said
about election day with only the information available at that time.  When the
true result is supplied, the forecasts are scored for accuracy, interval hits,
and the continuous ranked probability score (CRPS).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .config import ModelConfig
from .data import PollData, load_polls, polls_from_dataframe
from .model import ForecastResult, build_model, extract_results, run_inference

__all__ = ["BacktestPoint", "BacktestResult", "backtest"]


@dataclass
class BacktestPoint:
    """One candidate's election-day forecast made as of one past date.

    Attributes:
        as_of: The date the forecast was made as of; only polls on or before
            this date were used.
        candidate: Candidate name.
        n_polls: Number of polls available as of that date.
        mean, median, ci_lower, ci_upper: Election-day posterior summary in
            percentage points.
        actual: True election-day vote share, when known.
        error: ``mean - actual`` (signed, percentage points).
        abs_error: ``abs(error)``.
        covered: Whether the 90 % credible interval contained *actual*.
        crps: Continuous ranked probability score in percentage points. Lower
            is better; unlike absolute error, it rewards a full predictive
            distribution that is both sharp and well calibrated.
        converged: Whether the sampler reported a clean fit for this refit.
    """

    as_of: date
    candidate: str
    n_polls: int
    mean: float
    median: float
    ci_lower: float
    ci_upper: float
    actual: float | None = None
    error: float | None = None
    abs_error: float | None = None
    covered: bool | None = None
    crps: float | None = None
    converged: bool = True


@dataclass
class BacktestResult:
    """Scored output of a backtest run.

    Attributes:
        points: One entry per (as-of date, candidate) pair.
        election_date: The election being forecast.
        actual: Normalised true vote shares, when supplied.
        results: Full :class:`~kronikas.model.ForecastResult` per as-of date,
            populated only when ``keep_results=True`` was passed to
            :func:`backtest`.
    """

    points: list[BacktestPoint]
    election_date: date
    actual: dict[str, float] | None = None
    results: dict[date, ForecastResult] = field(default_factory=dict, repr=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Return every scored point as a tidy :class:`pandas.DataFrame`.

        One row per (as-of date, candidate), with columns matching the fields
        of :class:`BacktestPoint`.
        """
        return pd.DataFrame([vars(p) for p in self.points])

    def metrics(self) -> dict[str, Any]:
        """Aggregate accuracy and calibration statistics.

        Returns
        -------
        dict
            ``n_forecasts``, ``n_points`` and, when the true result is known,
            ``mae`` (mean absolute error, pp), ``rmse``, ``mean_crps``,
            ``interval_hit_rate_90`` (the observed share of 90 % intervals
            containing the truth) and ``bias_by_candidate``.

        Notes
        -----
        Bias is reported per candidate rather than pooled.  Forecast shares and
        actual shares both sum to 100, so signed errors cancel exactly across
        candidates and a pooled mean is identically zero — it would measure
        nothing.  Per candidate it is informative: a persistent positive value
        means the model overstated that candidate throughout the campaign.
        """
        as_of_dates = sorted({p.as_of for p in self.points})
        summary: dict[str, Any] = {
            "n_forecasts": len(as_of_dates),
            "n_points": len(self.points),
            "all_converged": all(p.converged for p in self.points),
        }
        scored = [p for p in self.points if p.error is not None]
        if scored:
            errors = np.array([p.error for p in scored], dtype=np.float64)
            covered = np.array([bool(p.covered) for p in scored], dtype=bool)
            crps = np.array([cast(float, p.crps) for p in scored], dtype=np.float64)
            names = [p.candidate for p in scored]
            bias_by_candidate: dict[str, float] = {}
            for name in dict.fromkeys(names):
                mask = np.array([n == name for n in names], dtype=bool)
                bias_by_candidate[name] = float(errors[mask].mean())
            summary.update(
                {
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "mean_crps": float(np.mean(crps)),
                    "interval_hit_rate_90": float(np.mean(covered)),
                    # Backward-compatible alias. The less ambitious name above
                    # is preferred because one election cannot establish
                    # repeated-sampling calibration.
                    "coverage_90": float(np.mean(covered)),
                    "bias_by_candidate": bias_by_candidate,
                }
            )
        return summary

    def summary(self) -> str:
        """Return a human-readable backtest report."""
        lines = ["=" * 60, "Backtest report", "=" * 60]
        lines.append(f"Election date: {self.election_date}")
        stats = self.metrics()
        lines.append(
            f"Forecasts: {stats['n_forecasts']} "
            f"({stats['n_points']} candidate-forecasts)"
        )
        if not stats["all_converged"]:
            lines.append("WARNING: at least one refit did not converge cleanly.")

        if "mae" in stats:
            lines.append("")
            lines.append("Accuracy vs. actual result")
            lines.append("-" * 26)
            lines.append(f"  MAE          {stats['mae']:5.2f} pp")
            lines.append(f"  RMSE         {stats['rmse']:5.2f} pp")
            lines.append(f"  Mean CRPS    {stats['mean_crps']:5.2f} pp")
            lines.append(
                f"  90% hit rate  {stats['interval_hit_rate_90']:6.1%}  "
                "(descriptive; many elections are needed for calibration)"
            )

            lines.append("")
            lines.append("Signed bias by candidate")
            lines.append("-" * 24)
            for name, bias in stats["bias_by_candidate"].items():
                lines.append(f"  {name:<20s} {bias:+5.2f} pp")

            lines.append("")
            lines.append("Mean absolute error by as-of date")
            lines.append("-" * 33)
            frame = self.to_dataframe()
            by_date = frame.groupby("as_of")["abs_error"].mean().sort_index()
            for as_of, mae in by_date.items():
                lines.append(f"  {as_of}  {mae:5.2f} pp")
        else:
            lines.append("")
            lines.append("No actual result supplied; forecasts recorded unscored.")

        lines.append("=" * 60)
        return "\n".join(lines)


def _coerce_poll_data(
    polls: PollData | pd.DataFrame | str | Path, **load_kwargs: Any
) -> PollData:
    """Accept poll data as a PollData, DataFrame, or path to a CSV."""
    if isinstance(polls, PollData):
        return polls
    if isinstance(polls, pd.DataFrame):
        load_kwargs.pop("decimal", None)
        return polls_from_dataframe(polls, **load_kwargs)
    return load_polls(polls, **load_kwargs)


def _normalise_actual(
    actual: dict[str, float], candidates: list[str]
) -> dict[str, float]:
    """Rescale the true result to percentage points summing to 100.

    Applies the same normalisation as the poll loader, so the comparison is
    like-for-like regardless of whether the caller passes fractions,
    percentages, or raw vote counts.
    """
    unknown = set(actual) - set(candidates)
    if unknown:
        raise ValueError(
            f"actual contains unknown candidates: {sorted(unknown)}. "
            f"Known candidates: {candidates}."
        )
    missing = set(candidates) - set(actual)
    if missing:
        raise ValueError(
            f"actual is missing candidates: {sorted(missing)}. Provide a value "
            "for every candidate so the shares can be normalised."
        )
    values = np.array([actual[name] for name in candidates], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("actual values must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("actual values must sum to a positive number.")
    return {name: float(actual[name]) / total * 100.0 for name in candidates}


def _crps_ensemble(samples: np.ndarray, actual: float) -> float:
    """Exact CRPS for an equally weighted empirical predictive distribution."""
    ordered = np.sort(np.asarray(samples, dtype=np.float64))
    n = ordered.size
    first = np.mean(np.abs(ordered - actual))
    weights = 2 * np.arange(1, n + 1) - n - 1
    pairwise_half = float(np.sum(weights * ordered)) / (n * n)
    return float(first - pairwise_half)


def backtest(
    polls: PollData | pd.DataFrame | str | Path,
    election_date: date,
    as_of_dates: list[date],
    *,
    actual: dict[str, float] | None = None,
    config: ModelConfig | None = None,
    keep_results: bool = False,
    **load_kwargs: Any,
) -> BacktestResult:
    """Refit the model as of each date and score its election-day forecast.

    Parameters
    ----------
    polls:
        Poll data, as a :class:`~kronikas.data.PollData`, a
        :class:`pandas.DataFrame`, or a path to a CSV.
    election_date:
        The election being forecast.
    as_of_dates:
        Dates to refit at.  For each, only polls published on or before that
        date are used, and the model's reference date is set to it.
    actual:
        True vote shares keyed by candidate name.  Any scale is accepted and
        normalised to sum to 100, matching how polls are handled.  When
        omitted, forecasts are recorded without accuracy scoring.
    config:
        Model and sampler configuration.  Note that a backtest runs one full
        MCMC fit per as-of date, so consider reducing ``num_draws`` or
        disabling ``progressbar``.
    keep_results:
        Retain the full :class:`~kronikas.model.ForecastResult` for each date
        on :attr:`BacktestResult.results`.  Off by default because each result
        holds a complete posterior trace.
    **load_kwargs:
        Forwarded to the loader when *polls* is a path or DataFrame (e.g.
        ``date_column``, ``decimal``).

    Returns
    -------
    BacktestResult

    Raises
    ------
    ValueError
        If *as_of_dates* is empty, or if an as-of date precedes the first poll.

    Examples
    --------
    >>> report = backtest(  # doctest: +SKIP
    ...     "polls.csv",
    ...     election_date=date(2024, 11, 5),
    ...     as_of_dates=[date(2024, 8, 1), date(2024, 10, 1)],
    ...     actual={"Alice": 48.2, "Bob": 47.1, "Carol": 4.7},
    ... )
    >>> print(report.summary())  # doctest: +SKIP
    """
    if not as_of_dates:
        raise ValueError("as_of_dates must contain at least one date.")

    poll_data = _coerce_poll_data(polls, **load_kwargs)
    config = config or ModelConfig()

    normalised_actual = (
        _normalise_actual(actual, list(poll_data.candidates)) if actual else None
    )

    points: list[BacktestPoint] = []
    kept: dict[date, ForecastResult] = {}

    for as_of in sorted(as_of_dates):
        if as_of >= election_date:
            warnings.warn(
                f"as-of date {as_of} is on or after election_date "
                f"({election_date}); skipping, as there is nothing to forecast.",
                stacklevel=2,
            )
            continue

        subset = poll_data.up_to(as_of)
        model, metadata = build_model(subset, election_date, as_of, config)
        trace = run_inference(model, config)
        # Convergence is recorded per point rather than warned about here: a
        # backtest deliberately refits on thin early data, where the occasional
        # ragged fit is expected and is itself part of the finding.
        result = extract_results(
            trace, subset, metadata, config=config, warn_on_convergence=False
        )
        if keep_results:
            kept[as_of] = result

        converged = result.diagnostics is None or result.diagnostics.converged
        n_polls = len(subset.dates)

        for candidate_idx, estimate in enumerate(result.election_day_estimates):
            truth = normalised_actual.get(estimate.name) if normalised_actual else None
            error = estimate.mean - truth if truth is not None else None
            points.append(
                BacktestPoint(
                    as_of=as_of,
                    candidate=estimate.name,
                    n_polls=n_polls,
                    mean=estimate.mean,
                    median=estimate.median,
                    ci_lower=estimate.ci_lower,
                    ci_upper=estimate.ci_upper,
                    actual=truth,
                    error=error,
                    abs_error=abs(error) if error is not None else None,
                    covered=(
                        bool(estimate.ci_lower <= truth <= estimate.ci_upper)
                        if truth is not None
                        else None
                    ),
                    crps=(
                        _crps_ensemble(
                            result._samples_for("election_day")[:, candidate_idx],
                            truth,
                        )
                        if truth is not None
                        else None
                    ),
                    converged=converged,
                )
            )

    return BacktestResult(
        points=points,
        election_date=election_date,
        actual=normalised_actual,
        results=kept,
    )
