"""Hierarchical Bayesian election forecast model.

The model consists of:

* **Logistic-normal random walk** – latent candidate support lives in
  log-ratio space (K-1 dimensions) and is mapped to the probability simplex
  via softmax.  This guarantees proportions are non-negative and sum to 1.
* **House effects** – per-pollster biases in log-ratio space (omitted when
  only one pollster is present).
* **Dirichlet observation model** – polls are Dirichlet-distributed around
  the latent proportions with concentration proportional to the stated
  sample size, times a learnt scaling factor that absorbs design effects
  and other sources of overdispersion.

Non-centred parameterisation is used for the random walk to improve NUTS
sampling geometry.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .config import ModelConfig, PollsterPrior
from .data import PollData
from .diagnostics import ConvergenceWarning, SamplingDiagnostics, compute_diagnostics

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CandidateEstimate:
    """Posterior summary for a single candidate at a point in time."""

    name: str
    mean: float
    median: float
    ci_lower: float  # 5th percentile (90 % credible interval)
    ci_upper: float  # 95th percentile


@dataclass
class ForecastResult:
    """Full forecast output.

    Attributes:
        today_estimates: Per-candidate posterior summaries for *today*.
        election_day_estimates: Per-candidate posterior summaries for
            election day.
        win_probabilities: Mapping of candidate name to the estimated
            probability of holding a **plurality of the vote share** on
            election day — that is, of polling higher than every other
            candidate.  This is not the probability of winning office under
            electoral systems that are not simple national plurality
            (multi-round runoffs, seat allocation from districts, electoral
            colleges, coalition formation).  Use it as a vote-share statistic
            and layer any seat or office model on top of
            :meth:`party_forecast_dataframe`.
        trace: Full ``arviz.InferenceData`` object for advanced inspection.
        candidates: Candidate names in column order.
        pollsters: Pollster names in index order.
        today_samples: ``(n_draws, n_candidates)`` vote shares at *today*.
        election_samples: ``(n_draws, n_candidates)`` shares at election day.
        house_effect_samples: ``(n_draws, n_pollsters, n_candidates)`` house
            effects, or *None* when the model omitted them.
        time_grid: Calendar date of every latent time-grid node.  The final
            entry is always the election date.
        election_date: The modelled election date, when known.
        today: The reference date used for "current" estimates, when known.
        diagnostics: Sampler convergence diagnostics, when available.
    """

    today_estimates: list[CandidateEstimate]
    election_day_estimates: list[CandidateEstimate]
    win_probabilities: dict[str, float]
    trace: az.InferenceData
    candidates: list[str] = field(default_factory=list)
    pollsters: list[str] = field(default_factory=list)
    today_samples: np.ndarray | None = field(default=None, repr=False)
    election_samples: np.ndarray | None = field(default=None, repr=False)
    house_effect_samples: np.ndarray | None = field(default=None, repr=False)
    time_grid: list[date] = field(default_factory=list)
    election_date: date | None = None
    today: date | None = None
    diagnostics: SamplingDiagnostics | None = field(default=None, repr=False)

    # -- sample access ----------------------------------------------------

    def _samples_for(self, day: str) -> np.ndarray:
        """Return the posterior sample matrix for ``"today"``/``"election_day"``."""
        if day == "today":
            samples = self.today_samples
        elif day == "election_day":
            samples = self.election_samples
        else:
            raise ValueError(f"day must be 'today' or 'election_day', got {day!r}")
        if samples is None or len(self.candidates) == 0:
            raise RuntimeError(
                "Samples not available.  Ensure ForecastResult was created "
                "via extract_results()."
            )
        return samples

    def threshold_probabilities(
        self,
        threshold: float,
        day: str = "election_day",
        *,
        inclusive: bool = True,
    ) -> dict[str, float]:
        """Probability that each candidate's vote share reaches *threshold*.

        Many electoral systems apply a minimum vote share below which a party
        receives no seats, and campaigns are often organised around any number
        of other round-number targets.  This computes, directly from the
        posterior draws, the probability that each candidate lands at or above
        a given share.

        Parameters
        ----------
        threshold:
            Vote share in **percentage points** (e.g. ``5.0`` for 5 %).
        day : {"election_day", "today"}
            Time point to evaluate.  Defaults to election day.
        inclusive:
            When True (default) compute ``P(share >= threshold)``; when False,
            ``P(share > threshold)``.  With continuous posterior draws the two
            differ only by a measure-zero set.

        Returns
        -------
        dict[str, float]
            Mapping of candidate name to probability in ``[0, 1]``.

        Examples
        --------
        >>> result.threshold_probabilities(5.0)  # doctest: +SKIP
        {'Alice': 1.0, 'Bob': 0.998, 'Carol': 0.412}
        """
        samples = self._samples_for(day)
        exceeds = samples >= threshold if inclusive else samples > threshold
        return {
            name: float(np.mean(exceeds[:, k]))
            for k, name in enumerate(self.candidates)
        }

    def lead_probability(
        self, candidate: str, opponent: str, day: str = "election_day"
    ) -> float:
        """Probability that *candidate* outpolls *opponent*.

        Parameters
        ----------
        candidate, opponent:
            Candidate names, which must both appear in :attr:`candidates`.
        day : {"election_day", "today"}
            Time point to evaluate.

        Returns
        -------
        float
            ``P(candidate > opponent)`` in ``[0, 1]``.
        """
        samples = self._samples_for(day)
        try:
            i = self.candidates.index(candidate)
            j = self.candidates.index(opponent)
        except ValueError as exc:
            raise KeyError(
                f"Unknown candidate; known candidates are {self.candidates}."
            ) from exc
        return float(np.mean(samples[:, i] > samples[:, j]))

    def party_forecast_dataframe(self, day: str = "today") -> pd.DataFrame:
        """Return a DataFrame of posterior vote-share samples per party.

        Each row represents one posterior draw (warmup excluded).  Each column
        is named after a party/candidate and contains vote-share values in
        percentage points.

        Parameters
        ----------
        day : {"today", "election_day"}
            Time point to extract samples for.  ``"today"`` returns samples at
            the reference date supplied to the model; ``"election_day"`` returns
            samples at the final time step (election date).

        Returns
        -------
        pandas.DataFrame
            Shape ``(n_draws, n_parties)``.
        """
        return pd.DataFrame(self._samples_for(day), columns=self.candidates)

    def latent_trend_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of latent trend percentiles over time.

        Returns
        -------
        pandas.DataFrame
            Contains the mean, 5th percentile, and 95th percentile of
            the latent trend in percentage points for each party at each
            time step.
        """
        pi = self.trace["posterior"]["pi"].values
        pi = pi.reshape(-1, pi.shape[2], pi.shape[3])

        pi_mean = np.mean(pi, axis=0) * 100.0
        pi_p5 = np.percentile(pi, 5, axis=0) * 100.0
        pi_p95 = np.percentile(pi, 95, axis=0) * 100.0

        records = []
        for t in range(pi.shape[1]):
            row = {}
            for k, name in enumerate(self.candidates):
                row[f"{name}_mean"] = pi_mean[t, k]
                row[f"{name}_p_5"] = pi_p5[t, k]
                row[f"{name}_p_95"] = pi_p95[t, k]
            records.append(row)

        df = pd.DataFrame(records)
        if self.time_grid:
            df.index = self.time_grid
        return df

    def house_effects_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of posterior house-effect samples per pollster and party.

        House effects represent systematic per-pollster biases in vote-share
        estimates.  Each value is the percentage-point deviation from a
        neutral equal-support baseline (all candidates at ``1/K``) produced
        by the pollster's bias term.  Positive values mean the pollster
        over-estimates a candidate relative to that neutral point; negative
        values indicate under-estimation.  Within each draw and pollster the
        values across all parties sum to zero.

        The DataFrame uses a two-level column ``MultiIndex``: the outer level
        is the pollster name and the inner level is the candidate/party name.

        Returns
        -------
        pandas.DataFrame
            Shape ``(n_draws, n_pollsters * n_parties)`` with a
            ``pandas.MultiIndex`` on the columns
            (``names=["pollster", "candidate"]``).  Values are in percentage
            points.

        Raises
        ------
        RuntimeError
            If the model was run with a single pollster (house effects are not
            identifiable) or if the result was not created via
            ``extract_results()``.
        """
        if self.house_effect_samples is None:
            if len(self.pollsters) <= 1:
                raise RuntimeError(
                    "House effects are not available: the model was run with "
                    "a single pollster and house effects are not identifiable."
                )
            raise RuntimeError(
                "House effect samples not available.  Ensure ForecastResult "
                "was created via extract_results()."
            )
        if not self.candidates or not self.pollsters:
            raise RuntimeError(
                "House effect samples not available.  Ensure ForecastResult "
                "was created via extract_results()."
            )

        n_draws, n_pollsters, _n_candidates = self.house_effect_samples.shape
        columns = pd.MultiIndex.from_product(
            [self.pollsters, self.candidates],
            names=["pollster", "candidate"],
        )
        data = self.house_effect_samples.reshape(n_draws, -1)
        return pd.DataFrame(data, columns=columns)

    def summary(self) -> str:
        """Return a human-readable forecast summary."""
        lines: list[str] = []
        lines.append("=" * 50)
        lines.append("Election Forecast Summary")
        lines.append("=" * 50)

        def _fmt_section(title: str, estimates: list[CandidateEstimate]) -> list[str]:
            out = [f"\n{title}"]
            out.append("-" * len(title))
            for e in estimates:
                out.append(
                    f"  {e.name:<20s} {e.mean:5.1f}%"
                    f"  (90% CI: {e.ci_lower:5.1f}% – {e.ci_upper:5.1f}%)"
                )
            return out

        lines.extend(_fmt_section("Current estimates", self.today_estimates))
        lines.extend(_fmt_section("Election-day forecast", self.election_day_estimates))

        lines.append("\nPlurality probabilities (election day)")
        lines.append("-" * 37)
        for name, prob in sorted(self.win_probabilities.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:<20s} {prob:6.1%}")

        if self.diagnostics is not None and not self.diagnostics.converged:
            lines.append("")
            lines.append(self.diagnostics.summary())

        lines.append("=" * 50)
        return "\n".join(lines)

    # -- export -----------------------------------------------------------

    def to_dict(self, thresholds: list[float] | None = None) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the forecast.

        Excludes the posterior draws and the trace, so the output stays small
        enough to publish directly.  Use :meth:`save` to persist the full
        posterior.

        Parameters
        ----------
        thresholds:
            Optional vote-share thresholds in percentage points.  For each one,
            :meth:`threshold_probabilities` is evaluated on election day and
            included under ``"threshold_probabilities"``, keyed by the
            threshold formatted as a string.
        """

        def _estimates(items: list[CandidateEstimate]) -> list[dict[str, Any]]:
            return [
                {
                    "name": e.name,
                    "mean": e.mean,
                    "median": e.median,
                    "ci_lower": e.ci_lower,
                    "ci_upper": e.ci_upper,
                }
                for e in items
            ]

        payload: dict[str, Any] = {
            "election_date": self.election_date.isoformat()
            if self.election_date
            else None,
            "today": self.today.isoformat() if self.today else None,
            "candidates": list(self.candidates),
            "pollsters": list(self.pollsters),
            "today_estimates": _estimates(self.today_estimates),
            "election_day_estimates": _estimates(self.election_day_estimates),
            "win_probabilities": dict(self.win_probabilities),
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
        }
        if thresholds:
            payload["threshold_probabilities"] = {
                f"{t:g}": self.threshold_probabilities(t) for t in thresholds
            }
        return payload

    def save(self, path: str | Path) -> Path:
        """Persist the full result to a netCDF file.

        Sampling can take minutes, so a run worth keeping should not have to be
        repeated.  The posterior trace is written via ``arviz.to_netcdf`` with
        the forecast metadata attached as an attribute, allowing :meth:`load`
        to reconstruct an equivalent :class:`ForecastResult`.

        Parameters
        ----------
        path:
            Destination file.  Parent directories must already exist.

        Returns
        -------
        pathlib.Path
            The path written to.
        """
        path = Path(path)
        metadata = {
            "candidates": list(self.candidates),
            "pollsters": list(self.pollsters),
            "today_idx": self._index_of_date(self.today),
            "election_idx": len(self.time_grid) - 1 if self.time_grid else None,
            "n_timesteps": len(self.time_grid),
            "include_house_effects": self.house_effect_samples is not None,
            "grid_start_date": self.time_grid[0].isoformat()
            if self.time_grid
            else None,
            "election_date": self.election_date.isoformat()
            if self.election_date
            else None,
            "today": self.today.isoformat() if self.today else None,
            "time_step_days": self._infer_step_days(),
        }
        self.trace.attrs["kronikas_metadata"] = json.dumps(metadata)
        az.to_netcdf(self.trace, str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> ForecastResult:
        """Reconstruct a result previously written by :meth:`save`."""
        path = Path(path)
        trace = az.from_netcdf(str(path))
        raw = trace.attrs.get("kronikas_metadata")
        if not raw:
            raise ValueError(
                f"{path} does not carry kronikas metadata; it was not written "
                "by ForecastResult.save()."
            )
        stored = json.loads(raw)
        metadata = {
            "today_idx": stored["today_idx"],
            "election_idx": stored["election_idx"],
            "n_timesteps": stored["n_timesteps"],
            "include_house_effects": stored["include_house_effects"],
            "grid_start_date": date.fromisoformat(stored["grid_start_date"])
            if stored.get("grid_start_date")
            else None,
            "election_date": date.fromisoformat(stored["election_date"])
            if stored.get("election_date")
            else None,
            "today": date.fromisoformat(stored["today"])
            if stored.get("today")
            else None,
            "time_step_days": stored["time_step_days"],
        }
        return _build_result(
            trace,
            candidates=stored["candidates"],
            pollsters=stored["pollsters"],
            metadata=metadata,
            warn_on_convergence=False,
        )

    def _index_of_date(self, target: date | None) -> int:
        """Index of the grid node nearest *target* (0 when unknown)."""
        if target is None or not self.time_grid:
            return 0
        deltas = [abs((node - target).days) for node in self.time_grid]
        return int(np.argmin(deltas))

    def _infer_step_days(self) -> int:
        """Recover the grid step from consecutive node dates."""
        if len(self.time_grid) < 2:
            return 1
        return (self.time_grid[1] - self.time_grid[0]).days


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------


def _np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax for NumPy arrays."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _pp_to_logit(pp: float, baseline: float = 0.5) -> float:
    """Convert a percentage-point bias to a logit-space shift.

    The shift is computed relative to *baseline*, the candidate's own support
    level, as ``logit(baseline + pp/100) - logit(baseline)``.  Anchoring on the
    candidate's actual level matters: a +3 pp bias is a logit shift of 0.12 for
    a candidate polling at 50 % but 0.50 for one polling at 5 %, so a fixed
    50 % baseline understates small-party biases roughly four-fold.

    Parameters
    ----------
    pp:
        Bias in percentage points.
    baseline:
        Candidate support as a proportion in ``(0, 1)``.  Defaults to 0.5, for
        which the formula reduces to plain ``logit(0.5 + pp/100)``.

    Raises
    ------
    ValueError
        If *baseline* is not a valid proportion, or if the bias would push the
        candidate outside ``(0 %, 100 %)``.
    """
    if not 0.0 < baseline < 1.0:
        raise ValueError(f"baseline must lie in (0, 1); got {baseline}.")
    shifted = baseline + pp / 100.0
    if not 0.0 < shifted < 1.0:
        raise ValueError(
            f"A mu_house bias of {pp} pp is impossible for a candidate at "
            f"{baseline * 100:.1f} %: it implies a support level of "
            f"{shifted * 100:.1f} %, outside (0, 100)."
        )
    return float(
        np.log(shifted / (1.0 - shifted)) - np.log(baseline / (1.0 - baseline))
    )


def _pt_softmax(x: pt.TensorVariable, axis: int = -1) -> pt.TensorVariable:
    """Numerically stable softmax for PyTensor tensors."""
    x_max = pt.max(x, axis=axis, keepdims=True)
    e_x = pt.exp(x - x_max)
    return e_x / pt.sum(e_x, axis=axis, keepdims=True)


@dataclass(frozen=True)
class TimeGrid:
    """A discrete time grid anchored on election day.

    The grid is laid out *backwards* from the election so that its final node
    falls exactly on ``election_date``.  Anchoring forwards from the first poll
    instead — as earlier versions did — leaves the last node up to
    ``step_days - 1`` days *after* the election, which reports a forecast for
    the wrong date and inflates its credible interval with a spurious extra
    step of random-walk drift.

    The trade-off is that ``start_date`` may precede the first poll by up to
    ``step_days - 1`` days, which is harmless: the latent trend simply begins
    slightly earlier.

    Attributes:
        start_date: Calendar date of grid node 0.
        n_timesteps: Total number of nodes, including both endpoints.
        step_days: Spacing between consecutive nodes, in days.
    """

    start_date: date
    n_timesteps: int
    step_days: int

    @property
    def end_date(self) -> date:
        """Calendar date of the final node (always the election date)."""
        return self.start_date + timedelta(days=(self.n_timesteps - 1) * self.step_days)

    def dates(self) -> list[date]:
        """Calendar date of every node, in order."""
        return [
            self.start_date + timedelta(days=i * self.step_days)
            for i in range(self.n_timesteps)
        ]

    def index_of(self, target: date) -> int:
        """Map a calendar date to the nearest node index, clipped to the grid."""
        offset = (target - self.start_date).days
        return max(0, min(round(offset / self.step_days), self.n_timesteps - 1))

    def offsets_to_indices(self, day_offsets: np.ndarray) -> np.ndarray:
        """Map day offsets from ``start_date`` to nearest node indices."""
        indices = np.rint(day_offsets / self.step_days).astype(np.int64)
        return np.clip(indices, 0, self.n_timesteps - 1)


def _build_time_grid(
    first_poll_date: date,
    election_date: date,
    time_step_days: int,
) -> TimeGrid:
    """Construct an election-anchored time grid covering all polls."""
    if time_step_days < 1:
        raise ValueError(
            f"time_step_days must be a positive integer, got {time_step_days}."
        )
    total_days = (election_date - first_poll_date).days
    if total_days <= 0:
        raise ValueError(
            f"election_date must be after the first poll date ({first_poll_date})."
        )
    n_steps = math.ceil(total_days / time_step_days)
    start_date = election_date - timedelta(days=n_steps * time_step_days)
    return TimeGrid(
        start_date=start_date,
        n_timesteps=n_steps + 1,
        step_days=time_step_days,
    )


def _warn_on_post_election_polls(poll_data: PollData, election_date: date) -> None:
    """Warn when polls postdate the election and are silently clipped.

    Such polls are pinned to the final grid node and therefore treated as
    election-day observations, which is almost never what the user intends.
    """
    poll_dates = np.array(
        [poll_data.first_poll_date + timedelta(days=int(d)) for d in poll_data.dates]
    )
    late = poll_dates > election_date
    if late.any():
        warnings.warn(
            f"{int(late.sum())} poll(s) are dated after election_date "
            f"({election_date}); the latest is {max(poll_dates[late])}. They "
            "will be pinned to the final time-grid node and treated as "
            "election-day observations. Drop them or check election_date.",
            stacklevel=3,
        )


def _warn_if_today_out_of_range(
    today: date, grid: TimeGrid, election_date: date
) -> None:
    """Warn when *today* falls outside the modelled window and is clamped."""
    if today < grid.start_date:
        warnings.warn(
            f"today ({today}) precedes the start of the time grid "
            f"({grid.start_date}); 'today' estimates will be clamped to the "
            "first time step.",
            stacklevel=3,
        )
    elif today > election_date:
        warnings.warn(
            f"today ({today}) is after election_date ({election_date}); "
            "'today' estimates will be clamped to election day.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_model(
    poll_data: PollData,
    election_date: date,
    today: date,
    config: ModelConfig,
) -> tuple[pm.Model, dict]:
    """Construct the PyMC model.

    Returns
    -------
    model
        A ``pymc.Model`` ready for sampling.
    metadata
        Dict with ``today_idx``, ``election_idx``, ``n_timesteps`` and
        ``include_house_effects``.
    """

    n_candidates = len(poll_data.candidates)
    n_pollsters = len(poll_data.pollsters)

    grid = _build_time_grid(
        poll_data.first_poll_date, election_date, config.time_step_days
    )
    n_timesteps = grid.n_timesteps

    # Poll offsets are stored relative to the first poll; the grid may start
    # earlier, so shift them onto the grid's own origin.
    grid_offset = (poll_data.first_poll_date - grid.start_date).days
    time_indices = grid.offsets_to_indices(poll_data.dates + grid_offset)

    _warn_on_post_election_polls(poll_data, election_date)
    _warn_if_today_out_of_range(today, grid, election_date)

    today_idx = grid.index_of(today)
    election_idx = n_timesteps - 1

    # ------------------------------------------------------------------
    # Derive initial log-ratios from earliest polls
    # ------------------------------------------------------------------
    early_mask = time_indices == 0
    if early_mask.sum() > 0:
        initial_props = poll_data.poll_values[early_mask].mean(axis=0) / 100.0
    else:
        initial_props = poll_data.poll_values.mean(axis=0) / 100.0

    initial_props = np.clip(initial_props, 1e-4, None)
    initial_props = initial_props / initial_props.sum()

    # K-1 log-ratios relative to the last candidate (reference)
    ref = initial_props[-1]
    initial_logratios = np.log(initial_props[:-1] / ref)

    # ------------------------------------------------------------------
    # Observed proportions on the simplex (fractions summing to 1)
    # ------------------------------------------------------------------
    observed_fractions = poll_data.poll_values / 100.0
    observed_fractions = np.clip(observed_fractions, 1e-6, None)
    observed_fractions = observed_fractions / observed_fractions.sum(
        axis=1, keepdims=True
    )

    include_house = n_pollsters > 1
    n_free = n_candidates - 1  # log-ratio dimensions

    # ------------------------------------------------------------------
    # Resolve per-pollster prior overrides
    # ------------------------------------------------------------------
    pollster_index = {name: i for i, name in enumerate(poll_data.pollsters)}
    resolved_priors: dict[int, PollsterPrior] = {}
    for name, prior in config.pollster_priors.items():
        if name in pollster_index:
            resolved_priors[pollster_index[name]] = prior
        else:
            warnings.warn(
                f"pollster_priors key {name!r} does not match any pollster "
                f"in the data (known: {poll_data.pollsters}). Ignoring.",
                stacklevel=2,
            )

    # Per-pollster kappa overrides
    has_custom_kappa = {
        j: p.kappa_log_sigma
        for j, p in resolved_priors.items()
        if p.kappa_log_sigma is not None
    }
    use_per_pollster_kappa = len(has_custom_kappa) > 0

    # Per-pollster house-effect overrides
    has_custom_house = {
        j: p.sigma_house
        for j, p in resolved_priors.items()
        if p.sigma_house is not None
    }

    # Per-pollster, per-party prior means for house effects.
    # Shape: (n_pollsters, n_candidates).  Defaults to 0.0; only entries with an
    # explicit mu_house override are non-zero.  These are in logit space, each
    # converted relative to that candidate's own support level.
    candidate_index = {name: i for i, name in enumerate(poll_data.candidates)}
    mu_matrix = np.zeros((n_pollsters, n_candidates))
    for j, prior in resolved_priors.items():
        if prior.mu_house:
            for party, mu in prior.mu_house.items():
                if party not in candidate_index:
                    warnings.warn(
                        f"mu_house key {party!r} does not match any "
                        f"candidate (known: {list(poll_data.candidates)}). "
                        f"Ignoring.",
                        stacklevel=2,
                    )
                else:
                    k = candidate_index[party]
                    mu_matrix[j, k] = _pp_to_logit(mu, baseline=initial_props[k])

    # ------------------------------------------------------------------
    # PyMC model
    # ------------------------------------------------------------------
    # `sigma_walk_prior` is expressed per `walk_reference_days`; rescale it to
    # the actual step length so the implied volatility over a fixed calendar
    # window does not change when `time_step_days` changes.
    walk_sigma_prior = config.per_step_walk_sigma

    with pm.Model() as model:
        # === Random-walk volatility ===
        if config.correlated_walk:
            # Correlated innovations: learn per-dimension SDs and an
            # inter-party correlation matrix via an LKJ prior.  The
            # Cholesky factor L of the covariance Σ = L·Lᵀ encodes both
            # scale and correlation.  sigma_walk_prior is reused as the
            # HalfNormal scale for each dimension's SD.
            chol, corr, sigmas = pm.LKJCholeskyCov(
                "chol_cov",
                n=n_free,
                eta=config.lkj_eta,
                sd_dist=pm.HalfNormal.dist(sigma=walk_sigma_prior),
                compute_corr=True,
            )
            pm.Deterministic("walk_corr", corr)
            pm.Deterministic("walk_sigmas", sigmas)
        else:
            sigma_walk = pm.HalfNormal("sigma_walk", sigma=walk_sigma_prior)

        # === Dirichlet concentration scaling ===
        if use_per_pollster_kappa:
            # Per-pollster kappa_log with individual prior SDs
            kappa_sigmas = [
                has_custom_kappa.get(j, config.kappa_log_sigma)
                for j in range(n_pollsters)
            ]
            kappa_log = pm.Normal(
                "kappa_log",
                mu=0.0,
                sigma=kappa_sigmas,
                shape=n_pollsters,
            )
            kappa_scale = pt.exp(kappa_log)  # (n_pollsters,)
        else:
            # Single shared kappa_log (original behaviour)
            kappa_log = pm.Normal("kappa_log", mu=0.0, sigma=config.kappa_log_sigma)
            kappa_scale = pt.exp(kappa_log)  # scalar

        # === Initial latent support (K-1 log-ratios) ===
        eta_init = pm.Normal(
            "eta_init",
            mu=initial_logratios,
            sigma=config.initial_sigma,
            shape=n_free,
        )

        # === Gaussian random walk (non-centred parameterisation) ===
        if n_timesteps > 1:
            innovations = pm.Normal(
                "innovations",
                0.0,
                1.0,
                shape=(n_timesteps - 1, n_free),
            )
            if config.correlated_walk:
                # Transform i.i.d. innovations through Cholesky factor:
                # each row innovations[t] @ L.T ~ N(0, Σ), producing
                # correlated steps across log-ratio dimensions.
                scaled = pt.dot(innovations, chol.T)
                eta_rest = eta_init[None, :] + pt.cumsum(scaled, axis=0)
            else:
                eta_rest = eta_init[None, :] + sigma_walk * pt.cumsum(
                    innovations, axis=0
                )
            eta = pt.concatenate([eta_init[None, :], eta_rest], axis=0)  # (T, K-1)
        else:
            eta = eta_init[None, :]  # (1, K-1)

        # Pad with zeros for the reference candidate, then softmax → simplex
        zeros_col = pt.zeros((eta.shape[0], 1))
        eta_full = pt.concatenate([eta, zeros_col], axis=1)  # (T, K)
        pi = _pt_softmax(eta_full, axis=1)  # (T, K)

        pm.Deterministic("pi", pi)

        # === House effects (log-ratio space, skip if single pollster) ===
        if include_house:
            if has_custom_house:
                # Build per-pollster sigma vector; pollsters without
                # overrides share a hierarchical sigma_house.
                needs_hierarchical = len(has_custom_house) < n_pollsters
                if needs_hierarchical:
                    sigma_house = pm.HalfNormal(
                        "sigma_house", sigma=config.sigma_house_prior
                    )
                sigma_parts = []
                for j in range(n_pollsters):
                    if j in has_custom_house:
                        sigma_parts.append(np.float64(has_custom_house[j]))
                    else:
                        sigma_parts.append(sigma_house)
                sigma_vec = pt.stack(sigma_parts)  # (n_pollsters,)
                delta_raw = pm.Normal(
                    "delta_raw",
                    mu_matrix,
                    sigma_vec[:, None],
                    shape=(n_pollsters, n_candidates),
                )
            else:
                sigma_house = pm.HalfNormal(
                    "sigma_house", sigma=config.sigma_house_prior
                )
                delta_raw = pm.Normal(
                    "delta_raw",
                    mu_matrix,
                    sigma_house,
                    shape=(n_pollsters, n_candidates),
                )
            # Zero-mean constrain house effects across all K parties for each pollster
            delta_full = pm.Deterministic(
                "delta", delta_raw - pt.mean(delta_raw, axis=1, keepdims=True)
            )

            eta_obs = eta_full[time_indices] + delta_full[poll_data.pollster_ids]
            mu_obs = _pt_softmax(eta_obs, axis=1)
        else:
            mu_obs = pi[time_indices]

        # === Dirichlet observation model ===
        sample_sizes = pt.as_tensor_variable(poll_data.sample_sizes.reshape(-1, 1))
        if use_per_pollster_kappa:
            # Index per-pollster kappa_scale by poll's pollster
            kappa = kappa_scale[poll_data.pollster_ids, None] * sample_sizes
        else:
            kappa = kappa_scale * sample_sizes  # (N, 1)
        alpha_dir = pt.maximum(mu_obs * kappa, 0.01)  # (N, K)

        pm.Dirichlet("obs", a=alpha_dir, observed=observed_fractions)

    metadata = {
        "today_idx": today_idx,
        "election_idx": election_idx,
        "n_timesteps": n_timesteps,
        "include_house_effects": include_house,
        "grid_start_date": grid.start_date,
        "election_date": election_date,
        "today": today,
        "first_poll_date": poll_data.first_poll_date,
        "time_step_days": config.time_step_days,
    }
    return model, metadata


def run_inference(model: pm.Model, config: ModelConfig) -> az.InferenceData:
    """Run NUTS sampling and return an ArviZ ``InferenceData``."""
    extra = dict(config.sampler_kwargs) if config.sampler_kwargs else {}

    if config.compute_log_likelihood:
        # Merge rather than overwrite: `sampler_kwargs` is the documented
        # escape hatch and may already carry unrelated idata_kwargs.
        idata_kwargs = dict(extra.get("idata_kwargs") or {})
        idata_kwargs.setdefault("log_likelihood", True)
        extra["idata_kwargs"] = idata_kwargs

    with model:
        trace = pm.sample(
            draws=config.num_draws,
            tune=config.num_tune,
            chains=config.num_chains,
            cores=config.cores,
            target_accept=config.target_accept,
            random_seed=config.random_seed,
            init=config.init_method,
            progressbar=config.progressbar,
            **extra,
        )
    return trace


def extract_results(
    trace: az.InferenceData,
    poll_data: PollData,
    metadata: dict,
    *,
    config: ModelConfig | None = None,
    warn_on_convergence: bool = True,
) -> ForecastResult:
    """Derive forecast summaries from the posterior trace.

    The ``pi`` Deterministic is already on the simplex (sums to 1), so no
    post-hoc normalisation is needed.

    Parameters
    ----------
    trace:
        Posterior samples from :func:`run_inference`.
    poll_data:
        The data the model was fitted to; supplies candidate and pollster names.
    metadata:
        The dict returned alongside the model by :func:`build_model`.
    config:
        Used only for the convergence thresholds.  Defaults are used when
        *None*.
    warn_on_convergence:
        Emit a :class:`~kronikas.diagnostics.ConvergenceWarning` when the
        posterior shows signs of non-convergence.
    """
    return _build_result(
        trace,
        candidates=list(poll_data.candidates),
        pollsters=list(poll_data.pollsters),
        metadata=metadata,
        config=config,
        warn_on_convergence=warn_on_convergence,
    )


def _build_result(
    trace: az.InferenceData,
    *,
    candidates: list[str],
    pollsters: list[str],
    metadata: dict,
    config: ModelConfig | None = None,
    warn_on_convergence: bool = True,
) -> ForecastResult:
    """Assemble a :class:`ForecastResult` from a trace and its metadata."""
    config = config or ModelConfig()

    # pi shape: (chains, draws, T, K)
    pi = trace["posterior"]["pi"].values
    # Flatten chains x draws → samples
    pi = pi.reshape(-1, pi.shape[2], pi.shape[3])

    # Convert from proportions (0-1) to percentage points (0-100)
    today_samples = pi[:, metadata["today_idx"], :] * 100.0
    election_samples = pi[:, metadata["election_idx"], :] * 100.0

    def _estimates(
        samples: np.ndarray, candidates: list[str]
    ) -> list[CandidateEstimate]:
        out = []
        for k, name in enumerate(candidates):
            col = samples[:, k]
            out.append(
                CandidateEstimate(
                    name=name,
                    mean=float(np.mean(col)),
                    median=float(np.median(col)),
                    ci_lower=float(np.percentile(col, 5)),
                    ci_upper=float(np.percentile(col, 95)),
                )
            )
        return out

    today_est = _estimates(today_samples, candidates)
    election_est = _estimates(election_samples, candidates)

    # Plurality probability: fraction of samples where the candidate leads
    winners = np.argmax(election_samples, axis=1)
    win_probs = {
        name: float(np.mean(winners == k)) for k, name in enumerate(candidates)
    }

    # House effects: percentage-point deviations from equal-support baseline
    house_effect_samples = None
    if metadata.get("include_house_effects") and "delta" in trace["posterior"]:
        # delta shape: (chains, draws, n_pollsters, K)
        delta_full = trace["posterior"]["delta"].values
        # Flatten chains × draws → samples: (n_samples, n_pollsters, K)
        delta_full = delta_full.reshape(-1, delta_full.shape[2], delta_full.shape[3])
        n_samples, n_pollsters, n_candidates = delta_full.shape
        # Softmax maps log-ratio offsets to proportions on the simplex
        pi_biased = _np_softmax(delta_full, axis=2)
        # Deviation from equal-support baseline (1/K per candidate)
        house_effect_samples = (pi_biased - 1.0 / n_candidates) * 100.0

    time_grid: list[date] = []
    start = metadata.get("grid_start_date")
    step = metadata.get("time_step_days")
    if start is not None and step is not None:
        time_grid = [
            start + timedelta(days=i * step) for i in range(metadata["n_timesteps"])
        ]

    diagnostics = compute_diagnostics(
        trace,
        r_hat_threshold=config.r_hat_threshold,
        ess_threshold=config.ess_threshold,
    )
    if warn_on_convergence and not diagnostics.converged:
        warnings.warn(
            "Posterior may not have converged; treat these estimates with "
            "caution.\n" + "\n".join(f"  - {issue}" for issue in diagnostics.issues),
            ConvergenceWarning,
            stacklevel=3,
        )

    return ForecastResult(
        today_estimates=today_est,
        election_day_estimates=election_est,
        win_probabilities=win_probs,
        trace=trace,
        candidates=list(candidates),
        pollsters=list(pollsters),
        today_samples=today_samples,
        election_samples=election_samples,
        house_effect_samples=house_effect_samples,
        time_grid=time_grid,
        election_date=metadata.get("election_date"),
        today=metadata.get("today"),
        diagnostics=diagnostics,
    )
