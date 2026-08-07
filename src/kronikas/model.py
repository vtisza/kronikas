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
import xarray as xr

from .config import ModelConfig, PollsterPrior, SharedBiasPrior
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

    def assume_shared_bias(self, offsets: dict[str, float]) -> ForecastResult:
        """Re-derive the forecast assuming the polls carry an industry-wide bias.

        A bias shared by every pollster is invisible to the model: the
        likelihood is exactly unchanged by shifting the latent trend one way and
        all house effects the other, so a common error passes into the forecast
        one-for-one.  That same invariance is what makes this method legitimate
        rather than a fudge — the shifted posterior is just as consistent with
        the observed polls as the original, so this reports a different point on
        a ridge the data cannot distinguish.

        Use it to state how fragile a call is: *this lead survives a 1 pp
        industry error and does not survive 3 pp.*

        No refit is needed; the stored posterior draws are shifted directly.

        .. note::
           This is an approximation to refitting with
           :class:`~kronikas.config.SharedBiasPrior`.  It is exact on the
           shares by construction, and on synthetic checks the mean estimate
           matched a refit to within 0.05 pp — but the prior also constrains
           the latent level, so the two do not coincide exactly and tail
           probabilities can differ by several points.  For exploring fragility
           this is the right tool; for a number you intend to publish, refit
           with ``ModelConfig(shared_bias=...)``.

        Parameters
        ----------
        offsets:
            Per-candidate bias in **percentage points**, keyed by name.
            Positive means the polls **over**-state that candidate, so the
            corrected forecast moves it *down*.  Candidates omitted are treated
            as 0.  Offsets need not sum to zero; the remainder is redistributed
            proportionally, matching how poll rows are normalised.

        Returns
        -------
        ForecastResult
            A new result with shifted samples, re-derived estimates and
            re-derived probabilities. ``trace`` and ``house_effect_samples``
            still describe the source fit, so :meth:`latent_trend_dataframe`
            reflects the original forecast. :meth:`save` stores the adjusted
            sample matrices too, so loading preserves the scenario exactly.

        Raises
        ------
        KeyError
            If *offsets* names a candidate that is not in the model.

        Examples
        --------
        >>> shifted = result.assume_shared_bias({"Alice": 3.0})  # doctest: +SKIP
        >>> shifted.win_probabilities  # doctest: +SKIP
        """
        unknown = set(offsets) - set(self.candidates)
        if unknown:
            raise KeyError(
                f"Unknown candidate(s) {sorted(unknown)}; known candidates are "
                f"{self.candidates}."
            )
        shift = np.array(
            [float(offsets.get(name, 0.0)) for name in self.candidates],
            dtype=np.float64,
        )
        # Candidates the caller did not name take up any residual, so a stated
        # correction lands at its stated size (see _balance_offsets).
        absorb = np.array([name not in offsets for name in self.candidates], dtype=bool)

        def _apply(samples: np.ndarray) -> np.ndarray:
            adjusted = samples - _balance_offsets(shift, samples, absorb)
            # A large correction can drive a small party negative in some
            # draws; floor it rather than emit impossible shares.
            floored = np.clip(adjusted, 1e-9, None)
            if not np.array_equal(floored, adjusted):
                warnings.warn(
                    "assume_shared_bias drove at least one candidate to a "
                    "non-positive share in some draws; those draws were "
                    "floored at ~0. The offsets may be too large for a small "
                    "party.",
                    stacklevel=3,
                )
            return floored / floored.sum(axis=1, keepdims=True) * 100.0

        today_samples = _apply(self._samples_for("today"))
        election_samples = _apply(self._samples_for("election_day"))

        return self._with_samples(today_samples, election_samples)

    def _with_samples(
        self, today_samples: np.ndarray, election_samples: np.ndarray
    ) -> ForecastResult:
        """Copy this result and rebuild all summaries from authoritative samples."""

        def _estimates(samples: np.ndarray) -> list[CandidateEstimate]:
            return [
                CandidateEstimate(
                    name=name,
                    mean=float(np.mean(samples[:, k])),
                    median=float(np.median(samples[:, k])),
                    ci_lower=float(np.percentile(samples[:, k], 5)),
                    ci_upper=float(np.percentile(samples[:, k], 95)),
                )
                for k, name in enumerate(self.candidates)
            ]

        winners = np.argmax(election_samples, axis=1)
        return ForecastResult(
            today_estimates=_estimates(today_samples),
            election_day_estimates=_estimates(election_samples),
            win_probabilities={
                name: float(np.mean(winners == k))
                for k, name in enumerate(self.candidates)
            },
            trace=self.trace,
            candidates=list(self.candidates),
            pollsters=list(self.pollsters),
            today_samples=today_samples,
            election_samples=election_samples,
            house_effect_samples=self.house_effect_samples,
            time_grid=list(self.time_grid),
            election_date=self.election_date,
            today=self.today,
            diagnostics=self.diagnostics,
        )

    def shared_bias_breakeven(
        self, day: str = "election_day", *, max_pp: float = 25.0
    ) -> float | None:
        """Smallest uniform polling error that would erase the leader's lead.

        Shifts support from the front-runner to the runner-up in equal measure
        and reports the size at which the front-runner's probability of leading
        falls to 50 %.  A small number means the call rests on the polls being
        collectively accurate.

        Parameters
        ----------
        day : {"election_day", "today"}
            Time point to evaluate.
        max_pp:
            Upper bound of the search, in percentage points.

        Returns
        -------
        float or None
            The break-even error in percentage points, or *None* if the lead
            survives a shift of *max_pp* (or if there is no clear leader).
        """
        samples = self._samples_for(day)
        means = samples.mean(axis=0)
        order = np.argsort(means)[::-1]
        leader, runner_up = int(order[0]), int(order[1])

        def probability(shift_pp: float) -> float:
            adjusted = samples.copy()
            adjusted[:, leader] -= shift_pp
            adjusted[:, runner_up] += shift_pp
            return float(np.mean(np.argmax(adjusted, axis=1) == leader))

        if probability(0.0) < 0.5:
            return 0.0
        if probability(max_pp) >= 0.5:
            return None

        low, high = 0.0, max_pp
        for _ in range(40):
            mid = (low + high) / 2.0
            if probability(mid) >= 0.5:
                low = mid
            else:
                high = mid
        return float((low + high) / 2.0)

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
        trace_to_save = self.trace.copy()
        trace_to_save.attrs["kronikas_metadata"] = json.dumps(metadata)
        if "kronikas_result" in trace_to_save.groups():
            delattr(trace_to_save, "kronikas_result")
        if self.today_samples is not None and self.election_samples is not None:
            trace_to_save.add_groups(
                {
                    "kronikas_result": xr.Dataset(
                        {
                            "today_samples": (
                                ("sample", "candidate"),
                                self.today_samples,
                            ),
                            "election_samples": (
                                ("sample", "candidate"),
                                self.election_samples,
                            ),
                        },
                        coords={"candidate": self.candidates},
                    )
                }
            )
        az.to_netcdf(trace_to_save, str(path))
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
        result = _build_result(
            trace,
            candidates=stored["candidates"],
            pollsters=stored["pollsters"],
            metadata=metadata,
            warn_on_convergence=False,
        )
        if "kronikas_result" in trace.groups():
            stored_samples = trace["kronikas_result"]
            result = result._with_samples(
                stored_samples["today_samples"].values,
                stored_samples["election_samples"].values,
            )
        return result

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


def _pp_sd_to_logit(sd_pp: float, baseline: float) -> float:
    """Convert a percentage-point standard deviation to log-ratio scale.

    Uses the local derivative ``d logit / d p = 1 / (p (1 - p))`` at *baseline*.
    This is a linearisation, exact only in the limit of small *sd_pp*, which is
    the regime that matters here: shared polling errors are a few points, not
    tens of points.
    """
    if not 0.0 < baseline < 1.0:
        raise ValueError(f"baseline must lie in (0, 1); got {baseline}.")
    if sd_pp < 0:
        raise ValueError(f"sd must be >= 0; got {sd_pp}.")
    return float((sd_pp / 100.0) / (baseline * (1.0 - baseline)))


def _balance_offsets(
    offsets_pp: np.ndarray, shares: np.ndarray, absorb: np.ndarray
) -> np.ndarray:
    """Spread the residual of *offsets_pp* so the offsets sum to zero.

    Shares must sum to a constant, so a bias statement has to balance: if one
    candidate is overstated, someone else is understated.  When the caller
    names only some candidates — "polls overstate A by 4" — the remaining 4
    points are taken from the candidates they did *not* name, in proportion to
    those candidates' support.

    Renormalising the whole vector afterwards instead would be wrong: scaling
    every share back up partially undoes the requested shift, so a stated 4 pp
    correction would land as roughly 2 pp.

    Parameters
    ----------
    offsets_pp:
        ``(K,)`` requested offsets in percentage points.
    shares:
        ``(N, K)`` share vectors the offsets apply to, used for the weights.
    absorb:
        ``(K,)`` boolean mask of candidates allowed to take up the residual —
        normally those the caller did not name.

    Returns
    -------
    numpy.ndarray
        ``(N, K)`` offsets that sum to zero along the candidate axis.
    """
    balanced = np.broadcast_to(offsets_pp, shares.shape).astype(np.float64).copy()
    residual = float(offsets_pp.sum())
    if abs(residual) < 1e-12:
        return balanced
    if not absorb.any():
        # Everything was named and it does not balance; fall back to spreading
        # across all candidates rather than silently ignoring the residual.
        absorb = np.ones_like(absorb, dtype=bool)
    block = shares[:, absorb]
    total = block.sum(axis=1, keepdims=True)
    uniform = np.full_like(block, 1.0 / max(int(absorb.sum()), 1))
    weights = np.divide(block, total, out=uniform, where=total > 0)
    balanced[:, absorb] -= residual * weights
    return balanced


def _shared_mean_to_logit(
    offsets_pp: np.ndarray, baseline: np.ndarray, absorb: np.ndarray | None = None
) -> np.ndarray:
    """Convert a *vector* of percentage-point offsets to a log-ratio shift.

    The offsets are a joint statement about the whole share vector — "polls
    show A 3 pp too high and B 3 pp too low" — so they must be converted
    jointly.  Converting each candidate independently with its own marginal
    logit double-counts: in a softmax the gap between two candidates moves by
    the *sum* of their individual shifts, so a 3 pp correction applied to both
    sides of a pair produces a 6 pp swing in their margin.

    Instead, build the corrected share vector directly and take the log-ratio
    difference, which lands exactly on the intended target::

        q = normalise(baseline - offsets)
        shift = log(baseline) - log(q)

    Offsets that do not sum to zero are balanced by :func:`_balance_offsets`,
    which takes the remainder from the candidates the caller did not name.
    """
    if absorb is None:
        absorb = offsets_pp == 0.0
    balanced = _balance_offsets(offsets_pp, baseline[None, :] * 100.0, absorb)[0]

    corrected = baseline - balanced / 100.0
    if np.any(corrected <= 0):
        bad = np.where(corrected <= 0)[0]
        raise ValueError(
            "shared_bias mean is too large: it drives "
            f"candidate index/indices {bad.tolist()} to a non-positive share "
            f"(baseline {np.round(baseline * 100, 1).tolist()} %, offsets "
            f"{offsets_pp.tolist()} pp)."
        )
    corrected = corrected / corrected.sum()
    shift = np.log(baseline) - np.log(corrected)
    # The additive constant is absorbed by the softmax; centre it for tidiness.
    return shift - shift.mean()


def _resolve_shared_bias(
    prior: SharedBiasPrior,
    candidates: list[str],
    initial_props: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate a SharedBiasPrior into log-ratio mean and SD vectors.

    The mean is converted jointly (see :func:`_shared_mean_to_logit`); the SD
    is converted per candidate with the local derivative, which is the right
    treatment for an independent perturbation of a single candidate.

    Returns
    -------
    mean_logit, sd_logit
        Arrays of shape ``(n_candidates,)``.
    """
    index = {name: i for i, name in enumerate(candidates)}
    for supplied in (prior.mean, prior.sd):
        for name in supplied:
            if name not in index:
                warnings.warn(
                    f"shared_bias key {name!r} does not match any candidate "
                    f"(known: {candidates}). Ignoring.",
                    stacklevel=3,
                )

    offsets_pp = np.array(
        [float(prior.mean.get(name, 0.0)) for name in candidates], dtype=np.float64
    )
    # Candidates the caller did not name take up any residual.
    absorb = np.array([name not in prior.mean for name in candidates], dtype=bool)
    mean_logit = (
        _shared_mean_to_logit(offsets_pp, initial_props, absorb)
        if np.any(offsets_pp)
        else np.zeros(len(candidates))
    )

    target_sd_pp = np.zeros(len(candidates))
    for name, k in index.items():
        sd_pp = float(prior.sd.get(name, prior.default_sd))
        target_sd_pp[k] = sd_pp

    sd_logit = np.zeros(len(candidates))
    if np.any(target_sd_pp > 0):
        corrected_baseline = _np_softmax(np.log(initial_props) - mean_logit)
        centring, sd_logit = _calibrate_shared_bias_spread(
            target_sd_pp, corrected_baseline
        )
        mean_logit = mean_logit + centring
    return mean_logit, sd_logit


def _calibrate_shared_bias_spread(
    target_sd_pp: np.ndarray, baseline: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate zero-sum logit scales to marginal share-space SDs.

    Softmax couples every candidate, so converting each requested percentage-
    point SD independently overstates the joint uncertainty. A deterministic
    Monte Carlo calibration accounts for that coupling and also removes the
    small Jensen shift, keeping the prior predictive mean at ``baseline``.

    A zero target means no *direct* latent error for that candidate. Its share
    can still move indirectly because all shares must sum to one.
    """
    target = np.asarray(target_sd_pp, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    n_candidates = baseline.size
    rng = np.random.default_rng(20240517)
    z = rng.normal(size=(50_000, n_candidates))
    z -= z.mean(axis=1, keepdims=True)
    z *= math.sqrt(n_candidates / (n_candidates - 1))
    # Eliminate finite-simulation scale noise so calibration is reproducible.
    z /= z.std(axis=0, keepdims=True)

    scales = np.array(
        [
            _pp_sd_to_logit(sd, float(base)) if sd > 0 else 0.0
            for sd, base in zip(target, baseline, strict=True)
        ]
    )
    centring = np.zeros(n_candidates)
    active = target > 0
    log_baseline = np.log(baseline)
    for _ in range(20):
        draws = _np_softmax(log_baseline - centring - z * scales, axis=1)
        achieved = draws.std(axis=0) * 100.0
        scales[active] *= target[active] / np.maximum(achieved[active], 1e-12)
        centring += np.log(draws.mean(axis=0) / baseline)
        centring -= centring.mean()
    return centring, scales


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

    if poll_data.last_poll_date > election_date:
        raise ValueError(
            f"Poll data contains observations after election_date "
            f"({election_date}); filter or correct those rows before building "
            "the model."
        )

    grid = _build_time_grid(
        poll_data.first_poll_date, election_date, config.time_step_days
    )
    n_timesteps = grid.n_timesteps

    # Poll offsets are stored relative to the first poll; the grid may start
    # earlier, so shift them onto the grid's own origin.
    grid_offset = (poll_data.first_poll_date - grid.start_date).days
    time_indices = grid.offsets_to_indices(poll_data.dates + grid_offset)

    _warn_if_today_out_of_range(today, grid, election_date)

    today_idx = grid.index_of(today)
    election_idx = n_timesteps - 1

    # ------------------------------------------------------------------
    # Derive initial log-ratios from earliest polls
    # ------------------------------------------------------------------
    # The backwards-anchored grid may begin before the first poll, so grid node
    # zero can legitimately be empty. Initialise from the earliest occupied
    # node, never from an average that leaks later campaign information back
    # into the initial-state prior.
    early_mask = time_indices == time_indices.min()
    initial_props = poll_data.poll_values[early_mask].mean(axis=0) / 100.0

    initial_props = np.clip(initial_props, 1e-4, None)
    initial_props = initial_props / initial_props.sum()

    # Use the largest candidate as the reference. The posterior is invariant to
    # this choice, but a tiny reference produces unnecessarily extreme
    # log-ratios and poor sampler geometry.
    reference_idx = int(np.argmax(initial_props))
    free_indices = np.array(
        [k for k in range(n_candidates) if k != reference_idx], dtype=np.int64
    )
    ref = initial_props[reference_idx]
    initial_logratios = np.log(initial_props[free_indices] / ref)

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

    # ------------------------------------------------------------------
    # Industry-wide shared bias
    # ------------------------------------------------------------------
    shared_mean_logit = np.zeros(n_candidates)
    shared_sd_logit = np.zeros(n_candidates)
    shared_bias_active = False
    if config.shared_bias is not None:
        shared_mean_logit, shared_sd_logit = _resolve_shared_bias(
            config.shared_bias, list(poll_data.candidates), initial_props
        )
        shared_bias_active = bool(
            np.any(shared_mean_logit != 0.0) or np.any(shared_sd_logit != 0.0)
        )

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

        # Insert a zero column for the reference candidate, preserving the
        # caller-visible candidate order.
        eta_full = pt.zeros((eta.shape[0], n_candidates))
        eta_full = pt.set_subtensor(eta_full[:, free_indices], eta)
        pi = _pt_softmax(eta_full, axis=1)  # (T, K)

        pm.Deterministic("pi", pi)

        # === House effects (log-ratio space, skip if single pollster) ===
        if include_house:
            needs_hierarchical = len(has_custom_house) < n_pollsters
            if needs_hierarchical:
                sigma_house = pm.HalfNormal(
                    "sigma_house", sigma=config.sigma_house_prior
                )
            sigma_parts = [
                np.float64(has_custom_house[j])
                if j in has_custom_house
                else sigma_house
                for j in range(n_pollsters)
            ]
            sigma_vec = pt.stack(sigma_parts)

            # There are only (P-1)*(K-1) identifiable house-effect dimensions.
            # Sampling a P*K Normal and centring it afterwards leaves P+K-1
            # flat directions in NUTS. ZeroSumNormal removes those dimensions
            # from the transform itself. Recentring after per-pollster scaling
            # preserves both constraints when custom scales are present.
            delta_raw = pm.ZeroSumNormal(
                "delta_raw",
                sigma=1.0,
                shape=(n_pollsters, n_candidates),
                n_zerosum_axes=2,
            )
            centred_mu = mu_matrix - mu_matrix.mean(axis=1, keepdims=True)
            centred_mu = centred_mu - centred_mu.mean(axis=0, keepdims=True)
            centred = pt.as_tensor_variable(centred_mu) + delta_raw * sigma_vec[:, None]
            centred = centred - pt.mean(centred, axis=1, keepdims=True)
            centred = centred - pt.mean(centred, axis=0, keepdims=True)
            delta_full = pm.Deterministic("delta", centred)

            eta_obs = eta_full[time_indices] + delta_full[poll_data.pollster_ids]
        else:
            eta_obs = eta_full[time_indices]

        # === Industry-wide bias (not identifiable; scale must be supplied) ===
        if shared_bias_active:
            shift = pt.as_tensor_variable(shared_mean_logit)
            if np.any(shared_sd_logit > 0):
                # Non-centred, and candidates with sd == 0 contribute nothing
                # extra, so a pure point scenario needs no special case.
                z = pm.ZeroSumNormal(
                    "shared_bias_z",
                    sigma=math.sqrt(n_candidates / (n_candidates - 1)),
                    shape=n_candidates,
                )
                shift = shift + z * pt.as_tensor_variable(shared_sd_logit)
            shared = pm.Deterministic("shared_bias", shift)
            # Polls show latent support PLUS the industry's common error, so
            # `pi` remains the bias-corrected estimate of true support.
            eta_obs = eta_obs + shared[None, :]

        mu_obs = _pt_softmax(eta_obs, axis=1)

        # === Dirichlet observation model ===
        sample_sizes = pt.as_tensor_variable(poll_data.sample_sizes.reshape(-1, 1))
        if use_per_pollster_kappa:
            # Index per-pollster kappa_scale by poll's pollster
            kappa = kappa_scale[poll_data.pollster_ids, None] * sample_sizes
        else:
            kappa = kappa_scale * sample_sizes  # (N, 1)
        raw_alpha = mu_obs * kappa
        alpha_dir = 0.01 + pt.softplus(100.0 * (raw_alpha - 0.01)) / 100.0

        pm.Dirichlet("obs", a=alpha_dir, observed=observed_fractions)

    metadata = {
        "today_idx": today_idx,
        "election_idx": election_idx,
        "n_timesteps": n_timesteps,
        "include_house_effects": include_house,
        "shared_bias_active": shared_bias_active,
        "grid_start_date": grid.start_date,
        "election_date": election_date,
        "today": today,
        "first_poll_date": poll_data.first_poll_date,
        "time_step_days": config.time_step_days,
        "reference_candidate_idx": reference_idx,
        "initial_props": initial_props.copy(),
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
