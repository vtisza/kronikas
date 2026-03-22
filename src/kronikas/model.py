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

import math
import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .config import ModelConfig, PollsterPrior
from .data import PollData

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
        win_probabilities: Mapping of candidate name to estimated
            probability of having the highest support on election day.
        trace: Full ``arviz.InferenceData`` object for advanced inspection.
    """

    today_estimates: list[CandidateEstimate]
    election_day_estimates: list[CandidateEstimate]
    win_probabilities: dict[str, float]
    trace: Any  # az.InferenceData
    candidates: list[str] = field(default_factory=list)
    pollsters: list[str] = field(default_factory=list)
    today_samples: np.ndarray = field(default=None, repr=False)
    election_samples: np.ndarray = field(default=None, repr=False)
    house_effect_samples: np.ndarray = field(default=None, repr=False)
    time_grid: list[date] = field(default_factory=list)

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
        import pandas as pd

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
        return pd.DataFrame(samples, columns=self.candidates)

    def latent_trend_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of latent trend percentiles over time.

        Returns
        -------
        pandas.DataFrame
            Contains the mean, 5th percentile, and 95th percentile of
            the latent trend in percentage points for each party at each
            time step.
        """
        import numpy as np
        import pandas as pd

        pi = self.trace.posterior["pi"].values
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
        import pandas as pd

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

        lines.append("\nWin probabilities")
        lines.append("-" * 17)
        for name, prob in sorted(self.win_probabilities.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name:<20s} {prob:6.1%}")

        lines.append("=" * 50)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------


def _np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax for NumPy arrays."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _pp_to_logit(pp: float) -> float:
    """Convert a percentage-point bias to a logit-space shift.

    Uses 50 % support as the baseline, which gives a good approximation
    across the typical electoral range (20–80 %).  The conversion is exact
    at 50 % and slightly underestimates the logit shift at extreme proportions.

    Parameters
    ----------
    pp:
        Bias in percentage points.  Must satisfy ``-50 < pp < 50``.
    """
    if abs(pp) >= 50.0:
        raise ValueError(
            f"mu_house values must be in the range (-50, 50) pp; got {pp}."
        )
    p = 0.5 + pp / 100.0
    return float(np.log(p / (1.0 - p)))


def _pt_softmax(x: pt.TensorVariable, axis: int = -1) -> pt.TensorVariable:
    """Numerically stable softmax for PyTensor tensors."""
    x_max = pt.max(x, axis=axis, keepdims=True)
    e_x = pt.exp(x - x_max)
    return e_x / pt.sum(e_x, axis=axis, keepdims=True)


def _build_time_grid(
    first_poll_date: date,
    election_date: date,
    time_step_days: int,
) -> int:
    """Return the number of discrete time steps in the grid."""
    total_days = (election_date - first_poll_date).days
    if total_days <= 0:
        raise ValueError(
            f"election_date must be after the first poll date ({first_poll_date})."
        )
    return math.ceil(total_days / time_step_days) + 1


def _map_polls_to_grid(
    poll_days: np.ndarray,
    n_timesteps: int,
    time_step_days: int,
) -> np.ndarray:
    """Map poll day-offsets to the nearest time-grid index."""
    indices = np.rint(poll_days / time_step_days).astype(np.int64)
    return np.clip(indices, 0, n_timesteps - 1)


def _date_to_index(
    target: date,
    first_poll_date: date,
    n_timesteps: int,
    time_step_days: int,
) -> int:
    """Map an arbitrary calendar date to the nearest grid index."""
    day_offset = (target - first_poll_date).days
    idx = round(day_offset / time_step_days)
    return max(0, min(idx, n_timesteps - 1))


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

    n_timesteps = _build_time_grid(
        poll_data.first_poll_date, election_date, config.time_step_days
    )
    time_indices = _map_polls_to_grid(
        poll_data.dates, n_timesteps, config.time_step_days
    )
    today_idx = _date_to_index(
        today, poll_data.first_poll_date, n_timesteps, config.time_step_days
    )
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
    # explicit mu_house override are non-zero.  These are in logit space.
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
                    mu_matrix[j, candidate_index[party]] = _pp_to_logit(mu)

    # ------------------------------------------------------------------
    # PyMC model
    # ------------------------------------------------------------------
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
                sd_dist=pm.HalfNormal.dist(sigma=config.sigma_walk_prior),
                compute_corr=True,
            )
            pm.Deterministic("walk_corr", corr)
            pm.Deterministic("walk_sigmas", sigmas)
        else:
            sigma_walk = pm.HalfNormal("sigma_walk", sigma=config.sigma_walk_prior)

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
        "first_poll_date": poll_data.first_poll_date,
        "time_step_days": config.time_step_days,
    }
    return model, metadata


def run_inference(model: pm.Model, config: ModelConfig) -> az.InferenceData:
    """Run NUTS sampling and return an ArviZ ``InferenceData``."""
    extra = dict(config.sampler_kwargs) if config.sampler_kwargs else {}
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
) -> ForecastResult:
    """Derive forecast summaries from the posterior trace.

    The ``pi`` Deterministic is already on the simplex (sums to 1), so no
    post-hoc normalisation is needed.
    """

    # pi shape: (chains, draws, T, K)
    pi = trace.posterior["pi"].values
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

    today_est = _estimates(today_samples, poll_data.candidates)
    election_est = _estimates(election_samples, poll_data.candidates)

    # Win probability: fraction of samples where candidate leads
    winners = np.argmax(election_samples, axis=1)
    win_probs = {
        name: float(np.mean(winners == k))
        for k, name in enumerate(poll_data.candidates)
    }

    # House effects: percentage-point deviations from equal-support baseline
    house_effect_samples = None
    if metadata.get("include_house_effects") and "delta" in trace.posterior:
        # delta shape: (chains, draws, n_pollsters, K)
        delta_full = trace.posterior["delta"].values
        # Flatten chains × draws → samples: (n_samples, n_pollsters, K)
        delta_full = delta_full.reshape(-1, delta_full.shape[2], delta_full.shape[3])
        n_samples, n_pollsters, n_candidates = delta_full.shape
        # Softmax maps log-ratio offsets to proportions on the simplex
        pi_biased = _np_softmax(delta_full, axis=2)
        # Deviation from equal-support baseline (1/K per candidate)
        house_effect_samples = (pi_biased - 1.0 / n_candidates) * 100.0

    time_grid = []
    if "first_poll_date" in metadata and "time_step_days" in metadata:
        from datetime import timedelta

        start = metadata["first_poll_date"]
        step = metadata["time_step_days"]
        time_grid = [
            start + timedelta(days=i * step) for i in range(metadata["n_timesteps"])
        ]

    return ForecastResult(
        today_estimates=today_est,
        election_day_estimates=election_est,
        win_probabilities=win_probs,
        trace=trace,
        candidates=list(poll_data.candidates),
        pollsters=list(poll_data.pollsters),
        today_samples=today_samples,
        election_samples=election_samples,
        house_effect_samples=house_effect_samples,
        time_grid=time_grid,
    )
