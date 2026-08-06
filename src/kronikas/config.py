"""Configuration and prior settings for the election forecast model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PollsterPrior:
    """Per-pollster prior overrides.

    Any field left as *None* inherits the global default from
    ``ModelConfig``.

    Attributes:
        sigma_house: Fixed SD for this pollster's house effect in logit
            space.  Overrides the hierarchical ``sigma_house`` for this
            pollster only.  Lower values express more trust in the
            pollster's accuracy; higher values allow larger systematic
            bias.
        kappa_log_sigma: SD of the ``Normal(0, .)`` prior on this
            pollster's log precision-scaling factor.  Smaller values
            constrain the pollster's effective sample size closer to the
            stated one; larger values allow more overdispersion.
        mu_house: Prior mean for this pollster's house effect, specified
            as a dict mapping candidate name to a bias in **percentage
            points**.  Only the candidates you want to deviate from zero
            need to be listed; all others default to 0.0.  A positive
            value encodes a belief that the pollster over-estimates that
            candidate; a negative value encodes under-estimation.  Must
            be in the range (-50, 50).  The reference candidate (last in
            the data) cannot be set and is always anchored at zero.
            Values are converted to logit space internally.
    """

    sigma_house: float | None = None
    kappa_log_sigma: float | None = None
    mu_house: dict[str, float] | None = None


@dataclass
class ModelConfig:
    """Configuration for the hierarchical Bayesian election forecast.

    The model operates in **log-ratio (logit) space** internally.  Latent
    candidate support is parameterised as K-1 log-ratios relative to a
    reference candidate and mapped to the probability simplex via softmax.
    The observation model is Dirichlet, which structurally guarantees that
    predicted shares sum to 100 %.

    Attributes:
        num_tune: Number of MCMC warmup (tuning) iterations per chain.
            During warmup the sampler adapts step size and mass matrix;
            these draws are discarded.
        num_draws: Number of posterior draws to keep **per chain** after
            warmup.  Total samples = ``num_draws × num_chains``.
        num_chains: Number of independent MCMC chains.  At least 2 is
            recommended to check convergence via R-hat.
        cores: CPU cores used for parallel chain sampling.  *None* lets
            PyMC auto-detect (typically ``min(num_chains, cpu_count)``).
            Set to 1 to force sequential sampling.
        target_accept: Target acceptance probability for NUTS.  Values of
            0.90–0.99 help prevent divergences in this model.
        random_seed: Seed for reproducibility.
        init_method: NUTS mass-matrix initialisation strategy passed to
            ``pymc.sample(init=...)``.  Common choices:
            ``"jitter+adapt_diag"`` (default, fast), ``"adapt_diag"``,
            ``"adapt_full"`` (slower but better for correlated posteriors).
        progressbar: Whether to display a progress bar during sampling.
        sampler_kwargs: Arbitrary extra keyword arguments forwarded verbatim
            to ``pymc.sample()``.  Use this for advanced settings not
            exposed above (e.g. ``nuts_sampler``, ``idata_kwargs``,
            ``compile_kwargs``).
        compute_log_likelihood: If True, ask PyMC to store the pointwise
            log-likelihood in the returned ``InferenceData``.  Required for
            model comparison with ``arviz.loo`` / ``arviz.waic``.  Adds an
            ``(chains, draws, n_polls)`` array to the trace, so it is off by
            default.
        time_step_days: Time discretisation granularity in days.  Polls are
            binned to the nearest step.  Weekly (7) balances resolution and
            speed.  The time grid is anchored so that its **last node falls
            exactly on election day**; the grid therefore starts on or just
            before the first poll.
        sigma_walk_prior: Scale (``HalfNormal``) for the random-walk SD in
            logit space, expressed **per ``walk_reference_days`` days** rather
            than per time step.  Smaller values produce smoother trends; 0.05
            corresponds to roughly 1 pp weekly movement at 50 % support.
            Because the prior is defined on a fixed calendar interval, changing
            ``time_step_days`` no longer changes the implied volatility of the
            trend (see ``walk_reference_days``).
        walk_reference_days: Calendar interval, in days, that
            ``sigma_walk_prior`` refers to.  The per-step prior scale actually
            handed to the sampler is
            ``sigma_walk_prior * sqrt(time_step_days / walk_reference_days)``,
            which keeps the variance accumulated over any fixed calendar
            window invariant to the grid resolution.  With the defaults
            (both 7) the per-step scale equals ``sigma_walk_prior`` exactly,
            matching the behaviour of earlier versions.
        sigma_house_prior: Scale (``HalfNormal``) for house-effect SD in
            logit space.  0.3 allows up to ~5 pp systematic pollster bias.
        initial_sigma: SD of the ``Normal`` prior on the initial log-ratios.
            0.5 allows moderate uncertainty about starting proportions.
        kappa_log_sigma: SD of the ``Normal(0, .)`` prior on the log
            concentration-scaling factor.  This controls how much noisier
            polls are compared to pure multinomial sampling.  At the prior
            mean (``kappa_scale = 1``), poll precision equals the multinomial
            expectation for the stated sample size; values < 1 imply extra
            noise (design effects, non-response, etc.).
        correlated_walk: If True, replace the scalar ``sigma_walk`` with a
            full covariance structure for the random-walk innovations using
            an ``LKJCholeskyCov`` prior.  This lets the model learn
            inter-party correlations — for example, that gains for one bloc
            typically coincide with losses for its main rival.  Each
            log-ratio dimension gets its own SD (prior set by
            ``sigma_walk_prior``, rescaled the same way) and a correlation
            matrix is estimated.  **Warning:** increases model complexity by
            O(K²) parameters and may require higher ``target_accept``
            (0.97–0.99) to avoid divergences.  Default ``False``.
        lkj_eta: Shape parameter of the LKJ prior on the correlation
            matrix, used only when ``correlated_walk`` is True.  Controls
            how strongly the prior favours identity-like (low correlation)
            matrices:

            - ``eta = 1.0``: uniform over all valid correlation matrices.
            - ``eta = 2.0``: mildly favours weaker correlations (default).
            - ``eta ≥ 5.0``: strongly favours near-independent dimensions.

            ``eta = 2.0`` is a reasonable default for most party systems: it
            lets a strong anti-correlation between two dominant parties
            emerge from the data while regularising poorly-identified
            small-party correlations toward zero.
        r_hat_threshold: Convergence warning threshold for the Gelman–Rubin
            statistic.  After sampling, a warning is emitted if any parameter
            exceeds this value.  Requires at least 2 chains to be computable.
        ess_threshold: Convergence warning threshold for the minimum bulk
            effective sample size across parameters.
    """

    # Sampler
    num_tune: int = 1500
    num_draws: int = 1000
    num_chains: int = 2
    cores: int | None = None
    target_accept: float = 0.95
    random_seed: int = 42
    init_method: str = "jitter+adapt_diag"
    progressbar: bool = True
    compute_log_likelihood: bool = False
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)

    # Time discretisation
    time_step_days: int = 7

    # Priors (logit / log-ratio scale)
    sigma_walk_prior: float = 0.05
    walk_reference_days: int = 7
    sigma_house_prior: float = 0.3
    initial_sigma: float = 0.5
    kappa_log_sigma: float = 0.5

    # Convergence warning thresholds
    r_hat_threshold: float = 1.01
    ess_threshold: float = 400.0

    # Correlated random walk
    correlated_walk: bool = False
    lkj_eta: float = 2.0

    # Per-pollster prior overrides
    pollster_priors: dict[str, PollsterPrior] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time_step_days < 1:
            raise ValueError(
                f"time_step_days must be a positive integer, got {self.time_step_days}."
            )
        if self.walk_reference_days < 1:
            raise ValueError(
                "walk_reference_days must be a positive integer, "
                f"got {self.walk_reference_days}."
            )
        for name in ("sigma_walk_prior", "sigma_house_prior", "initial_sigma"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

    @property
    def per_step_walk_sigma(self) -> float:
        """Prior scale for a single random-walk step.

        ``sigma_walk_prior`` is expressed per ``walk_reference_days`` days.
        Rescaling by ``sqrt(time_step_days / walk_reference_days)`` keeps the
        variance accumulated over a fixed calendar window independent of the
        grid resolution, so changing ``time_step_days`` alters the time
        resolution of the trend without also changing how volatile the prior
        says it is.
        """
        return self.sigma_walk_prior * math.sqrt(
            self.time_step_days / self.walk_reference_days
        )
