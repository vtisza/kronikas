"""Sampler convergence diagnostics.

PyMC already runs its own convergence checks inside ``pm.sample()`` and emits
``SamplerWarning`` messages when it finds problems.  Those warnings are easy to
miss in a script that goes on to print confident-looking numbers, so this
module re-derives the headline statistics into a small, inspectable object that
travels with the forecast and can be escalated deliberately.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np

__all__ = [
    "ConvergenceWarning",
    "SamplingDiagnostics",
    "compute_diagnostics",
]


class ConvergenceWarning(UserWarning):
    """Warning raised when the posterior shows signs of non-convergence."""


@dataclass(frozen=True)
class SamplingDiagnostics:
    """Headline convergence statistics for a posterior trace.

    Attributes:
        max_r_hat: Largest Gelman–Rubin statistic across all parameters, or
            *None* when it cannot be computed (fewer than 2 chains).
        max_r_hat_variable: Name of the variable attaining ``max_r_hat``.
        min_ess_bulk: Smallest bulk effective sample size across parameters.
        min_ess_bulk_variable: Name of the variable attaining ``min_ess_bulk``.
        min_ess_tail: Smallest tail effective sample size across parameters.
        n_divergences: Number of divergent transitions recorded by NUTS.
        n_chains: Number of chains in the trace.
        n_draws: Posterior draws retained per chain.
        r_hat_threshold: Threshold used to judge ``max_r_hat``.
        ess_threshold: Threshold used to judge ``min_ess_bulk``.
    """

    max_r_hat: float | None
    max_r_hat_variable: str | None
    min_ess_bulk: float | None
    min_ess_bulk_variable: str | None
    min_ess_tail: float | None
    n_divergences: int
    n_chains: int
    n_draws: int
    r_hat_threshold: float = 1.01
    ess_threshold: float = 400.0

    @property
    def issues(self) -> list[str]:
        """Human-readable descriptions of every problem detected.

        Only positive evidence of a problem appears here.  Circumstances that
        merely *limit* what can be checked are reported by :attr:`notes`.
        """
        found: list[str] = []
        if self.max_r_hat is not None and self.max_r_hat > self.r_hat_threshold:
            found.append(
                f"R-hat {self.max_r_hat:.4f} exceeds {self.r_hat_threshold} "
                f"(worst: {self.max_r_hat_variable}); chains have not mixed."
            )
        if self.min_ess_bulk is not None and self.min_ess_bulk < self.ess_threshold:
            found.append(
                f"Bulk ESS {self.min_ess_bulk:.0f} is below {self.ess_threshold:.0f} "
                f"(worst: {self.min_ess_bulk_variable}); estimates are noisy."
            )
        if self.n_divergences > 0:
            found.append(
                f"{self.n_divergences} divergent transition(s); the posterior "
                "geometry is being explored unreliably. Try raising "
                "target_accept."
            )
        return found

    @property
    def notes(self) -> list[str]:
        """Caveats that limit how much the diagnostics can establish.

        A single-chain run, for instance, is not evidence of non-convergence —
        it simply leaves convergence unchecked, which is worth saying out loud
        without labelling the fit as bad.
        """
        found: list[str] = []
        if self.max_r_hat is None:
            found.append(
                f"R-hat is not computable with {self.n_chains} chain(s); "
                "convergence is unverified. Use num_chains >= 2 to check it."
            )
        return found

    @property
    def converged(self) -> bool:
        """True when no convergence problem was detected.

        A ``True`` value with a non-empty :attr:`notes` means "nothing looks
        wrong", not "convergence was verified".
        """
        return not self.issues

    def summary(self) -> str:
        """Return a human-readable diagnostics report."""
        lines = ["Sampling diagnostics", "-" * 20]
        lines.append(f"  chains x draws     {self.n_chains} x {self.n_draws}")
        r_hat = "n/a" if self.max_r_hat is None else f"{self.max_r_hat:.4f}"
        worst = f" ({self.max_r_hat_variable})" if self.max_r_hat_variable else ""
        lines.append(f"  max R-hat          {r_hat}{worst}")
        ess = "n/a" if self.min_ess_bulk is None else f"{self.min_ess_bulk:.0f}"
        ess_worst = (
            f" ({self.min_ess_bulk_variable})" if self.min_ess_bulk_variable else ""
        )
        lines.append(f"  min ESS (bulk)     {ess}{ess_worst}")
        tail = "n/a" if self.min_ess_tail is None else f"{self.min_ess_tail:.0f}"
        lines.append(f"  min ESS (tail)     {tail}")
        lines.append(f"  divergences        {self.n_divergences}")
        if self.converged:
            lines.append("  status             OK")
        else:
            lines.append("  status             PROBLEMS DETECTED")
            lines.extend(f"    - {issue}" for issue in self.issues)
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "max_r_hat": self.max_r_hat,
            "max_r_hat_variable": self.max_r_hat_variable,
            "min_ess_bulk": self.min_ess_bulk,
            "min_ess_bulk_variable": self.min_ess_bulk_variable,
            "min_ess_tail": self.min_ess_tail,
            "n_divergences": self.n_divergences,
            "n_chains": self.n_chains,
            "n_draws": self.n_draws,
            "r_hat_threshold": self.r_hat_threshold,
            "ess_threshold": self.ess_threshold,
            "converged": self.converged,
            "issues": self.issues,
            "notes": self.notes,
        }


def _extreme_over_dataset(
    dataset: Any, *, largest: bool
) -> tuple[float | None, str | None]:
    """Reduce an xarray Dataset of per-parameter stats to a single extreme.

    Returns the extreme value and the name of the variable attaining it,
    ignoring all-NaN variables (which arise when a statistic is undefined).
    """
    reducer = np.nanmax if largest else np.nanmin
    best: float | None = None
    best_name: str | None = None
    for name, values in dataset.data_vars.items():
        array = np.asarray(values.values, dtype=np.float64)
        if array.size == 0 or np.all(np.isnan(array)):
            continue
        candidate = float(reducer(array))
        if np.isnan(candidate):
            continue
        if best is None or (candidate > best if largest else candidate < best):
            best = candidate
            best_name = str(name)
    return best, best_name


def compute_diagnostics(
    trace: az.InferenceData,
    *,
    r_hat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
    var_names: list[str] | None = None,
) -> SamplingDiagnostics:
    """Compute convergence diagnostics for a posterior trace.

    Parameters
    ----------
    trace:
        Posterior samples as returned by :func:`kronikas.model.run_inference`.
    r_hat_threshold, ess_threshold:
        Thresholds recorded on the result and used by
        :attr:`SamplingDiagnostics.converged`.
    var_names:
        Restrict the computation to these variables.  Defaults to every
        posterior variable.

    Returns
    -------
    SamplingDiagnostics
    """
    posterior = trace["posterior"]
    n_chains = int(posterior.sizes.get("chain", 0))
    n_draws = int(posterior.sizes.get("draw", 0))

    kwargs = {"var_names": var_names} if var_names else {}

    # ArviZ warns (and returns NaN) when R-hat is undefined for a single
    # chain; that case is reported through `issues` instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n_chains >= 2:
            max_r_hat, max_r_hat_var = _extreme_over_dataset(
                az.rhat(trace, **kwargs), largest=True
            )
        else:
            max_r_hat, max_r_hat_var = None, None

        min_ess_bulk, min_ess_bulk_var = _extreme_over_dataset(
            az.ess(trace, method="bulk", **kwargs), largest=False
        )
        min_ess_tail, _ = _extreme_over_dataset(
            az.ess(trace, method="tail", **kwargs), largest=False
        )

    n_divergences = 0
    sample_stats = getattr(trace, "sample_stats", None)
    if sample_stats is not None and "diverging" in sample_stats:
        n_divergences = int(np.asarray(sample_stats["diverging"].values).sum())

    return SamplingDiagnostics(
        max_r_hat=max_r_hat,
        max_r_hat_variable=max_r_hat_var,
        min_ess_bulk=min_ess_bulk,
        min_ess_bulk_variable=min_ess_bulk_var,
        min_ess_tail=min_ess_tail,
        n_divergences=n_divergences,
        n_chains=n_chains,
        n_draws=n_draws,
        r_hat_threshold=r_hat_threshold,
        ess_threshold=ess_threshold,
    )
