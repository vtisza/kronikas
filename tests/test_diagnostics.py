"""Tests for convergence diagnostics."""

from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import pytest

from kronikas.diagnostics import (
    ConvergenceWarning,
    SamplingDiagnostics,
    compute_diagnostics,
)


def _trace(
    n_chains: int = 2, n_draws: int = 4000, *, diverging: int = 0, seed: int = 0
) -> az.InferenceData:
    """Build a well-mixed synthetic trace, optionally with divergences.

    Draws are i.i.d., so R-hat should sit at ~1.0 — but only once the chains
    are long enough for the statistic to settle; short chains scatter above
    the 1.01 threshold by chance alone.
    """
    rng = np.random.default_rng(seed)
    posterior = {
        "alpha": rng.normal(size=(n_chains, n_draws)),
        "beta": rng.normal(size=(n_chains, n_draws, 3)),
    }
    flags = np.zeros((n_chains, n_draws), dtype=bool)
    if diverging:
        flags.reshape(-1)[:diverging] = True
    return az.from_dict(posterior, sample_stats={"diverging": flags})


class TestComputeDiagnostics:
    def test_clean_trace_reports_converged(self):
        diagnostics = compute_diagnostics(_trace(), ess_threshold=50.0)
        assert diagnostics.converged
        assert diagnostics.issues == []
        assert diagnostics.n_chains == 2
        assert diagnostics.n_draws == 4000
        assert diagnostics.max_r_hat is not None

    def test_divergences_are_counted_and_flagged(self):
        diagnostics = compute_diagnostics(_trace(diverging=7), ess_threshold=50.0)
        assert diagnostics.n_divergences == 7
        assert not diagnostics.converged
        assert any("divergent" in issue for issue in diagnostics.issues)

    def test_low_ess_is_flagged(self):
        diagnostics = compute_diagnostics(_trace(), ess_threshold=1e9)
        assert not diagnostics.converged
        assert any("ESS" in issue for issue in diagnostics.issues)
        assert diagnostics.min_ess_bulk_variable in {"alpha", "beta"}

    def test_high_r_hat_is_flagged(self):
        diagnostics = compute_diagnostics(_trace(), r_hat_threshold=0.0)
        assert not diagnostics.converged
        assert any("R-hat" in issue for issue in diagnostics.issues)

    def test_single_chain_is_a_note_not_a_failure(self):
        """One chain leaves convergence unverified; that is not a defect."""
        diagnostics = compute_diagnostics(_trace(n_chains=1), ess_threshold=50.0)
        assert diagnostics.max_r_hat is None
        assert diagnostics.converged
        assert diagnostics.issues == []
        assert any("not computable" in note for note in diagnostics.notes)

    def test_summary_mentions_status(self):
        assert "OK" in compute_diagnostics(_trace(), ess_threshold=50.0).summary()
        assert (
            "PROBLEMS"
            in compute_diagnostics(_trace(diverging=3), ess_threshold=50.0).summary()
        )

    def test_to_dict_is_json_serialisable(self):
        import json

        payload = compute_diagnostics(_trace(), ess_threshold=50.0).to_dict()
        assert json.loads(json.dumps(payload))["converged"] is True

    def test_missing_sample_stats_is_tolerated(self):
        trace = az.from_dict({"alpha": np.random.default_rng(1).normal(size=(2, 50))})
        assert compute_diagnostics(trace, ess_threshold=1.0).n_divergences == 0


class TestSamplingDiagnosticsDataclass:
    def _make(self, **overrides) -> SamplingDiagnostics:
        base = {
            "max_r_hat": 1.0,
            "max_r_hat_variable": "alpha",
            "min_ess_bulk": 1000.0,
            "min_ess_bulk_variable": "alpha",
            "min_ess_tail": 900.0,
            "n_divergences": 0,
            "n_chains": 4,
            "n_draws": 1000,
        }
        return SamplingDiagnostics(**{**base, **overrides})

    def test_converged_when_all_thresholds_met(self):
        assert self._make().converged

    def test_r_hat_above_threshold_fails(self):
        assert not self._make(max_r_hat=1.05).converged

    def test_ess_below_threshold_fails(self):
        assert not self._make(min_ess_bulk=10.0).converged


class TestConvergenceWarningIntegration:
    @pytest.mark.slow
    def test_warns_through_extract_results(
        self, poll_data, election_date, today, fast_config
    ):
        from kronikas.model import build_model, extract_results, run_inference

        model, meta = build_model(poll_data, election_date, today, fast_config)
        trace = run_inference(model, fast_config)

        strict = type(fast_config)(
            **{
                **vars(fast_config),
                "ess_threshold": 1e9,
                "r_hat_threshold": 1.01,
            }
        )
        with pytest.warns(ConvergenceWarning, match="may not have converged"):
            extract_results(trace, poll_data, meta, config=strict)

    @pytest.mark.slow
    def test_can_be_suppressed(self, poll_data, election_date, today, fast_config):
        from kronikas.model import build_model, extract_results, run_inference

        model, meta = build_model(poll_data, election_date, today, fast_config)
        trace = run_inference(model, fast_config)

        strict = type(fast_config)(**{**vars(fast_config), "ess_threshold": 1e9})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = extract_results(
                trace, poll_data, meta, config=strict, warn_on_convergence=False
            )
        assert not [w for w in caught if issubclass(w.category, ConvergenceWarning)]
        assert result.diagnostics is not None
        assert not result.diagnostics.converged
