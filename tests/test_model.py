"""Tests for kronikas.model (Bayesian model construction and inference)."""

from datetime import date

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from kronikas.config import ModelConfig, PollsterPrior
from kronikas.data import PollData
from kronikas.model import (
    CandidateEstimate,
    ForecastResult,
    build_model,
    extract_results,
    run_inference,
)


class TestBuildModel:
    def test_returns_model_and_metadata(self, poll_data, election_date, today, fast_config):
        model, meta = build_model(poll_data, election_date, today, fast_config)
        assert isinstance(model, pm.Model)
        assert "today_idx" in meta
        assert "election_idx" in meta
        assert "n_timesteps" in meta

    def test_model_has_expected_variables(self, poll_data, election_date, today, fast_config):
        model, _ = build_model(poll_data, election_date, today, fast_config)
        names = {v.name for v in model.free_RVs}
        assert "sigma_walk" in names
        assert "kappa_log" in names
        assert "eta_init" in names
        assert "innovations" in names

    def test_house_effects_with_multiple_pollsters(
        self, poll_data, election_date, today, fast_config
    ):
        model, meta = build_model(poll_data, election_date, today, fast_config)
        assert meta["include_house_effects"] is True
        names = {v.name for v in model.free_RVs}
        assert "sigma_house" in names
        assert "delta" in names

    def test_no_house_effects_single_pollster(
        self, single_pollster_csv, election_date, today, fast_config
    ):
        from kronikas.data import load_polls

        data = load_polls(single_pollster_csv)
        model, meta = build_model(data, election_date, today, fast_config)
        assert meta["include_house_effects"] is False
        names = {v.name for v in model.free_RVs}
        assert "delta" not in names

    def test_election_before_polls_raises(self, poll_data, fast_config):
        with pytest.raises(ValueError, match="after the first poll"):
            build_model(
                poll_data,
                election_date=date(2023, 1, 1),
                today=date(2024, 3, 1),
                config=fast_config,
            )

    def test_timesteps_reasonable(self, poll_data, election_date, today, fast_config):
        _, meta = build_model(poll_data, election_date, today, fast_config)
        # Election ~138 days from first poll, step=30 → ceil(138/30)+1 = 6
        assert 2 <= meta["n_timesteps"] <= 20
        assert 0 <= meta["today_idx"] < meta["n_timesteps"]
        assert meta["election_idx"] == meta["n_timesteps"] - 1


class TestPollsterPriors:
    """Per-pollster prior overrides in model construction."""

    def test_custom_house_effect_for_one_pollster(
        self, poll_data, election_date, today
    ):
        """When one pollster has a custom sigma_house, hierarchical
        sigma_house is still created for the other pollster."""
        config = ModelConfig(
            num_tune=50, num_draws=50, num_chains=1, cores=1,
            time_step_days=30, progressbar=False,
            pollster_priors={"PollCo": PollsterPrior(sigma_house=0.1)},
        )
        model, meta = build_model(poll_data, election_date, today, config)
        names = {v.name for v in model.free_RVs}
        assert "delta" in names
        # sigma_house still needed for the non-overridden pollster
        assert "sigma_house" in names
        assert meta["include_house_effects"] is True

    def test_custom_house_effect_for_all_pollsters(
        self, poll_data, election_date, today
    ):
        """When every pollster has a custom sigma_house, the hierarchical
        sigma_house is omitted."""
        config = ModelConfig(
            num_tune=50, num_draws=50, num_chains=1, cores=1,
            time_step_days=30, progressbar=False,
            pollster_priors={
                "PollCo": PollsterPrior(sigma_house=0.1),
                "SurveyInc": PollsterPrior(sigma_house=0.2),
            },
        )
        model, meta = build_model(poll_data, election_date, today, config)
        names = {v.name for v in model.free_RVs}
        assert "delta" in names
        # No hierarchical sigma_house needed
        assert "sigma_house" not in names

    def test_custom_kappa_creates_per_pollster_kappa(
        self, poll_data, election_date, today
    ):
        """Per-pollster kappa_log_sigma produces a vector kappa_log."""
        config = ModelConfig(
            num_tune=50, num_draws=50, num_chains=1, cores=1,
            time_step_days=30, progressbar=False,
            pollster_priors={"PollCo": PollsterPrior(kappa_log_sigma=0.1)},
        )
        model, _ = build_model(poll_data, election_date, today, config)
        # kappa_log should be a vector (one per pollster)
        for rv in model.free_RVs:
            if rv.name == "kappa_log":
                assert rv.eval().shape == (len(poll_data.pollsters),)
                break
        else:
            pytest.fail("kappa_log not found in model")

    def test_no_custom_kappa_keeps_scalar(
        self, poll_data, election_date, today, fast_config
    ):
        """Without per-pollster kappa overrides, kappa_log is scalar."""
        model, _ = build_model(poll_data, election_date, today, fast_config)
        for rv in model.free_RVs:
            if rv.name == "kappa_log":
                assert rv.eval().shape == ()
                break

    def test_unknown_pollster_warns(
        self, poll_data, election_date, today
    ):
        config = ModelConfig(
            num_tune=50, num_draws=50, num_chains=1, cores=1,
            time_step_days=30, progressbar=False,
            pollster_priors={"NonExistent": PollsterPrior(sigma_house=0.1)},
        )
        with pytest.warns(UserWarning, match="NonExistent"):
            build_model(poll_data, election_date, today, config)

    def test_combined_house_and_kappa_overrides(
        self, poll_data, election_date, today
    ):
        """Both sigma_house and kappa_log_sigma can be set together."""
        config = ModelConfig(
            num_tune=50, num_draws=50, num_chains=1, cores=1,
            time_step_days=30, progressbar=False,
            pollster_priors={
                "PollCo": PollsterPrior(sigma_house=0.1, kappa_log_sigma=0.2),
            },
        )
        model, _ = build_model(poll_data, election_date, today, config)
        names = {v.name for v in model.free_RVs}
        assert "delta" in names
        assert "kappa_log" in names
        assert "sigma_house" in names  # still needed for SurveyInc


class TestInferenceEndToEnd:
    """Run actual (minimal) MCMC to verify the pipeline doesn't crash."""

    @pytest.mark.slow
    def test_sample_and_extract(self, poll_data, election_date, today, fast_config):
        model, meta = build_model(poll_data, election_date, today, fast_config)
        trace = run_inference(model, fast_config)

        result = extract_results(trace, poll_data, meta)
        assert isinstance(result, ForecastResult)

        # One estimate per candidate
        assert len(result.today_estimates) == len(poll_data.candidates)
        assert len(result.election_day_estimates) == len(poll_data.candidates)

        # Each estimate has plausible values
        for est in result.today_estimates:
            assert isinstance(est, CandidateEstimate)
            assert 0 <= est.ci_lower <= est.mean <= est.ci_upper <= 100

        # Win probabilities sum to ~1
        total_prob = sum(result.win_probabilities.values())
        assert abs(total_prob - 1.0) < 1e-6

    @pytest.mark.slow
    def test_summary_string(self, poll_data, election_date, today, fast_config):
        model, meta = build_model(poll_data, election_date, today, fast_config)
        trace = run_inference(model, fast_config)
        result = extract_results(trace, poll_data, meta)

        text = result.summary()
        assert "Election Forecast Summary" in text
        assert "Win probabilities" in text
        for c in poll_data.candidates:
            assert c in text


class TestPartyForecastDataframe:
    """Tests for ForecastResult.party_forecast_dataframe()."""

    @pytest.mark.slow
    def test_today_dataframe_shape_and_columns(
        self, poll_data, election_date, today, fast_config
    ):
        model, meta = build_model(poll_data, election_date, today, fast_config)
        trace = run_inference(model, fast_config)
        result = extract_results(trace, poll_data, meta)

        df = result.party_forecast_dataframe(day="today")
        expected_draws = fast_config.num_draws * fast_config.num_chains
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (expected_draws, len(poll_data.candidates))
        assert list(df.columns) == poll_data.candidates

    @pytest.mark.slow
    def test_election_day_dataframe_shape_and_columns(
        self, poll_data, election_date, today, fast_config
    ):
        model, meta = build_model(poll_data, election_date, today, fast_config)
        trace = run_inference(model, fast_config)
        result = extract_results(trace, poll_data, meta)

        df = result.party_forecast_dataframe(day="election_day")
        expected_draws = fast_config.num_draws * fast_config.num_chains
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (expected_draws, len(poll_data.candidates))
        assert list(df.columns) == poll_data.candidates

    @pytest.mark.slow
    def test_values_are_percentage_points(
        self, poll_data, election_date, today, fast_config
    ):
        model, meta = build_model(poll_data, election_date, today, fast_config)
        trace = run_inference(model, fast_config)
        result = extract_results(trace, poll_data, meta)

        for day in ("today", "election_day"):
            df = result.party_forecast_dataframe(day=day)
            assert (df.values >= 0).all(), f"{day}: negative values found"
            assert (df.values <= 100).all(), f"{day}: values exceed 100%"
            row_sums = df.sum(axis=1)
            np.testing.assert_allclose(
                row_sums, 100.0, atol=1e-6,
                err_msg=f"{day}: rows do not sum to 100%",
            )

    @pytest.mark.slow
    def test_default_day_is_today(
        self, poll_data, election_date, today, fast_config
    ):
        model, meta = build_model(poll_data, election_date, today, fast_config)
        trace = run_inference(model, fast_config)
        result = extract_results(trace, poll_data, meta)

        df_default = result.party_forecast_dataframe()
        df_today = result.party_forecast_dataframe(day="today")
        pd.testing.assert_frame_equal(df_default, df_today)

    def test_invalid_day_raises(self, poll_data, election_date, today, fast_config):
        """Invalid day argument raises ValueError without running MCMC."""
        import numpy as np

        # Construct a minimal ForecastResult with sample arrays
        dummy_samples = np.zeros((10, len(poll_data.candidates)))
        result = ForecastResult(
            today_estimates=[],
            election_day_estimates=[],
            win_probabilities={},
            trace=None,
            candidates=poll_data.candidates,
            today_samples=dummy_samples,
            election_samples=dummy_samples,
        )
        with pytest.raises(ValueError, match="day must be"):
            result.party_forecast_dataframe(day="invalid")


class TestHouseEffectsDataframe:
    """Tests for ForecastResult.house_effects_dataframe()."""

    @pytest.mark.slow
    def test_shape_and_columns(self, poll_data, election_date, today, fast_config):
        model, meta = build_model(poll_data, election_date, today, fast_config)
        trace = run_inference(model, fast_config)
        result = extract_results(trace, poll_data, meta)

        df = result.house_effects_dataframe()
        expected_draws = fast_config.num_draws * fast_config.num_chains
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (
            expected_draws,
            len(poll_data.pollsters) * len(poll_data.candidates),
        )
        assert df.columns.names == ["pollster", "candidate"]
        assert list(df.columns.get_level_values("pollster").unique()) == poll_data.pollsters
        assert list(df.columns.get_level_values("candidate").unique()) == poll_data.candidates

    @pytest.mark.slow
    def test_values_sum_to_zero_per_pollster(
        self, poll_data, election_date, today, fast_config
    ):
        """Within each draw and pollster, deviations across candidates sum to zero."""
        model, meta = build_model(poll_data, election_date, today, fast_config)
        trace = run_inference(model, fast_config)
        result = extract_results(trace, poll_data, meta)

        df = result.house_effects_dataframe()
        for pollster in poll_data.pollsters:
            row_sums = df[pollster].sum(axis=1)
            np.testing.assert_allclose(
                row_sums, 0.0, atol=1e-6,
                err_msg=f"House effects for {pollster!r} do not sum to zero",
            )

    @pytest.mark.slow
    def test_values_within_bounds(self, poll_data, election_date, today, fast_config):
        """House effects in percentage points are bounded by ±100."""
        model, meta = build_model(poll_data, election_date, today, fast_config)
        trace = run_inference(model, fast_config)
        result = extract_results(trace, poll_data, meta)

        df = result.house_effects_dataframe()
        assert (df.values > -100).all(), "House effect below -100pp"
        assert (df.values < 100).all(), "House effect above 100pp"

    def test_single_pollster_raises(
        self, single_pollster_csv, election_date, today, fast_config
    ):
        """RuntimeError is raised when no house effects were estimated."""
        from kronikas.data import load_polls

        data = load_polls(single_pollster_csv)
        # Build a minimal ForecastResult as if it came from a single-pollster run
        result = ForecastResult(
            today_estimates=[],
            election_day_estimates=[],
            win_probabilities={},
            trace=None,
            candidates=data.candidates,
            pollsters=list(data.pollsters),
            today_samples=None,
            election_samples=None,
            house_effect_samples=None,
        )
        with pytest.raises(RuntimeError, match="single pollster"):
            result.house_effects_dataframe()
