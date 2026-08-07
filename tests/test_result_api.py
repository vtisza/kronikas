"""Tests for ForecastResult's query, export, and serialisation surface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kronikas.model import ForecastResult, build_model, extract_results, run_inference


@pytest.fixture()
def result(poll_data, election_date, today, fast_config):
    """A real (if tiny) fitted result."""
    model, meta = build_model(poll_data, election_date, today, fast_config)
    trace = run_inference(model, fast_config)
    return extract_results(trace, poll_data, meta, config=fast_config)


@pytest.mark.slow
class TestThresholdProbabilities:
    def test_probabilities_are_valid(self, result):
        probs = result.threshold_probabilities(5.0)
        assert set(probs) == set(result.candidates)
        assert all(0.0 <= p <= 1.0 for p in probs.values())

    def test_zero_threshold_is_certain(self, result):
        probs = result.threshold_probabilities(0.0)
        assert all(p == pytest.approx(1.0) for p in probs.values())

    def test_impossible_threshold_is_never_met(self, result):
        probs = result.threshold_probabilities(100.0)
        assert all(p == pytest.approx(0.0) for p in probs.values())

    def test_monotonic_in_threshold(self, result):
        low = result.threshold_probabilities(10.0)
        high = result.threshold_probabilities(40.0)
        assert all(high[name] <= low[name] for name in result.candidates)

    def test_matches_manual_computation(self, result):
        samples = result.party_forecast_dataframe(day="election_day")
        expected = float((samples[result.candidates[0]] >= 30.0).mean())
        got = result.threshold_probabilities(30.0)[result.candidates[0]]
        assert got == pytest.approx(expected)

    def test_today_and_election_day_differ_in_source(self, result):
        today = result.threshold_probabilities(30.0, day="today")
        election = result.threshold_probabilities(30.0, day="election_day")
        assert set(today) == set(election)

    def test_exclusive_bound_is_supported(self, result):
        inclusive = result.threshold_probabilities(0.0, inclusive=True)
        exclusive = result.threshold_probabilities(0.0, inclusive=False)
        assert all(exclusive[name] <= inclusive[name] for name in result.candidates)

    def test_invalid_day_raises(self, result):
        with pytest.raises(ValueError, match="day must be"):
            result.threshold_probabilities(5.0, day="tomorrow")


@pytest.mark.slow
class TestLeadProbability:
    def test_complementary_pair(self, result):
        a, b = result.candidates[0], result.candidates[1]
        assert result.lead_probability(a, b) + result.lead_probability(b, a) == (
            pytest.approx(1.0)
        )

    def test_self_comparison_is_zero(self, result):
        a = result.candidates[0]
        assert result.lead_probability(a, a) == pytest.approx(0.0)

    def test_unknown_candidate_raises(self, result):
        with pytest.raises(KeyError, match="Unknown candidate"):
            result.lead_probability("Nobody", result.candidates[0])


@pytest.mark.slow
class TestToDict:
    def test_is_json_serialisable(self, result):
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["candidates"] == result.candidates
        assert set(payload["win_probabilities"]) == set(result.candidates)

    def test_includes_dates(self, result, election_date, today):
        payload = result.to_dict()
        assert payload["election_date"] == election_date.isoformat()
        assert payload["today"] == today.isoformat()

    def test_thresholds_are_optional_and_keyed(self, result):
        assert "threshold_probabilities" not in result.to_dict()
        payload = result.to_dict(thresholds=[5.0, 12.5])
        assert set(payload["threshold_probabilities"]) == {"5", "12.5"}

    def test_carries_diagnostics(self, result):
        assert result.to_dict()["diagnostics"]["n_chains"] == 1


@pytest.mark.slow
class TestSaveLoad:
    def test_roundtrip_preserves_estimates(self, result, tmp_path: Path):
        path = result.save(tmp_path / "forecast.nc")
        assert path.is_file()

        restored = ForecastResult.load(path)
        assert restored.candidates == result.candidates
        assert restored.pollsters == result.pollsters
        assert restored.time_grid == result.time_grid
        assert restored.election_date == result.election_date
        assert restored.today == result.today

        for before, after in zip(
            result.election_day_estimates, restored.election_day_estimates, strict=True
        ):
            assert after.name == before.name
            assert after.mean == pytest.approx(before.mean)
            assert after.ci_lower == pytest.approx(before.ci_lower)
            assert after.ci_upper == pytest.approx(before.ci_upper)

    def test_roundtrip_preserves_samples(self, result, tmp_path: Path):
        restored = ForecastResult.load(result.save(tmp_path / "f.nc"))
        assert np.allclose(restored.today_samples, result.today_samples)
        assert np.allclose(restored.election_samples, result.election_samples)
        assert np.allclose(restored.house_effect_samples, result.house_effect_samples)

    def test_roundtrip_preserves_win_probabilities(self, result, tmp_path: Path):
        restored = ForecastResult.load(result.save(tmp_path / "f.nc"))
        for name, prob in result.win_probabilities.items():
            assert restored.win_probabilities[name] == pytest.approx(prob)

    def test_roundtrip_preserves_shared_bias_scenario(self, result, tmp_path: Path):
        shifted = result.assume_shared_bias(
            {result.candidates[0]: 3.0, result.candidates[1]: -3.0}
        )
        restored = ForecastResult.load(shifted.save(tmp_path / "scenario.nc"))

        assert np.allclose(restored.today_samples, shifted.today_samples)
        assert np.allclose(restored.election_samples, shifted.election_samples)
        assert restored.win_probabilities == shifted.win_probabilities

    def test_last_grid_node_is_still_election_day(self, result, tmp_path: Path):
        restored = ForecastResult.load(result.save(tmp_path / "f.nc"))
        assert restored.time_grid[-1] == restored.election_date

    def test_load_rejects_foreign_netcdf(self, result, tmp_path: Path):
        import arviz as az

        path = tmp_path / "plain.nc"
        az.to_netcdf(result.trace, str(path))
        # Written without save(), so the metadata attribute is absent.
        result.trace.attrs.pop("kronikas_metadata", None)
        az.to_netcdf(result.trace, str(path))
        with pytest.raises(ValueError, match="does not carry kronikas metadata"):
            ForecastResult.load(path)


@pytest.mark.slow
class TestLatentTrendDataframe:
    def test_one_row_per_grid_node(self, result):
        frame = result.latent_trend_dataframe()
        assert len(frame) == len(result.time_grid)

    def test_indexed_by_calendar_date(self, result):
        frame = result.latent_trend_dataframe()
        assert list(frame.index) == result.time_grid
        assert frame.index[-1] == result.election_date

    def test_three_columns_per_candidate(self, result):
        frame = result.latent_trend_dataframe()
        for name in result.candidates:
            assert {f"{name}_mean", f"{name}_p_5", f"{name}_p_95"} <= set(frame.columns)
        assert len(frame.columns) == 3 * len(result.candidates)

    def test_means_sum_to_100_at_every_step(self, result):
        frame = result.latent_trend_dataframe()
        means = frame[[f"{n}_mean" for n in result.candidates]].sum(axis=1)
        assert np.allclose(means, 100.0)

    def test_percentiles_bracket_the_mean(self, result):
        frame = result.latent_trend_dataframe()
        for name in result.candidates:
            assert (frame[f"{name}_p_5"] <= frame[f"{name}_p_95"]).all()
            assert (frame[f"{name}_p_5"] <= frame[f"{name}_mean"] + 1e-9).all()
            assert (frame[f"{name}_mean"] <= frame[f"{name}_p_95"] + 1e-9).all()

    def test_values_are_percentage_points(self, result):
        frame = result.latent_trend_dataframe()
        assert frame.to_numpy().min() >= 0.0
        assert frame.to_numpy().max() <= 100.0

    def test_agrees_with_the_election_day_estimate(self, result):
        """The final row must match election_day_estimates."""
        frame = result.latent_trend_dataframe()
        for estimate in result.election_day_estimates:
            assert frame[f"{estimate.name}_mean"].iloc[-1] == pytest.approx(
                estimate.mean, abs=1e-9
            )


@pytest.mark.slow
class TestSummaryText:
    def test_names_plurality_not_victory(self, result):
        """The statistic is a vote-share plurality, and should say so."""
        text = result.summary()
        assert "Plurality probabilities" in text
        assert "Election Forecast Summary" in text
