"""Tests for prior-mean conversion and log-likelihood plumbing."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from kronikas.config import ModelConfig, PollsterPrior
from kronikas.model import _pp_to_logit, build_model


class TestPpToLogit:
    def test_default_baseline_is_fifty_percent(self):
        """Unchanged from earlier versions when no baseline is supplied."""
        assert _pp_to_logit(3.0) == pytest.approx(np.log(0.53 / 0.47))
        assert _pp_to_logit(3.0) == pytest.approx(_pp_to_logit(3.0, 0.5))

    def test_zero_bias_is_zero_shift_at_any_baseline(self):
        for baseline in (0.01, 0.05, 0.3, 0.5, 0.9):
            assert _pp_to_logit(0.0, baseline) == pytest.approx(0.0)

    def test_small_parties_get_a_much_larger_shift(self):
        """A fixed 50 % baseline understates small-party bias several-fold."""
        at_fifty = _pp_to_logit(3.0, 0.50)
        at_five = _pp_to_logit(3.0, 0.05)
        assert at_five > at_fifty
        assert at_five / at_fifty == pytest.approx(4.18, abs=0.05)

    def test_shift_is_exact_for_its_baseline(self):
        """logit(baseline) + shift must land exactly on the intended share."""
        for baseline, pp in [(0.05, 3.0), (0.42, -6.0), (0.7, 5.0)]:
            shift = _pp_to_logit(pp, baseline)
            start = np.log(baseline / (1 - baseline))
            recovered = 1.0 / (1.0 + np.exp(-(start + shift)))
            assert recovered == pytest.approx(baseline + pp / 100.0)

    def test_sign_is_preserved(self):
        assert _pp_to_logit(-4.0, 0.3) < 0
        assert _pp_to_logit(4.0, 0.3) > 0

    @pytest.mark.parametrize("pp", [50.0, -50.0, 75.0])
    def test_impossible_bias_at_default_baseline_raises(self, pp: float):
        with pytest.raises(ValueError, match="impossible"):
            _pp_to_logit(pp)

    def test_impossible_bias_for_a_small_party_raises(self):
        with pytest.raises(ValueError, match="impossible"):
            _pp_to_logit(-6.0, 0.05)

    @pytest.mark.parametrize("baseline", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_baseline_raises(self, baseline: float):
        with pytest.raises(ValueError, match="baseline must lie"):
            _pp_to_logit(1.0, baseline)


class TestMuHouseUsesCandidateBaseline:
    def test_small_party_bias_exceeds_fixed_baseline_equivalent(
        self, poll_data, election_date, today
    ):
        """Candidate_C sits near 11 %, so its shift must beat the 50 % value."""
        config = ModelConfig(
            pollster_priors={"PollCo": PollsterPrior(mu_house={"Candidate_C": 3.0})}
        )
        build_model(poll_data, election_date, today, config)

        baseline_share = poll_data.poll_values[0, 2] / 100.0
        expected = _pp_to_logit(3.0, baseline_share)
        assert expected > _pp_to_logit(3.0, 0.5)

    def test_unknown_candidate_still_warns(self, poll_data, election_date, today):
        config = ModelConfig(
            pollster_priors={"PollCo": PollsterPrior(mu_house={"Nobody": 3.0})}
        )
        with pytest.warns(UserWarning, match="does not match any"):
            build_model(poll_data, election_date, today, config)

    def test_impossible_bias_surfaces_as_error(self, poll_data, election_date, today):
        """Candidate_C is near 11 %, so -30 pp cannot be expressed."""
        config = ModelConfig(
            pollster_priors={"PollCo": PollsterPrior(mu_house={"Candidate_C": -30.0})}
        )
        with pytest.raises(ValueError, match="impossible"):
            build_model(poll_data, election_date, today, config)


class TestLogLikelihoodOption:
    def test_defaults_to_off(self):
        assert ModelConfig().compute_log_likelihood is False

    @pytest.mark.slow
    def test_populates_log_likelihood_group(
        self, poll_data, election_date, today, fast_config
    ):
        from kronikas.model import run_inference

        config = ModelConfig(**{**vars(fast_config), "compute_log_likelihood": True})
        model, _ = build_model(poll_data, election_date, today, config)
        trace = run_inference(model, config)
        assert hasattr(trace, "log_likelihood")
        assert "obs" in trace.log_likelihood

    @pytest.mark.slow
    def test_absent_by_default(self, poll_data, election_date, today, fast_config):
        from kronikas.model import run_inference

        model, _ = build_model(poll_data, election_date, today, fast_config)
        trace = run_inference(model, fast_config)
        assert not hasattr(trace, "log_likelihood")

    def test_does_not_clobber_user_idata_kwargs(self):
        config = ModelConfig(
            compute_log_likelihood=True,
            sampler_kwargs={"idata_kwargs": {"dims": {"x": ["y"]}}},
        )
        # Mirrors the merge performed inside run_inference.
        extra = dict(config.sampler_kwargs)
        idata_kwargs = dict(extra.get("idata_kwargs") or {})
        idata_kwargs.setdefault("log_likelihood", True)
        assert idata_kwargs == {"dims": {"x": ["y"]}, "log_likelihood": True}


class TestGridAnchoringWithPriors:
    def test_election_index_lands_on_election_day(
        self, poll_data, election_date, today
    ):
        config = ModelConfig(time_step_days=11)
        _, meta = build_model(poll_data, election_date, today, config)
        from datetime import timedelta

        node = meta["grid_start_date"] + timedelta(
            days=meta["election_idx"] * meta["time_step_days"]
        )
        assert node == election_date
        assert isinstance(meta["grid_start_date"], date)
