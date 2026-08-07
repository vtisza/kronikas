"""Tests for industry-wide shared bias: the model term and the scenario tool."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from kronikas import ElectionForecast, ModelConfig, SharedBiasPrior
from kronikas.model import (
    ForecastResult,
    _calibrate_shared_bias_spread,
    _np_softmax,
    _shared_mean_to_logit,
    build_model,
)

TRUE = np.array([45.0, 45.0, 10.0])
NAMES = ["A", "B", "C"]
ELECTION = date(2024, 11, 5)
START = date(2024, 4, 1)


@pytest.fixture(scope="module")
def biased_polls() -> pd.DataFrame:
    """Polls where every pollster shades ~3 pp toward A. Truth is a dead heat."""
    bias = {"PollCo": 4.0, "SurveyInc": 3.0, "Trio": 2.0}
    rng = np.random.default_rng(7)
    rows = []
    for i in range(30):
        firm = list(bias)[i % 3]
        p = TRUE.copy()
        p[0] += bias[firm]
        p[1] -= bias[firm]
        rows.append(
            {
                "date": (START + timedelta(days=6 * i)).isoformat(),
                "pollster": firm,
                "sample_size": 1500,
                **dict(zip(NAMES, rng.multinomial(1500, p / 100.0), strict=True)),
            }
        )
    return pd.DataFrame(rows)


def _fit(df: pd.DataFrame, shared_bias=None, **overrides):
    config = ModelConfig(
        num_tune=400,
        num_draws=300,
        num_chains=2,
        cores=1,
        time_step_days=14,
        progressbar=False,
        random_seed=11,
        ess_threshold=1.0,
        r_hat_threshold=100.0,
        shared_bias=shared_bias,
        **overrides,
    )
    return ElectionForecast.from_dataframe(
        df, election_date=ELECTION, today=ELECTION - timedelta(days=1), config=config
    ).run()


class TestSharedBiasPriorValidation:
    def test_defaults_are_inert(self):
        assert SharedBiasPrior().is_inert()

    def test_a_mean_alone_is_not_inert(self):
        assert not SharedBiasPrior(mean={"A": 2.0}).is_inert()

    def test_a_spread_alone_is_not_inert(self):
        assert not SharedBiasPrior(default_sd=2.0).is_inert()
        assert not SharedBiasPrior(sd={"A": 2.0}).is_inert()

    def test_zero_values_stay_inert(self):
        assert SharedBiasPrior(mean={"A": 0.0}, sd={"A": 0.0}).is_inert()

    def test_negative_sd_rejected(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            SharedBiasPrior(sd={"A": -1.0})
        with pytest.raises(ValueError, match="default_sd must be >= 0"):
            SharedBiasPrior(default_sd=-1.0)


class TestMeanConversion:
    """The mean must be converted jointly, not per candidate."""

    def test_lands_on_the_intended_share_vector(self):
        baseline = np.array([0.476, 0.424, 0.100])
        offsets = np.array([3.0, -3.0, 0.0])
        shift = _shared_mean_to_logit(offsets, baseline)

        corrected = np.exp(np.log(baseline) - shift)
        corrected /= corrected.sum()
        expected = baseline - offsets / 100.0
        assert np.allclose(corrected, expected, atol=1e-9)

    def test_pairwise_correction_is_not_double_counted(self):
        """A 3 pp correction on both sides must move the margin by 3, not 6."""
        baseline = np.array([0.476, 0.424, 0.100])
        shift = _shared_mean_to_logit(np.array([3.0, -3.0, 0.0]), baseline)
        observed_gap = np.log(baseline[0] / baseline[1])
        corrected_gap = observed_gap - (shift[0] - shift[1])
        assert corrected_gap == pytest.approx(np.log(0.446 / 0.454), abs=1e-6)

    def test_zero_offsets_give_zero_shift(self):
        baseline = np.array([0.5, 0.3, 0.2])
        assert np.allclose(_shared_mean_to_logit(np.zeros(3), baseline), 0.0)

    def test_unnamed_candidates_absorb_the_remainder(self):
        """A stated 4 pp correction must land as 4 pp, not be diluted."""
        baseline = np.array([0.5, 0.3, 0.2])
        absorb = np.array([False, True, True])
        shift = _shared_mean_to_logit(np.array([4.0, 0.0, 0.0]), baseline, absorb)
        corrected = np.exp(np.log(baseline) - shift)
        corrected /= corrected.sum()

        assert corrected[0] == pytest.approx(0.46, abs=1e-9)
        # the 4 points go to B and C in proportion to their support
        assert corrected[1] == pytest.approx(0.324, abs=1e-9)
        assert corrected[2] == pytest.approx(0.216, abs=1e-9)
        assert corrected.sum() == pytest.approx(1.0)

    def test_impossible_offset_raises(self):
        with pytest.raises(ValueError, match="too large"):
            _shared_mean_to_logit(np.array([0.0, 0.0, 25.0]), np.array([0.5, 0.3, 0.2]))


class TestSpreadConversion:
    def test_default_sd_is_a_marginal_share_space_sd(self):
        baseline = np.array([0.45, 0.45, 0.10])
        target = np.array([2.5, 2.5, 2.5])
        centre, scales = _calibrate_shared_bias_spread(target, baseline)

        rng = np.random.default_rng(99)
        z = rng.normal(size=(200_000, 3))
        z -= z.mean(axis=1, keepdims=True)
        z *= np.sqrt(3 / 2)
        draws = _np_softmax(np.log(baseline) - centre - z * scales, axis=1)

        assert draws.mean(axis=0) * 100 == pytest.approx(baseline * 100, abs=0.08)
        assert draws.std(axis=0) * 100 == pytest.approx(target, abs=0.08)


class TestModelWiring:
    def test_absent_by_default(self, poll_data, election_date, today):
        model, meta = build_model(poll_data, election_date, today, ModelConfig())
        assert meta["shared_bias_active"] is False
        assert "shared_bias" not in model.named_vars

    def test_inert_prior_does_not_activate(self, poll_data, election_date, today):
        config = ModelConfig(shared_bias=SharedBiasPrior())
        _, meta = build_model(poll_data, election_date, today, config)
        assert meta["shared_bias_active"] is False

    def test_spread_creates_a_random_variable(self, poll_data, election_date, today):
        config = ModelConfig(shared_bias=SharedBiasPrior(default_sd=2.0))
        model, meta = build_model(poll_data, election_date, today, config)
        assert meta["shared_bias_active"] is True
        assert "shared_bias_z" in model.named_vars

    def test_pure_scenario_needs_no_random_variable(
        self, poll_data, election_date, today
    ):
        config = ModelConfig(shared_bias=SharedBiasPrior(mean={"Candidate_A": 2.0}))
        model, meta = build_model(poll_data, election_date, today, config)
        assert meta["shared_bias_active"] is True
        assert "shared_bias" in model.named_vars
        assert "shared_bias_z" not in model.named_vars

    def test_unknown_candidate_warns(self, poll_data, election_date, today):
        config = ModelConfig(shared_bias=SharedBiasPrior(mean={"Nobody": 2.0}))
        with pytest.warns(UserWarning, match="does not match any candidate"):
            _, meta = build_model(poll_data, election_date, today, config)
        assert meta["shared_bias_active"] is False


@pytest.mark.slow
class TestSharedBiasChangesTheForecast:
    def test_baseline_misses_a_shared_bias(self, biased_polls):
        """Establishes the problem the feature exists to address."""
        result = _fit(biased_polls)
        a = {e.name: e for e in result.election_day_estimates}["A"]
        assert a.mean > 46.0  # truth is 45.0
        assert not (a.ci_lower <= 45.0 <= a.ci_upper)

    def test_symmetric_spread_widens_without_shifting(self, biased_polls):
        base = _fit(biased_polls)
        wide = _fit(biased_polls, SharedBiasPrior(default_sd=2.5))
        a0 = {e.name: e for e in base.election_day_estimates}["A"]
        a1 = {e.name: e for e in wide.election_day_estimates}["A"]

        assert a1.mean == pytest.approx(a0.mean, abs=1.0)  # centre roughly held
        assert (a1.ci_upper - a1.ci_lower) > 2 * (a0.ci_upper - a0.ci_lower)
        assert a1.ci_lower <= 45.0 <= a1.ci_upper  # now covers the truth

    def test_directional_centre_moves_the_estimate_toward_truth(self, biased_polls):
        corrected = _fit(
            biased_polls, SharedBiasPrior(mean={"A": 3.0, "B": -3.0}, sd={"A": 1.0})
        )
        a = {e.name: e for e in corrected.election_day_estimates}["A"]
        assert a.mean == pytest.approx(45.0, abs=1.5)
        assert a.ci_lower <= 45.0 <= a.ci_upper

    def test_centre_is_not_forced_to_be_symmetric(self, biased_polls):
        """A one-sided belief is expressible and moves only that direction."""
        up = _fit(biased_polls, SharedBiasPrior(mean={"A": 2.0}))
        down = _fit(biased_polls, SharedBiasPrior(mean={"A": -2.0}))
        a_up = {e.name: e for e in up.election_day_estimates}["A"]
        a_down = {e.name: e for e in down.election_day_estimates}["A"]
        assert a_up.mean < a_down.mean


@pytest.mark.slow
class TestAssumeSharedBias:
    @pytest.fixture(scope="class")
    def base(self, biased_polls):
        return _fit(biased_polls)

    def test_shift_moves_the_estimate_one_for_one(self, base):
        a0 = {e.name: e for e in base.election_day_estimates}["A"].mean
        shifted = base.assume_shared_bias({"A": 3.0, "B": -3.0})
        a1 = {e.name: e for e in shifted.election_day_estimates}["A"].mean
        assert a1 == pytest.approx(a0 - 3.0, abs=0.15)

    def test_original_result_is_untouched(self, base):
        before = {e.name: e for e in base.election_day_estimates}["A"].mean
        base.assume_shared_bias({"A": 5.0})
        after = {e.name: e for e in base.election_day_estimates}["A"].mean
        assert after == before

    def test_shares_still_sum_to_100(self, base):
        shifted = base.assume_shared_bias({"A": 3.0, "B": -3.0})
        assert np.allclose(shifted.election_samples.sum(axis=1), 100.0)
        assert np.allclose(shifted.today_samples.sum(axis=1), 100.0)

    def test_probabilities_move_monotonically(self, base):
        probs = [
            base.assume_shared_bias({"A": pp, "B": -pp}).win_probabilities["A"]
            for pp in (0.0, 1.0, 2.0, 3.0, 4.0)
        ]
        assert probs == sorted(probs, reverse=True)

    def test_one_sided_offset_lands_at_its_stated_size(self, base):
        """{"A": 4} must move A by 4 pp, with the rest taken from the others."""
        a0 = {e.name: e for e in base.election_day_estimates}["A"].mean
        shifted = base.assume_shared_bias({"A": 4.0})
        a1 = {e.name: e for e in shifted.election_day_estimates}["A"].mean
        assert a1 == pytest.approx(a0 - 4.0, abs=0.2)

    def test_zero_offset_is_a_no_op(self, base):
        same = base.assume_shared_bias({})
        for before, after in zip(
            base.election_day_estimates, same.election_day_estimates, strict=True
        ):
            assert after.mean == pytest.approx(before.mean)

    def test_metadata_is_carried_over(self, base):
        shifted = base.assume_shared_bias({"A": 1.0})
        assert shifted.candidates == base.candidates
        assert shifted.election_date == base.election_date
        assert shifted.time_grid == base.time_grid

    def test_unknown_candidate_raises(self, base):
        with pytest.raises(KeyError, match="Unknown candidate"):
            base.assume_shared_bias({"Nobody": 1.0})


@pytest.mark.slow
class TestBreakeven:
    def test_reports_where_the_lead_dies(self, biased_polls):
        base = _fit(biased_polls)
        breakeven = base.shared_bias_breakeven()
        assert breakeven is not None
        assert 0.0 < breakeven < 10.0

        just_under = base.assume_shared_bias(
            {"A": breakeven - 0.5, "B": -(breakeven - 0.5)}
        )
        just_over = base.assume_shared_bias(
            {"A": breakeven + 0.5, "B": -(breakeven + 0.5)}
        )
        assert just_under.win_probabilities["A"] > 0.5
        assert just_over.win_probabilities["A"] < 0.5

    def test_returns_none_for_an_unassailable_lead(self, biased_polls):
        base = _fit(biased_polls)
        assert base.shared_bias_breakeven(max_pp=0.01) is None

    def test_counts_every_opponent_when_deciding_if_leader_survives(self):
        import arviz as az

        result = ForecastResult(
            today_estimates=[],
            election_day_estimates=[],
            win_probabilities={},
            trace=az.InferenceData(),
            candidates=["Leader", "Runner", "Third"],
            today_samples=np.array([[39.0, 16.0, 45.0], [60.0, 30.0, 10.0]]),
            election_samples=np.array(
                [[39.0, 16.0, 45.0]] * 6 + [[60.0, 30.0, 10.0]] * 4
            ),
        )
        # Leader beats Runner in every draw, but has the plurality in only 40%.
        assert result.shared_bias_breakeven() == 0.0
