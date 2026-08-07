"""Tests for the backtesting harness."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from kronikas.backtesting import _normalise_actual, backtest
from kronikas.data import load_polls


class TestNormaliseActual:
    def test_percentages_pass_through(self):
        got = _normalise_actual({"A": 50.0, "B": 30.0, "C": 20.0}, ["A", "B", "C"])
        assert got == {"A": 50.0, "B": 30.0, "C": 20.0}

    def test_raw_counts_are_rescaled(self):
        got = _normalise_actual({"A": 500, "B": 300, "C": 200}, ["A", "B", "C"])
        assert got["A"] == pytest.approx(50.0)
        assert sum(got.values()) == pytest.approx(100.0)

    def test_fractions_are_rescaled(self):
        got = _normalise_actual({"A": 0.5, "B": 0.5}, ["A", "B"])
        assert got == {"A": pytest.approx(50.0), "B": pytest.approx(50.0)}

    def test_unknown_candidate_raises(self):
        with pytest.raises(ValueError, match="unknown candidates"):
            _normalise_actual({"A": 1.0, "Z": 1.0}, ["A", "B"])

    def test_missing_candidate_raises(self):
        with pytest.raises(ValueError, match="missing candidates"):
            _normalise_actual({"A": 1.0}, ["A", "B"])

    def test_non_positive_total_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _normalise_actual({"A": 0.0, "B": 0.0}, ["A", "B"])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    def test_invalid_values_raise(self, bad):
        with pytest.raises(ValueError, match="finite and non-negative"):
            _normalise_actual({"A": bad, "B": 1.0}, ["A", "B"])


class TestBacktestValidation:
    def test_empty_dates_raise(self, polls_csv: Path, fast_config):
        with pytest.raises(ValueError, match="at least one date"):
            backtest(polls_csv, date(2024, 6, 1), [], config=fast_config)

    def test_as_of_before_first_poll_raises(self, polls_csv: Path, fast_config):
        with pytest.raises(ValueError, match="No polls on or before"):
            backtest(
                polls_csv,
                date(2024, 6, 1),
                [date(2020, 1, 1)],
                config=fast_config,
            )

    def test_as_of_after_election_is_skipped(self, polls_csv: Path, fast_config):
        with pytest.warns(UserWarning, match="on or after election_date"):
            report = backtest(
                polls_csv,
                date(2024, 6, 1),
                [date(2024, 7, 1)],
                config=fast_config,
            )
        assert report.points == []


@pytest.mark.slow
class TestBacktestRun:
    @pytest.fixture()
    def report(self, polls_csv: Path, fast_config):
        return backtest(
            polls_csv,
            election_date=date(2024, 6, 1),
            as_of_dates=[date(2024, 2, 15), date(2024, 3, 15)],
            actual={"Candidate_A": 47.0, "Candidate_B": 41.0, "Candidate_C": 12.0},
            config=fast_config,
        )

    def test_one_point_per_date_and_candidate(self, report):
        assert len(report.points) == 2 * 3
        assert {p.as_of for p in report.points} == {
            date(2024, 2, 15),
            date(2024, 3, 15),
        }

    def test_only_earlier_polls_are_used(self, report, polls_csv: Path):
        """Each refit must see exactly the polls available at that time."""
        data = load_polls(polls_csv)
        for as_of in {p.as_of for p in report.points}:
            expected = sum(1 for d in data.poll_dates if d <= as_of)
            got = {p.n_polls for p in report.points if p.as_of == as_of}
            assert got == {expected}

    def test_errors_are_scored(self, report):
        for point in report.points:
            assert point.actual is not None
            assert point.error == pytest.approx(point.mean - point.actual)
            assert point.abs_error == pytest.approx(abs(point.error))
            assert point.covered == (point.ci_lower <= point.actual <= point.ci_upper)
            assert point.crps is not None and point.crps >= 0.0

    def test_metrics_are_reported(self, report):
        stats = report.metrics()
        assert stats["n_forecasts"] == 2
        assert stats["n_points"] == 6
        assert stats["mae"] >= 0.0
        assert stats["rmse"] >= stats["mae"] - 1e-9
        assert stats["mean_crps"] >= 0.0
        assert 0.0 <= stats["interval_hit_rate_90"] <= 1.0

    def test_bias_is_per_candidate_not_pooled(self, report):
        """Pooled signed bias is identically zero for compositional data."""
        bias = report.metrics()["bias_by_candidate"]
        assert set(bias) == {"Candidate_A", "Candidate_B", "Candidate_C"}
        assert sum(bias.values()) == pytest.approx(0.0, abs=1e-9)

    def test_dataframe_is_tidy(self, report):
        frame = report.to_dataframe()
        assert len(frame) == 6
        assert {"as_of", "candidate", "mean", "actual", "abs_error"} <= set(
            frame.columns
        )

    def test_summary_describes_hit_rate_without_claiming_calibration(self, report):
        text = report.summary()
        assert "Backtest report" in text
        assert "90% hit rate" in text
        assert "many elections are needed for calibration" in text
        assert "Signed bias by candidate" in text

    def test_results_not_kept_by_default(self, report):
        assert report.results == {}


@pytest.mark.slow
class TestBacktestInputForms:
    def test_accepts_dataframe(self, polls_csv: Path, fast_config):
        frame = pd.read_csv(polls_csv)
        report = backtest(
            frame,
            election_date=date(2024, 6, 1),
            as_of_dates=[date(2024, 3, 15)],
            config=fast_config,
        )
        assert len(report.points) == 3
        assert all(p.actual is None for p in report.points)

    def test_accepts_poll_data_and_keeps_results(self, polls_csv: Path, fast_config):
        report = backtest(
            load_polls(polls_csv),
            election_date=date(2024, 6, 1),
            as_of_dates=[date(2024, 3, 15)],
            config=fast_config,
            keep_results=True,
        )
        assert set(report.results) == {date(2024, 3, 15)}
        assert report.results[date(2024, 3, 15)].time_grid[-1] == date(2024, 6, 1)
