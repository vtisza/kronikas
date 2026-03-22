"""Integration tests for kronikas.forecast.ElectionForecast."""

from datetime import date
from pathlib import Path

import pytest

from kronikas.config import ModelConfig
from kronikas.forecast import ElectionForecast, _parse_date


class TestParseDate:
    def test_from_string(self):
        assert _parse_date("2024-11-05", "x") == date(2024, 11, 5)

    def test_from_date(self):
        d = date(2024, 11, 5)
        assert _parse_date(d, "x") is d

    def test_bad_string(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_date("nope", "x")

    def test_bad_type(self):
        with pytest.raises(TypeError, match="must be a str"):
            _parse_date(12345, "x")  # type: ignore[arg-type]


class TestElectionForecastInit:
    def test_loads_data(self, polls_csv: Path):
        ef = ElectionForecast(
            polls_csv=polls_csv,
            election_date="2024-06-01",
            today="2024-03-20",
        )
        assert len(ef.poll_data.candidates) == 3
        assert ef.election_date == date(2024, 6, 1)
        assert ef.today == date(2024, 3, 20)

    def test_custom_config(self, polls_csv: Path):
        cfg = ModelConfig(num_draws=500)
        ef = ElectionForecast(
            polls_csv=polls_csv,
            election_date="2024-06-01",
            config=cfg,
        )
        assert ef.config.num_draws == 500

    def test_bad_csv_path(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ElectionForecast(
                polls_csv=tmp_path / "nope.csv",
                election_date="2024-06-01",
            )


class TestElectionForecastRun:
    @pytest.mark.slow
    def test_end_to_end(self, polls_csv: Path):
        ef = ElectionForecast(
            polls_csv=polls_csv,
            election_date="2024-06-01",
            today="2024-03-20",
            config=ModelConfig(
                num_tune=50,
                num_draws=50,
                num_chains=1,
                cores=1,
                time_step_days=30,
                random_seed=99,
                progressbar=False,
            ),
        )
        result = ef.run()
        assert len(result.today_estimates) == 3
        assert len(result.election_day_estimates) == 3
        assert abs(sum(result.win_probabilities.values()) - 1.0) < 1e-6
