"""Tests for the election-anchored time grid and its boundary warnings."""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from kronikas.config import ModelConfig
from kronikas.data import polls_from_dataframe
from kronikas.model import _build_time_grid, build_model


class TestTimeGridAnchoring:
    @pytest.mark.parametrize("step", [1, 2, 3, 7, 10, 14, 30])
    @pytest.mark.parametrize(
        "first_poll,election",
        [
            (date(2024, 1, 15), date(2024, 11, 5)),
            (date(2024, 1, 1), date(2024, 1, 8)),
            (date(2023, 6, 30), date(2026, 4, 12)),
        ],
    )
    def test_last_node_is_exactly_election_day(
        self, step: int, first_poll: date, election: date
    ):
        """The whole point of anchoring backwards: no overshoot, ever."""
        grid = _build_time_grid(first_poll, election, step)
        assert grid.end_date == election
        assert grid.dates()[-1] == election

    @pytest.mark.parametrize("step", [1, 3, 7, 14, 30])
    def test_grid_covers_the_first_poll(self, step: int):
        grid = _build_time_grid(date(2024, 1, 15), date(2024, 11, 5), step)
        assert grid.start_date <= date(2024, 1, 15)
        # ...but never wastes more than one step of runway.
        assert (date(2024, 1, 15) - grid.start_date).days < step

    def test_nodes_are_evenly_spaced(self):
        grid = _build_time_grid(date(2024, 1, 15), date(2024, 11, 5), 7)
        dates = grid.dates()
        assert len(dates) == grid.n_timesteps
        gaps = {(b - a).days for a, b in zip(dates, dates[1:], strict=False)}
        assert gaps == {7}

    def test_index_of_is_clipped(self):
        grid = _build_time_grid(date(2024, 1, 15), date(2024, 3, 15), 7)
        assert grid.index_of(grid.start_date) == 0
        assert grid.index_of(grid.end_date) == grid.n_timesteps - 1
        assert grid.index_of(date(2020, 1, 1)) == 0
        assert grid.index_of(date(2030, 1, 1)) == grid.n_timesteps - 1

    def test_index_of_rounds_to_nearest(self):
        grid = _build_time_grid(date(2024, 1, 15), date(2024, 3, 15), 10)
        assert grid.index_of(grid.start_date + timedelta(days=4)) == 0
        assert grid.index_of(grid.start_date + timedelta(days=6)) == 1

    def test_rejects_non_positive_step(self):
        with pytest.raises(ValueError, match="positive integer"):
            _build_time_grid(date(2024, 1, 1), date(2024, 6, 1), 0)

    def test_rejects_election_before_first_poll(self):
        with pytest.raises(ValueError, match="must be after"):
            _build_time_grid(date(2024, 6, 1), date(2024, 1, 1), 7)


def _frame(dates: list[str], pollsters: list[str] | None = None) -> pd.DataFrame:
    """Minimal two-candidate, two-pollster frame with the given poll dates."""
    n = len(dates)
    cycle = pollsters or [["PollCo", "SurveyInc"][i % 2] for i in range(n)]
    return pd.DataFrame(
        {
            "date": dates,
            "pollster": cycle,
            "sample_size": [1000] * n,
            "Alpha": [52.0, 51.0, 53.0, 50.0, 52.5][:n],
            "Beta": [48.0, 49.0, 47.0, 50.0, 47.5][:n],
        }
    )


class TestBuildModelGridMetadata:
    def test_metadata_grid_ends_on_election_day(self):
        data = polls_from_dataframe(_frame(["2024-01-15", "2024-02-15"]))
        config = ModelConfig(time_step_days=7)
        _, meta = build_model(data, date(2024, 11, 5), date(2024, 3, 1), config)

        grid_end = meta["grid_start_date"] + timedelta(
            days=meta["election_idx"] * meta["time_step_days"]
        )
        assert grid_end == date(2024, 11, 5)
        assert meta["election_date"] == date(2024, 11, 5)
        assert meta["today"] == date(2024, 3, 1)

    def test_poll_indices_stay_within_grid(self):
        data = polls_from_dataframe(_frame(["2024-01-15", "2024-04-02", "2024-08-30"]))
        config = ModelConfig(time_step_days=7)
        model, meta = build_model(data, date(2024, 11, 5), date(2024, 9, 1), config)
        assert 0 <= meta["today_idx"] <= meta["election_idx"]
        assert meta["n_timesteps"] == meta["election_idx"] + 1


class TestBoundaryWarnings:
    def test_warns_on_polls_after_election_date(self):
        data = polls_from_dataframe(_frame(["2024-01-15", "2024-07-01"]))
        config = ModelConfig(time_step_days=7)
        with pytest.raises(ValueError, match="after election_date"):
            build_model(data, date(2024, 6, 1), date(2024, 3, 1), config)


class TestInitialState:
    def test_uses_earliest_occupied_node_not_all_polls(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-15", "2024-05-15"],
                "pollster": ["P", "P"],
                "sample_size": [1000, 1000],
                "Alpha": [70.0, 20.0],
                "Beta": [20.0, 70.0],
                "Small": [10.0, 10.0],
            }
        )
        data = polls_from_dataframe(frame)
        _, meta = build_model(
            data,
            election_date=date(2024, 6, 1),
            today=date(2024, 5, 15),
            config=ModelConfig(time_step_days=30),
        )
        assert meta["initial_props"] == pytest.approx([0.7, 0.2, 0.1])

    def test_largest_early_candidate_is_reference(self):
        data = polls_from_dataframe(_frame(["2024-01-15", "2024-02-15"]))
        _, meta = build_model(
            data,
            election_date=date(2024, 6, 1),
            today=date(2024, 2, 15),
            config=ModelConfig(),
        )
        assert meta["reference_candidate_idx"] == int(np.argmax(meta["initial_props"]))

    def test_no_warning_when_everything_is_in_range(self):
        data = polls_from_dataframe(_frame(["2024-01-15", "2024-03-01"]))
        config = ModelConfig(time_step_days=7)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_model(data, date(2024, 6, 1), date(2024, 3, 1), config)
        messages = [str(w.message) for w in caught]
        assert not [m for m in messages if "election_date" in m or "time grid" in m]

    def test_warns_when_today_after_election(self):
        data = polls_from_dataframe(_frame(["2024-01-15", "2024-03-01"]))
        config = ModelConfig(time_step_days=7)
        with pytest.warns(UserWarning, match="after election_date"):
            build_model(data, date(2024, 6, 1), date(2024, 7, 1), config)

    def test_warns_when_today_precedes_grid(self):
        data = polls_from_dataframe(_frame(["2024-01-15", "2024-03-01"]))
        config = ModelConfig(time_step_days=7)
        with pytest.warns(UserWarning, match="precedes the start of the time grid"):
            build_model(data, date(2024, 6, 1), date(2023, 1, 1), config)


class TestWalkSigmaScaling:
    def test_default_matches_legacy_behaviour(self):
        """With the shipped defaults the per-step scale is untouched."""
        config = ModelConfig()
        assert config.per_step_walk_sigma == pytest.approx(config.sigma_walk_prior)

    @pytest.mark.parametrize("step", [1, 2, 3, 7, 14, 30])
    def test_implied_volatility_is_grid_invariant(self, step: int):
        """Variance accumulated over a week must not depend on the step size."""
        config = ModelConfig(time_step_days=step)
        weekly_variance = (7.0 / step) * config.per_step_walk_sigma**2
        assert np.sqrt(weekly_variance) == pytest.approx(config.sigma_walk_prior)

    def test_reference_window_is_configurable(self):
        config = ModelConfig(
            time_step_days=30, walk_reference_days=30, sigma_walk_prior=0.2
        )
        assert config.per_step_walk_sigma == pytest.approx(0.2)

    def test_rejects_invalid_settings(self):
        with pytest.raises(ValueError, match="time_step_days"):
            ModelConfig(time_step_days=0)
        with pytest.raises(ValueError, match="walk_reference_days"):
            ModelConfig(walk_reference_days=0)
        with pytest.raises(ValueError, match="sigma_walk_prior"):
            ModelConfig(sigma_walk_prior=0.0)
