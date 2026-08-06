"""Tests for in-memory poll input and date-based filtering."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kronikas import ElectionForecast, polls_from_dataframe
from kronikas.data import load_polls


class TestPollsFromDataFrame:
    def test_matches_csv_loader(self, polls_csv: Path):
        from_csv = load_polls(polls_csv)
        from_frame = polls_from_dataframe(pd.read_csv(polls_csv))

        assert from_frame.candidates == from_csv.candidates
        assert from_frame.pollsters == from_csv.pollsters
        assert from_frame.first_poll_date == from_csv.first_poll_date
        assert np.array_equal(from_frame.dates, from_csv.dates)
        assert np.array_equal(from_frame.pollster_ids, from_csv.pollster_ids)
        assert np.allclose(from_frame.poll_values, from_csv.poll_values)

    def test_does_not_mutate_the_caller_frame(self, polls_csv: Path):
        frame = pd.read_csv(polls_csv)
        before = frame.copy(deep=True)
        polls_from_dataframe(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_validates_like_the_csv_loader(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "pollster": ["PollCo"],
                "sample_size": [1000],
                "Only": [100.0],
            }
        )
        with pytest.raises(ValueError, match="At least 2 candidate columns"):
            polls_from_dataframe(frame)

    def test_rejects_negative_values(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "pollster": ["PollCo"],
                "sample_size": [1000],
                "A": [-5.0],
                "B": [105.0],
            }
        )
        with pytest.raises(ValueError, match="negative values"):
            polls_from_dataframe(frame)

    def test_rows_are_normalised(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-02-01"],
                "pollster": ["PollCo", "SurveyInc"],
                "sample_size": [1000, 1000],
                "A": [30.0, 3.0],
                "B": [30.0, 3.0],
            }
        )
        data = polls_from_dataframe(frame)
        assert np.allclose(data.poll_values.sum(axis=1), 100.0)
        assert np.allclose(data.poll_values[0], data.poll_values[1])


class TestPollDates:
    def test_poll_dates_round_trip(self, polls_csv: Path):
        data = load_polls(polls_csv)
        assert data.poll_dates[0] == data.first_poll_date
        assert data.poll_dates[-1] == data.last_poll_date
        assert data.poll_dates == sorted(data.poll_dates)


class TestUpTo:
    def test_filters_to_cutoff(self, polls_csv: Path):
        data = load_polls(polls_csv)
        subset = data.up_to(date(2024, 2, 15))
        assert len(subset.dates) == 3
        assert all(d <= date(2024, 2, 15) for d in subset.poll_dates)

    def test_inclusive_of_cutoff(self, polls_csv: Path):
        data = load_polls(polls_csv)
        assert date(2024, 2, 15) in data.up_to(date(2024, 2, 15)).poll_dates

    def test_full_range_is_a_faithful_copy(self, polls_csv: Path):
        data = load_polls(polls_csv)
        subset = data.up_to(date(2030, 1, 1))
        assert np.array_equal(subset.dates, data.dates)
        assert np.array_equal(subset.pollster_ids, data.pollster_ids)
        assert subset.pollsters == data.pollsters

    def test_rebases_dates_on_new_first_poll(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-02-01", "2024-03-01"],
                "pollster": ["A", "B", "A"],
                "sample_size": [1000] * 3,
                "X": [50.0, 51.0, 52.0],
                "Y": [50.0, 49.0, 48.0],
            }
        )
        subset = polls_from_dataframe(frame).up_to(date(2024, 2, 15))
        assert subset.first_poll_date == date(2024, 1, 1)
        assert subset.dates[0] == 0

    def test_drops_and_renumbers_absent_pollsters(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-03-01"],
                "pollster": ["Early", "Late"],
                "sample_size": [1000, 1000],
                "X": [50.0, 52.0],
                "Y": [50.0, 48.0],
            }
        )
        subset = polls_from_dataframe(frame).up_to(date(2024, 2, 1))
        assert subset.pollsters == ["Early"]
        assert subset.pollster_ids.tolist() == [0]

    def test_empty_selection_raises(self, polls_csv: Path):
        with pytest.raises(ValueError, match="No polls on or before"):
            load_polls(polls_csv).up_to(date(2000, 1, 1))

    def test_does_not_mutate_the_source(self, polls_csv: Path):
        data = load_polls(polls_csv)
        original = data.dates.copy()
        data.up_to(date(2024, 2, 15))
        assert np.array_equal(data.dates, original)


class TestElectionForecastFromDataFrame:
    def test_builds_equivalent_poll_data(self, polls_csv: Path):
        frame = pd.read_csv(polls_csv)
        forecast = ElectionForecast.from_dataframe(
            frame, election_date="2024-06-01", today="2024-03-20"
        )
        assert forecast.election_date == date(2024, 6, 1)
        assert forecast.today == date(2024, 3, 20)
        assert forecast.poll_data.candidates == load_polls(polls_csv).candidates

    def test_accepts_column_overrides(self):
        frame = pd.DataFrame(
            {
                "poll_date": ["2024-01-01", "2024-02-01"],
                "firm": ["A", "B"],
                "n": [1000, 1200],
                "X": [50.0, 51.0],
                "Y": [50.0, 49.0],
            }
        )
        forecast = ElectionForecast.from_dataframe(
            frame,
            election_date="2024-06-01",
            date_column="poll_date",
            pollster_column="firm",
            sample_size_column="n",
        )
        assert forecast.poll_data.candidates == ["X", "Y"]
        assert forecast.poll_data.pollsters == ["A", "B"]
