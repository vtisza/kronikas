"""Tests for kronikas.data (CSV loading, validation, normalisation)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kronikas.data import PollData, load_polls


class TestLoadPollsHappyPath:
    def test_returns_poll_data(self, polls_csv: Path):
        result = load_polls(polls_csv)
        assert isinstance(result, PollData)

    def test_shape(self, polls_csv: Path):
        result = load_polls(polls_csv)
        assert result.poll_values.shape == (5, 3)
        assert result.dates.shape == (5,)
        assert result.pollster_ids.shape == (5,)
        assert result.sample_sizes.shape == (5,)

    def test_candidates(self, polls_csv: Path):
        result = load_polls(polls_csv)
        assert result.candidates == ["Candidate_A", "Candidate_B", "Candidate_C"]

    def test_pollsters(self, polls_csv: Path):
        result = load_polls(polls_csv)
        assert set(result.pollsters) == {"PollCo", "SurveyInc"}

    def test_sorted_by_date(self, polls_csv: Path):
        result = load_polls(polls_csv)
        assert (np.diff(result.dates) >= 0).all()

    def test_first_poll_date(self, polls_csv: Path):
        from datetime import date

        result = load_polls(polls_csv)
        assert result.first_poll_date == date(2024, 1, 15)


class TestNormalisation:
    def test_rows_sum_to_100(self, polls_csv: Path):
        result = load_polls(polls_csv)
        row_sums = result.poll_values.sum(axis=1)
        np.testing.assert_allclose(row_sums, 100.0, atol=1e-10)

    def test_values_non_negative(self, polls_csv: Path):
        result = load_polls(polls_csv)
        assert (result.poll_values >= 0).all()


class TestExplicitCandidateColumns:
    def test_subset(self, polls_csv: Path):
        result = load_polls(polls_csv, candidate_columns=["Candidate_A", "Candidate_B"])
        assert result.candidates == ["Candidate_A", "Candidate_B"]
        assert result.poll_values.shape[1] == 2

    def test_bad_column(self, polls_csv: Path):
        with pytest.raises(ValueError, match="not found"):
            load_polls(polls_csv, candidate_columns=["Nonexistent"])


class TestValidationErrors:
    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_polls(tmp_path / "nope.csv")

    def test_missing_column(self, tmp_path: Path):
        df = pd.DataFrame({"date": ["2024-01-01"], "X": [50], "Y": [50]})
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_polls(path)

    def test_too_few_candidates(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "pollster": ["P"],
                "sample_size": [100],
                "Only": [50],
            }
        )
        path = tmp_path / "one_cand.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="At least 2"):
            load_polls(path)

    def test_negative_candidate_value(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "pollster": ["P"],
                "sample_size": [100],
                "A": [-5],
                "B": [50],
            }
        )
        path = tmp_path / "neg.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="negative"):
            load_polls(path)

    def test_zero_sample_size(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "pollster": ["P"],
                "sample_size": [0],
                "A": [50],
                "B": [50],
            }
        )
        path = tmp_path / "zero_n.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="positive"):
            load_polls(path)

    def test_bad_date(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "date": ["not-a-date"],
                "pollster": ["P"],
                "sample_size": [100],
                "A": [50],
                "B": [50],
            }
        )
        path = tmp_path / "bad_date.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="parse"):
            load_polls(path, date_format="%Y-%m-%d")


class TestDecimalParameter:
    def test_default_dot_decimal(self, tmp_path: Path):
        """Default decimal='.' reads standard dot-decimal CSVs correctly."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-02-01"],
                "pollster": ["P", "P"],
                "sample_size": [1000, 1000],
                "A": [45.5, 44.0],
                "B": [54.5, 56.0],
            }
        )
        path = tmp_path / "dot_decimal.csv"
        df.to_csv(path, index=False)
        result = load_polls(path)
        assert result.poll_values.shape == (2, 2)

    def test_comma_decimal(self, tmp_path: Path):
        """decimal=',' parses comma-decimal values (e.g. European CSVs)."""
        path = tmp_path / "comma_decimal.csv"
        path.write_text(
            "date,pollster,sample_size,A,B\n"
            '2024-01-01,P,1000,"45,5","54,5"\n'
            '2024-02-01,P,800,"40,0","60,0"\n'
        )
        result = load_polls(path, decimal=",")
        np.testing.assert_allclose(
            result.poll_values[0], [45.5 / 100.0 * 100, 54.5 / 100.0 * 100]
        )

    def test_comma_decimal_normalisation(self, tmp_path: Path):
        """Rows loaded with decimal=',' still normalise to 100%."""
        path = tmp_path / "comma_decimal_norm.csv"
        path.write_text(
            'date,pollster,sample_size,A,B\n2024-01-01,P,1000,"30,0","70,0"\n'
        )
        result = load_polls(path, decimal=",")
        np.testing.assert_allclose(result.poll_values.sum(axis=1), 100.0, atol=1e-10)


class TestMissingValues:
    def test_nan_rows_dropped(self, tmp_path: Path):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "pollster": ["P", "P", "P"],
                "sample_size": [100, 100, 100],
                "A": [50, None, 55],
                "B": [50, 60, 45],
            }
        )
        path = tmp_path / "nan.csv"
        df.to_csv(path, index=False)
        with pytest.warns(UserWarning, match="Dropped 1"):
            result = load_polls(path)
        assert result.poll_values.shape[0] == 2
