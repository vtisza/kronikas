"""Shared test fixtures."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from kronikas.config import ModelConfig
from kronikas.data import load_polls


@pytest.fixture()
def polls_csv(tmp_path: Path) -> Path:
    """Write a small but realistic poll CSV and return its path."""
    df = pd.DataFrame(
        {
            "date": [
                "2024-01-15",
                "2024-02-01",
                "2024-02-15",
                "2024-03-01",
                "2024-03-15",
            ],
            "pollster": [
                "PollCo",
                "SurveyInc",
                "PollCo",
                "SurveyInc",
                "PollCo",
            ],
            "sample_size": [1000, 1200, 800, 1500, 1000],
            "Candidate_A": [45, 44, 46, 43, 47],
            "Candidate_B": [40, 42, 39, 43, 38],
            "Candidate_C": [10, 11, 12, 10, 13],
        }
    )
    path = tmp_path / "polls.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def single_pollster_csv(tmp_path: Path) -> Path:
    """CSV with a single pollster (house effects should be skipped)."""
    df = pd.DataFrame(
        {
            "date": ["2024-01-10", "2024-02-10", "2024-03-10"],
            "pollster": ["OnlyPoll", "OnlyPoll", "OnlyPoll"],
            "sample_size": [1000, 1000, 1000],
            "Alpha": [52, 51, 53],
            "Beta": [48, 49, 47],
        }
    )
    path = tmp_path / "single_pollster.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def poll_data(polls_csv: Path):
    """Loaded PollData from the standard fixture CSV."""
    return load_polls(polls_csv)


@pytest.fixture()
def fast_config() -> ModelConfig:
    """Minimal sampler settings for fast test runs."""
    return ModelConfig(
        num_tune=50,
        num_draws=50,
        num_chains=1,
        cores=1,
        target_accept=0.8,
        time_step_days=30,
        random_seed=123,
        progressbar=False,
    )


@pytest.fixture()
def election_date() -> date:
    return date(2024, 6, 1)


@pytest.fixture()
def today() -> date:
    return date(2024, 3, 20)
