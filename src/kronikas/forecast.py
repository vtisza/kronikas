"""High-level forecasting interface."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from .config import ModelConfig
from .data import PollData, load_polls
from .model import ForecastResult, build_model, extract_results, run_inference


class ElectionForecast:
    """One-call interface for running an election forecast.

    Parameters
    ----------
    polls_csv:
        Path to the CSV file containing opinion polls.
    election_date:
        Date of the election (ISO-8601 string or ``datetime.date``).
    today:
        Reference date for "current" estimates.  Defaults to today.
    config:
        Model / sampler configuration.  Uses ``ModelConfig()`` defaults
        when *None*.
    date_column, pollster_column, sample_size_column:
        Column-name overrides for the poll CSV (see ``load_polls``).
    candidate_columns:
        Explicit candidate column names.  When *None*, inferred
        automatically.
    date_format:
        Optional ``strftime``-style format for parsing dates.
    decimal:
        Character used as the decimal point in the CSV (default ``"."``).
        Use ``","`` for European-style CSVs.

    Examples
    --------
    >>> forecast = ElectionForecast(
    ...     polls_csv="polls.csv",
    ...     election_date="2024-11-05",
    ... )
    >>> result = forecast.run()
    >>> print(result.summary())
    """

    def __init__(
        self,
        polls_csv: str | Path,
        election_date: str | date,
        *,
        today: str | date | None = None,
        config: ModelConfig | None = None,
        date_column: str = "date",
        pollster_column: str = "pollster",
        sample_size_column: str = "sample_size",
        candidate_columns: list[str] | None = None,
        date_format: str | None = None,
        decimal: str = ".",
    ) -> None:
        self.config = config or ModelConfig()
        self.election_date = _parse_date(election_date, "election_date")
        self.today = _parse_date(today, "today") if today else date.today()
        self.poll_data: PollData = load_polls(
            polls_csv,
            date_column=date_column,
            pollster_column=pollster_column,
            sample_size_column=sample_size_column,
            candidate_columns=candidate_columns,
            date_format=date_format,
            decimal=decimal,
        )

    def run(self) -> ForecastResult:
        """Build the model, sample, and return a ``ForecastResult``."""
        model, metadata = build_model(
            self.poll_data, self.election_date, self.today, self.config
        )
        trace = run_inference(model, self.config)
        return extract_results(trace, self.poll_data, metadata)


def _parse_date(value: str | date | None, name: str) -> date:
    """Coerce a string or date to ``datetime.date``."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            raise ValueError(
                f"Cannot parse '{name}' as a date: {value!r}. "
                "Use YYYY-MM-DD format."
            ) from None
    raise TypeError(f"'{name}' must be a str or datetime.date, got {type(value)}")
