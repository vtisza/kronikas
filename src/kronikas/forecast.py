"""High-level forecasting interface."""

from __future__ import annotations

import warnings
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .config import ModelConfig
from .data import PollData, load_polls, polls_from_dataframe
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
    undecided_column:
        Optional undecided-response column. When supplied, it is excluded from
        candidates and used to reduce each poll's effective sample size.

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
        undecided_column: str | None = None,
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
            undecided_column=undecided_column,
        )

    @classmethod
    def from_dataframe(
        cls,
        polls: pd.DataFrame,
        election_date: str | date,
        *,
        today: str | date | None = None,
        config: ModelConfig | None = None,
        date_column: str = "date",
        pollster_column: str = "pollster",
        sample_size_column: str = "sample_size",
        candidate_columns: list[str] | None = None,
        date_format: str | None = None,
        undecided_column: str | None = None,
    ) -> ElectionForecast:
        """Build a forecast from an in-memory :class:`pandas.DataFrame`.

        Equivalent to the constructor but skips the CSV round-trip, for data
        that already lives in memory.

        Examples
        --------
        >>> forecast = ElectionForecast.from_dataframe(  # doctest: +SKIP
        ...     polls_frame, election_date="2024-11-05"
        ... )
        >>> result = forecast.run()  # doctest: +SKIP
        """
        instance = cls.__new__(cls)
        instance.config = config or ModelConfig()
        instance.election_date = _parse_date(election_date, "election_date")
        instance.today = _parse_date(today, "today") if today else date.today()
        instance.poll_data = polls_from_dataframe(
            polls,
            date_column=date_column,
            pollster_column=pollster_column,
            sample_size_column=sample_size_column,
            candidate_columns=candidate_columns,
            date_format=date_format,
            undecided_column=undecided_column,
        )
        return instance

    def run(self) -> ForecastResult:
        """Build the model, sample, and return a ``ForecastResult``.

        Emits a :class:`~kronikas.diagnostics.ConvergenceWarning` if the
        posterior shows signs of not having converged.
        """
        cutoff = min(self.today, self.election_date)
        fit_data = self.poll_data
        if self.poll_data.last_poll_date > cutoff:
            warnings.warn(
                f"Ignoring polls after the forecast cutoff ({cutoff}); only "
                "information available by that date may enter the fit.",
                stacklevel=2,
            )
            fit_data = self.poll_data.up_to(cutoff)
        model, metadata = build_model(
            fit_data, self.election_date, self.today, self.config
        )
        trace = run_inference(model, self.config)
        return extract_results(trace, fit_data, metadata, config=self.config)


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
                f"Cannot parse '{name}' as a date: {value!r}. Use YYYY-MM-DD format."
            ) from None
    raise TypeError(f"'{name}' must be a str or datetime.date, got {type(value)}")
