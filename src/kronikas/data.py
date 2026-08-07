"""Poll data loading, validation and normalisation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PollData:
    """Validated and normalised poll data ready for modelling.

    All percentage values are on a 0-100 scale and sum to 100 per row.

    Attributes:
        dates: Integer array of shape ``(N,)`` – days since the first poll.
        pollster_ids: Integer array of shape ``(N,)`` – pollster index per poll.
        sample_sizes: Float array of shape ``(N,)`` – sample sizes.
        poll_values: Float array of shape ``(N, K)`` – normalised candidate
            support in percentage points.
        candidates: Candidate names in column order.
        pollsters: Unique pollster names in index order.
        first_poll_date: Calendar date of the earliest poll.
    """

    dates: np.ndarray
    pollster_ids: np.ndarray
    sample_sizes: np.ndarray
    poll_values: np.ndarray
    candidates: list[str]
    pollsters: list[str]
    first_poll_date: date

    @property
    def poll_dates(self) -> list[date]:
        """Calendar date of every poll, in row order."""
        return [self.first_poll_date + timedelta(days=int(d)) for d in self.dates]

    @property
    def last_poll_date(self) -> date:
        """Calendar date of the most recent poll."""
        return self.first_poll_date + timedelta(days=int(self.dates.max()))

    def up_to(self, cutoff: date) -> PollData:
        """Return a copy containing only polls taken on or before *cutoff*.

        Used by :func:`kronikas.backtesting.backtest` to reconstruct the
        information available at a past point in time.  Pollsters that drop out
        entirely are removed and the remaining pollster indices renumbered, so
        the result is indistinguishable from data loaded fresh.

        Parameters
        ----------
        cutoff:
            Inclusive upper bound on poll dates.

        Raises
        ------
        ValueError
            If no polls remain on or before *cutoff*.
        """
        mask = np.array([d <= cutoff for d in self.poll_dates], dtype=bool)
        if not mask.any():
            raise ValueError(
                f"No polls on or before {cutoff}; the earliest poll is "
                f"{self.first_poll_date}."
            )

        names = [self.pollsters[i] for i in self.pollster_ids[mask]]
        remaining = sorted(set(names))
        remap = {name: i for i, name in enumerate(remaining)}

        kept_dates = [d for d, keep in zip(self.poll_dates, mask, strict=True) if keep]
        new_first = min(kept_dates)

        return PollData(
            dates=np.array([(d - new_first).days for d in kept_dates], dtype=np.int64),
            pollster_ids=np.array([remap[n] for n in names], dtype=np.int64),
            sample_sizes=self.sample_sizes[mask].copy(),
            poll_values=self.poll_values[mask].copy(),
            candidates=list(self.candidates),
            pollsters=remaining,
            first_poll_date=new_first,
        )


def load_polls(
    csv_path: str | Path,
    *,
    date_column: str = "date",
    pollster_column: str = "pollster",
    sample_size_column: str = "sample_size",
    candidate_columns: list[str] | None = None,
    date_format: str | None = None,
    decimal: str = ".",
) -> PollData:
    """Read and validate a CSV of opinion polls.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    date_column:
        Name of the column containing poll dates.
    pollster_column:
        Name of the column identifying the polling firm.
    sample_size_column:
        Name of the column with sample sizes.
    candidate_columns:
        Explicit list of candidate column names.  When *None* (default),
        every column that is not *date_column*, *pollster_column* or
        *sample_size_column* is treated as a candidate.
    date_format:
        Optional ``strftime``-style format string passed to
        ``pd.to_datetime``.
    decimal:
        Character used as the decimal point in the CSV (default ``"."``).
        Use ``","`` for European-style CSVs where numbers are written as
        ``"45,3"`` instead of ``"45.3"``.

    Returns
    -------
    PollData
        Validated, normalised, date-sorted poll data.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        On schema violations (missing columns, non-numeric candidates,
        negative values, etc.).
    """

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Poll CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, decimal=decimal)

    return polls_from_dataframe(
        df,
        date_column=date_column,
        pollster_column=pollster_column,
        sample_size_column=sample_size_column,
        candidate_columns=candidate_columns,
        date_format=date_format,
    )


def polls_from_dataframe(
    df: pd.DataFrame,
    *,
    date_column: str = "date",
    pollster_column: str = "pollster",
    sample_size_column: str = "sample_size",
    candidate_columns: list[str] | None = None,
    date_format: str | None = None,
) -> PollData:
    """Validate and normalise polls held in a :class:`pandas.DataFrame`.

    Identical to :func:`load_polls` but takes an in-memory frame, so data
    arriving from a database, an API, or an upstream cleaning step does not
    have to be round-tripped through a temporary CSV.  The input is not
    modified.

    Parameters
    ----------
    df:
        One row per poll.  See :func:`load_polls` for the expected schema.
    date_column, pollster_column, sample_size_column, candidate_columns,
    date_format:
        As documented on :func:`load_polls`.

    Returns
    -------
    PollData
        Validated, normalised, date-sorted poll data.

    Raises
    ------
    ValueError
        On schema violations (missing columns, non-numeric candidates,
        negative values, etc.).

    Examples
    --------
    >>> import pandas as pd
    >>> from kronikas import polls_from_dataframe
    >>> frame = pd.DataFrame(
    ...     {
    ...         "date": ["2024-01-15", "2024-02-01"],
    ...         "pollster": ["PollCo", "SurveyInc"],
    ...         "sample_size": [1000, 1200],
    ...         "Alice": [45.0, 44.0],
    ...         "Bob": [55.0, 56.0],
    ...     }
    ... )
    >>> data = polls_from_dataframe(frame)
    >>> data.candidates
    ['Alice', 'Bob']
    """
    df = df.copy()

    # --- required columns ---------------------------------------------------
    meta_cols = {date_column, pollster_column, sample_size_column}
    missing = meta_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- candidate columns ---------------------------------------------------
    if candidate_columns is not None:
        bad = set(candidate_columns) - set(df.columns)
        if bad:
            raise ValueError(f"Candidate columns not found in CSV: {sorted(bad)}")
        cand_cols = list(candidate_columns)
    else:
        cand_cols = [c for c in df.columns if c not in meta_cols]

    if len(cand_cols) < 2:
        raise ValueError(
            f"At least 2 candidate columns are required, found: {cand_cols}"
        )

    # --- parse dates ---------------------------------------------------------
    try:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    except Exception as exc:
        raise ValueError(
            f"Could not parse dates in column '{date_column}'. "
            "Try specifying date_format."
        ) from exc

    # --- drop rows with missing values ---------------------------------------
    required_cols = [date_column, pollster_column, sample_size_column, *cand_cols]
    mask = df[required_cols].notna().all(axis=1)
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped} row(s) with missing values.",
            stacklevel=2,
        )
        df = df.loc[mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid poll rows remain after dropping NaNs.")

    # --- validate types and values -------------------------------------------
    for col in cand_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Candidate column '{col}' must be numeric.")
        if (df[col] < 0).any():
            raise ValueError(f"Candidate column '{col}' contains negative values.")

    if not pd.api.types.is_numeric_dtype(df[sample_size_column]):
        raise ValueError(f"'{sample_size_column}' must be numeric.")
    if (df[sample_size_column] <= 0).any():
        raise ValueError("Sample sizes must be positive.")

    # --- normalise to 100% --------------------------------------------------
    raw = df[cand_cols].to_numpy(dtype=np.float64)
    row_sums = raw.sum(axis=1, keepdims=True)
    if (row_sums == 0).any():
        raise ValueError(
            "At least one poll has zero total support across all candidates."
        )
    normalised = raw / row_sums * 100.0

    # --- sort by date --------------------------------------------------------
    sort_idx = np.argsort(df[date_column].values)
    df = df.iloc[sort_idx].reset_index(drop=True)
    normalised = normalised[sort_idx]

    # --- encode pollsters ----------------------------------------------------
    pollsters = sorted(df[pollster_column].unique())
    pollster_map = {p: i for i, p in enumerate(pollsters)}

    first_date = df[date_column].min().date()
    days = (df[date_column] - df[date_column].min()).dt.days.to_numpy(dtype=np.int64)

    return PollData(
        dates=days,
        pollster_ids=df[pollster_column].map(pollster_map).to_numpy(dtype=np.int64),
        sample_sizes=df[sample_size_column].to_numpy(dtype=np.float64),
        poll_values=normalised,
        candidates=cand_cols,
        pollsters=pollsters,
        first_poll_date=first_date,
    )
