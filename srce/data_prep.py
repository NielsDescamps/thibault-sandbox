from pathlib import Path

import pandas as pd


def select_first_n_days(input_path, output_path=None, days=5, date_col="dates"):
    """Load a time-series CSV and keep only the first `days` days of data.

    Parameters
    ----------
    input_path : str or Path
        CSV file to read.
    output_path : str or Path, optional
        If given, write the trimmed data to this CSV path.
    days : int
        Number of days (from the earliest timestamp) to keep.
    date_col : str
        Name of the datetime column to parse, sort, and filter on.

    Returns
    -------
    pd.DataFrame
        The trimmed data, sorted by `date_col`.
    """
    df = pd.read_csv(input_path, parse_dates=[date_col])
    df = df.sort_values(date_col)

    start = df[date_col].min()
    end = start + pd.Timedelta(days=days)
    trimmed = df[(df[date_col] >= start) & (df[date_col] < end)]

    if output_path is not None:
        trimmed.to_csv(output_path, index=False)

    return trimmed


def select_representative_weeks(input_path, output_path=None, date_col="dates", days_per_month=7):
    """Load a time-series CSV and keep only the first `days_per_month` days of each calendar month.

    With the default `days_per_month=7`, this samples the first week of every
    month, giving a ~12-week (84-day) subset that spans the whole year's
    seasonality at a fraction of the data -- useful for optimizations that
    don't scale to a full year (e.g. a MILP with per-interval binaries).

    Parameters
    ----------
    input_path : str or Path
        CSV file to read.
    output_path : str or Path, optional
        If given, write the sampled data to this CSV path.
    date_col : str
        Name of the datetime column to parse, sort, and filter on.
    days_per_month : int
        Number of days from the start of each calendar month to keep.

    Returns
    -------
    pd.DataFrame
        The sampled data, sorted by `date_col`.
    """
    df = pd.read_csv(input_path, parse_dates=[date_col])
    df = df.sort_values(date_col)

    sample = df[df[date_col].dt.day <= days_per_month]

    if output_path is not None:
        sample.to_csv(output_path, index=False)

    return sample


def select_date_range(input_path, start, end, output_path=None, date_col="dates"):
    """Load a time-series CSV and keep only rows within [start, end).

    Parameters
    ----------
    input_path : str or Path
        CSV file to read.
    start : str or datetime-like
        Start of the period to keep (inclusive).
    end : str or datetime-like
        End of the period to keep (exclusive).
    output_path : str or Path, optional
        If given, write the trimmed data to this CSV path.
    date_col : str
        Name of the datetime column to parse, sort, and filter on.

    Returns
    -------
    pd.DataFrame
        The trimmed data, sorted by `date_col`.
    """
    df = pd.read_csv(input_path, parse_dates=[date_col])
    df = df.sort_values(date_col)

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    tz = df[date_col].dt.tz
    if tz is not None:
        if start.tz is None:
            start = start.tz_localize(tz)
        if end.tz is None:
            end = end.tz_localize(tz)

    trimmed = df[(df[date_col] >= start) & (df[date_col] < end)]

    if output_path is not None:
        trimmed.to_csv(output_path, index=False)

    return trimmed
