import pandas as pd


def analyze_load_profile(data, date_col="dates", cols=("con", "gen", "off", "inj")):
    """Compute summary metrics for a con/gen/off/inj load profile.

    Values in `cols` are ENERGY per 15-minute interval, in kWh -- not
    instantaneous power (kW). This matches how this project's source data is
    stored (see the off/1000 -> MWh conversion used elsewhere). Every value
    below derived directly from `cols` (totals, peaks, the daily-profile
    average, per-day sums) is therefore also in kWh (or kWh-per-interval);
    nothing here is a kW/power figure unless explicitly converted by the
    caller (kW = kWh_per_interval / 0.25, i.e. x4 for a 15-minute interval).

    Parameters
    ----------
    data : str, Path, or pd.DataFrame
        CSV path to read, or an already-loaded DataFrame.
    date_col : str
        Name of the datetime column.
    cols : sequence of str
        Columns to analyze. Columns not present in the data are skipped.

    Returns
    -------
    dict with keys:
        total_kwh : dict[col, float]
            Sum over the whole period, per column, in kWh.
        peaks : dict[col, dict]
            {"value_kwh": ..., "timestamp": ...} for the max of each column
            -- the single highest interval's energy, in kWh (not kW).
        quarter_hour_profile : pd.DataFrame
            Average kWh-per-interval, per column, for each quarter-hour of
            the day (96 rows, "HH:MM"), i.e. a typical daily profile.
        daily_totals : pd.DataFrame
            Sum per column for each calendar day, in kWh.
        daily_totals_summary : pd.DataFrame
            mean/min/max/std of daily_totals (kWh/day), per column.
        load_factor : dict[col, float]
            mean / peak, per column (unitless ratio; closer to 1 = flatter profile).
        self_consumption_ratio : float, optional
            Share of generated energy consumed on-site rather than injected
            to the grid. Only present if both "gen" and "inj" are in `cols`
            and total generation is nonzero.
        n_days : int
            Number of distinct calendar days covered.
        date_range : tuple
            (min date, max date) in the data.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(data, parse_dates=[date_col])
    df = df.sort_values(date_col)

    cols = [c for c in cols if c in df.columns and df[c].notna().any()]

    total_kwh = {c: df[c].sum() for c in cols}

    peaks = {}
    for c in cols:
        idx = df[c].idxmax()
        peaks[c] = {"value_kwh": df.loc[idx, c], "timestamp": df.loc[idx, date_col]}

    time_of_day = df[date_col].dt.strftime("%H:%M")
    quarter_hour_profile = df.groupby(time_of_day)[cols].mean().sort_index()

    day = df[date_col].dt.date
    daily_totals = df.groupby(day)[cols].sum()
    daily_totals_summary = daily_totals.agg(["mean", "min", "max", "std"])

    load_factor = {
        c: (df[c].mean() / df[c].max()) if df[c].max() else float("nan")
        for c in cols
    }

    metrics = {
        "total_kwh": total_kwh,
        "peaks": peaks,
        "quarter_hour_profile": quarter_hour_profile,
        "daily_totals": daily_totals,
        "daily_totals_summary": daily_totals_summary,
        "load_factor": load_factor,
        "n_days": daily_totals.shape[0],
        "date_range": (df[date_col].min(), df[date_col].max()),
    }

    if "gen" in cols and "inj" in cols and df["gen"].sum() > 0:
        self_consumed = df["gen"].sum() - df["inj"].sum()
        metrics["self_consumption_ratio"] = self_consumed / df["gen"].sum()

    return metrics
