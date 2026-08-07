from datetime import timezone, timedelta
from pathlib import Path

import pandas as pd

# Fixed CET (UTC+1, no DST): the source platform ignores daylight saving and
# always exports in a constant UTC+1 offset. Confirmed by checking each
# profile's raw timestamps across both 2024 DST transition dates: a fixed-
# offset source runs straight through 02:00-03:00 with no gap or duplicate
# on either transition day, while a real DST-aware (Europe/Brussels) source
# skips the 02:00-02:45 slot on the spring-forward date and repeats/collides
# on the fall-back date.
FIXED_CET_NO_DST = timezone(timedelta(hours=1))

# Real, DST-aware Belgian local time.
BRUSSELS_LOCAL = "Europe/Brussels"

# Known site profiles: excel file name (in data/profiles/), the sheet that
# holds the dates/con/gen/off/inj columns, and the timezone its raw
# timestamps are actually in (verified per-profile, see FIXED_CET_NO_DST
# above -- don't assume, check new profiles the same way before adding them).
PROFILES = {
    "carmeuse": {"file": "carmeuse.xlsx", "sheet": "Site_Profile", "source_tz": FIXED_CET_NO_DST},
    "montea": {"file": "Montea Gent BTM 2.xlsx", "sheet": "Sheet1", "source_tz": BRUSSELS_LOCAL},
    "lemahieu": {"file": "Lemahieu Oostende - Profile Input 2024.xlsx", "sheet": "Sheet1", "source_tz": FIXED_CET_NO_DST},
}

PROFILE_COLS = ("dates", "con", "gen", "off", "inj")

# Backwards-compatible alias.
DEFAULT_SOURCE_TZ = FIXED_CET_NO_DST


def load_profile_excel(
    input_path,
    sheet_name,
    date_col="dates",
    cols=PROFILE_COLS,
    source_tz=FIXED_CET_NO_DST,
    round_freq="15min",
):
    """Load a site load-profile Excel export and convert its timestamps to UTC.

    Parameters
    ----------
    input_path : str or Path
        Excel file to read.
    sheet_name : str
        Sheet containing the `dates`/`con`/`gen`/`off`/`inj` columns.
    date_col : str
        Name of the datetime column.
    cols : sequence of str
        Columns to keep. Output column order follows the source sheet, not
        this argument -- pandas' usecols does not reorder columns.
    source_tz : tzinfo or str
        Timezone the source timestamps are naive-local in. Use
        `FIXED_CET_NO_DST` (the default) for sources that ignore daylight
        saving, or `BRUSSELS_LOCAL` ("Europe/Brussels") for sources that use
        real DST-aware local time -- check which one applies (see the
        `PROFILES` comment) before picking one for a new source.
    round_freq : str or None
        Round timestamps to this frequency after conversion, to correct for
        floating-point drift in Excel-stored datetimes (observed e.g. on the
        Lemahieu file, where timestamps land a few ms before the quarter
        hour). Set to None to disable.

    Returns
    -------
    pd.DataFrame
        `date_col` as a tz-aware UTC column (not the index), sorted. Rows
        whose timestamp fell in an unresolvable DST transition, or that
        collide with another row on the same timestamp after conversion,
        are dropped with a printed warning rather than guessed at.
    """
    # Note: intentionally not using read_excel's parse_dates here -- on pandas
    # 3.0 it can turn an already-native datetime column into plain strings
    # for some source files (observed on the Lemahieu profile). Parsing
    # explicitly afterward is robust to both cases.
    df = pd.read_excel(input_path, sheet_name=sheet_name, usecols=list(cols))
    df[date_col] = pd.to_datetime(df[date_col])

    # ambiguous/nonexistent="NaT" rather than guessing: for a DST-aware
    # source, the spring-forward gap is expected to have no data at all
    # (that wall-clock hour never happened) and doesn't trigger this; the
    # fall-back hour only becomes unresolvable if the source already
    # collapsed its two real occurrences into a single labeled row (as seen
    # on the Montea file), in which case we can't know which one it kept.
    df[date_col] = (
        df[date_col]
        .dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )

    nat_count = df[date_col].isna().sum()
    if nat_count:
        print(
            f"Warning: {input_path} has {nat_count} row(s) landing on an unresolvable "
            "DST transition (ambiguous fall-back hour) after localizing to "
            f"{source_tz!r} -- dropping them."
        )
        df = df.dropna(subset=[date_col])

    if round_freq is not None:
        df[date_col] = df[date_col].dt.round(round_freq)

    df = df.sort_values(date_col)

    dup_count = df[date_col].duplicated().sum()
    if dup_count:
        print(
            f"Warning: {input_path} has {dup_count} duplicate timestamp(s) after conversion "
            "(same UTC instant, different values in the source) -- keeping the first occurrence."
        )
        df = df.drop_duplicates(subset=date_col, keep="first")

    return df.reset_index(drop=True)


def load_market_prices(input_path, date_col="dates"):
    """Load a market data CSV that is already in UTC (e.g. data/market/da_belgium.csv)."""
    df = pd.read_csv(input_path, parse_dates=[date_col])
    return df.sort_values(date_col).reset_index(drop=True)


def merge_profile_with_market(profile_df, market_df, date_col="dates", how="inner"):
    """Join a site profile with market data on their shared UTC timestamps."""
    merged = pd.merge(profile_df, market_df, on=date_col, how=how)
    return merged.sort_values(date_col).reset_index(drop=True)


def build_profile_dataset(
    profile_name,
    data_dir,
    market_filename="da_belgium.csv",
    output_filename=None,
    date_col="dates",
    source_tz=None,
    verbose=True,
):
    """Run the full pipeline for one of the known site profiles: Excel -> UTC -> merge with market data.

    Replaces doing this by hand across a *_to_CSV.ipynb + csvmerger.ipynb
    pair of notebooks.

    Parameters
    ----------
    profile_name : str
        Key into `PROFILES` (e.g. "carmeuse", "montea", "lemahieu").
    data_dir : str or Path
        Path to the project's `data` directory (containing `profiles/`,
        `market/`, and `csvs/` subfolders).
    market_filename : str
        CSV file in `data_dir/market/` to merge against.
    output_filename : str, optional
        If given, write the merged result to `data_dir/csvs/<output_filename>`.
    source_tz : tzinfo or str, optional
        Forwarded to `load_profile_excel`. Defaults to the profile's
        verified timezone from `PROFILES`; only pass this to override it.
    verbose : bool
        Print row-count / match-rate diagnostics, useful for spotting
        timezone or data-gap problems.

    Returns
    -------
    pd.DataFrame
        The merged profile + market dataset, in the same shape as the
        existing merged*.csv files (dates, profile columns, then market
        columns).
    """
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile {profile_name!r}. Known profiles: {sorted(PROFILES)}")

    data_dir = Path(data_dir)
    config = PROFILES[profile_name]
    profile_path = data_dir / "profiles" / config["file"]
    market_path = data_dir / "market" / market_filename
    if source_tz is None:
        source_tz = config["source_tz"]

    profile_df = load_profile_excel(profile_path, sheet_name=config["sheet"], date_col=date_col, source_tz=source_tz)
    market_df = load_market_prices(market_path, date_col=date_col)
    merged = merge_profile_with_market(profile_df, market_df, date_col=date_col)

    if verbose:
        match_rate = len(merged) / len(profile_df) if len(profile_df) else float("nan")
        print(
            f"{profile_name}: {len(profile_df)} profile rows, {len(market_df)} market rows "
            f"-> {len(merged)} merged rows ({match_rate:.1%} of profile matched)"
        )
        if match_rate < 0.99:
            print(
                "Warning: a significant share of profile rows did not find a matching market "
                "timestamp -- double check source_tz and the profile's date range."
            )

    if output_filename is not None:
        output_path = data_dir / "csvs" / output_filename
        merged.to_csv(output_path, index=False)
        if verbose:
            print(f"Saved to {output_path}")

    return merged
