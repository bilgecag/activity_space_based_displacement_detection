"""Day/night duration split and relevance scores.

Nighttime presence identifies residential areas and close social ties, so
ASA weights locations by the time spent in them at night (default window
22:00-07:00). ``day_night_durations`` splits each stay's duration into day
and night hours with a closed-form cumulative computation:

    D(t) = 15 * days_since_epoch(t) + clip(hour_of_day(t) - 7, 0, 15)

is the cumulative number of daytime hours from the epoch up to t, hence
day = D(end) - D(start) and night = total - day (for the default window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_NS_PER_HOUR = 3_600_000_000_000


def _cumulative_day_hours(ts: pd.Series, night_start: int, night_end: int) -> np.ndarray:
    day_len = night_start - night_end          # daytime hours per day
    ns = ts.values.astype("datetime64[ns]").astype(np.int64)
    hours = ns / _NS_PER_HOUR
    days = np.floor(hours / 24.0)
    hod = hours - days * 24.0
    return day_len * days + np.clip(hod - night_end, 0.0, day_len)


def day_night_durations(df: pd.DataFrame,
                        start_col: str = "start_time",
                        end_col: str = "end_time",
                        night_start_hour: int = 22,
                        night_end_hour: int = 7,
                        legacy_accounting: bool = False) -> pd.DataFrame:
    """Hours of daytime and nighttime within each [start, end] interval.

    legacy_accounting reproduces the reference implementation, which omits
    the final morning night hours of multi-day stays ending at or after the
    night start hour.
    """
    day = (_cumulative_day_hours(df[end_col], night_start_hour, night_end_hour)
           - _cumulative_day_hours(df[start_col], night_start_hour, night_end_hour))
    total = (df[end_col] - df[start_col]).dt.total_seconds().to_numpy() / 3600.0
    night = total - day
    if legacy_accounting:
        multi_day = (df[start_col].dt.normalize().to_numpy()
                     != df[end_col].dt.normalize().to_numpy())
        ends_in_night = (df[end_col].dt.hour >= night_start_hour).to_numpy()
        night = night - np.where(multi_day & ends_in_night, float(night_end_hour), 0.0)
    return pd.DataFrame({"day_duration": day, "night_duration": night}, index=df.index)


def _group_keys(df: pd.DataFrame) -> list:
    """Users are scored separately per period when a 'period' column exists."""
    return ["user_id", "period"] if "period" in df.columns else ["user_id"]


def aggregate_durations_per_location(stays: pd.DataFrame) -> pd.DataFrame:
    """Total duration/day/night per (user[, period], stay location), attached
    back to every stay.

    The stay location is identified by its ordered location-id list; repeated
    visits to the same location share one aggregate.
    """
    df = stays.copy()
    df["_loc"] = df["location_ids"].astype(str)
    keys = _group_keys(df) + ["_loc"]
    agg = (
        df.groupby(keys)[["duration", "day_duration", "night_duration"]]
        .sum()
        .rename(columns=lambda c: f"{c}_aggregated")
        .reset_index()
    )
    return df.merge(agg, on=keys, how="left").drop(columns="_loc")


def add_relevance(df: pd.DataFrame) -> pd.DataFrame:
    """Relevance % of each row = its aggregate share of the user's total
    (per period when a 'period' column exists).

    Adds total_relevance / day_relevance / night_relevance. The denominator
    is the sum of the *_aggregated columns over the user's rows, so at stay
    level a location visited by several stays contributes once per stay; at
    activity-space level (one row per space) the scores are exact shares.
    """
    df = df.copy()
    keys = _group_keys(df)
    for col, name in [("duration_aggregated", "total_relevance"),
                      ("day_duration_aggregated", "day_relevance"),
                      ("night_duration_aggregated", "night_relevance")]:
        totals = df.groupby(keys)[col].transform("sum")
        df[name] = df[col] / totals * 100.0
    return df


def prepare_stays_with_relevance(stays: pd.DataFrame,
                                 night_start_hour: int = 22,
                                 night_end_hour: int = 7,
                                 legacy_accounting: bool = False) -> pd.DataFrame:
    """day/night split -> per-location aggregates -> relevance scores."""
    dn = day_night_durations(stays, night_start_hour=night_start_hour,
                             night_end_hour=night_end_hour,
                             legacy_accounting=legacy_accounting)
    out = pd.concat([dn, stays], axis=1)
    out = aggregate_durations_per_location(out)
    return add_relevance(out)
