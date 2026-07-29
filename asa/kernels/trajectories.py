"""Daily familiarity trajectories.

For every user and calendar day, the trajectory value is the mean familiarity
score of the stays present that day — nighttime stays if any (night hours in
the 22:00-07:00 window), otherwise daytime stays. Binarizing the value at the
relevance threshold yields the familiar/unfamiliar location sequence consumed
by the segment-based displacement detector.

Hour counts are inclusive of both interval endpoints and computed with
closed-form cumulative tick counts (exact for hour-aligned timestamps, which
is the native resolution of hourly-binned CDR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_NS_PER_HOUR = 3_600_000_000_000
_NS_PER_DAY = 24 * _NS_PER_HOUR


def _cum_night_ticks(ns: np.ndarray) -> np.ndarray:
    hours = ns // _NS_PER_HOUR
    days = hours // 24
    h = hours - days * 24
    return 9 * days + np.minimum(h + 1, 7) + np.maximum(h - 21, 0)


def _cum_day_ticks(ns: np.ndarray) -> np.ndarray:
    hours = ns // _NS_PER_HOUR
    days = hours // 24
    h = hours - days * 24
    return 15 * days + np.clip(h - 6, 0, 15)


def daily_relevance_series(stay_relevance: pd.DataFrame) -> pd.DataFrame:
    """Daily familiarity per user.

    Parameters
    ----------
    stay_relevance : DataFrame
        [user_id, stay_id, start_time, end_time, habitual_night_relevance_*]
        (one familiarity column per DBSCAN threshold).

    Returns
    -------
    DataFrame [user_id, date, location_1, location_2, ...] where location_i
    is the daily mean familiarity for the i-th threshold column.
    """
    df = stay_relevance.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df = df.dropna(subset=["start_time", "end_time"]).reset_index(drop=True)

    rel_cols = [c for c in df.columns if c.startswith("habitual_night_relevance_")]
    if not rel_cols:
        raise ValueError("no habitual_night_relevance_* columns found")
    if df.empty:
        return pd.DataFrame(columns=["user_id", "date"]
                            + [f"location_{i+1}" for i in range(len(rel_cols))])

    start_ns = df["start_time"].values.astype("datetime64[ns]").astype(np.int64)
    end_ns = df["end_time"].values.astype("datetime64[ns]").astype(np.int64)

    # expand each stay to the days it overlaps (start < day+1d AND end > day)
    first_day = start_ns // _NS_PER_DAY
    last_day = np.maximum((end_ns - 1) // _NS_PER_DAY, first_day)
    n_days = (last_day - first_day + 1).astype(np.int64)

    idx = np.repeat(np.arange(len(df)), n_days)
    offsets = np.concatenate([np.arange(n) for n in n_days])
    day = (first_day[idx] + offsets) * _NS_PER_DAY

    a = np.maximum(start_ns[idx], day)
    b = np.minimum(end_ns[idx], day + _NS_PER_DAY)
    keep = a < b
    idx, day, a, b = idx[keep], day[keep], a[keep], b[keep]

    night_hours = _cum_night_ticks(b) - _cum_night_ticks(a - _NS_PER_HOUR)
    day_hours = _cum_day_ticks(b) - _cum_day_ticks(a - _NS_PER_HOUR)

    expanded = pd.DataFrame({
        "user_id": df["user_id"].to_numpy()[idx],
        "date": day.astype("datetime64[ns]"),
        "is_night": night_hours > 0,
        "is_day": day_hours > 0,
    })
    for c in rel_cols:
        expanded[c] = df[c].to_numpy()[idx]

    grp = ["user_id", "date"]
    any_night = expanded.groupby(grp)["is_night"].transform("any")
    selected = expanded[expanded["is_night"] | (~any_night & expanded["is_day"])]

    result = selected.groupby(grp)[rel_cols].mean().reset_index()
    result = result.rename(columns={c: f"location_{i+1}" for i, c in enumerate(rel_cols)})
    result["date"] = result["date"].dt.date
    return result.sort_values(["user_id", "date"]).reset_index(drop=True)


def binarize_trajectories(daily: pd.DataFrame, location_col: str,
                          relevance_threshold: float) -> pd.DataFrame:
    """Familiar/unfamiliar daily sequence for the displacement detector.

    location = 0 (unfamiliar) if the daily familiarity is at or below the
    threshold, else 1 (familiar). Dates are encoded YYYYMMDD.
    """
    out = daily[["date", "user_id", location_col]].copy()
    out[location_col] = np.where(out[location_col] <= relevance_threshold, 0, 1)
    out["date"] = out["date"].astype(str).str.replace("-", "", regex=False)
    return out.rename(columns={location_col: "location"})[["date", "user_id", "location"]]
