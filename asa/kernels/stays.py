"""Stay-location detection kernel.

A *stay location* is a period during which a user's signals remain within a
spatial threshold D of both the previous signal and the first signal of the
stay. A stay is *established* once at least T seconds have elapsed since its
first signal; a signal arriving earlier restarts the stay (zero-length stays
are discarded). Once established, a stay continues for every subsequent
in-range signal — there is no maximum gap.

Each stay records the ordered set of location ids (towers/clusters) observed
during it; the stay's footprint polygon is the convex hull of those
locations' service areas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


def _stay_breaks(times_s: np.ndarray, x: np.ndarray, y: np.ndarray,
                 dist_m: float, dur_s: float) -> np.ndarray:
    """Boolean array marking the first signal of each stay."""
    n = len(times_s)
    breaks = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return breaks
    breaks[0] = True
    start_i = 0
    for i in range(1, n):
        dxp = x[i] - x[i - 1]
        dyp = y[i] - y[i - 1]
        d_prev = (dxp * dxp + dyp * dyp) ** 0.5
        dxs = x[i] - x[start_i]
        dys = y[i] - y[start_i]
        d_start = (dxs * dxs + dys * dys) ** 0.5
        elapsed = times_s[i] - times_s[start_i]
        if d_prev <= dist_m and d_start <= dist_m and elapsed >= dur_s:
            continue
        breaks[i] = True
        start_i = i
    return breaks


if _HAS_NUMBA:  # pragma: no cover
    _stay_breaks = njit(cache=True)(_stay_breaks)


def detect_stays(signals: pd.DataFrame,
                 distance_threshold_m: float,
                 duration_threshold_s: float) -> pd.DataFrame:
    """Detect stays for every user in ``signals``.

    Parameters
    ----------
    signals : DataFrame [user_id, time, location_id, x, y]
        x/y in a metric CRS. May contain several users.

    Returns
    -------
    DataFrame [user_id, start_time, end_time, duration (hours),
               location_ids (list), stay_id]
    where stay_id is the per-user sequence number of the stay.
    """
    df = signals.sort_values(["user_id", "time"], kind="mergesort").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["user_id", "start_time", "end_time",
                                     "duration", "location_ids", "stay_id"])

    times_s = df["time"].values.astype("datetime64[s]").astype(np.int64)
    x = df["x"].to_numpy(np.float64)
    y = df["y"].to_numpy(np.float64)

    breaks = np.zeros(len(df), dtype=bool)
    users = df["user_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(df)]
    for s, e in zip(starts, ends):
        breaks[s:e] = _stay_breaks(times_s[s:e], x[s:e], y[s:e],
                                   float(distance_threshold_m),
                                   float(duration_threshold_s))

    df["_stay"] = np.cumsum(breaks) - 1
    grouped = df.groupby("_stay", sort=True)
    stays = grouped.agg(
        user_id=("user_id", "first"),
        start_time=("time", "first"),
        end_time=("time", "last"),
    )
    stays["location_ids"] = grouped["location_id"].agg(lambda s: list(dict.fromkeys(s)))
    stays["duration"] = (stays["end_time"] - stays["start_time"]).dt.total_seconds() / 3600.0

    stays = stays[stays["duration"] != 0].reset_index(drop=True)
    stays["stay_id"] = stays.groupby("user_id").cumcount()
    return stays


def split_pre_post(stays: pd.DataFrame, split_ts: pd.Timestamp) -> pd.DataFrame:
    """Assign each stay to the pre- and/or post-disaster period.

    A stay overlapping the split instant contributes to both periods with its
    times clipped; durations are recomputed. Adds a ``period`` column with
    values 'pre'/'post'.
    """
    pre = stays[stays["start_time"] < split_ts].copy()
    pre["end_time"] = pre["end_time"].clip(upper=split_ts)
    pre["period"] = "pre"

    post = stays[stays["end_time"] > split_ts].copy()
    post["start_time"] = post["start_time"].clip(lower=split_ts)
    post["period"] = "post"

    out = pd.concat([pre, post], ignore_index=True)
    out["duration"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 3600.0
    return out
