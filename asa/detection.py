"""Displacement detection and origin/destination attribution.

The daily familiar/unfamiliar trajectories are scanned with the segment-based
migration detector of Chi et al. (2020, PLoS ONE): contiguous segments of the
same location value are identified (allowing small gaps), and a migration
event is a transition between two stable segments. In ASA the two "locations"
are familiar (1) and unfamiliar (0) space, so an event leaving familiar space
after the disaster is a displacement.

The detector itself is an external package (``migration_detector``); this
module builds its input directly from an in-memory trajectory table, labels
the detected events, and attaches the stay locations of each displaced
person's origin and destination periods.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .params import ASAParams


def _import_migration_detector(extra_paths: list | None = None):
    for p in extra_paths or []:
        if p not in sys.path:
            sys.path.insert(0, p)
    from tqdm import tqdm

    tqdm.pandas()  # the detector uses progress_apply
    import migration_detector as md

    return md


def trajectory_from_frame(binary_df: pd.DataFrame, detector_paths: list | None = None):
    """Build a detector TrajRecord from a (date, user_id, location) table."""
    md = _import_migration_detector(detector_paths)
    df = binary_df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["date"] = df["date"].astype(int)

    start = pd.to_datetime(str(df["date"].min()), format="%Y%m%d")
    end = pd.to_datetime(str(df["date"].max()), format="%Y%m%d")
    all_dates = [int(d.strftime("%Y%m%d")) for d in pd.date_range(start, end)]
    date2index = dict(zip(all_dates, range(len(all_dates))))
    index2date = dict(zip(range(len(all_dates)), all_dates))
    long_dates = [int(d.strftime("%Y%m%d"))
                  for d in pd.date_range(start, end + pd.Timedelta(days=200))]
    date_num_long = pd.DataFrame({"date": long_dates,
                                  "date_num": range(len(long_dates))})

    df["date_num"] = df["date"].map(date2index)
    agg = df.groupby(["user_id", "location"])["date_num"].agg(list).reset_index()
    user_loc_agg = agg.groupby("user_id").apply(
        lambda x: dict(zip(x["location"], x["date_num"]))
    ).reset_index(name="all_record")
    return md.TrajRecord(user_loc_agg, df, index2date, date_num_long)


def find_migrants(binary_df: pd.DataFrame, params: ASAParams,
                  detector_paths: list | None = None) -> pd.DataFrame:
    """Run the segment detector on a binary trajectory table."""
    traj = trajectory_from_frame(binary_df, detector_paths)
    return traj.find_migrants(
        num_stayed_days_migrant=params.k_days,
        num_days_missing_gap=params.epsilon_days,
        small_seg_len=params.small_seg_len,
        seg_prop=params.seg_prop,
        min_overlap_part_len=params.min_overlap_part_len,
        max_gap_home_des=params.max_gap_home_des,
    )


def classify_events(migrants: pd.DataFrame, disaster_date: str) -> pd.DataFrame:
    """Label each detected event.

    home==1 & destination==0 : leaving familiar space — a displacement if it
    happens on/after the disaster date, otherwise ordinary migration.
    home==0 & destination==1 : returning to familiar space — a
    return_displacement when the user was displaced before, else
    return_migration. One-hot movement_type_* columns are appended.
    """
    df = migrants.sort_values(["user_id", "migration_date"]).copy()
    df["migration_date"] = pd.to_datetime(df["migration_date"], format="%Y%m%d")
    disaster = pd.to_datetime(disaster_date, format="%Y%m%d")

    leaving = (df["home"] == 1) & (df["destination"] == 0)
    returning = (df["home"] == 0) & (df["destination"] == 1)

    df["movement_type"] = None
    df.loc[leaving & (df["migration_date"] < disaster), "movement_type"] = "migration"
    df.loc[leaving & (df["migration_date"] >= disaster), "movement_type"] = "displacement"

    is_disp = (df["movement_type"] == "displacement").astype(int)
    disp_before = is_disp.groupby(df["user_id"]).cumsum() - is_disp
    df.loc[returning & (disp_before > 0), "movement_type"] = "return_displacement"
    df.loc[returning & (disp_before == 0), "movement_type"] = "return_migration"

    df["migration_date"] = df["migration_date"].dt.strftime("%Y%m%d")
    dummies = pd.get_dummies(df["movement_type"], prefix="movement_type").astype(int)
    return pd.concat([df.drop(columns="movement_type"), dummies], axis=1)


def match_stays_with_events(labeled: pd.DataFrame, stays: pd.DataFrame,
                            match_type: str = "origin") -> pd.DataFrame:
    """Stays overlapping each event's origin (home) or destination period."""
    if match_type == "origin":
        start_col, end_col = "home_start_date", "home_end_date"
    elif match_type == "destination":
        start_col, end_col = "destination_start_date", "destination_end_date"
    else:
        raise ValueError("match_type must be 'origin' or 'destination'")

    lm = labeled.copy()
    lm[start_col] = pd.to_datetime(lm[start_col], format="%Y%m%d")
    lm[end_col] = pd.to_datetime(lm[end_col], format="%Y%m%d")
    lm["user_id"] = lm["user_id"].astype(np.int64)

    st = stays.copy()
    st["start_time"] = pd.to_datetime(st["start_time"])
    st["end_time"] = pd.to_datetime(st["end_time"])

    merged = lm.merge(st, on="user_id", how="inner")
    mask = ((merged["start_time"] <= merged[end_col])
            & (merged["end_time"] >= merged[start_col]))
    return merged[mask].reset_index(drop=True)


def weighted_midpoints(stays: pd.DataFrame, relevance_threshold: float,
                       side: str = "destination") -> pd.DataFrame:
    """Night-duration-weighted centroid of a user's origin or destination stays.

    origin: familiar stays (night_relevance > threshold);
    destination: unfamiliar stays (habitual_night_relevance < threshold).
    """
    import shapely
    from shapely.geometry import Point

    if side == "destination":
        sub = stays[stays["habitual_night_relevance"] < relevance_threshold].copy()
    else:
        sub = stays[stays["night_relevance"] > relevance_threshold].copy()

    cent = shapely.centroid(sub["geometry"].values)
    sub["_cx"] = shapely.get_x(cent)
    sub["_cy"] = shapely.get_y(cent)

    g = sub.groupby("user_id")
    wsum = g["night_duration"].sum()
    wx = g.apply(lambda d: (d["_cx"] * d["night_duration"]).sum())
    wy = g.apply(lambda d: (d["_cy"] * d["night_duration"]).sum())
    ok = wsum > 0
    return pd.DataFrame({
        "user_id": wsum.index[ok],
        f"{side}_midpoint": [Point(a / c, b / c)
                             for a, b, c in zip(wx[ok], wy[ok], wsum[ok])],
    }).reset_index(drop=True)


def midpoint_distances(origin_midpoints: pd.DataFrame,
                       destination_midpoints: pd.DataFrame) -> pd.DataFrame:
    """Displacement distance = distance between the two weighted midpoints."""
    import shapely

    df = origin_midpoints.merge(destination_midpoints, on="user_id", how="inner")
    df["distance"] = shapely.distance(df["origin_midpoint"].values,
                                      df["destination_midpoint"].values)
    return df
