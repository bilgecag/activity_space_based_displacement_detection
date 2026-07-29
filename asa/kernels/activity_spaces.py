"""Activity spaces and the pre/post-disaster relevance transfer.

An *activity space* groups a user's nearby stay locations: DBSCAN with
min_samples=1 over the stay-polygon centroids, one convex hull per cluster.
Activity spaces computed on the pre-disaster stays describe habitual living
space; post-disaster spaces are scored by their spatial overlap with the
pre-disaster ones:

    R_post = mean over overlapping pre spaces of
             R_pre * area(pre ∩ post) / area(pre)

Every stay then inherits the score of its activity space, yielding the
per-stay familiarity used to build daily trajectories.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .relevance import add_relevance

STAY_INPUT_COLS = ["user_id", "stay_id", "geometry", "duration",
                   "day_duration", "night_duration"]


def build_activity_spaces(stays: pd.DataFrame, eps_m: float) -> pd.DataFrame:
    """One convex-hull activity space per DBSCAN cluster per user.

    ``stays.geometry`` must hold shapely polygons in a metric CRS.
    Returns [user_id, space_id, geometry, num_points, stay_id_aggregated,
    duration_aggregated, day/night_duration_aggregated, *_relevance].
    """
    import shapely

    df = stays[STAY_INPUT_COLS].copy()
    if df.empty:
        return pd.DataFrame(columns=["user_id", "space_id", "num_points",
                                     "stay_id_aggregated", "duration_aggregated",
                                     "day_duration_aggregated",
                                     "night_duration_aggregated", "geometry",
                                     "total_relevance", "day_relevance",
                                     "night_relevance"])
    cent = shapely.centroid(df["geometry"].values)
    df["_x"] = shapely.get_x(cent)
    df["_y"] = shapely.get_y(cent)
    df = df.sort_values("user_id", kind="mergesort").reset_index(drop=True)

    users = df["user_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(df)]
    coords = df[["_x", "_y"]].to_numpy()
    labels = np.empty(len(df), dtype=np.int64)
    for s, e in zip(starts, ends):
        if e - s == 1:
            labels[s:e] = 0
        else:
            labels[s:e] = DBSCAN(eps=eps_m, min_samples=1).fit(coords[s:e]).labels_
    df["space_id"] = labels

    keys = ["user_id", "space_id"]
    agg = df.groupby(keys).agg(
        num_points=("stay_id", "size"),
        stay_id_aggregated=("stay_id", list),
        duration_aggregated=("duration", "sum"),
        day_duration_aggregated=("day_duration", "sum"),
        night_duration_aggregated=("night_duration", "sum"),
    )
    unions = df.groupby(keys)["geometry"].agg(
        lambda g: shapely.convex_hull(shapely.union_all(list(g))))
    agg["geometry"] = unions
    return add_relevance(agg.reset_index())


def overlay_pre_post(origin_spaces: pd.DataFrame,
                     destination_spaces: pd.DataFrame) -> pd.DataFrame:
    """Score every post space by its overlap with the user's pre spaces."""
    import shapely

    pre = origin_spaces[["user_id", "space_id", "geometry",
                         "stay_id_aggregated", "night_relevance"]]
    post = destination_spaces[["user_id", "space_id", "geometry",
                               "stay_id_aggregated"]]
    if pre.empty or post.empty:
        return pd.DataFrame(columns=["user_id", "habitual_night_relevance",
                                     "destination_stay_ids"])
    pairs = pre.merge(post, on="user_id", suffixes=("_pre", "_post"))
    if pairs.empty:
        return pd.DataFrame(columns=["user_id", "habitual_night_relevance",
                                     "destination_stay_ids"])

    inter = shapely.area(shapely.intersection(pairs["geometry_pre"].values,
                                              pairs["geometry_post"].values))
    pairs["intersection_area"] = inter
    pairs = pairs[pairs["intersection_area"] > 0].reset_index(drop=True)
    pre_area = shapely.area(pairs["geometry_pre"].values)
    pairs["habitual_night_relevance"] = (
        pairs["night_relevance"] * pairs["intersection_area"] / pre_area
    )
    return pairs.rename(columns={"stay_id_aggregated_post": "destination_stay_ids"})[
        ["user_id", "habitual_night_relevance", "destination_stay_ids"]
    ]


def stay_relevance_for_eps(pre_stays: pd.DataFrame, post_stays: pd.DataFrame,
                           eps_m: float, label: str) -> pd.DataFrame:
    """Per-stay familiarity score for one DBSCAN threshold.

    Pre-disaster stays inherit their space's night relevance; post-disaster
    stays average the overlap scores of the spaces containing them
    (0 when a post space overlaps no pre space).
    Returns [user_id, stay_id, habitual_night_relevance_<label>].
    """
    col = f"habitual_night_relevance_{label}"
    if pre_stays.empty and post_stays.empty:
        return pd.DataFrame(columns=["user_id", "stay_id", col])

    origin_spaces = build_activity_spaces(pre_stays, eps_m)
    destination_spaces = build_activity_spaces(post_stays, eps_m)
    overlay = overlay_pre_post(origin_spaces, destination_spaces)

    o = pre_stays[["user_id", "stay_id"]].merge(
        origin_spaces[["user_id", "stay_id_aggregated", "night_relevance"]]
        .explode("stay_id_aggregated")
        .rename(columns={"stay_id_aggregated": "stay_id"}),
        on=["user_id", "stay_id"], how="left",
    ).rename(columns={"night_relevance": col})

    t = post_stays[["user_id", "stay_id"]].merge(
        overlay.explode("destination_stay_ids")
        .rename(columns={"destination_stay_ids": "stay_id"}),
        on=["user_id", "stay_id"], how="left",
    )
    t["habitual_night_relevance"] = t["habitual_night_relevance"].fillna(0)
    t = (t.groupby(["user_id", "stay_id"])["habitual_night_relevance"]
         .mean().reset_index().rename(columns={"habitual_night_relevance": col}))

    return pd.concat([o, t], ignore_index=True)


def habitual_region_share(pre_stays: pd.DataFrame, region_geom,
                          eps_m: float) -> pd.DataFrame:
    """Share of each user's pre-disaster habitual space inside a region.

    Builds the pre-disaster activity spaces and sums the night relevance of
    those intersecting ``region_geom``. Users with a share above ~90% can be
    treated as residents of the region (excluding tourists and transit
    visitors from the displacement analysis).

    Returns [user_id, region_night_relevance].
    """
    import shapely

    spaces = build_activity_spaces(pre_stays, eps_m)
    if spaces.empty:
        return pd.DataFrame(columns=["user_id", "region_night_relevance"])
    hits = shapely.intersects(spaces["geometry"].values, region_geom)
    out = (spaces[hits].groupby("user_id")["night_relevance"].sum()
           .rename("region_night_relevance").reset_index())
    return out


def stay_relevance_table(split_stays: pd.DataFrame,
                         eps_list_m: list,
                         stay_times: pd.DataFrame) -> pd.DataFrame:
    """Per-stay familiarity for several DBSCAN thresholds, plus stay times.

    Parameters
    ----------
    split_stays : output of the relevance stage, with a 'period' column
        ('pre'/'post'), shapely geometry, durations and relevance scores.
    stay_times : [user_id, stay_id, start_time, end_time] — the original
        (unclipped) stay intervals used for the daily trajectories.
    """
    pre = split_stays[split_stays["period"] == "pre"].reset_index(drop=True)
    post = split_stays[split_stays["period"] == "post"].reset_index(drop=True)

    out = None
    for eps in eps_list_m:
        label = f"{int(eps) // 1000}km"
        tbl = stay_relevance_for_eps(pre, post, float(eps), label)
        out = tbl if out is None else out.merge(tbl, on=["user_id", "stay_id"])
    return out.merge(stay_times, on=["user_id", "stay_id"], how="left")
