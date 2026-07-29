"""Spatial context indices and per-person origin/destination features.

Cell-level indices (urbanization, infrastructure damage, border proximity,
city population) are transferred onto stay polygons by area/density-weighted
intersection, then summarized per displaced person as nighttime-duration-
weighted means over the origin (familiar) and destination (unfamiliar) stays:

    index(person, side) = sum_s index_s * night_duration_s / sum_s night_duration_s

The destination-minus-origin differences of these indices describe how the
person's living space changed with the displacement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

DAMAGE_WEIGHTS = {
    "collapsed_buildings": 1.0,
    "needs_demolished_buildings": 0.8,
    "heavily_damaged_buildings": 0.7,
    "slightly_damaged_buildings": 0.3,
}
URBANIZATION_WEIGHTS = {
    "DENSE URBAN": 1.0,
    "URBAN": 0.8,
    "CITY SEASONAL": 0.8,
    "SUBURBAN": 0.4,
    "CITY SUBURBAN": 0.4,
    "SEASONAL": 0.4,
    "RURAL": 0.0,
}
URBANIZATION_CATEGORIES = list(URBANIZATION_WEIGHTS) + ["Unknown"]


# --------------------------------------------------------------------------- #
# Cell-level indices
# --------------------------------------------------------------------------- #
def urbanization_shares_per_cell(cells: gpd.GeoDataFrame,
                                 site_context: pd.DataFrame,
                                 site_areas: gpd.GeoDataFrame) -> pd.DataFrame:
    """Area-weighted urbanization category shares (%) per cell.

    cells:        [location_id, geometry] service areas used in the analysis
    site_context: [site_id, context] operator urbanization label per site
    site_areas:   [site_id, geometry] raw per-site service areas
    """
    towers = site_areas.merge(site_context, on="site_id", how="left")
    towers["context"] = towers["context"].fillna("Unknown")
    overlay = gpd.overlay(cells[["location_id", "geometry"]], towers, how="intersection")
    overlay["_area"] = overlay.geometry.area

    dummies = pd.get_dummies(overlay["context"]).mul(overlay["_area"], axis=0)
    dummies["location_id"] = overlay["location_id"].to_numpy()
    shares = dummies.groupby("location_id").sum()
    total = overlay.groupby("location_id")["_area"].sum()
    return (shares.div(total, axis=0) * 100.0).reset_index()


def urbanization_index(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted urbanization index in [0, 1] from category shares."""
    cols = [c for c in URBANIZATION_CATEGORIES if c in df.columns]
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    valid = vals[[c for c in cols if c != "Unknown"]].sum(axis=1)
    weighted = sum(vals[c] * w for c, w in URBANIZATION_WEIGHTS.items() if c in vals)
    out = df.copy()
    out["urbanization_index"] = weighted / valid
    return out


def damage_counts_per_cell(damage_files: dict, cells: gpd.GeoDataFrame,
                           crs: str) -> pd.DataFrame:
    """Damaged-building point counts per cell and damage category.

    damage_files maps category name -> point file path (categories must match
    DAMAGE_WEIGHTS keys without the '_buildings' suffix).
    """
    out = cells[["location_id"]].copy()
    for category, path in damage_files.items():
        pts = gpd.read_file(path).set_crs("EPSG:4326", allow_override=True).to_crs(crs)
        joined = gpd.sjoin(pts, cells[["location_id", "geometry"]],
                           how="inner", predicate="within")
        counts = joined.groupby("location_id").size()
        out[f"{category}_buildings"] = out["location_id"].map(counts).fillna(0)
    return out


def damage_index(df: pd.DataFrame, area: pd.Series | None = None) -> pd.DataFrame:
    """Severity-weighted damage measure; optionally density-normalized to [0,1]."""
    cols = list(DAMAGE_WEIGHTS)
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    weighted = sum(vals[c] * w for c, w in DAMAGE_WEIGHTS.items())
    out = df.copy()
    out["damage_index"] = np.where(vals.sum(axis=1) > 0, weighted, 0.0)
    if area is not None:
        out["damage_index"] = out["damage_index"] / area
        lo, hi = out["damage_index"].min(), out["damage_index"].max()
        out["damage_index"] = (out["damage_index"] - lo) / (hi - lo)
    return out


def border_flag(gdf: gpd.GeoDataFrame, border_line, buffer_m: float = 20_000) -> pd.Series:
    """1 when the geometry lies within ``buffer_m`` of the given border line."""
    zone = border_line.buffer(buffer_m)
    return gdf.geometry.intersects(zone).astype(int)


# --------------------------------------------------------------------------- #
# Transfer to stay polygons
# --------------------------------------------------------------------------- #
def transfer_to_stays(stay_polygons: gpd.GeoDataFrame,
                      cell_features: gpd.GeoDataFrame,
                      count_cols: list,
                      percentage_cols: list,
                      dummy_cols: list) -> pd.DataFrame:
    """Aggregate cell features onto stay polygons through their intersections.

    counts      -> cell density x intersected area, summed per stay
    percentages -> intersected-area-weighted mean, renormalized to 100
    dummies     -> max
    """
    cells = cell_features.copy()
    cells["_cell_area"] = cells.geometry.area

    stays = stay_polygons[["user_id", "stay_id", "geometry"]].copy()
    stays["stay_area"] = stays.geometry.area

    inter = gpd.overlay(stays, cells, how="intersection", keep_geom_type=False)
    inter["_ia"] = inter.geometry.area
    inter = inter[inter["_ia"] >= 1e-6]

    keys = ["user_id", "stay_id"]
    out = inter.groupby(keys).agg(stay_area=("stay_area", "first"))
    for col in dummy_cols:
        out[col] = inter.groupby(keys)[col].max()
    for col in count_cols:
        dens = inter[col] / inter["_cell_area"]
        out[col] = (dens * inter["_ia"]).groupby([inter[k] for k in keys]).sum()
    weighted = pd.DataFrame({
        c: (inter[c] * inter["_ia"]).groupby([inter[k] for k in keys]).sum()
        for c in percentage_cols
    })
    total = weighted.sum(axis=1)
    for col in percentage_cols:
        out[col] = weighted[col] / total * 100.0
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Per-person weighted features and differences
# --------------------------------------------------------------------------- #
def weight_indices(stay_features: pd.DataFrame,
                   side: str = "origin",
                   relevance_threshold: float = 5.0,
                   index_cols: tuple = ("syria_border", "city_type",
                                        "urbanization_index", "damage_index")
                   ) -> pd.DataFrame:
    """Nighttime-duration-weighted mean of each index over the relevant stays.

    origin: familiar stays (night_relevance > threshold);
    destination: unfamiliar stays (habitual_night_relevance < threshold).
    Falls back to daytime weights for users with zero night duration.
    """
    rel_col = "night_relevance" if side == "origin" else "habitual_night_relevance"
    df = stay_features.copy()
    if side == "origin":
        df = df[df[rel_col] > relevance_threshold]
    else:
        df = df[df[rel_col] < relevance_threshold]

    def _one(group: pd.DataFrame) -> pd.Series:
        w = group["night_duration"]
        if w.sum() == 0:
            w = group["day_duration"]
        total = w.sum()
        weights = w / total if total > 0 else 0
        return pd.Series({f"weighted_{c}": (group[c] * weights).sum()
                          for c in index_cols if c in group.columns})

    return df.groupby("user_id").apply(_one).reset_index()


def origin_destination_differences(df: pd.DataFrame) -> pd.DataFrame:
    """destination - origin for every weighted index (columns *_diff)."""
    out = df.copy()
    for var in ["weighted_syria_border", "weighted_city_type",
                "weighted_urbanization_index", "weighted_damage_index"]:
        o, d = f"{var}_origin", f"{var}_dest"
        if o in df.columns and d in df.columns:
            out[f"{var}_diff"] = df[d] - df[o]
    return out
