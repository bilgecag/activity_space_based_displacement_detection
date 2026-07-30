"""Post-processing: publishable indicator datasets on the H3 grid (res 6).

Builds, from the pipeline outputs and the raw context data:

1. Displaced-people counts per H3 resolution-6 cell, separately for the
   origin and the destination side and for the Syrian and Turkish groups.
   Pre-disaster stay areas of a displaced person are their origin,
   post-disaster stay areas their destination. A stay location is not a
   person, so each observed displaced person contributes one unit of mass
   distributed over the H3 cells of their stay centroids proportionally to
   the nights they spent in each (their night-spent share). Cell values
   start as the sums of these per-person shares: the spatial distribution
   of the actual stay locations.

   Publication happens in three steps, separately per group and side:

   a. Primary k-anonymity (k = 10): only cells to which at least ten
      distinct observed people of the group contribute are kept.
   b. The paper's extrapolated population estimates are distributed over
      the protected cells proportionally to their observed mass —
      860,000 Turkish DPs (560,000 to other cities + 300,000 within
      city boundaries) and 70,000 Syrian DPs (55,000 + 15,000) — and
      rounded to whole people. Published values are therefore estimated
      displaced-population counts, not observed device counts.
   c. Secondary cut: any cell whose extrapolated count is below ten is
      removed, so no number below ten is ever reported.

   The extrapolated total, the published total, and the suppressed
   remainder of every file are reported in the metadata file.

2. The urbanization and infrastructure-damage indices of the paper,
   aggregated to the same H3 resolution-6 cells (intersection-area-weighted
   mean for urbanization; severity-weighted damaged-building density,
   min-max normalized, for damage).

Outputs (written to <release>/indicator_dataset/, GeoParquet in EPSG:4326
with the hexagon polygon of every cell, ready for direct use in QGIS):
    displaced_persons_origin_h3r6_syrian.parquet
    displaced_persons_origin_h3r6_turkish.parquet
    displaced_persons_destination_h3r6_syrian.parquet
    displaced_persons_destination_h3r6_turkish.parquet
    urbanization_index_h3r6.parquet
    damage_index_h3r6.parquet
    metadata.json

Usage: python build_h3_indicators.py [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb as swkb
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import local_config as cfg  # noqa: E402
from asa import indices as idx  # noqa: E402

RUN = cfg.OUTPUT_DIR
H3_RESOLUTION = 6
K_ANONYMITY = 10

# Extrapolated displaced-population estimates declared in the publication:
# Turkish 560,000 to other cities + 300,000 within city boundaries;
# Syrian 55,000 to other cities + 15,000 within city boundaries.
EXTRAPOLATED_TOTALS = {"turkish": 560_000 + 300_000, "syrian": 55_000 + 15_000}
SEGMENTS = {"turkish": cfg.SEGMENT_TURKISH, "syrian": cfg.SEGMENT_SYRIAN}


# --------------------------------------------------------------------------- #
# h3 compatibility (v3 / v4 APIs)
# --------------------------------------------------------------------------- #
import h3  # noqa: E402

if hasattr(h3, "latlng_to_cell"):        # h3 v4
    def latlng_to_cell(lat, lng, res):
        return h3.latlng_to_cell(lat, lng, res)

    def cell_boundary(cell):
        return h3.cell_to_boundary(cell)  # ((lat, lng), ...)

    def neighbours(cell):
        return h3.grid_disk(cell, 1)
else:                                     # h3 v3
    def latlng_to_cell(lat, lng, res):
        return h3.geo_to_h3(lat, lng, res)

    def cell_boundary(cell):
        return h3.h3_to_geo_boundary(cell)

    def neighbours(cell):
        return h3.k_ring(cell, 1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hex_polygon(cell):
    """Shapely polygon of an H3 cell in geographic coordinates."""
    return Polygon([(lng, lat) for lat, lng in cell_boundary(cell)])


def to_geoparquet(df: pd.DataFrame, path: str):
    """Attach hexagon geometries and write a GeoParquet file."""
    geoms = [hex_polygon(c) for c in df["h3_index"]]
    gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326").to_parquet(path)


# --------------------------------------------------------------------------- #
# Displaced-people counts per cell
# --------------------------------------------------------------------------- #
def load_side(name: str) -> gpd.GeoDataFrame:
    df = pd.read_parquet(os.path.join(RUN, name))
    df = df.drop_duplicates(["user_id", "stay_id"]).reset_index(drop=True)
    return gpd.GeoDataFrame(df.drop(columns=["geometry_wkb"]),
                            geometry=[swkb.loads(b) for b in df["geometry_wkb"]],
                            crs=cfg.METRIC_CRS)


def stay_allocation(stays: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per-person unit mass split over stay cells by night-spent share.

    Weight of a stay = the person's nighttime hours there divided by their
    total nighttime hours. People with no nighttime hours at all fall back
    to total-duration shares, then to uniform shares.
    """
    sub = stays.copy()
    w = sub["night_duration_aggregated"].clip(lower=0).astype(float)
    totals = w.groupby(sub["user_id"]).transform("sum")
    fb = sub["duration_aggregated"].clip(lower=0).astype(float)
    fb_totals = fb.groupby(sub["user_id"]).transform("sum")
    sub["weight"] = np.where(totals > 0, w / totals,
                             np.where(fb_totals > 0, fb / fb_totals, np.nan))
    sub["weight"] = sub["weight"].fillna(
        1.0 / sub.groupby("user_id")["stay_id"].transform("count"))

    cent = gpd.GeoSeries(sub.geometry.centroid, crs=cfg.METRIC_CRS).to_crs("EPSG:4326")
    sub["h3_index"] = [latlng_to_cell(p.y, p.x, H3_RESOLUTION) for p in cent]
    return sub[["user_id", "h3_index", "weight"]]


def dp_counts_per_cell(alloc: pd.DataFrame, users: set,
                       extrapolated_total: int) -> tuple:
    """Extrapolated DP counts per H3 cell, k-anonymized in two steps."""
    sub = alloc[alloc["user_id"].isin(users)]
    mass = sub.groupby("h3_index")["weight"].sum()
    contributors = sub.groupby("h3_index")["user_id"].nunique()

    # step a: primary k-anonymity — keep cells with >= k distinct observed
    # contributing people
    protected = mass[contributors >= K_ANONYMITY]

    # step b: distribute the paper's extrapolated population total over the
    # protected cells proportionally to their observed mass; largest-remainder
    # rounding keeps the pre-cut sum exactly equal to the total
    factor = extrapolated_total / protected.sum()
    exact = protected * factor
    scaled = np.floor(exact).astype(int)
    short = extrapolated_total - int(scaled.sum())
    for cell in (exact - scaled).sort_values(ascending=False).index[:short]:
        scaled[cell] += 1

    # step c: secondary cut — never report a count below k
    published = scaled[scaled >= K_ANONYMITY].sort_values(ascending=False)

    out = published.rename("dp_count").reset_index()
    meta = {
        "users_in_run": int(sub["user_id"].nunique()),
        "extrapolated_total": extrapolated_total,
        "people_per_observed_unit": round(float(factor), 2),
        "published_total": int(published.sum()),
        "suppressed_total": extrapolated_total - int(published.sum()),
        "suppressed_cells": int(len(mass) - len(published)),
        "published_cells": int(len(published)),
    }
    return out, meta


# --------------------------------------------------------------------------- #
# Indices per cell
# --------------------------------------------------------------------------- #
def h3_grid_over(cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """H3 res-6 hexagons covering the given service areas."""
    lonlat = gpd.GeoSeries(cells.geometry.centroid, crs=cfg.METRIC_CRS).to_crs("EPSG:4326")
    seeds = {latlng_to_cell(p.y, p.x, H3_RESOLUTION) for p in lonlat}
    ring = set()
    for c in seeds:
        ring.update(neighbours(c))
    hexes = sorted(seeds | ring)
    polys = [Polygon([(lng, lat) for lat, lng in cell_boundary(c)]) for c in hexes]
    return gpd.GeoDataFrame({"h3_index": hexes}, geometry=polys,
                            crs="EPSG:4326").to_crs(cfg.METRIC_CRS)


def indices_per_cell() -> tuple:
    log("building cell-level indices from raw context data ...")
    df = pd.read_csv(cfg.CLUSTER_VORONOI_CSV, index_col=0)
    df["voronoi_geometry"] = df["voronoi_geometry"].astype(str)
    df = df[df["voronoi_geometry"] != "nan"]
    geo = gpd.GeoSeries.from_wkt(df["voronoi_geometry"], crs="EPSG:4326").to_crs(cfg.METRIC_CRS)
    cells = gpd.GeoDataFrame({"location_id": df["cluster"].astype(int).to_numpy()},
                             geometry=geo.values, crs=cfg.METRIC_CRS)
    cells = cells.drop_duplicates("location_id").reset_index(drop=True)

    site_areas = gpd.read_file(cfg.TURKCELL_VORONOI_SHP).rename(columns={"matcher": "site_id"})
    site_areas.crs = "EPSG:5636"
    site_areas = site_areas.to_crs(cfg.METRIC_CRS)[["site_id", "geometry"]]
    context = pd.read_csv(cfg.CONTEXT_CELLS_CSV)

    urb = idx.urbanization_shares_per_cell(cells, context, site_areas)
    feats = cells.merge(urb, on="location_id", how="left")
    for c in idx.URBANIZATION_CATEGORIES:
        if c not in feats.columns:
            feats[c] = 0.0
        feats[c] = feats[c].fillna(0.0)
    feats.loc[feats[idx.URBANIZATION_CATEGORIES].sum(axis=1) == 0, "Unknown"] = 100.0
    feats = idx.urbanization_index(feats)
    feats["urbanization_index"] = feats["urbanization_index"].fillna(0.0)

    dmg = idx.damage_counts_per_cell(cfg.DAMAGE_FILES, cells, cfg.METRIC_CRS)
    feats = feats.merge(dmg, on="location_id", how="left")
    count_cols = [f"{k}_buildings" for k in cfg.DAMAGE_FILES]
    feats[count_cols] = feats[count_cols].fillna(0)
    feats["damage_weighted"] = sum(
        feats[f"{k}_buildings"] * w
        for k, w in [("collapsed", 1.0), ("needs_demolished", 0.8),
                     ("heavily_damaged", 0.7), ("slightly_damaged", 0.3)])
    feats["cell_area"] = feats.geometry.area

    log("intersecting service areas with the H3 grid ...")
    hexes = h3_grid_over(cells)
    inter = gpd.overlay(hexes,
                        feats[["location_id", "urbanization_index",
                               "damage_weighted", "cell_area", "geometry"]],
                        how="intersection", keep_geom_type=False)
    inter["_ia"] = inter.geometry.area
    inter = inter[inter["_ia"] > 0]

    urban = ((inter["urbanization_index"] * inter["_ia"]).groupby(inter["h3_index"]).sum()
             / inter.groupby("h3_index")["_ia"].sum())
    urban_df = urban.rename("urbanization_index").round(4).reset_index()

    dmg_mass = (inter["damage_weighted"] / inter["cell_area"] * inter["_ia"]).groupby(
        inter["h3_index"]).sum()
    hex_area = hexes.set_index("h3_index").geometry.area
    density = (dmg_mass / hex_area.reindex(dmg_mass.index)).fillna(0)
    lo, hi = density.min(), density.max()
    damage_df = (((density - lo) / (hi - lo)).rename("damage_index")
                 .round(6).reset_index())
    return urban_df, damage_df


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # default: the indicator_dataset folder at the repository root
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "indicator_dataset"))
    args = ap.parse_args()
    if not os.path.isdir(RUN):
        sys.exit(f"pipeline outputs not found at {RUN} — run "
                 "replication/run_pipeline.py and "
                 "replication/make_replication_report.py first "
                 "(the shipped indicator_dataset/ was built from such a run)")
    os.makedirs(args.out, exist_ok=True)

    feats = pd.read_csv(os.path.join(RUN, "person_features.csv"))
    feats = feats[feats["distance"] > 10]
    metadata = {"h3_resolution": H3_RESOLUTION, "k_anonymity": K_ANONYMITY,
                "format": "GeoParquet, hexagon polygons in EPSG:4326",
                "allocation": "per-person unit mass split over stay cells by night-spent share",
                "protection": "cells with fewer than k distinct observed contributors removed first",
                "extrapolation": "the paper's estimated displaced-population totals distributed "
                                 "proportionally over the protected cells",
                "secondary_cut": "no extrapolated count below k is published",
                "datasets": {}}

    sides = {"origin": stay_allocation(load_side("origin_stays.parquet")),
             "destination": stay_allocation(load_side("destination_stays.parquet"))}
    for group, seg in SEGMENTS.items():
        users = set(feats[feats["segment"] == seg]["user_id"].astype("int64"))
        for side, alloc in sides.items():
            name = f"displaced_persons_{side}_h3r6_{group}"
            out, meta = dp_counts_per_cell(alloc, users, EXTRAPOLATED_TOTALS[group])
            to_geoparquet(out, os.path.join(args.out, f"{name}.parquet"))
            metadata["datasets"][name] = meta
            log(f"{name}: {meta['published_cells']} cells, "
                f"published {meta['published_total']:,} of "
                f"{meta['extrapolated_total']:,} extrapolated "
                f"({meta['suppressed_cells']} cells suppressed, "
                f"{meta['suppressed_total']:,} people)")

    urban_df, damage_df = indices_per_cell()
    to_geoparquet(urban_df, os.path.join(args.out, "urbanization_index_h3r6.parquet"))
    to_geoparquet(damage_df, os.path.join(args.out, "damage_index_h3r6.parquet"))
    metadata["datasets"]["urbanization_index_h3r6"] = {"cells": int(len(urban_df))}
    metadata["datasets"]["damage_index_h3r6"] = {"cells": int(len(damage_df))}

    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    log("done")
