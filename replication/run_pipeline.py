"""End-to-end ASA run for the 2023 Türkiye-Syria earthquake study.

Runs the full Spark pipeline from the raw CDR files to detected displaced
people and their origin/destination stay locations. Each stage persists its
output under OUTPUT_DIR so the run can resume after interruption:

    signals            (implicit; not persisted)
    stays.parquet
    stays_polygons.parquet
    split_stays.parquet
    region_share.parquet
    stay_relevance.parquet
    daily_series.parquet
    binary_trajectory.csv
    displaced_events.csv         (all classified events)
    displaced_users.csv          (analysis population, with segment)
    origin_stays.parquet / destination_stays.parquet

Usage:
    python run_pipeline.py [--stage all|stays|relevance|daily|detect|match]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import local_config as cfg  # noqa: E402  (same directory)
from asa.spark import (  # noqa: E402
    build_session, normalize_signals, attach_sites, filter_residents,
    detect_stays, attach_polygons, split_stays, habitual_region_share,
    stay_relevance, daily_series,
)
from asa.schemas import SiteSchema  # noqa: E402
from asa.kernels.trajectories import binarize_trajectories  # noqa: E402
from asa.detection import (  # noqa: E402
    find_migrants, classify_events, match_stays_with_events,
)

from pyspark.sql import functions as F  # noqa: E402

P = cfg.PARAMS
OUT = cfg.OUTPUT_DIR


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def path(name: str) -> str:
    return os.path.join(OUT, name)


# --------------------------------------------------------------------------- #
# Reference tables
# --------------------------------------------------------------------------- #
def load_site_table() -> pd.DataFrame:
    """site_id -> tower cluster (location_id), metric coordinates, province id."""
    import geopandas as gpd

    match = pd.read_csv(cfg.SITE_CLUSTER_MATCH, index_col=0)

    clusters = gpd.read_file(cfg.CLUSTERS_SHP).to_crs(cfg.METRIC_CRS)
    centroids = clusters.drop_duplicates("cluster").set_index("cluster")

    tower = pd.read_csv(cfg.TOWER_TXT, sep="|", header=0, encoding="ISO-8859-1")
    tower = tower.drop(columns=[c for c in tower.columns if c.startswith("Unnamed")])
    tower = tower.iloc[1:, :].apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    tower = tower.rename(columns=lambda x: x.strip()).rename(columns={"matcher": "site_id"})
    tower["site_id"] = tower["site_id"].astype(int)
    tower["city_id"] = tower.groupby("city").ngroup() + 1

    sites = match.merge(tower[["site_id", "city_id", "city"]].drop_duplicates("site_id"),
                        on="site_id", how="left")
    sites["x"] = sites["cluster"].map(centroids.geometry.x)
    sites["y"] = sites["cluster"].map(centroids.geometry.y)
    sites = sites.dropna(subset=["x", "y"]).reset_index(drop=True)

    affected = sites[sites["city_id"].isin(cfg.AFFECTED_CITY_IDS)]["city"].dropna().unique()
    log(f"affected city ids resolve to: {sorted(affected)}")
    return sites


def load_cluster_areas() -> pd.DataFrame:
    """location_id -> service-area polygon (WKB, metric CRS)."""
    import geopandas as gpd

    df = pd.read_csv(cfg.CLUSTER_VORONOI_CSV, index_col=0)
    df["voronoi_geometry"] = df["voronoi_geometry"].astype(str)
    df = df[df["voronoi_geometry"] != "nan"]
    geo = gpd.GeoSeries.from_wkt(df["voronoi_geometry"], crs="EPSG:4326").to_crs(cfg.METRIC_CRS)
    out = pd.DataFrame({"location_id": df["cluster"].astype(int).to_numpy(),
                        "area_wkb": [g.wkb for g in geo]})
    return out.drop_duplicates("location_id").reset_index(drop=True)


def affected_region_wkb() -> bytes:
    """Union of the ten affected provinces (metric CRS, WKB)."""
    import geopandas as gpd
    from shapely.ops import unary_union

    cities = gpd.read_file(cfg.CITY_MAP_SHP)
    name_col = next(c for c in ["adm1_en", "ADM1_EN", "NAME_1", "city"] if c in cities.columns)
    cities["_name"] = cities[name_col].astype(str).str.upper()
    sel = cities[cities["_name"].isin(cfg.AFFECTED_CITY_NAMES)]
    if len(sel) < len(cfg.AFFECTED_CITY_NAMES):
        log(f"WARNING: only {len(sel)} affected provinces matched in {name_col}")
    return unary_union(sel.to_crs(cfg.METRIC_CRS).geometry.values).wkb


# --------------------------------------------------------------------------- #
# Spark stages
# --------------------------------------------------------------------------- #
def build_signals(spark):
    frames = []
    for f in cfg.CDR_FILES:
        df = spark.read.option("header", True).csv(f)
        if "DAY" in df.columns:  # raw export: timestamp from day + hour bin
            df = df.withColumn("time", F.to_timestamp(
                F.concat_ws(" ", F.col("DAY"), F.col("HOUR")), "d/M/yyyy H"))
        else:
            df = df.withColumn("time", F.to_timestamp(cfg.CDR_SCHEMA.time))
        frames.append(df.withColumn("_source", F.lit(os.path.basename(f))))
    raw = frames[0]
    for df in frames[1:]:
        raw = raw.unionByName(df)
    signals = normalize_signals(raw, cfg.CDR_SCHEMA,
                                min_signals_per_source=cfg.MIN_SIGNALS_PER_FILE,
                                source_col="_source")
    signals = (signals
               .withColumn("user_id", F.col("user_id").cast("long"))
               .withColumn("site_id", F.col("site_id").cast("long"))
               .withColumn("group", F.col("group").cast("int"))
               .select("user_id", "time", "site_id", "group"))
    return signals


def stage_stays(spark):
    sites = load_site_table()
    site_schema = SiteSchema(site_id="site_id", location_id="cluster",
                             x="x", y="y", region_id="city_id")
    signals = build_signals(spark)
    signals = attach_sites(signals, spark.createDataFrame(
        sites[["site_id", "cluster", "x", "y", "city_id"]]), site_schema)
    signals = filter_residents(signals, cfg.AFFECTED_CITY_IDS, *cfg.RESIDENT_WINDOW)
    signals.cache()

    # one segment label per user (first observed record)
    segments = (signals.groupBy("user_id")
                .agg(F.first("group").alias("segment")))
    segments.toPandas().to_csv(path("user_segments.csv"), index=False)

    stays = detect_stays(signals.select("user_id", "time", "location_id", "x", "y"), P)
    stays.write.mode("overwrite").parquet(path("stays.parquet"))
    log(f"stays written: {spark.read.parquet(path('stays.parquet')).count():,} rows")

    areas = load_cluster_areas()
    stays = spark.read.parquet(path("stays.parquet"))
    with_polys = attach_polygons(stays, areas)
    with_polys.write.mode("overwrite").parquet(path("stays_polygons.parquet"))
    log(f"stay polygons written: "
        f"{spark.read.parquet(path('stays_polygons.parquet')).count():,} rows")


def stage_relevance(spark):
    stays = spark.read.parquet(path("stays_polygons.parquet"))

    split = split_stays(stays, P)
    split.write.mode("overwrite").parquet(path("split_stays.parquet"))
    log(f"split stays written: "
        f"{spark.read.parquet(path('split_stays.parquet')).count():,} rows")

    share = habitual_region_share(stays, P, affected_region_wkb(), P.dbscan_eps_m)
    share.write.mode("overwrite").parquet(path("region_share.parquet"))

    rel = stay_relevance(stays, P, cfg.DBSCAN_EPS_LIST)
    rel.write.mode("overwrite").parquet(path("stay_relevance.parquet"))
    log(f"stay relevance written: "
        f"{spark.read.parquet(path('stay_relevance.parquet')).count():,} rows")


def stage_daily(spark):
    rel = spark.read.parquet(path("stay_relevance.parquet"))
    daily = daily_series(rel, n_thresholds=len(cfg.DBSCAN_EPS_LIST))
    daily.write.mode("overwrite").parquet(path("daily_series.parquet"))
    log(f"daily series written: "
        f"{spark.read.parquet(path('daily_series.parquet')).count():,} rows")


# --------------------------------------------------------------------------- #
# Driver-side stages
# --------------------------------------------------------------------------- #
def stage_detect(spark):
    daily = spark.read.parquet(path("daily_series.parquet")).toPandas()
    loc_col = f"location_{cfg.DBSCAN_EPS_LIST.index(P.dbscan_eps_m) + 1}"
    binary = binarize_trajectories(daily, loc_col, P.relevance_threshold)
    binary.to_csv(path("binary_trajectory.csv"), index=False)
    log(f"binary trajectory: {len(binary):,} rows, "
        f"{binary['user_id'].nunique():,} users")

    migrants = find_migrants(binary, P, cfg.DETECTOR_PATHS)
    events = classify_events(
        migrants.sort_values(["user_id", "migration_date"])[[
            "user_id", "migration_date", "home_start_date", "home_end_date",
            "destination_start_date", "destination_end_date", "home", "destination",
        ]],
        disaster_date=P.disaster_date.strftime("%Y%m%d"),
    )
    events.to_csv(path("displaced_events.csv"), index=False)

    # analysis population: residents with >= 90% habitual space in the region
    share = spark.read.parquet(path("region_share.parquet")).toPandas()
    population = set(share[share["region_night_relevance"]
                           > cfg.HABITUAL_REGION_SHARE_MIN]["user_id"])
    log(f"analysis population (>{cfg.HABITUAL_REGION_SHARE_MIN}% habitual space "
        f"in region): {len(population):,} users")

    segments = pd.read_csv(path("user_segments.csv"))
    events["user_id_int"] = events["user_id"].astype("int64")
    displaced = events[(events.get("movement_type_displacement", 0) == 1)
                       & events["user_id_int"].isin(population)]
    displaced_users = (displaced[["user_id_int"]].drop_duplicates()
                       .rename(columns={"user_id_int": "user_id"})
                       .merge(segments, on="user_id", how="left"))
    displaced_users.to_csv(path("displaced_users.csv"), index=False)
    log(f"displaced users in population: {len(displaced_users):,} "
        f"(segments: {displaced_users['segment'].value_counts().to_dict()})")

    if "movement_type_return_displacement" in events.columns:
        ret = events[(events["movement_type_return_displacement"] == 1)
                     & events["user_id_int"].isin(population)]["user_id"].nunique()
        log(f"returnee rate: {ret / max(len(displaced_users), 1):.4f}")


def stage_match(spark):
    events = pd.read_csv(path("displaced_events.csv"))
    displaced_users = pd.read_csv(path("displaced_users.csv"))
    disp_events = events[(events["movement_type_displacement"] == 1)
                         & events["user_id"].astype("int64")
                         .isin(set(displaced_users["user_id"]))]

    split = spark.read.parquet(path("split_stays.parquet"))
    users_df = spark.createDataFrame(displaced_users[["user_id"]])
    split_dp = split.join(users_df, on="user_id", how="inner").toPandas()

    rel = spark.read.parquet(path("stay_relevance.parquet"))
    loc_label = f"{int(P.dbscan_eps_m) // 1000}km"
    rel_dp = (rel.join(users_df, on="user_id", how="inner")
              .select("user_id", "stay_id",
                      f"habitual_night_relevance_{loc_label}")
              .toPandas()
              .drop_duplicates(["user_id", "stay_id"])
              .rename(columns={f"habitual_night_relevance_{loc_label}":
                               "habitual_night_relevance"}))

    pre = split_dp[split_dp["period"] == "pre"]
    post = split_dp[split_dp["period"] == "post"].merge(
        rel_dp, on=["user_id", "stay_id"], how="left")

    origins = match_stays_with_events(disp_events, pre, "origin")
    destinations = match_stays_with_events(disp_events, post, "destination")
    origins.to_parquet(path("origin_stays.parquet"), index=False)
    destinations.to_parquet(path("destination_stays.parquet"), index=False)
    log(f"origin stays: {len(origins):,} rows / "
        f"{origins['user_id'].nunique():,} users; "
        f"destination stays: {len(destinations):,} rows / "
        f"{destinations['user_id'].nunique():,} users")


STAGES = {
    "stays": stage_stays,
    "relevance": stage_relevance,
    "daily": stage_daily,
    "detect": stage_detect,
    "match": stage_match,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all"] + list(STAGES))
    ap.add_argument("--smoke", action="store_true",
                    help="run on the small 1000-customer CDR sample")
    args = ap.parse_args()

    if args.smoke:
        cfg.CDR_FILES = [
            f.replace("Fine_grained/FGM_", "Fine_grained/summary/FGM_")
            for f in cfg.CDR_FILES
        ]
        # the sample files are already normalized to lowercase column names
        from asa import CDRSchema

        cfg.CDR_SCHEMA = CDRSchema(user_id="customer_id", time="time",
                                   site_id="site_id_caller",
                                   group="segment_caller")
        OUT = OUT + "_smoke"

    os.makedirs(OUT, exist_ok=True)
    spark = build_session("asa-replication", driver_memory="10g",
                          local_dir=os.path.join(OUT, "_spark_tmp"))
    spark.sparkContext.setLogLevel("WARN")

    t0 = time.time()
    todo = list(STAGES) if args.stage == "all" else [args.stage]
    for name in todo:
        log(f"=== stage {name} ===")
        STAGES[name](spark)
    log(f"done in {(time.time() - t0) / 60:.1f} min")
