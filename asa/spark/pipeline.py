"""The distributed ASA pipeline stages.

Users are independent throughout the pipeline, so every heavy stage groups
the data into user buckets and applies the single-machine kernels
(``asa.kernels``) to each bucket with ``applyInPandas``. Geometries travel
between stages as WKB bytes.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from pyspark.sql import DataFrame, functions as F, types as T

from ..params import ASAParams
from ..schemas import CDRSchema, SiteSchema

N_BUCKETS = 96


def _bucketed(df: DataFrame, col: str = "user_id") -> DataFrame:
    return df.withColumn("_bucket", F.pmod(F.hash(F.col(col)), F.lit(N_BUCKETS)))


# --------------------------------------------------------------------------- #
# Input normalization
# --------------------------------------------------------------------------- #
def normalize_signals(df: DataFrame, schema: CDRSchema,
                      min_signals_per_source: int = 0,
                      source_col: str | None = None) -> DataFrame:
    """Map the caller's columns to [user_id, time, site_id (, group)] and
    optionally drop users with fewer than ``min_signals_per_source`` signals
    per input source (e.g. per monthly file)."""
    for old, new in schema.rename_map().items():
        if old != new:
            df = df.withColumnRenamed(old, new)
    if min_signals_per_source > 0:
        # the activity filter counts every signal, located or not
        keys = ["user_id"] + ([source_col] if source_col else [])
        counts = df.groupBy(*keys).agg(F.count("*").alias("_n"))
        df = df.join(counts.filter(F.col("_n") >= min_signals_per_source).drop("_n"),
                     on=keys, how="inner")
    return df.filter(F.col("site_id").isNotNull())


def attach_sites(signals: DataFrame, sites: DataFrame, schema: SiteSchema) -> DataFrame:
    """Join the site reference table: adds location_id, x, y (, region_id)."""
    for old, new in schema.rename_map().items():
        if old != new:
            sites = sites.withColumnRenamed(old, new)
    keep = ["site_id", "location_id", "x", "y"]
    if "region_id" in sites.columns:
        keep.append("region_id")
    return signals.join(sites.select(*keep).dropDuplicates(["site_id"]),
                        on="site_id", how="inner")


def filter_residents(signals: DataFrame,
                     affected_region_ids: list,
                     window_start: dt.datetime,
                     window_end: dt.datetime) -> DataFrame:
    """Keep users whose LAST signal inside the window lies in the affected
    regions — the population at risk of displacement."""
    from pyspark.sql import Window

    w = signals.filter((F.col("time") > F.lit(window_start))
                       & (F.col("time") < F.lit(window_end)))
    latest = Window.partitionBy("user_id").orderBy(F.col("time").desc())
    last = (w.withColumn("_rn", F.row_number().over(latest))
             .filter(F.col("_rn") == 1)
             .filter(F.col("region_id").isin(list(affected_region_ids)))
             .select("user_id"))
    return signals.join(last, on="user_id", how="inner")


# --------------------------------------------------------------------------- #
# Stage 1 — stay locations
# --------------------------------------------------------------------------- #
_STAYS_SCHEMA = T.StructType([
    T.StructField("user_id", T.LongType()),
    T.StructField("start_time", T.TimestampType()),
    T.StructField("end_time", T.TimestampType()),
    T.StructField("duration", T.DoubleType()),
    T.StructField("location_ids", T.ArrayType(T.LongType())),
    T.StructField("stay_id", T.LongType()),
])


def detect_stays(signals: DataFrame, params: ASAParams) -> DataFrame:
    """Stay locations for every user (see asa.kernels.stays)."""
    dist, dur = params.stay_distance_m, params.stay_duration_s

    def _kernel(pdf: pd.DataFrame) -> pd.DataFrame:
        from ..kernels.stays import detect_stays as _detect

        out = _detect(pdf[["user_id", "time", "location_id", "x", "y"]], dist, dur)
        out["location_ids"] = out["location_ids"].map(
            lambda ids: [int(i) for i in ids])
        return out[["user_id", "start_time", "end_time", "duration",
                    "location_ids", "stay_id"]]

    return (_bucketed(signals)
            .groupBy("_bucket")
            .applyInPandas(_kernel, schema=_STAYS_SCHEMA))


# --------------------------------------------------------------------------- #
# Stage 1b — stay footprint polygons
# --------------------------------------------------------------------------- #
def attach_polygons(stays: DataFrame, site_areas: pd.DataFrame) -> DataFrame:
    """Stay polygon = convex hull of the service areas of its locations.

    site_areas : pandas DataFrame [location_id, area_wkb] with one service-
    area polygon per location id (broadcast to the executors).
    """
    spark = stays.sparkSession
    lookup = dict(zip(site_areas["location_id"].astype(int),
                      site_areas["area_wkb"]))
    b_lookup = spark.sparkContext.broadcast(lookup)

    out_schema = T.StructType(
        [f for f in stays.schema.fields]
        + [T.StructField("geometry_wkb", T.BinaryType())]
    )
    out_cols = [f.name for f in out_schema.fields]

    def _kernel(pdf: pd.DataFrame) -> pd.DataFrame:
        import shapely
        from shapely import wkb as swkb

        areas = b_lookup.value
        cache: dict = {}

        def hull(ids):
            key = tuple(sorted(set(ids)))
            if key not in cache:
                geoms = [swkb.loads(areas[i]) for i in key if i in areas]
                cache[key] = (shapely.convex_hull(shapely.union_all(geoms)).wkb
                              if geoms else None)
            return cache[key]

        pdf["geometry_wkb"] = pdf["location_ids"].map(hull)
        return pdf[pdf["geometry_wkb"].notna()][out_cols]

    return (_bucketed(stays)
            .groupBy("_bucket")
            .applyInPandas(_kernel, schema=out_schema))


# --------------------------------------------------------------------------- #
# Stage 2+3 — activity spaces and per-stay familiarity
# --------------------------------------------------------------------------- #
def _relevance_schema(eps_list_m: list) -> T.StructType:
    fields = [
        T.StructField("user_id", T.LongType()),
        T.StructField("stay_id", T.LongType()),
    ]
    fields += [T.StructField(f"habitual_night_relevance_{int(e) // 1000}km",
                             T.DoubleType()) for e in eps_list_m]
    fields += [
        T.StructField("start_time", T.TimestampType()),
        T.StructField("end_time", T.TimestampType()),
    ]
    return T.StructType(fields)


def stay_relevance(stays_with_polygons: DataFrame,
                   params: ASAParams,
                   eps_list_m: list | None = None) -> DataFrame:
    """Per-stay familiarity scores for one or more DBSCAN thresholds.

    Splits each user's stays at the disaster instant, computes day/night
    durations and relevance shares per period, builds pre/post activity
    spaces, overlays them and transfers the scores back to the stays.
    """
    if eps_list_m is None:
        eps_list_m = [params.dbscan_eps_m]
    split_ts = params.disaster_timestamp
    night_start = params.night_start_hour
    night_end = params.night_end_hour
    legacy = params.legacy_day_night_accounting
    out_schema = _relevance_schema(eps_list_m)
    rel_cols = [f.name for f in out_schema.fields]

    def _kernel(pdf: pd.DataFrame) -> pd.DataFrame:
        from shapely import wkb as swkb

        from ..kernels.stays import split_pre_post
        from ..kernels.relevance import prepare_stays_with_relevance
        from ..kernels.activity_spaces import stay_relevance_table

        stay_times = pdf[["user_id", "stay_id", "start_time", "end_time"]]

        split = split_pre_post(
            pdf[["user_id", "stay_id", "start_time", "end_time", "duration",
                 "location_ids", "geometry_wkb"]],
            pd.Timestamp(split_ts),
        )
        split = split[split["geometry_wkb"].notna()].reset_index(drop=True)
        split["geometry"] = split["geometry_wkb"].map(swkb.loads)
        split = prepare_stays_with_relevance(
            split, night_start_hour=night_start, night_end_hour=night_end,
            legacy_accounting=legacy,
        )
        table = stay_relevance_table(split, eps_list_m, stay_times)
        return table[rel_cols]

    return (_bucketed(stays_with_polygons)
            .groupBy("_bucket")
            .applyInPandas(_kernel, schema=out_schema))


# --------------------------------------------------------------------------- #
# Stage 2b — split stays with durations and relevance (for O/D attribution)
# --------------------------------------------------------------------------- #
_SPLIT_SCHEMA = T.StructType([
    T.StructField("user_id", T.LongType()),
    T.StructField("stay_id", T.LongType()),
    T.StructField("period", T.StringType()),
    T.StructField("start_time", T.TimestampType()),
    T.StructField("end_time", T.TimestampType()),
    T.StructField("duration", T.DoubleType()),
    T.StructField("day_duration", T.DoubleType()),
    T.StructField("night_duration", T.DoubleType()),
    T.StructField("duration_aggregated", T.DoubleType()),
    T.StructField("day_duration_aggregated", T.DoubleType()),
    T.StructField("night_duration_aggregated", T.DoubleType()),
    T.StructField("total_relevance", T.DoubleType()),
    T.StructField("day_relevance", T.DoubleType()),
    T.StructField("night_relevance", T.DoubleType()),
    T.StructField("geometry_wkb", T.BinaryType()),
])


def split_stays(stays_with_polygons: DataFrame, params: ASAParams) -> DataFrame:
    """Pre/post-disaster stays with day/night durations and relevance scores.

    This is the per-stay view used later to attribute origin and destination
    locations to each displaced person and to compute spatial indices.
    """
    split_ts = params.disaster_timestamp
    night_start, night_end = params.night_start_hour, params.night_end_hour
    legacy = params.legacy_day_night_accounting
    out_cols = [f.name for f in _SPLIT_SCHEMA.fields]

    def _kernel(pdf: pd.DataFrame) -> pd.DataFrame:
        from ..kernels.stays import split_pre_post
        from ..kernels.relevance import prepare_stays_with_relevance

        split = split_pre_post(
            pdf[["user_id", "stay_id", "start_time", "end_time", "duration",
                 "location_ids", "geometry_wkb"]],
            pd.Timestamp(split_ts),
        )
        split = split[split["geometry_wkb"].notna()].reset_index(drop=True)
        split = prepare_stays_with_relevance(
            split, night_start_hour=night_start, night_end_hour=night_end,
            legacy_accounting=legacy,
        )
        return split[out_cols]

    return (_bucketed(stays_with_polygons)
            .groupBy("_bucket")
            .applyInPandas(_kernel, schema=_SPLIT_SCHEMA))


# --------------------------------------------------------------------------- #
# Stage 2c — resident filter on habitual space
# --------------------------------------------------------------------------- #
def habitual_region_share(stays_with_polygons: DataFrame,
                          params: ASAParams,
                          region_wkb: bytes,
                          eps_m: float) -> DataFrame:
    """Share (%) of each user's pre-disaster habitual space inside a region."""
    split_ts = params.disaster_timestamp
    night_start, night_end = params.night_start_hour, params.night_end_hour
    legacy = params.legacy_day_night_accounting
    out_schema = T.StructType([
        T.StructField("user_id", T.LongType()),
        T.StructField("region_night_relevance", T.DoubleType()),
    ])

    def _kernel(pdf: pd.DataFrame) -> pd.DataFrame:
        from shapely import wkb as swkb

        from ..kernels.stays import split_pre_post
        from ..kernels.relevance import prepare_stays_with_relevance
        from ..kernels.activity_spaces import habitual_region_share as _share

        region = swkb.loads(region_wkb)
        split = split_pre_post(
            pdf[["user_id", "stay_id", "start_time", "end_time", "duration",
                 "location_ids", "geometry_wkb"]],
            pd.Timestamp(split_ts),
        )
        split = split[split["geometry_wkb"].notna()].reset_index(drop=True)
        split = prepare_stays_with_relevance(
            split, night_start_hour=night_start, night_end_hour=night_end,
            legacy_accounting=legacy,
        )
        pre = split[split["period"] == "pre"].reset_index(drop=True)
        if pre.empty:
            return pd.DataFrame(columns=["user_id", "region_night_relevance"])
        from shapely import wkb as _wkb

        pre = pre.assign(geometry=pre["geometry_wkb"].map(_wkb.loads))
        return _share(pre, region, eps_m)

    return (_bucketed(stays_with_polygons)
            .groupBy("_bucket")
            .applyInPandas(_kernel, schema=out_schema))


# --------------------------------------------------------------------------- #
# Stage 4a — daily trajectories
# --------------------------------------------------------------------------- #
def daily_series(stay_relevance_df: DataFrame, n_thresholds: int) -> DataFrame:
    """Daily familiarity per user (one location_i column per threshold)."""
    fields = [T.StructField("user_id", T.LongType()),
              T.StructField("date", T.DateType())]
    fields += [T.StructField(f"location_{i+1}", T.DoubleType())
               for i in range(n_thresholds)]
    out_schema = T.StructType(fields)

    def _kernel(pdf: pd.DataFrame) -> pd.DataFrame:
        from ..kernels.trajectories import daily_relevance_series

        out = daily_relevance_series(pdf)
        return out[[f.name for f in out_schema.fields]]

    return (_bucketed(stay_relevance_df)
            .groupBy("_bucket")
            .applyInPandas(_kernel, schema=out_schema))
