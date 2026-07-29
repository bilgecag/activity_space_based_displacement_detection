"""Distributed ASA pipeline on Apache Spark.

Stages (each takes and returns a Spark DataFrame):

    normalize_signals   -> canonical signal table
    attach_sites        -> adds location_id and metric coordinates
    filter_residents    -> keeps users last observed in the affected area
    detect_stays        -> stay locations per user
    attach_polygons     -> stay footprint polygons (WKB)
    stay_relevance      -> per-stay familiarity scores (Eq. 1)
    daily_series        -> daily familiarity trajectories

The remaining steps (binarization, segment-based detection, origin/
destination matching) operate on small aggregated data and run on the
driver — see ``asa.detection``.
"""
from .session import build_session  # noqa: F401
from .pipeline import (  # noqa: F401
    normalize_signals,
    attach_sites,
    filter_residents,
    detect_stays,
    attach_polygons,
    split_stays,
    habitual_region_share,
    stay_relevance,
    daily_series,
)
