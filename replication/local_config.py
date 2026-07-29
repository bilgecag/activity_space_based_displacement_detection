"""Local configuration for replicating the 2023 Türkiye-Syria earthquake study.

Everything machine- and study-specific lives here: file locations, the
column layout of the Turkcell CDR export, the disaster instant, and the ten
affected provinces. The ``asa`` library itself is data-agnostic.

Setting up on a new machine
---------------------------
Only the "external data roots" block below refers to files outside this
repository. Point each root at your copy of the data, either by editing
the defaults here or by exporting the corresponding environment variable
(the environment variable always wins):

    ASA_MOBILE_DATA          raw Turkcell CDR + tower data (proprietary,
                             not distributable; required by run_pipeline.py)
    ASA_PAPER_RESULTS        cached result files of the original study
                             (only the Fig 11 TMB baseline in
                             make_paper_figures.py)
    ASA_TURKSTAT_DIR         TURKSTAT migration/population tables
                             (only the Fig 11 validation)
    ASA_ADMIN_BOUNDARIES     Türkiye admin boundary shapefiles
                             (tur_polbnda_adm1/adm2, OCHA/HDX)
    ASA_MIGRATION_DETECTOR   directory containing the migration_detector
                             package (Chi et al.)

Reference data that may be redistributed (the refugee-camp points) ships
inside the repository under reference_data/ and needs no setup.
"""
from __future__ import annotations

import datetime as dt
import os

from asa import ASAParams, CDRSchema

HOME = os.path.expanduser("~")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(var: str, default: str) -> str:
    return os.environ.get(var, default)


# --------------------------------------------------------------------------- #
# external data roots — EDIT HERE (or export the environment variables)
# --------------------------------------------------------------------------- #
MOBILE_DATA = _env("ASA_MOBILE_DATA", f"{HOME}/Downloads/mobile_data")
EPJ_RESULTS = _env("ASA_PAPER_RESULTS",
                   f"{HOME}/Downloads/PHD GRADUATION/thesis_manuscript/Articles/"
                   "EPJ Data Science /results")
DEBIAS_DIR = _env("ASA_TURKSTAT_DIR",
                  f"{HOME}/Downloads/PHD GRADUATION/thesis_manuscript/Articles/"
                  "EPJ Data Science /EPJ Data Science - Literature/Debiasing")
ADMIN_DIR = _env("ASA_ADMIN_BOUNDARIES",
                 f"{HOME}/Desktop/Geodirectory/turkey_administrativelevels0_1_2")
DETECTOR_PATHS = [_env("ASA_MIGRATION_DETECTOR",
                       f"{HOME}/Desktop/mobile_phone_indicators")]

# raw outgoing CDR files covering 2023-01-01 .. 2023-03-15
CDR_FILES = [
    f"{MOBILE_DATA}/Data-CDR/Outgoing/Fine_grained/FGM_{i}.txt"
    for i in ["7", "7_2", "7_3", "7_4", "7_5"]
]
CDR_SCHEMA = CDRSchema(user_id="CUSTOMER_ID", time="time",
                       site_id="CALER_SITE_ID", group="CALLER_SEGMENT")
MIN_SIGNALS_PER_FILE = 3

# tower reference data
SITE_CLUSTER_MATCH = f"{MOBILE_DATA}/Cell_Tower_Locations/clustered_towers/site_cluster_match.csv"
CLUSTERS_SHP = f"{MOBILE_DATA}/Cell_Tower_Locations/clustered_towers/clusters.shp"
CLUSTER_VORONOI_CSV = f"{MOBILE_DATA}/Cell_Tower_Locations/clustered_towers/cluster_voronoi.csv"
TOWER_TXT = f"{MOBILE_DATA}/Cell_Tower_Locations/cell_city_district.txt"
TURKCELL_VORONOI_SHP = f"{MOBILE_DATA}/Cell_Tower_Locations/turkcell_voronoi/voronoi.shp"
CONTEXT_CELLS_CSV = f"{MOBILE_DATA}/Cell_Tower_Locations/context_cells.csv"
CITY_MAP_SHP = os.path.join(ADMIN_DIR, "tur_polbnda_adm1.shp")
DISTRICT_MAP_SHP = os.path.join(ADMIN_DIR, "tur_polbnda_adm2.shp")

DAMAGE_FILES = {
    "collapsed": f"{MOBILE_DATA}/damaged/fsq_studio_Collapsed.geojson.geojson",
    "heavily_damaged": f"{MOBILE_DATA}/damaged/fsq_studio_Heavily_damaged.geojson.geojson",
    "needs_demolished": f"{MOBILE_DATA}/damaged/fsq_studio_Needs_demolished.geojson.geojson",
    "slightly_damaged": f"{MOBILE_DATA}/damaged/fsq_studio_Slightly.geojson.geojson",
}
# bundled with the repository — no setup needed
CAMPS_SHP = os.path.join(REPO_ROOT, "reference_data", "refugee_camps",
                         "tur_pntcntr_camps.shp")

METRIC_CRS = "EPSG:32636"

# the ten affected provinces; ids follow the alphabetical city numbering of
# the tower reference table
AFFECTED_CITY_NAMES = ["KAHRAMANMARAS", "HATAY", "GAZIANTEP", "ADIYAMAN", "MALATYA",
                       "OSMANIYE", "ADANA", "DIYARBAKIR", "SANLIURFA", "KILIS"]
AFFECTED_CITY_IDS = [42, 33, 37, 65, 2, 69, 26, 56, 48, 1]

# users must have been observed in the affected provinces in the month before
# the disaster (last signal within this window)
RESIDENT_WINDOW = (dt.datetime(2023, 1, 6, 4), dt.datetime(2023, 2, 6, 4))
# and hold >= 90% of their pre-disaster habitual space inside them
HABITUAL_REGION_SHARE_MIN = 90.0

# ASA hyperparameters of the published experiment
PARAMS = ASAParams(
    disaster_date=dt.date(2023, 2, 6),
    disaster_time=dt.time(4, 0),          # the first earthquake struck at ~04:17
    stay_distance_m=2_000.0,
    stay_duration_s=7_200.0,
    dbscan_eps_m=5_000.0,
    relevance_threshold=5.0,
    k_days=14,
    epsilon_days=14,
    legacy_day_night_accounting=True,     # match the published computation
)
DBSCAN_EPS_LIST = [3_000.0, 5_000.0, 10_000.0, 20_000.0]

# population segments in the CDR
SEGMENT_TURKISH = 1
SEGMENT_SYRIAN = 2

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "output", "spark_run_2km_2hours")
