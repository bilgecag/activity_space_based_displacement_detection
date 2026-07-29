"""Paper-styled figures: runs the EXACT figure code of the original notebook
("A - Last analysis.ipynb", cells stored verbatim in paper_figure_code/) on the
outputs of the Spark pipeline rerun.

This script only prepares the input variables the notebook cells expect
(gdf_origins, diff_syrians, plot_damage, df_adjusted_*, ...) from
output/spark_run_2km_2hours/, then executes the unmodified cell code with
the figure paths redirected to output/figures/.

Produces: paper_fig4_kde_maps.png, paper_fig8_distance_timing.png,
paper_fig9_distributions.png, paper_fig10_border_map.png,
paper_fig11_validation.png

Usage: python make_paper_figures.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from shapely import wkb as swkb  # noqa: E402
from shapely.geometry import Point  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import local_config as cfg  # noqa: E402
import turkstat  # noqa: E402
from asa.detection import find_migrants  # noqa: E402

RUN = cfg.OUTPUT_DIR
FIGDIR = os.path.join(os.path.dirname(RUN), "figures")
CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_figure_code")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cell(name: str, ns: dict, save_as: str | None = None):
    src = open(os.path.join(CELLS, name)).read()
    src = src.replace("%matplotlib inline", "")
    # redirect the notebook's Desktop save paths into the figure directory
    src = src.replace('"/Desktop/',
                      f'"{FIGDIR}/paper_')
    log(f"executing {name} ...")
    exec(compile(src, name, "exec"), ns)
    if save_as:
        ns["plt"].savefig(os.path.join(FIGDIR, save_as), dpi=200,
                          bbox_inches="tight")
    ns["plt"].close("all")


# --------------------------------------------------------------------------- #
# Notebook input variables from the Spark run
# --------------------------------------------------------------------------- #
def load_inputs() -> dict:
    log("loading run outputs ...")
    feats = pd.read_csv(os.path.join(RUN, "person_features.csv"))
    feats = feats.rename(columns={"user_id": "customer_id"})

    both = feats[feats["distance"] > 10].reset_index(drop=True)
    turkish = both[both["segment"] == 1].reset_index(drop=True)
    syrians = both[both["segment"] == 2].reset_index(drop=True)
    diff_turkish, diff_syrians = turkish, syrians  # *_diff columns present

    segments = pd.read_csv(os.path.join(RUN, "user_segments.csv")).rename(
        columns={"user_id": "customer_id"})
    df_affected_customers = segments

    def load_stays(name):
        df = pd.read_parquet(os.path.join(RUN, name))
        df = df.rename(columns={"user_id": "customer_id"})
        g = gpd.GeoDataFrame(df.drop(columns=["geometry_wkb"]),
                             geometry=[swkb.loads(b) for b in df["geometry_wkb"]],
                             crs=cfg.METRIC_CRS)
        return g.drop_duplicates(["customer_id", "stay_id"]).reset_index(drop=True)

    gdf_origins = load_stays("origin_stays.parquet")
    gdf_destinations = load_stays("destination_stays.parquet")

    log("building district/city reference layers ...")
    adm2 = gpd.read_file(cfg.DISTRICT_MAP_SHP)
    adm2["city"] = adm2["adm1_en"].astype(str).str.upper().map(turkstat.turkish_to_english)
    adm2["district"] = adm2["adm2_en"].astype(str).str.upper().map(turkstat.turkish_to_english)
    adm2["city_district"] = adm2["city"] + "_" + adm2["district"]
    gdf_tower_unique = adm2[["city_district", "city", "district", "geometry"]].to_crs(cfg.METRIC_CRS)
    gdf_tower_unique["is_in_earthquake_area"] = gdf_tower_unique["city"].isin(
        cfg.AFFECTED_CITY_NAMES).astype(int)

    eq_cities = gdf_tower_unique[gdf_tower_unique["is_in_earthquake_area"] == 1]
    eq_cities_unary = gpd.GeoDataFrame(
        geometry=[unary_union(eq_cities.geometry.values)], crs=cfg.METRIC_CRS)

    city_centers = [
        ("Hatay", 36.2003, 36.1600), ("Kilis", 36.7165, 37.1150),
        ("Gaziantep", 37.0662, 37.3833), ("Şanlıurfa", 37.1591, 38.7969),
        ("Adana", 37.0000, 35.3213), ("Diyarbakır", 37.9158, 40.2189),
        ("Osmaniye", 37.0742, 36.2472), ("Adıyaman", 37.7648, 38.2765),
        ("Malatya", 38.3552, 38.3095), ("Kahramanmaraş", 37.5753, 36.9371),
        ("Mersin", 36.8000, 34.6333),
    ]
    df_cc = pd.DataFrame(city_centers, columns=["City", "Latitude", "Longitude"])
    gdf_city_centers = gpd.GeoDataFrame(
        df_cc, geometry=[Point(xy) for xy in zip(df_cc["Longitude"], df_cc["Latitude"])],
        crs="EPSG:4326").to_crs(cfg.METRIC_CRS)

    refugee_camps = gpd.read_file(cfg.CAMPS_SHP)
    tix = refugee_camps[refugee_camps["cmp_en"] == "TURKOGLU"].index
    refugee_camps.loc[tix, "geometry"] = Point(37.016083, 37.441337)
    refugee_camps = refugee_camps[(refugee_camps["Status"] == "ACTIVE")
                                  & (refugee_camps["Type"] == "Refugee Camp")]
    refugee_camps = refugee_camps.set_crs("EPSG:4326").to_crs(cfg.METRIC_CRS)

    # per-person frame for the Fig 8 prep cell (merge key: customer_id+segment)
    origins_grouped = both[["customer_id", "segment"]].drop_duplicates()

    # border-shifted Syrians and destination stays clipped to districts (Fig 10)
    sinira_giden_suriyeliler = diff_syrians[
        diff_syrians["weighted_syria_border_diff"] == 1]["customer_id"].unique().tolist()
    log("overlaying destination stays with districts (Fig 10) ...")
    gdf_frame_overlayed_destinations = gpd.overlay(
        gdf_destinations[["customer_id", "stay_id", "geometry"]],
        gdf_tower_unique[["city_district", "geometry"]],
        how="intersection", keep_geom_type=False)

    return dict(
        turkish=turkish, syrians=syrians,
        diff_turkish=diff_turkish, diff_syrians=diff_syrians,
        df_affected_customers=df_affected_customers,
        gdf_origins=gdf_origins, gdf_destinations=gdf_destinations,
        gdf_tower_unique=gdf_tower_unique, eq_cities_unary=eq_cities_unary,
        gdf_city_centers=gdf_city_centers, refugee_camps=refugee_camps,
        origins_grouped=origins_grouped,
        sinira_giden_suriyeliler=sinira_giden_suriyeliler,
        gdf_frame_overlayed_destinations=gdf_frame_overlayed_destinations,
    )


# --------------------------------------------------------------------------- #
# Fig 11 inputs: debiased ASA + TMB estimates vs TURKSTAT
# --------------------------------------------------------------------------- #
def classify_home_location_events(migrants: pd.DataFrame,
                                  disaster: str = "20230206") -> pd.DataFrame:
    """Displacement labeling for multi-location (city id) trajectories."""
    df = migrants.sort_values(["user_id", "migration_date"]).copy()
    df["migration_date"] = pd.to_datetime(df["migration_date"], format="%Y%m%d")
    eq = pd.to_datetime(disaster, format="%Y%m%d")
    original_home = df.groupby("user_id")["home"].transform("first")
    leaving = df["home"] == original_home
    returning = df["destination"] == original_home
    df["movement_type"] = "secondary_movement"
    df.loc[leaving & (df["migration_date"] < eq), "movement_type"] = "migration"
    df.loc[leaving & (df["migration_date"] >= eq), "movement_type"] = "displacement"
    df.loc[~leaving & returning, "movement_type"] = "return"
    return df


def build_adjusted_frames(ns: dict) -> tuple:
    log("computing city flows and debiased estimates ...")
    cities = ns["gdf_tower_unique"][["city", "geometry"]].dissolve("city").reset_index()

    def dominant_city(gdf, weight_col):
        cent = gpd.GeoDataFrame(gdf[["customer_id", weight_col]].copy(),
                                geometry=gdf.geometry.centroid, crs=cfg.METRIC_CRS)
        j = gpd.sjoin(cent, cities, how="inner", predicate="within")
        per = j.groupby(["customer_id", "city"])[weight_col].sum().reset_index()
        return (per.sort_values(weight_col, ascending=False)
                .drop_duplicates("customer_id")[["customer_id", "city"]])

    o_city = dominant_city(ns["gdf_origins"], "night_duration_aggregated").rename(
        columns={"city": "origin_city"})
    d_city = dominant_city(ns["gdf_destinations"], "night_duration_aggregated").rename(
        columns={"city": "destination_city"})
    seg = ns["df_affected_customers"]
    flows_as = (o_city.merge(d_city, on="customer_id").merge(seg, on="customer_id")
                .groupby(["origin_city", "destination_city", "segment"])
                .size().rename("observed_rate").reset_index())

    # per-city sample sizes (dominant pre-disaster city of every resident)
    split = pd.read_parquet(os.path.join(RUN, "split_stays.parquet"),
                            columns=["user_id", "period", "duration", "geometry_wkb"])
    pre = split[split["period"] == "pre"]
    cent = gpd.GeoDataFrame(
        pre[["user_id", "duration"]].rename(columns={"user_id": "customer_id"}),
        geometry=[swkb.loads(b).centroid for b in pre["geometry_wkb"]], crs=cfg.METRIC_CRS)
    j = gpd.sjoin(cent, cities, how="inner", predicate="within")
    per = j.groupby(["customer_id", "city"])["duration"].sum().reset_index()
    dom = per.sort_values("duration", ascending=False).drop_duplicates("customer_id")
    dom = dom.merge(seg, on="customer_id")
    sample = (dom.groupby(["city", "segment"])["customer_id"].nunique()
              .rename("customers").reset_index())

    # official statistics
    inflow_df = turkstat.read_migration_by_reason(
        os.path.join(cfg.DEBIAS_DIR, "migrant-inflows-2023.csv"), "inflow")
    outflow_df = turkstat.read_migration_by_reason(
        os.path.join(cfg.DEBIAS_DIR, "migrant-outflows-2023.csv"), "outflow")
    tr_pop = turkstat.read_turkish_population(
        os.path.join(cfg.DEBIAS_DIR, "turkish_pop_2023.csv"))
    syr_pop = turkstat.read_syrian_population(
        os.path.join(cfg.DEBIAS_DIR, "syrian_pop_04-23.txt"))

    rates = (sample.pivot(index="city", columns="segment", values="customers")
             .rename(columns={1: "total_turkish_customer", 2: "total_syrian_customer"})
             .reset_index().merge(tr_pop, on="city").merge(syr_pop, on="city"))
    rates["eff_tur"] = rates["total_turkish_customer"] / (rates["turkish_population"] * (1 - 0.22))
    rates["eff_syr"] = rates["total_syrian_customer"] / (rates["syrian_population"] * (1 - 0.41))

    def adjust(flows):
        f = flows[flows["origin_city"] != flows["destination_city"]].merge(
            rates[["city", "eff_tur", "eff_syr"]], left_on="origin_city",
            right_on="city", how="left")
        f["adjusted"] = np.where(f["segment"] == 1,
                                 f["observed_rate"] / f["eff_tur"],
                                 f["observed_rate"] / f["eff_syr"])
        aff = f[f["origin_city"].isin(cfg.AFFECTED_CITY_NAMES) & (f["segment"] == 1)]
        out = aff.groupby("origin_city")["adjusted"].sum().rename(
            "turkish_adjusted_outflow_rate")
        inn = aff.groupby("destination_city")["adjusted"].sum().rename(
            "turkish_adjusted_inflow_rate")
        frame = pd.concat([out, inn], axis=1).reset_index().rename(
            columns={"index": "city"}).fillna(0)
        return frame

    df_adjusted_as = adjust(flows_as)

    # TMB baseline: city-level daily modal locations + the same detector
    log("running TMB city-level detection (cached daily home locations) ...")
    tmb_path = os.path.join(cfg.EPJ_RESULTS, "algorithm_test",
                            "home_location_approach_city.csv")
    tmb_traj = pd.read_csv(tmb_path)[["date", "user_id", "location"]].dropna()
    tmb_traj["location"] = tmb_traj["location"].astype(int)
    migrants = find_migrants(tmb_traj, cfg.PARAMS, cfg.DETECTOR_PATHS)
    events = classify_home_location_events(
        migrants[["user_id", "migration_date", "home", "destination"]].copy())
    filt = pd.read_csv(os.path.join(cfg.EPJ_RESULTS, "filtered_customers_with_segments.csv"),
                       index_col=0)
    events["customer_id"] = events["user_id"].astype("int64")
    events = events.merge(filt, on="customer_id", how="inner")
    disp = events[events["movement_type"] == "displacement"]

    tower = pd.read_csv(cfg.TOWER_TXT, sep="|", header=0, encoding="ISO-8859-1")
    tower = tower.drop(columns=[c for c in tower.columns if c.startswith("Unnamed")])
    tower = tower.iloc[1:, :].apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    tower = tower.rename(columns=lambda x: x.strip())
    tower["city_id"] = tower.groupby("city").ngroup() + 1
    id2city = tower.drop_duplicates("city_id").set_index("city_id")["city"].map(
        lambda s: turkstat.turkish_to_english(str(s).upper()))
    disp = disp.assign(origin_city=disp["home"].astype(int).map(id2city),
                       destination_city=disp["destination"].astype(int).map(id2city))
    flows_tmb = (disp.groupby(["origin_city", "destination_city", "segment"])
                 .size().rename("observed_rate").reset_index())
    df_adjusted_tmb = adjust(flows_tmb)

    other_in = inflow_df[["city", "other"]].rename(columns={"other": "other_inflow"})
    other_out = outflow_df[["city", "other"]].rename(columns={"other": "other_outflow"})
    df_adjusted_as = df_adjusted_as.merge(other_in, on="city", how="left").merge(
        other_out, on="city", how="left").fillna(0)
    df_adjusted_tmb = df_adjusted_tmb.merge(other_in, on="city", how="left").merge(
        other_out, on="city", how="left").fillna(0)
    return df_adjusted_tmb, df_adjusted_as


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(FIGDIR, exist_ok=True)

    ns = {"np": np, "pd": pd, "gpd": gpd, "plt": plt, "sns": sns,
          "LinearSegmentedColormap": LinearSegmentedColormap,
          "ListedColormap": ListedColormap, "gaussian_kde": gaussian_kde,
          "Point": Point, "unary_union": unary_union}
    ns.update(load_inputs())

    run_cell("fig08_prep_damage_categories.py", ns)
    run_cell("fig08_distance_and_timing.py", ns, "paper_fig8_distance_timing.png")
    run_cell("fig09_index_difference_distributions.py", ns, "paper_fig9_distributions.png")
    run_cell("fig04_kde_maps.py", ns, "paper_fig4_kde_maps.png")
    run_cell("fig10_border_density_map.py", ns, "paper_fig10_border_map.png")

    tmb, asa_adj = build_adjusted_frames(ns)
    ns["df_adjusted_tmb"], ns["df_adjusted_as"] = tmb, asa_adj
    run_cell("fig11_turkstat_validation.py", ns, "paper_fig11_validation.png")

    log(f"done in {(time.time() - t0) / 60:.1f} min")
