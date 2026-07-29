"""Replication report: per-person feature table + summary figures/numbers.

This is the post-processing step that follows run_pipeline.py. It has two
roles:

1. Build the per-person origin/destination feature table
   (person_features.csv: weighted spatial indices, displacement distance,
   timing, returns). The other post-processing scripts
   (build_h3_indicators.py, make_paper_figures.py) read this table.

2. Produce quick-look versions of the paper's figures — drawn with the
   asa.figures library helpers, NOT the publication's original plotting
   code — together with the headline statistics used to compare the rerun
   against the published results:

    fig4_kde_maps.png            origin/destination stay-location density
    fig8_distance_timing.png     distance by damage exposure + timing
    fig9_index_differences.png   destination-origin index differences
    fig10_border_map.png         border-shifted Syrian DPs
    fig11_validation.png         debiased estimates vs TURKSTAT
    table1_syrian_flows.csv      Syrian DP inflows/outflows by city
    headline_numbers.json        headline statistics for comparison

For figures identical to the publication, run make_paper_figures.py, which
executes the original notebook plotting code (paper_figure_code/) instead.

Usage: python make_replication_report.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb as swkb
from shapely.geometry import Point
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import local_config as cfg  # noqa: E402
from city_populations import CITY_POPULATIONS  # noqa: E402
import turkstat  # noqa: E402
from asa import figures as figs  # noqa: E402
from asa import indices as idx  # noqa: E402
from asa import debias as deb  # noqa: E402
from asa.detection import weighted_midpoints, midpoint_distances  # noqa: E402

RUN = cfg.OUTPUT_DIR
FIGDIR = os.path.join(os.path.dirname(RUN), "figures")
os.makedirs(FIGDIR, exist_ok=True)

BORDER_GATES = [
    ("Karkamış", 36.8345, 37.9983), ("Yayladağı", 35.9025, 36.0606),
    ("Cilvegözü", 36.2338, 36.6797), ("Öncüpınar", 36.6439, 37.0872),
    ("Çobanbey", 36.6325, 37.4728), ("Akçakale", 36.7072, 38.9491),
    ("Ceylanpınar", 36.8461, 40.0489),
]
CITY_CENTERS = [
    ("Hatay", 36.2003, 36.1600), ("Kilis", 36.7165, 37.1150),
    ("Gaziantep", 37.0662, 37.3833), ("Şanlıurfa", 37.1591, 38.7969),
    ("Adana", 37.0000, 35.3213), ("Diyarbakır", 37.9158, 40.2189),
    ("Osmaniye", 37.0742, 36.2472), ("Adıyaman", 37.7648, 38.2765),
    ("Malatya", 38.3552, 38.3095), ("Kahramanmaraş", 37.5753, 36.9371),
    ("Mersin", 36.8000, 34.6333),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Reference layers
# --------------------------------------------------------------------------- #
def load_cells() -> gpd.GeoDataFrame:
    df = pd.read_csv(cfg.CLUSTER_VORONOI_CSV, index_col=0)
    df["voronoi_geometry"] = df["voronoi_geometry"].astype(str)
    df = df[df["voronoi_geometry"] != "nan"]
    geo = gpd.GeoSeries.from_wkt(df["voronoi_geometry"], crs="EPSG:4326").to_crs(cfg.METRIC_CRS)
    cells = gpd.GeoDataFrame({"location_id": df["cluster"].astype(int).to_numpy()},
                             geometry=geo.values, crs=cfg.METRIC_CRS)
    return cells.drop_duplicates("location_id").reset_index(drop=True)


def load_cities() -> gpd.GeoDataFrame:
    cities = gpd.read_file(cfg.CITY_MAP_SHP)
    name_col = next(c for c in ["adm1_en", "ADM1_EN", "NAME_1", "city"] if c in cities.columns)
    cities["city"] = cities[name_col].astype(str).str.upper().map(turkstat.turkish_to_english)
    return cities[["city", "geometry"]].to_crs(cfg.METRIC_CRS)


def build_cell_features() -> gpd.GeoDataFrame:
    """Urbanization, damage, border and city-population indices per cell."""
    cells = load_cells()

    site_areas = gpd.read_file(cfg.TURKCELL_VORONOI_SHP).rename(
        columns={"matcher": "site_id"})
    site_areas.crs = "EPSG:5636"
    site_areas = site_areas.to_crs(cfg.METRIC_CRS)[["site_id", "geometry"]]
    context = pd.read_csv(cfg.CONTEXT_CELLS_CSV)

    log("urbanization shares per cell ...")
    urb = idx.urbanization_shares_per_cell(cells, context, site_areas)
    feats = cells.merge(urb, on="location_id", how="left")
    for c in idx.URBANIZATION_CATEGORIES:
        if c not in feats.columns:
            feats[c] = 0.0
        feats[c] = feats[c].fillna(0.0)
    feats.loc[feats[idx.URBANIZATION_CATEGORIES].sum(axis=1) == 0, "Unknown"] = 100.0
    feats = idx.urbanization_index(feats)
    feats["urbanization_index"] = feats["urbanization_index"].fillna(0.0)

    log("damage counts per cell ...")
    dmg = idx.damage_counts_per_cell(cfg.DAMAGE_FILES, cells, cfg.METRIC_CRS)
    feats = feats.merge(dmg, on="location_id", how="left").fillna(
        {f"{k}_buildings": 0 for k in cfg.DAMAGE_FILES})
    feats = idx.damage_index(feats, area=feats.geometry.area)

    log("border flag per cell ...")
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres")).to_crs(cfg.METRIC_CRS)
    tur = world.loc[world["name"] == "Turkey"].geometry.iloc[0]
    syr = world.loc[world["name"] == "Syria"].geometry.iloc[0]
    border_line = tur.intersection(syr).simplify(0.001)
    feats["syria_border"] = idx.border_flag(feats, border_line)

    log("city population per cell ...")
    cities = load_cities()
    cent = feats.copy()
    cent["geometry"] = feats.geometry.centroid
    joined = gpd.sjoin(cent, cities, how="left", predicate="within")
    feats["city"] = joined["city"].values
    feats["city_type"] = feats["city"].map(
        {k: v[1] for k, v in CITY_POPULATIONS.items()})
    return feats


# --------------------------------------------------------------------------- #
# Per-person feature table
# --------------------------------------------------------------------------- #
def build_feature_table() -> pd.DataFrame:
    users = pd.read_csv(os.path.join(RUN, "displaced_users.csv"))
    events = pd.read_csv(os.path.join(RUN, "displaced_events.csv"))
    origins = pd.read_parquet(os.path.join(RUN, "origin_stays.parquet"))
    dests = pd.read_parquet(os.path.join(RUN, "destination_stays.parquet"))
    log(f"displaced users {len(users):,}; origin stays {len(origins):,}; "
        f"destination stays {len(dests):,}")

    feats = build_cell_features()
    count_cols = [f"{k}_buildings" for k in cfg.DAMAGE_FILES]
    pct_cols = idx.URBANIZATION_CATEGORIES
    dummy_cols = ["syria_border"]

    def prep(stays):
        g = gpd.GeoDataFrame(stays.copy(),
                             geometry=[swkb.loads(b) for b in stays["geometry_wkb"]],
                             crs=cfg.METRIC_CRS)
        t = idx.transfer_to_stays(g, feats, count_cols, pct_cols, dummy_cols)
        t = idx.urbanization_index(t)
        t = idx.damage_index(t, area=t["stay_area"])
        # city population of the stay (centroid within province)
        cent = g[["user_id", "stay_id"]].copy()
        cent = gpd.GeoDataFrame(cent, geometry=g.geometry.centroid, crs=cfg.METRIC_CRS)
        joined = gpd.sjoin(cent, load_cities(), how="left", predicate="within")
        joined = joined.drop_duplicates(["user_id", "stay_id"])
        t = t.merge(joined[["user_id", "stay_id", "city"]],
                    on=["user_id", "stay_id"], how="left")
        t["city_type"] = t["city"].map({k: v[1] for k, v in CITY_POPULATIONS.items()})
        t = t.merge(g.drop(columns="geometry_wkb"), on=["user_id", "stay_id"], how="left")
        return g, t

    log("transferring indices to origin stays ...")
    g_orig, t_orig = prep(origins.drop_duplicates(["user_id", "stay_id"]))
    log("transferring indices to destination stays ...")
    g_dest, t_dest = prep(dests.drop_duplicates(["user_id", "stay_id"]))

    log("weighting per person ...")
    w_orig = idx.weight_indices(t_orig, side="origin",
                                relevance_threshold=cfg.PARAMS.relevance_threshold)
    w_dest = idx.weight_indices(t_dest, side="destination",
                                relevance_threshold=cfg.PARAMS.relevance_threshold)
    table = w_orig.merge(w_dest, on="user_id", how="left", suffixes=("_origin", "_dest"))

    log("distances ...")
    o_mid = weighted_midpoints(t_orig.set_geometry("geometry"),
                               cfg.PARAMS.relevance_threshold + 0.1, "origin")
    d_mid = weighted_midpoints(t_dest.set_geometry("geometry"),
                               cfg.PARAMS.relevance_threshold + 0.1, "destination")
    dist = midpoint_distances(o_mid, d_mid)
    table = table.merge(dist[["user_id", "distance"]], on="user_id", how="left")

    log("timing ...")
    ev = events[events["movement_type_displacement"] == 1].copy()
    ev["user_id"] = ev["user_id"].astype("int64")
    ref = pd.Timestamp(cfg.PARAMS.disaster_date)
    ev["displacement_date"] = np.maximum(
        (pd.to_datetime(ev["home_end_date"], format="%Y%m%d") - ref).dt.days + 1, 1)
    ev["destination_date"] = np.maximum(
        (pd.to_datetime(ev["destination_start_date"], format="%Y%m%d") - ref).dt.days + 1, 1)
    ret_cols = [c for c in ["movement_type_return_displacement",
                            "movement_type_return_migration"] if c in events.columns]
    events["user_id_int"] = events["user_id"].astype("int64")
    has_return = (events.groupby("user_id_int")[ret_cols].max().max(axis=1)
                  if ret_cols else 0)
    timing = ev.groupby("user_id")[["displacement_date", "destination_date"]].max()
    table = table.merge(timing, on="user_id", how="left")
    table["has_return"] = table["user_id"].map(has_return).fillna(0).astype(int)

    table = table.merge(users, on="user_id", how="left")
    table = idx.origin_destination_differences(table)
    table.to_csv(os.path.join(RUN, "person_features.csv"), index=False)

    # keep the geo frames for the maps
    return table, g_orig, g_dest, t_orig, t_dest


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_fig4(g_orig, g_dest, table):
    boundary = load_cities().to_crs("EPSG:4326")
    aff = boundary[boundary["city"].isin(cfg.AFFECTED_CITY_NAMES)]
    xmin, ymin, xmax, ymax = aff.total_bounds
    bounds = (xmin - 0.4, xmax + 0.4, ymin - 0.3, ymax + 0.3)

    camps = gpd.read_file(cfg.CAMPS_SHP)
    tix = camps[camps["cmp_en"] == "TURKOGLU"].index
    camps.loc[tix, "geometry"] = Point(37.016083, 37.441337)
    camps = camps[(camps["Status"] == "ACTIVE") & (camps["Type"] == "Refugee Camp")]
    camps = camps.set_crs("EPSG:4326")

    centers = pd.DataFrame(CITY_CENTERS, columns=["name", "lat", "lon"])
    pts = {
        "City centers": (centers["lon"], centers["lat"], "o", "red"),
        "TACs": (camps.geometry.x, camps.geometry.y, "*", "yellow"),
    }

    panels = []
    for gdf, kind in [(g_orig, "Origin"), (g_dest, "Destination")]:
        g4 = gdf.to_crs("EPSG:4326")
        g4 = g4[g4["night_duration_aggregated"] > 0]
        seg = g4.merge(table[["user_id", "segment"]].drop_duplicates(), on="user_id")
        for s, label, color in [(cfg.SEGMENT_SYRIAN, "Syrian", "blue"),
                                (cfg.SEGMENT_TURKISH, "Turkish", "green")]:
            sub = seg[seg["segment"] == s]
            cent = sub.geometry.centroid
            panels.append({
                "title": f"{label} {kind} Stay Locations",
                "x": cent.x.to_numpy(), "y": cent.y.to_numpy(),
                "weights": sub["night_duration_aggregated"].to_numpy(),
                "color": color, "bounds": bounds,
            })
    # order: Syrian origin, Turkish origin, Syrian destination, Turkish destination
    fig = figs.kde_maps(panels, boundary=aff.boundary, points=pts,
                        density_threshold=0.5)
    fig.savefig(os.path.join(FIGDIR, "fig4_kde_maps.png"), dpi=200)
    log("fig4 saved")


def make_fig8_9(table):
    t = table[table["distance"] > 10]
    groups = {"Syrian": t[t["segment"] == cfg.SEGMENT_SYRIAN],
              "Turkish": t[t["segment"] == cfg.SEGMENT_TURKISH]}

    fig = figs.distance_and_timing(groups)
    fig.savefig(os.path.join(FIGDIR, "fig8_distance_timing.png"), dpi=200)

    cols = {
        "weighted_damage_index_diff": "Damage Index (difference)",
        "weighted_urbanization_index_diff": "Urbanization Index (difference)",
        "weighted_city_type_diff": "Population (difference)",
        "weighted_syria_border_diff": "Border Index (difference)",
    }
    fig = figs.index_difference_kdes(groups, cols)
    fig.savefig(os.path.join(FIGDIR, "fig9_index_differences.png"), dpi=200)
    log("fig8 + fig9 saved")

    return {
        "turkish_day1_share": float((groups["Turkish"]["displacement_date"] <= 1).mean()),
        "syrian_day1_share": float((groups["Syrian"]["displacement_date"] <= 1).mean()),
        "median_turkish_damage_diff":
            float(groups["Turkish"]["weighted_damage_index_diff"].median()),
        "n_features_turkish": int(len(groups["Turkish"])),
        "n_features_syrian": int(len(groups["Syrian"])),
    }


def make_fig10(table, g_dest):
    t = table[table["distance"] > 10]
    border_syr = t[(t["segment"] == cfg.SEGMENT_SYRIAN)
                   & (t["weighted_syria_border_diff"] == 1)]["user_id"].unique()
    border_tur = t[(t["segment"] == cfg.SEGMENT_TURKISH)
                   & (t["weighted_syria_border_diff"] == 1)]["user_id"].unique()

    cities = load_cities().to_crs("EPSG:4326")
    border_cities = cities[cities["city"].isin(["SANLIURFA", "GAZIANTEP", "KILIS", "HATAY"])]
    xmin, ymin, xmax, ymax = border_cities.total_bounds
    bounds = (xmin - 0.1, xmax + 0.1, ymin - 0.1, ymax + 0.35)

    sub = g_dest[g_dest["user_id"].isin(set(border_syr))].to_crs("EPSG:4326")
    cent = sub.geometry.centroid

    gates = pd.DataFrame(BORDER_GATES, columns=["name", "lat", "lon"])
    camps = gpd.read_file(cfg.CAMPS_SHP)
    camps.loc[camps["cmp_en"] == "TURKOGLU", "geometry"] = Point(37.016083, 37.441337)
    camps = camps[(camps["Status"] == "ACTIVE")
                  & (camps["Type"] == "Refugee Camp")].set_crs("EPSG:4326")
    centers = pd.DataFrame(CITY_CENTERS, columns=["name", "lat", "lon"])
    pts = {
        "Border gates": (gates["lon"], gates["lat"], "s", "red"),
        "TACs": (camps.geometry.x, camps.geometry.y, "*", "yellow"),
        "City centers": (centers["lon"], centers["lat"], "o", "darkred"),
    }
    fig = figs.kde_maps([{
        "title": f"Distribution of Syrian DPs Across Syrian Border "
                 f"(total number: {len(border_syr)})",
        "x": cent.x.to_numpy(), "y": cent.y.to_numpy(),
        "weights": sub["night_duration_aggregated"].to_numpy(),
        "color": "blue", "bounds": bounds,
    }], boundary=border_cities.boundary, points=pts, density_threshold=0.4)
    fig.savefig(os.path.join(FIGDIR, "fig10_border_map.png"), dpi=200)
    log("fig10 saved")
    return {"border_shifted_syrian": int(len(border_syr)),
            "border_shifted_turkish": int(len(border_tur))}


def make_fig11_table1(t_orig, t_dest, table):
    """Debias the DP flows and compare against TURKSTAT."""
    # dominant night city per person and side
    def dominant_city(t):
        d = t.dropna(subset=["city"]).sort_values("night_duration_aggregated",
                                                  ascending=False)
        return d.drop_duplicates("user_id")[["user_id", "city"]]

    o_city = dominant_city(t_orig).rename(columns={"city": "origin_city"})
    d_city = dominant_city(t_dest).rename(columns={"city": "destination_city"})
    flows = (o_city.merge(d_city, on="user_id")
             .merge(table[["user_id", "segment"]].drop_duplicates(), on="user_id"))
    pair_flows = (flows.groupby(["origin_city", "destination_city", "segment"])
                  .size().rename("observed_rate").reset_index())

    # sample sizes: dominant pre-disaster city per resident user
    log("computing per-city sample sizes from split stays ...")
    split = pd.read_parquet(os.path.join(RUN, "split_stays.parquet"),
                            columns=["user_id", "stay_id", "period", "duration",
                                     "geometry_wkb"])
    pre = split[split["period"] == "pre"]
    del split
    cent = gpd.GeoDataFrame(
        pre[["user_id", "duration"]].copy(),
        geometry=[swkb.loads(b).centroid for b in pre["geometry_wkb"]],
        crs=cfg.METRIC_CRS)
    joined = gpd.sjoin(cent, load_cities(), how="inner", predicate="within")
    per_city = joined.groupby(["user_id", "city"])["duration"].sum().reset_index()
    dom = per_city.sort_values("duration", ascending=False).drop_duplicates("user_id")
    segments = pd.read_csv(os.path.join(RUN, "user_segments.csv"))
    dom = dom.merge(segments, on="user_id", how="left")
    sample = (dom.groupby(["city", "segment"])["user_id"].nunique()
              .rename("customers").reset_index())

    # official data
    inflow_df = turkstat.read_migration_by_reason(
        os.path.join(cfg.DEBIAS_DIR, "migrant-inflows-2023.csv"), "inflow")
    outflow_df = turkstat.read_migration_by_reason(
        os.path.join(cfg.DEBIAS_DIR, "migrant-outflows-2023.csv"), "outflow")
    tr_pop = turkstat.read_turkish_population(
        os.path.join(cfg.DEBIAS_DIR, "turkish_pop_2023.csv")).rename(
        columns={"city": "region", "turkish_population": "population"})
    syr_pop = turkstat.read_syrian_population(
        os.path.join(cfg.DEBIAS_DIR, "syrian_pop_04-23.txt")).rename(
        columns={"city": "region", "syrian_population": "population"})

    results = {}
    adj_frames = {}
    for seg, pop, delta, label in [
        (cfg.SEGMENT_TURKISH, tr_pop, 0.21 + 0.01, "turkish"),
        (cfg.SEGMENT_SYRIAN, syr_pop, 0.40 + 0.01, "syrian"),
    ]:
        counts = sample[sample["segment"] == seg].rename(
            columns={"city": "region"})[["region", "customers"]]
        rates = deb.effective_sampling_rates(counts, pop, delta)
        f = pair_flows[(pair_flows["segment"] == seg)
                       & (pair_flows["origin_city"] != pair_flows["destination_city"])]
        adjusted = deb.debias_flows(f.rename(columns={"origin_city": "origin_region",
                                                      "destination_city":
                                                      "destination_region"}),
                                    rates, count_col="observed_rate")
        out = deb.adjusted_outflows(adjusted, cfg.AFFECTED_CITY_NAMES)
        inn = deb.adjusted_inflows(adjusted, cfg.AFFECTED_CITY_NAMES)
        # inflows including affected destinations (Table 1 convention)
        inn_all = (adjusted[adjusted["origin_region"].isin(cfg.AFFECTED_CITY_NAMES)
                            & (adjusted["origin_region"] != adjusted["destination_region"])]
                   .groupby("destination_region")["adjusted_flow"].sum()
                   .sort_values(ascending=False).rename("adjusted_inflow").reset_index())
        adj_frames[label] = (out, inn, inn_all)
        results[f"{label}_outflow_total"] = float(out["adjusted_outflow"].sum())

    # Table 1: Syrian flows
    out_s, _, in_s = adj_frames["syrian"]
    table1 = pd.concat([
        in_s.head(10).rename(columns={"destination_region": "Destination City",
                                      "adjusted_inflow": "DP Inflows"}).reset_index(drop=True),
        out_s.head(10).rename(columns={"origin_region": "Origin City",
                                       "adjusted_outflow": "DP Outflows"}).reset_index(drop=True),
    ], axis=1)
    table1.to_csv(os.path.join(FIGDIR, "table1_syrian_flows.csv"), index=False)

    # Fig 11: Turkish estimates vs TURKSTAT
    out_t, in_t, _ = adj_frames["turkish"]
    official_out = outflow_df[["city", "other"]].rename(
        columns={"other": "official"})
    official_in = inflow_df[["city", "other"]].rename(
        columns={"other": "official"})
    ov = out_t.rename(columns={"origin_region": "city",
                               "adjusted_outflow": "estimate"}).merge(
        official_out, on="city", how="inner")
    iv = in_t.rename(columns={"destination_region": "city",
                              "adjusted_inflow": "estimate"}).merge(
        official_in, on="city", how="inner")
    fig = figs.validation_scatter(iv, ov, label="ASA (spark rerun)")
    fig.savefig(os.path.join(FIGDIR, "fig11_validation.png"), dpi=200)
    log("fig11 + table1 saved")

    from scipy.stats import pearsonr

    results["pearson_outflow"] = float(pearsonr(ov["estimate"], ov["official"])[0])
    results["pearson_inflow"] = float(pearsonr(iv["estimate"], iv["official"])[0])
    results["syrian_top_inflows"] = in_s.head(10).round(0).to_dict("records")
    results["syrian_top_outflows"] = out_s.head(10).round(0).to_dict("records")
    return results


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    t0 = time.time()
    table, g_orig, g_dest, t_orig, t_dest = build_feature_table()

    headline = {
        "n_displaced_users": int(table["user_id"].nunique()),
        "n_displaced_by_segment":
            table.drop_duplicates("user_id")["segment"].value_counts().to_dict(),
    }
    headline.update(make_fig8_9(table))
    headline.update(make_fig10(table, g_dest))
    make_fig4(g_orig, g_dest, table)
    headline.update(make_fig11_table1(t_orig, t_dest, table))

    with open(os.path.join(FIGDIR, "headline_numbers.json"), "w") as f:
        json.dump(headline, f, indent=2, default=str)
    log(json.dumps(headline, indent=2, default=str))
    log(f"done in {(time.time() - t0) / 60:.1f} min")
