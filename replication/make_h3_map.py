"""Interactive map of the H3 indicator datasets.

Builds a single self-contained HTML file (Leaflet via folium) with:

* the four displaced-people layers (Turkish/Syrian x origin/destination),
* the urbanization and damage indices,
* the active refugee camps (with the TURKOGLU coordinate correction
  described in the paper),

each as a toggleable layer with hover tooltips.

Usage: python make_h3_map.py [--out <html file>]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import geopandas as gpd
import folium
from matplotlib import colormaps
from matplotlib.colors import to_hex
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import local_config as cfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DP_LAYERS = [
    ("displaced_persons_origin_h3r6_turkish", "Turkish origin", "Greens"),
    ("displaced_persons_origin_h3r6_syrian", "Syrian origin", "Blues"),
    ("displaced_persons_destination_h3r6_turkish", "Turkish destination", "Oranges"),
    ("displaced_persons_destination_h3r6_syrian", "Syrian destination", "Purples"),
]
INDEX_LAYERS = [
    ("damage_index_h3r6", "Damage index", "Reds", "damage_index"),
    ("urbanization_index_h3r6", "Urbanization index", "viridis", "urbanization_index"),
]


def colorize(gdf, column, cmap_name, log=False):
    """Precompute a hex fill color per cell from the (optionally log) value."""
    vals = gdf[column].astype(float)
    v = np.log1p(vals) if log else vals
    vmax = v.max() if v.max() > 0 else 1.0
    cmap = colormaps[cmap_name]
    gdf = gdf.copy()
    gdf["_color"] = [to_hex(cmap(0.15 + 0.85 * x / vmax)) for x in v]
    return gdf


def add_hex_layer(m, gdf, name, value_col, alias, show):
    layer = folium.GeoJson(
        gdf[["h3_index", value_col, "_color", "geometry"]],
        name=name,
        show=show,
        style_function=lambda f: {
            "fillColor": f["properties"]["_color"],
            "color": f["properties"]["_color"],
            "weight": 0.5,
            "fillOpacity": 0.65,
        },
        tooltip=folium.GeoJsonTooltip(fields=["h3_index", value_col],
                                      aliases=["H3 cell", alias]),
    )
    layer.add_to(m)


def load_camps():
    camps = gpd.read_file(cfg.CAMPS_SHP)
    tix = camps[camps["cmp_en"] == "TURKOGLU"].index
    camps.loc[tix, "geometry"] = Point(37.016083, 37.441337)
    camps = camps[(camps["Status"] == "ACTIVE") & (camps["Type"] == "Refugee Camp")]
    return camps.set_crs("EPSG:4326", allow_override=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # defaults: the indicator_dataset folder and map file at the repo root
    ap.add_argument("--data", default=os.path.join(ROOT, "indicator_dataset"))
    ap.add_argument("--out", default=os.path.join(ROOT, "h3_indicator_map.html"))
    args = ap.parse_args()
    DATASET_DIR = args.data

    m = folium.Map(location=[37.6, 36.5], zoom_start=7, tiles="cartodbpositron",
                   prefer_canvas=True)

    for stem, name, cmap in DP_LAYERS:
        gdf = gpd.read_parquet(os.path.join(DATASET_DIR, f"{stem}.parquet"))
        gdf = colorize(gdf, "dp_count", cmap, log=True)
        add_hex_layer(m, gdf, f"DPs — {name}", "dp_count", "Displaced people",
                      show=(name == "Syrian destination"))

    for stem, name, cmap, col in INDEX_LAYERS:
        gdf = gpd.read_parquet(os.path.join(DATASET_DIR, f"{stem}.parquet"))
        gdf = gdf[gdf[col] > 0]
        gdf = colorize(gdf, col, cmap)
        add_hex_layer(m, gdf, name, col, name, show=False)

    camps = load_camps()
    fg = folium.FeatureGroup(name="Refugee camps", show=True)
    for _, r in camps.iterrows():
        folium.CircleMarker(
            location=[r.geometry.y, r.geometry.x], radius=5,
            color="black", weight=1.5, fill=True, fill_color="red",
            fill_opacity=0.9,
            tooltip=f"{r['cmp_en']} ({r['adm1_en']})",
        ).add_to(fg)
    fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    m.save(args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
