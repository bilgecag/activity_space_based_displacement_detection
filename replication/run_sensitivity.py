"""Hyperparameter sensitivity of the detected displacement counts.

From the daily familiarity trajectories of the main run, re-detects
displacements across:

    * DBSCAN activity-space distances (3/5/10/20 km — the four trajectory
      columns of the main run)
    * relevance thresholds (0/5/10/15 %)
    * detector parameters k and epsilon (3/5/7/10/14 days)

and draws boxplots of the displaced-people counts (paper Figs 6 and 7,
DBSCAN/relevance/k/epsilon panels; the stay spatio-temporal threshold panel
requires separate pipeline runs per threshold and is out of scope here).

Usage: python run_sensitivity.py
"""
from __future__ import annotations

import dataclasses
import itertools
import os
import sys
import time

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import local_config as cfg  # noqa: E402
from asa.kernels.trajectories import binarize_trajectories  # noqa: E402
from asa.detection import find_migrants, classify_events  # noqa: E402

RUN = cfg.OUTPUT_DIR
FIGDIR = os.path.join(os.path.dirname(RUN), "figures")

RELEVANCE_THRESHOLDS = [0.0, 5.0, 10.0, 15.0]
K_EPS_PAIRS = ([(k, 14) for k in [3, 5, 7, 10, 14]]
               + [(14, e) for e in [3, 5, 7, 10]])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if __name__ == "__main__":
    daily = pd.read_parquet(os.path.join(RUN, "daily_series.parquet"))
    share = pd.read_parquet(os.path.join(RUN, "region_share.parquet"))
    population = set(share[share["region_night_relevance"]
                           > cfg.HABITUAL_REGION_SHARE_MIN]["user_id"])
    log(f"daily series {len(daily):,} rows; population {len(population):,}")

    rows = []
    for (i, eps), thr, (k, e) in itertools.product(
            enumerate(cfg.DBSCAN_EPS_LIST), RELEVANCE_THRESHOLDS, K_EPS_PAIRS):
        binary = binarize_trajectories(daily, f"location_{i+1}", thr)
        params = dataclasses.replace(cfg.PARAMS, k_days=k, epsilon_days=e)
        migrants = find_migrants(binary, params, cfg.DETECTOR_PATHS)
        events = classify_events(
            migrants.sort_values(["user_id", "migration_date"])[[
                "user_id", "migration_date", "home_start_date", "home_end_date",
                "destination_start_date", "destination_end_date",
                "home", "destination"]],
            disaster_date=cfg.PARAMS.disaster_date.strftime("%Y%m%d"))
        events["uid"] = events["user_id"].astype("int64")
        disp = events[(events.get("movement_type_displacement", 0) == 1)
                      & events["uid"].isin(population)]
        n = disp["user_id"].nunique()
        rows.append({"dbscan_km": int(eps) // 1000, "relevance_pct": thr,
                     "k": k, "epsilon": e, "n_displaced": n})
        log(f"eps={int(eps)//1000}km thr={thr:>4} k={k:>2} e={e:>2} -> {n:,}")

    counts = pd.DataFrame(rows)
    counts.to_csv(os.path.join(FIGDIR, "sensitivity_counts.csv"), index=False)

    from asa.figures import sensitivity_boxplots

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    base = counts[(counts["k"] == 14) & (counts["epsilon"] == 14)]
    sensitivity_boxplots(base, "dbscan_km", "Impact of DBSCAN Distance (km)", axes[0, 0])
    sensitivity_boxplots(base, "relevance_pct", "Impact of Relevance Threshold (%)",
                         axes[0, 1])
    sensitivity_boxplots(counts[counts["epsilon"] == 14], "k",
                         "ASA: Impact of k", axes[1, 0])
    sensitivity_boxplots(counts[counts["k"] == 14], "epsilon",
                         "ASA: Impact of ε", axes[1, 1])
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig6_7_sensitivity.png"), dpi=200)
    log("sensitivity figure saved")
    log(f"count range: {counts['n_displaced'].min():,} - {counts['n_displaced'].max():,}")
