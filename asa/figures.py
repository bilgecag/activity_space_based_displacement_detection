"""Standard ASA result figures.

Plot helpers for the main displacement analyses: kernel-density maps of
origin/destination stay locations, displacement-distance distributions,
cumulative displacement timing, index-difference distributions, and the
validation scatter against official statistics. All functions take plain
DataFrames and return the matplotlib figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402


def weighted_kde_grid(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None,
                      bounds: tuple, grid: int = 400,
                      bw_scale: float = 0.3) -> tuple:
    """Weighted Gaussian KDE evaluated on a regular grid.

    bounds = (x_min, x_max, y_min, y_max). Bandwidth: Silverman's rule
    scaled by ``bw_scale`` to preserve local hotspots.
    """
    kde = gaussian_kde(np.vstack([x, y]),
                       weights=None if weights is None else weights / weights.sum(),
                       bw_method="silverman")
    kde.set_bandwidth(kde.factor * bw_scale)
    xs = np.linspace(bounds[0], bounds[1], grid)
    ys = np.linspace(bounds[2], bounds[3], grid)
    xx, yy = np.meshgrid(xs, ys)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def _hybrid_cmap(base: str, threshold: float) -> LinearSegmentedColormap:
    """Transparent -> base color, shifting hue above the threshold share."""
    if base == "blue":
        light, mid, dark = ((0.75, 0.82, 0.96, 0.35), (0.42, 0.5, 0.88, 0.7),
                            (0.29, 0.08, 0.55, 0.95))
    else:  # green
        light, mid, dark = ((0.78, 0.92, 0.77, 0.35), (0.35, 0.68, 0.35, 0.7),
                            (0.0, 0.35, 0.15, 0.95))
    return LinearSegmentedColormap.from_list(
        f"hybrid_{base}",
        [(0.0, (1, 1, 1, 0)), (0.03, light), (threshold, mid), (1.0, dark)])


def kde_maps(panels: list, boundary=None, points: dict | None = None,
             suptitle: str | None = None, grid: int = 400,
             bw_scale: float = 0.3, density_threshold: float = 0.5):
    """Grid of KDE maps.

    panels: list of dicts with keys
        title, x, y, weights (optional), color ('blue'|'green'), bounds
    boundary: optional GeoSeries plotted as context outline (lon/lat)
    points: optional {label: (x_arr, y_arr, marker, color)} overlays
    """
    n = len(panels)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    grids = []
    for p in panels:
        xx, yy, zz = weighted_kde_grid(p["x"], p["y"], p.get("weights"),
                                       p["bounds"], grid, bw_scale)
        grids.append((xx, yy, zz))

    for ax, p, (xx, yy, zz) in zip(axes, panels, grids):
        if boundary is not None:
            boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5)
        cmap = _hybrid_cmap(p.get("color", "blue"), density_threshold)
        ax.imshow(zz / zz.max(), extent=(p["bounds"][0], p["bounds"][1],
                                     p["bounds"][2], p["bounds"][3]),
                  origin="lower", cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for label, (px, py, marker, color) in (points or {}).items():
            ax.scatter(px, py, marker=marker, c=color, s=28, label=label,
                       edgecolors="black", linewidths=0.4, zorder=5)
        ax.set_title(p["title"])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(p["bounds"][0], p["bounds"][1])
        ax.set_ylim(p["bounds"][2], p["bounds"][3])
    for ax in axes[n:]:
        ax.axis("off")
    if points:
        axes[0].legend(loc="lower right", fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


DAMAGE_CATEGORY_BINS = [(-np.inf, 0.01, "Low", "#1f77b4"),
                        (0.01, 0.1, "Medium", "#ff7f0e"),
                        (0.1, 0.2, "High", "#d62728"),
                        (0.2, np.inf, "Very High", "#9467bd")]


def damage_category(damage_index: pd.Series) -> pd.Series:
    """Origin damage exposure category from the weighted damage index."""
    out = pd.Series(index=damage_index.index, dtype=object)
    for lo, hi, label, _ in DAMAGE_CATEGORY_BINS:
        out[(damage_index > lo) & (damage_index <= hi)] = label
    return out


def distance_and_timing(features_by_group: dict):
    """2x2 figure: displacement-distance KDE by damage category (top) and
    cumulative displacement timing (bottom), one column per group.

    features_by_group: {group_label: DataFrame with distance (m),
                        weighted_damage_index_origin, displacement_date}
    """
    fig, axes = plt.subplots(2, len(features_by_group), figsize=(16, 10))
    axes = np.atleast_2d(axes)

    for j, (label, df) in enumerate(features_by_group.items()):
        cats = damage_category(df["weighted_damage_index_origin"])
        ax = axes[0, j]
        for lo, hi, cat, color in DAMAGE_CATEGORY_BINS:
            vals = df.loc[cats == cat, "distance"] / 1000.0
            if len(vals) > 5:
                kde = gaussian_kde(vals)
                xs = np.linspace(0, 1200, 400)
                ax.plot(xs, kde(xs), color=color, label=f"{cat} (n={len(vals)})")
        ax.set_title(f"Distance distribution by damage category — {label}")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Density")
        ax.legend(title="Damage category", fontsize=8)

        ax = axes[1, j]
        for lo, hi, cat, color in DAMAGE_CATEGORY_BINS:
            days = df.loc[cats == cat, "displacement_date"].astype(float)
            if len(days) > 5:
                xs = np.arange(1, 26)
                cum = [(days <= d).mean() * 100 for d in xs]
                ax.plot(xs, cum, marker="o", ms=3, color=color,
                        label=f"{cat} (n={len(days)})")
        ax.set_title(f"Cumulative displacement timing — {label}")
        ax.set_xlabel("Days after the disaster")
        ax.set_ylabel("Cumulative share of DPs (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def index_difference_kdes(diff_by_group: dict, columns: dict):
    """KDE of destination-minus-origin index differences per group.

    diff_by_group: {group_label: DataFrame}; columns: {column: panel title}.
    """
    n = len(columns)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 4.5 * ((n + 1) // 2)))
    axes = np.atleast_1d(axes).ravel()
    colors = {"Syrian": ("#4c72b0", 0.35), "Turkish": ("#2e7d32", 0.35)}

    for ax, (col, title) in zip(axes, columns.items()):
        for label, df in diff_by_group.items():
            vals = df[col].dropna().astype(float)
            if len(vals) < 5 or vals.std() == 0:
                continue
            kde = gaussian_kde(vals)
            xs = np.linspace(vals.quantile(0.001), vals.quantile(0.999), 400)
            color, alpha = colors.get(label, ("#666666", 0.3))
            ax.plot(xs, kde(xs), color=color, label=label)
            ax.fill_between(xs, kde(xs), color=color, alpha=alpha)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("Density")
        ax.legend()
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def validation_scatter(inflows: pd.DataFrame, outflows: pd.DataFrame,
                       label: str = "ASA"):
    """Scatter of population-scaled CDR estimates vs official statistics.

    inflows/outflows: DataFrames with columns [city, estimate, official].
    """
    from scipy.stats import pearsonr

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, df, title in [(axes[0], inflows, "DP inflow to non-affected cities"),
                          (axes[1], outflows, "DP outflow from the affected region")]:
        d = df.dropna(subset=["estimate", "official"])
        r, p = pearsonr(d["estimate"], d["official"])
        ax.scatter(d["estimate"], d["official"], c="tab:blue", label=label)
        lim = max(d["estimate"].max(), d["official"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="45° line")
        for _, row in d.nlargest(8, "official").iterrows():
            ax.annotate(row["city"], (row["estimate"], row["official"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.set_title(f"{title}\nPearson r = {r:.2f}{stars}")
        ax.set_xlabel(f"{label} estimate (people)")
        ax.set_ylabel("TURKSTAT")
        ax.legend()
    fig.tight_layout()
    return fig


def sensitivity_boxplots(counts: pd.DataFrame, by: str, title: str, ax=None):
    """Boxplot of detected-DP counts grouped by one hyperparameter."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    groups = sorted(counts[by].unique())
    data = [counts.loc[counts[by] == g, "n_displaced"] for g in groups]
    ax.boxplot(data, labels=[str(g) for g in groups], patch_artist=True,
               boxprops=dict(facecolor="#87CEEB"))
    ax.set_title(title)
    ax.set_xlabel(by)
    ax.set_ylabel("Number of displaced people")
    ax.set_ylim(bottom=0)
    return ax
