"""Scaling detected displacement counts to population level.

Mobile-operator samples over- or under-represent population groups and
regions. Detected origin/destination flows are debiased with per-region,
per-group effective sampling rates:

    effective_rate(region, group) = sample_size / (population * (1 - delta))
    adjusted_flow = observed_flow / effective_rate(origin region)

where delta is the share of the population not represented in the data
(e.g. children and the very old, who rarely hold subscriptions).
Inflows to unaffected regions are the sums of the adjusted flows from the
affected origins.
"""
from __future__ import annotations

import pandas as pd


def effective_sampling_rates(sample_counts: pd.DataFrame,
                             population: pd.DataFrame,
                             ineligible_share: float) -> pd.DataFrame:
    """Per-region effective sampling rate for one population group.

    sample_counts: [region, customers]; population: [region, population].
    """
    df = sample_counts.merge(population, on="region", how="inner")
    df["eligible_population"] = df["population"] * (1.0 - ineligible_share)
    df["effective_sampling_rate"] = df["customers"] / df["eligible_population"]
    return df


def debias_flows(flows: pd.DataFrame, rates: pd.DataFrame,
                 origin_col: str = "origin_region",
                 count_col: str = "flow") -> pd.DataFrame:
    """Adjusted flow = observed flow / effective sampling rate of the origin."""
    out = flows.merge(rates[["region", "effective_sampling_rate"]],
                      left_on=origin_col, right_on="region", how="left").drop(columns="region")
    out["adjusted_flow"] = out[count_col] / out["effective_sampling_rate"]
    return out


def adjusted_inflows(adjusted: pd.DataFrame, affected_regions: list,
                     origin_col: str = "origin_region",
                     destination_col: str = "destination_region") -> pd.DataFrame:
    """Population-level inflow to each unaffected region from the affected ones."""
    aff = adjusted[adjusted[origin_col].isin(affected_regions)]
    return (aff[~aff[destination_col].isin(affected_regions)]
            .groupby(destination_col)["adjusted_flow"].sum()
            .sort_values(ascending=False)
            .rename("adjusted_inflow")
            .reset_index())


def adjusted_outflows(adjusted: pd.DataFrame, affected_regions: list,
                      origin_col: str = "origin_region",
                      destination_col: str = "destination_region",
                      between_regions_only: bool = True) -> pd.DataFrame:
    """Population-level outflow of each affected region."""
    aff = adjusted[adjusted[origin_col].isin(affected_regions)]
    if between_regions_only:
        aff = aff[aff[origin_col] != aff[destination_col]]
    return (aff.groupby(origin_col)["adjusted_flow"].sum()
            .sort_values(ascending=False)
            .rename("adjusted_outflow")
            .reset_index())
