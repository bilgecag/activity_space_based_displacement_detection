"""Input schemas.

ASA works on two tables that the caller provides in any format Spark can
read; the schema objects map the caller's column names onto the canonical
names used internally.

Canonical signal columns:   user_id, time, site_id [, group]
Canonical site columns:     site_id, location_id, x, y [, region_id, area_wkt]

``location_id`` is the spatial unit of stay detection — typically a tower or
a privacy-preserving tower cluster. ``x``/``y`` must be in a metric CRS.
``area_wkt`` optionally carries the service-area polygon (e.g. a Voronoi
cell) used to build stay-location polygons; without it, stay locations are
represented by point convex hulls of the site coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CDRSchema:
    """Column mapping for the raw signal table."""

    user_id: str = "user_id"
    time: str = "time"
    site_id: str = "site_id"
    group: str | None = None          # optional population segment column

    def rename_map(self) -> dict:
        m = {self.user_id: "user_id", self.time: "time", self.site_id: "site_id"}
        if self.group:
            m[self.group] = "group"
        return m


@dataclass(frozen=True)
class SiteSchema:
    """Column mapping for the site/tower reference table."""

    site_id: str = "site_id"
    location_id: str = "location_id"
    x: str = "x"
    y: str = "y"
    region_id: str | None = None      # optional admin area of the site
    area_wkt: str | None = None       # optional service-area polygon (WKT)

    def rename_map(self) -> dict:
        m = {self.site_id: "site_id", self.location_id: "location_id",
             self.x: "x", self.y: "y"}
        if self.region_id:
            m[self.region_id] = "region_id"
        if self.area_wkt:
            m[self.area_wkt] = "area_wkt"
        return m
