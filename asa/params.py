"""Parameters of the Activity Space Approach (ASA).

ASA detects disaster-induced displacements from mobile positioning data
(CDR/xDR) in four steps:

    1. Stay locations    — spatio-temporal clustering of raw signals
    2. Activity spaces   — DBSCAN grouping of each user's stay locations,
                           before and after the disaster
    3. Relevance scores  — nighttime familiarity of post-disaster locations,
                           from the spatial overlap with pre-disaster spaces
    4. Displacement      — segment-based detection on the daily binary
                           familiar/unfamiliar trajectory

Everything data-specific (file locations, column names, study area) is
supplied by the caller; see ``asa.schemas.CDRSchema`` and the function
arguments of ``asa.spark``.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class ASAParams:
    """Hyperparameters of the four ASA stages.

    Attributes
    ----------
    disaster_date : datetime.date
        Date of the disaster. Signals are split into a pre- and a
        post-disaster period at this date (see ``disaster_time``).
    disaster_time : datetime.time
        Optional time-of-day of the split instant, for disasters where the
        exact onset matters. Defaults to midnight.
    stay_distance_m : float
        Spatial threshold of stay detection: a signal extends the current
        stay only if it is within this distance of both the previous signal
        and the first signal of the stay.
    stay_duration_s : float
        Temporal threshold of stay detection: minimum elapsed time from the
        first signal before a stay is established.
    dbscan_eps_m : float
        Maximum distance between stay locations clustered into the same
        activity space (DBSCAN eps; min_samples is fixed to 1 so no stay is
        treated as noise).
    relevance_threshold : float
        Nighttime-relevance percentage below which a location counts as
        unfamiliar (the binary trajectory value is 0).
    night_start_hour / night_end_hour : int
        Nighttime window (default 22:00-07:00), used both for the relevance
        weighting and for choosing the representative stays of each day.
    k_days : int
        Minimum number of days spent at a location for the segment-based
        detector to accept it as an origin/destination segment.
    epsilon_days : int
        Maximum gap of missing days bridged inside a segment.
    small_seg_len, seg_prop, min_overlap_part_len :
        Remaining parameters of the segment detector (Chi et al. 2020),
        with the defaults used throughout the ASA experiments.
    legacy_day_night_accounting : bool
        When True, reproduces the day/night split of the reference
        implementation, which does not count the final morning night hours
        (00:00-07:00) of multi-day stays ending at or after 22:00. Leave
        False for the mathematically exact split.
    """

    disaster_date: dt.date = dt.date(2023, 2, 6)
    disaster_time: dt.time = dt.time(0, 0)

    stay_distance_m: float = 2_000.0
    stay_duration_s: float = 7_200.0

    dbscan_eps_m: float = 5_000.0

    relevance_threshold: float = 5.0
    night_start_hour: int = 22
    night_end_hour: int = 7

    k_days: int = 14
    epsilon_days: int = 14
    small_seg_len: int = 2
    seg_prop: float = 0.5
    min_overlap_part_len: int = 0

    legacy_day_night_accounting: bool = False

    @property
    def disaster_timestamp(self) -> dt.datetime:
        """The pre/post split instant."""
        return dt.datetime.combine(self.disaster_date, self.disaster_time)

    @property
    def max_gap_home_des(self) -> int:
        """Maximum gap between origin and destination segments (detector)."""
        return self.epsilon_days
