"""Activity Space Approach (ASA) — displacement detection from mobile
positioning data.

ASA measures disaster-induced displacement as a sustained loss of familiar
living space rather than a change of a single home location. The library is
data-agnostic: callers describe their signal and site tables with
``asa.schemas`` and supply all study-specific inputs (disaster date, affected
regions, spatial parameters) through ``asa.params.ASAParams`` and function
arguments.

Modules
-------
asa.params      hyperparameters of the four ASA stages
asa.schemas     input column mappings
asa.spark       distributed pipeline (signals -> daily trajectories)
asa.kernels     single-machine per-user algorithms used by the pipeline
asa.detection   segment-based displacement detection, O/D attribution
asa.indices     spatial context indices and per-person O/D features
asa.debias      scaling of detected counts to population level
asa.figures     standard result figures
"""
from .params import ASAParams  # noqa: F401
from .schemas import CDRSchema, SiteSchema  # noqa: F401
