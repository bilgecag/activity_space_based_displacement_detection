# Activity Space Approach (ASA) — displacement detection from mobile phone data

Code and aggregated datasets accompanying:

> *A novel activity space approach to discover displacement patterns via
> mobile phone data: An analysis of the 2023 Türkiye–Syria Earthquakes*
> (EPJ Data Science).

The repository contains a generic, data-agnostic Spark library implementing
the Activity Space Approach (`asa/`), the study-specific replication scripts
(`replication/`), and privacy-preserving indicator datasets aggregated on the
H3 hexagonal grid (`indicator_dataset/`).

## Layout

```
asa/                    generic library (no study-specific paths or constants)
  params.py             ASAParams: disaster date/time, thresholds, hyperparameters
  schemas.py            column-name mappings for arbitrary CDR / tower exports
  kernels/              pandas/numba kernels: stay detection, day-night
                        accounting, activity spaces (DBSCAN + hulls),
                        relevance overlay, daily binary trajectories
  spark/                distributed pipeline (bucketed applyInPandas)
  detection.py          displacement event detection and origin/destination
                        stay matching, weighted midpoints, distances
  indices.py            urbanization / damage / border indices and per-person
                        weighted features
  debias.py             observed-rate debiasing to population estimates
  figures.py            figure helpers
replication/            everything specific to the 2023 Türkiye–Syria study
  local_config.py       file locations, CDR column layout, disaster instant,
                        affected provinces, published hyperparameters
  run_pipeline.py       end-to-end run: raw CDR -> displaced people + stays
  make_replication_report.py
                        builds the per-person feature table
                        (person_features.csv, needed by the scripts below)
                        plus quick-look figures and the headline numbers
                        used to compare the rerun against the publication
  make_paper_figures.py exact paper figures: executes the publication's
                        original plotting code (kept verbatim in
                        paper_figure_code/, one file per figure)
  build_h3_indicators.py post-processing: builds indicator_dataset/ (see below)
  make_h3_map.py        interactive HTML map of the indicator datasets
  run_sensitivity.py    hyperparameter sensitivity sweep
reference_data/
  refugee_camps/        Türkiye refugee-camp point locations (bundled;
                        used by the map and the paper-figure overlays)
indicator_dataset/      aggregated, k-anonymized datasets (see below)
h3_indicator_map.html   interactive map of the indicator datasets
```

## Running the pipeline

```bash
pip install -r requirements.txt
python replication/run_pipeline.py             # raw CDR -> displaced people
python replication/make_replication_report.py  # per-person feature table
                                                # + summary figures/numbers
python replication/make_paper_figures.py       # exact paper figures
python replication/build_h3_indicators.py      # H3 indicator datasets
python replication/make_h3_map.py              # interactive map
```

The scripts run in this order: everything after `run_pipeline.py` reads its
outputs; `make_paper_figures.py` and `build_h3_indicators.py` additionally
need the per-person feature table written by `make_replication_report.py`,
and `make_h3_map.py` draws from `indicator_dataset/`.

All input locations live in `replication/local_config.py`; the `asa` library
itself never references study-specific files. To apply the method to another
setting, provide your own config module (a `CDRSchema` describing your CDR
columns, an `ASAParams` with the disaster date, and tower/polygon reference
files).

### Configuring data locations

The repository is self-contained for everything that may be redistributed:
the aggregated datasets, the map, and the refugee-camp reference points
ship inside it. The raw study inputs cannot be shared and must be pointed
to on your machine — either edit the "external data roots" block at the
top of `replication/local_config.py`, or export the environment variables
(the environment always wins):

| variable | points to | needed by |
|---|---|---|
| `ASA_MOBILE_DATA` | raw Turkcell CDR + tower data (proprietary) | `run_pipeline.py` and everything downstream |
| `ASA_ADMIN_BOUNDARIES` | Türkiye admin boundaries (`tur_polbnda_adm1/adm2`, OCHA/HDX) | feature table, figures |
| `ASA_MIGRATION_DETECTOR` | directory containing the `migration_detector` package | `run_pipeline.py` (detection stage) |
| `ASA_TURKSTAT_DIR` | TURKSTAT migration/population tables | Fig 11 validation only |
| `ASA_PAPER_RESULTS` | cached result files of the original study | Fig 11 TMB baseline in `make_paper_figures.py` only |

Regenerating the shipped artifacts (`indicator_dataset/`, the map) does not
require any of these once a pipeline run exists; the map alone needs
nothing but `indicator_dataset/` and the bundled camp points.

## indicator_dataset/

Aggregated datasets safe for public sharing. All spatial units are
**H3 resolution-6 hexagons** (~36 km²), identified by the `h3_index` column.
Files are **GeoParquet** with the hexagon polygon of every cell in
EPSG:4326 — they can be dragged directly into QGIS (3.28+) or read with
`geopandas.read_parquet`.

### Displaced-people counts

| file | content |
|---|---|
| `displaced_persons_origin_h3r6_turkish.parquet` | Turkish DPs by origin cell |
| `displaced_persons_origin_h3r6_syrian.parquet` | Syrian DPs by origin cell |
| `displaced_persons_destination_h3r6_turkish.parquet` | Turkish DPs by destination cell |
| `displaced_persons_destination_h3r6_syrian.parquet` | Syrian DPs by destination cell |

Columns: `h3_index`, `dp_count`, `geometry`.

`dp_count` is an **estimated displaced-population count** — the paper's
extrapolated, debiased estimates distributed over the observed spatial
pattern — not a count of observed devices.

Construction (see `replication/build_h3_indicators.py`), separately per
group and side:

1. Pre-disaster stay areas of a displaced person are their origin,
   post-disaster stay areas their destination. A stay location is not a
   person, so each observed displaced person contributes one unit of mass
   distributed over the H3 cells of their stay-location centroids
   proportionally to the nights spent in each (their night-spent share).
   This gives the spatial distribution of the actual stay locations, the
   same support as the density maps in the paper.
2. **Primary k-anonymity (k = 10)**: only cells to which at least ten
   distinct observed people of the group contribute are kept.
3. The paper's extrapolated displaced-population estimates are
   distributed over the protected cells proportionally to their observed
   mass: **860,000 Turkish DPs** (560,000 displaced to other cities +
   300,000 within their city boundaries) and **70,000 Syrian DPs**
   (55,000 + 15,000), rounded to whole people so each file sums exactly
   to the group total.
4. **Secondary cut**: any cell whose extrapolated count is below ten is
   removed, so no number below ten appears anywhere in the data (this
   removes at most a few tens of people per file; exact figures in
   `metadata.json`, together with the extrapolated and published totals
   and suppressed-cell counts).

### Context indices

| file | content |
|---|---|
| `urbanization_index_h3r6.parquet` | urbanization index (0–1) per hexagon |
| `damage_index_h3r6.parquet` | earthquake damage index (0–1) per hexagon |

The urbanization index is the intersection-area-weighted mean of the
cell-context weights used in the paper (dense urban = 1.0, urban = 0.8, …,
rural = 0). The damage index is the severity-weighted damaged-building
density (collapsed = 1.0, needs demolition = 0.8, heavily damaged = 0.7,
slightly damaged = 0.3), min-max normalized over all hexagons.

`metadata.json` records the resolution, the k-anonymity parameter, the
extrapolated and published totals, and the suppressed-cell counts per
dataset.

### Interactive map

`h3_indicator_map.html` (repository root) shows all six indicator layers
and the active refugee camps as toggleable layers with hover tooltips.
Open it in any browser — the file is self-contained except for the
basemap tiles, which are fetched online. Regenerate it from the datasets
with `python replication/make_h3_map.py`.

## Privacy

No individual-level records are included. The raw CDR data cannot be shared;
all published datasets are spatial aggregates with k-anonymity (k = 10)
applied as described above.
