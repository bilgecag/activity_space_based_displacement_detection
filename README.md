# activity_space_based_displacement_detection

This library provide tools to process mobile‐phone CDR/XDR/GPS data for measuring disaster‐induced displacements with Activity Space Approach (ASA). Key components include:

# Clone the repo
git clone https://github.com/bilgecag/activity_space_based_displacement_detection.git
cd activity_space_based_displacement_detection

# (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

python main.py \
  --cdr_folder data/cdr \
  --tower_file data/towers.csv \
  --voronoi_shapefile data/voronoi.shp \
  --clusters_file data/clusters.csv \
  --output_dir results/

.

├── activity_space_approach/      # Computes activity-space centroids
├── clustering/                   # Clustering utilities for towers & users
├── home_location_approach/       # Derives “home” locations pre- and post-event
├── migration_detector/           # Detects displacement based on sequences
├── cdr_processing.py             # Read & filter CDR/XDR, customer‐signal analysis :contentReference[oaicite:1]{index=1}
├── cell_tower_processing.py      # Read towers, build Voronoi polygons
├── spatial_calculations.py       # Geospatial helper functions
├── main.py                       # CLI entry point to run end-to-end pipeline
├── requirements.txt              # Python dependencies
└── README.md                     # (This file)
