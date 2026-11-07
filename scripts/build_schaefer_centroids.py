"""build_schaefer_centroids.py
==============================
One-off setup script: fetches the Schaefer-400 (17-network) atlas via nilearn
and writes its 400 parcel centroid coordinates (MNI mm, canonical parcel
order) to data/atlases/schaefer400_17networks/centroids.json.

The order must match the fMRI data columns. The RBC timeseries are
atlas-Schaefer2018p400n17, and the 7- and 17-network atlases number the same
parcels differently, so a 7-network file builds a graph between the wrong parcels.

Run this once, on a machine with internet access:

    python scripts/build_schaefer_centroids.py

After that, main.py's --graph_mode spatial --dataset fmri reads the cached
JSON directly (see data.rbc_dataset.schaefer_centroids) -- no nilearn atlas
download happens inside the training loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "atlases"
    / "schaefer400_17networks"
    / "centroids.json"
)


def main() -> None:
    from nilearn import datasets, plotting

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=400, yeo_networks=17, resolution_mm=2
    )
    coords, labels = plotting.find_parcellation_cut_coords(
        atlas.maps, return_label_names=True
    )
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != (400, 3):
        raise RuntimeError(
            f"Expected 400 parcel centroids, got {coords.shape}. The fetched "
            "atlas may not match Schaefer-400/17-network."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "yeo_networks": 17,
                "centroids": coords.tolist(),
                "labels": [str(l) for l in labels],
            }
        )
    )
    print(f"Wrote {OUT_PATH} ({coords.shape[0]} centroids)")


if __name__ == "__main__":
    main()
