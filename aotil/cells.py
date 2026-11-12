#!/usr/bin/env python3
"""
Load a cell written by zoo.py.

A cell is one model population (state, negative protocol, training zone, seed; the paper's is FL_band_AE_s0):
the models trained on Zone AE, with their scores on the in-distribution (ID, Zone AE) and out-of-distribution
(OOD, Zone A) points of the same held-out tiles. Inputs: <cells_dir>/<tag>.npz and <cells_dir>/<tag>_models.csv.
Every stage reads cells through load_cell(cells_dir, tag).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Optional per-point arrays: one value per OOD point (zoo.SPATIAL_COLS), and the
# ID half's point keys and labels (zoo.ID_COLS stored as <col>_id, plus y_id).
SPATIAL_COLS = ["point_id", "lon", "lat", "block_id", "tile_50km", "county_fips",
                "meta_comid", "meta_source_cit"]
ID_KEYS = ["point_id_id", "lon_id", "lat_id", "y_id"]


class Cell:
    """The arrays of one cell and its model table."""

    def __init__(self, tag: str, npz, models: pd.DataFrame):
        self.tag = tag
        self.S_ood = npz["S_ood"]                 # (n_models, n_ood_points)
        self.S_id = npz["S_id"]                   # (n_models, n_id_points)
        self.y_ood = npz["y_ood"]
        self.models = models
        self.n_models, self.d = self.S_ood.shape

        # One attribute per optional key, named after the key; a key the cell lacks loads as None.
        for key in SPATIAL_COLS + ID_KEYS + ["tiles_ood"]:
            value = np.asarray(npz[key]) if key in npz.files else None
            setattr(self, key, value)
        if self.tile_50km is None and self.tiles_ood is not None:
            self.tile_50km = self.tiles_ood

    def has_spatial(self) -> bool:
        return self.lon is not None and self.lat is not None

    def require_spatial(self) -> None:
        if not self.has_spatial():
            raise SystemExit(f"{self.tag}: the cell has no lon/lat; rebuild it with "
                             f"run_experiments.py --stage zoo --force.")


def load_cell(cells_dir: str | Path, tag: str) -> Cell:
    cells_dir = Path(cells_dir)
    npz_path = cells_dir / f"{tag}.npz"
    csv_path = cells_dir / f"{tag}_models.csv"
    if not npz_path.exists():
        raise SystemExit(f"missing cell {npz_path}: run `run_experiments.py --stage zoo` first")
    if not csv_path.exists():
        raise SystemExit(f"missing {csv_path} (the model table that pairs with the npz): "
                         f"run `run_experiments.py --stage zoo --force`")
    models = pd.read_csv(csv_path)
    return Cell(tag, np.load(npz_path, allow_pickle=True), models)
