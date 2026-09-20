#!/usr/bin/env python3
"""
Loading a cell and splitting its model population.

A "cell" is one (state, protocol, train-domain, seed): a population of models
trained on one domain, with their per-example scores on the in-domain and
out-of-domain halves of the same held-out tiles. zoo.py writes it; everything
downstream reads it through here so the model splits are defined in exactly one
place.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

# Tree ensembles vs everything else. The family-disjoint split exists because a
# RANDOM split of the model population does not make the halves independent when
# the population varies mainly along one axis (overall quality): a subset tuned
# to anti-correlate with quality in one half keeps anti-correlating in the other
# whether or not any spurious feature is involved. Disjoint learner families
# break that shared axis. This is the analogue of the paper's ResNet/ViT
# ablation, and it is the control that decides whether a result is real.
TREE_FAMILIES = {"hgb", "rf", "et", "dt", "xgb"}

SPATIAL_COLS = ["lon", "lat", "block_id", "tile_50km", "county_fips",
                "meta_comid", "meta_source_cit"]


class Cell:
    """Arrays for one cell, plus the model-split bookkeeping."""

    def __init__(self, tag: str, npz, models: pd.DataFrame):
        self.tag = tag
        self.S_ood = npz["S_ood"]                 # (n_models, n_ood_points)
        self.S_id = npz["S_id"]
        self.y_ood = npz["y_ood"]
        self.models = models
        self.n_models, self.d = self.S_ood.shape
        for c in SPATIAL_COLS + ["tiles_ood"]:
            setattr(self, c, np.asarray(npz[c]) if c in npz.files else None)
        if self.tile_50km is None and self.tiles_ood is not None:
            self.tile_50km = self.tiles_ood       # pilot-era cells

    @property
    def x(self) -> np.ndarray:
        """Probit ID performance, one value per model."""
        from oodselect import probit
        return probit(self.models.score_id.to_numpy())

    def has_spatial(self) -> bool:
        return self.lon is not None and self.lat is not None

    def require_spatial(self) -> None:
        if not self.has_spatial():
            raise SystemExit(
                f"{self.tag}: no lon/lat in the cell. These were added to zoo.py "
                f"for the spatial work; regenerate the cell with the current "
                f"zoo.py before running anything that needs a graph.")

    def splits(self, seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(selection, validation, reporting) model indices, by split rule.

        'random-split' is the paper's 60/20/20. The two family splits hold out
        an entire learner family; they have no separate validation block, so the
        restart is chosen on the training objective there.
        """
        rng = np.random.default_rng(seed)
        N = self.n_models
        perm = rng.permutation(N)
        n_tr, n_va = int(0.6 * N), int(0.2 * N)
        out = {"random-split": (perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:])}

        is_tree = self.models.kind.isin(TREE_FAMILIES).to_numpy()
        tr_i, nt_i = np.where(is_tree)[0], np.where(~is_tree)[0]
        if len(tr_i) >= 8 and len(nt_i) >= 8:
            out["trees->others"] = (tr_i, np.array([], int), nt_i)
            out["others->trees"] = (nt_i, np.array([], int), tr_i)
        return out


def cell_tag(state: str, protocol: str, train_dom: str, seed: int) -> str:
    return f"{state}_{protocol}_{train_dom}_s{seed}"


def result_tag(state: str, protocol: str, train_dom: str, seed: int,
               mu: float, gph: str) -> str:
    """The naming every result file carries, so a row can be traced to its run."""
    mu_s = f"{mu:g}".replace(".", "p")
    return f"{state}_{protocol}_{train_dom}_s{seed}_mu{mu_s}_{gph}"


def load_cell(cells_dir: str | Path, tag: str) -> Cell:
    cells_dir = Path(cells_dir)
    npz_p = cells_dir / f"{tag}.npz"
    csv_p = cells_dir / f"{tag}_models.csv"
    if not npz_p.exists():
        raise SystemExit(f"missing cell {npz_p} -- run zoo.py for this cell first")
    if not csv_p.exists():
        raise SystemExit(f"missing {csv_p} (the model table that pairs with the npz)")
    return Cell(tag, np.load(npz_p, allow_pickle=True), pd.read_csv(csv_p))


def list_cells(cells_dir: str | Path) -> list[str]:
    return sorted(p.stem for p in Path(cells_dir).glob("*.npz"))
