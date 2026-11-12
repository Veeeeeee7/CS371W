"""Shared paths, colours, matplotlib style and model-scoring helpers for the scripts in this folder.

Importing it selects the Agg backend and applies the paper's style. Correctness, the model split
and r follow aotil/oodselect_paper.py. Run the scripts from this folder, e.g.
    cd results/2026-11-08/paper_figures && ../../../.venv/bin/python fig_maps_combined.py
This copy reads the run of docs/TASK-REVISED-EXPERIMENTS-2026-11-07.md: the cell cells_v4_2026-11-07
on data/processed_v4, split 0 only. Copied on 2026-11-08 from results/2026-11-07/paper_figures for the
redrawn maps figure; only this docstring changed.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Repo root as a string ending in '/' (this file is results/<date>/paper_figures/common.py).
R = str(Path(__file__).resolve().parents[3]) + '/'
EPS = 1e-6  # probit clip and variance floor of the OODSelect objective

# The run these scripts read, relative to R: the cells, the selections, the regions and the
# regions table of the v3 rebuild, and its one split.
RUN = 'results/2026-11-07/'
CELLS_DIR = 'data/interim/aotil/cells_v4_2026-11-07/'
SELECTIONS_DIR = RUN + 'oodselect_paper/'
REGIONS_DIR = RUN + 'regions/'
REGIONS_LAND = RUN + 'o5/regions_land.csv'
DATASET = 'data/processed_v4/dataset_FL.parquet'
SEEDS = [0]

BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#8a8984'
GRID = '#e4e3df'

plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 7.5,
    'axes.titlesize': 8, 'axes.labelsize': 7.5, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'pdf.fonttype': 42,  # embed TrueType fonts; paper checkers reject Type 3
    'savefig.dpi': 300,
})


def correctness(S, y):
    """1 where a model is right on a point (float32, models x points).

    S is the stored true-class probability. argmax picks class 0 on a tie, so S == 0.5 is right
    only when y == 0."""
    y = np.asarray(y)
    right = (S > 0.5) | ((S == 0.5) & (y[None, :] == 0))
    return right.astype(np.float32)


def model_split(n):
    """Train, validation and test model indices: the 60/20/20 split of oodselect_paper."""
    idx = np.arange(n)
    rest, te = train_test_split(idx, test_size=0.2, random_state=42)
    tr, va = train_test_split(rest, test_size=0.25, random_state=42)
    return tr, va, te


def probit_np(x):
    """Probit of x clipped to [EPS, 1 - EPS], in float64, with torch's erfinv as OODSelect uses."""
    x = np.clip(np.asarray(x, float), EPS, 1 - EPS)
    return math.sqrt(2) * torch.erfinv(torch.as_tensor(2 * x - 1)).numpy()


def probit_t(v):
    """Probit of a torch tensor clamped to [EPS, 1 - EPS]."""
    return math.sqrt(2) * torch.erfinv(2 * torch.clamp(v, EPS, 1 - EPS) - 1)


def r_mask(Z, acc, mask):
    """The OODSelect objective in float32 torch: Pearson r between the probit of each model's ID
    accuracy and the probit of its accuracy on the points weighted by mask (Z: models x points)."""
    scores = torch.as_tensor(Z.astype(np.float32))
    id_acc = torch.as_tensor(acc.astype(np.float32))
    weights = torch.as_tensor(mask.astype(np.float32))
    mask_acc = (scores @ weights) / (weights.sum() + 1e-6)
    x = probit_t(mask_acc)
    y = probit_t(id_acc)
    x_centred = x - x.mean()
    y_centred = y - y.mean()
    cov = (x_centred * y_centred).mean()
    x_var = (x_centred ** 2).mean() + EPS
    y_var = (y_centred ** 2).mean() + EPS
    return float(cov / torch.sqrt(x_var * y_var))


def load_seed(s):
    """The split's cell (the open npz), the correctness on its Zone A test points and each model's
    accuracy on its held-out Zone AE points."""
    z = np.load(R + CELLS_DIR + f'FL_band_AE_s{s}.npz', allow_pickle=True)
    Z_ood = correctness(z['S_ood'], z['y_ood'])
    Z_id = correctness(z['S_id'], z['y_id'])
    return z, Z_ood, Z_id.mean(1)


def topS(v, s, S=500):
    """0/1 float mask of the S points with the largest theta in the S = 500 run of OODSelect
    variant v on seed s; ties go to the earlier point."""
    theta = np.load(R + SELECTIONS_DIR + f'{v}/FL_band_AE_s{s}_S500_theta.npy')
    mask = np.zeros(len(theta))
    mask[np.argsort(-theta, kind='stable')[:S]] = 1
    return mask


def region_names():
    """(seed, component) -> the names of the counties the region lies in, e.g. 'Dixie/Lafayette',
    from REGIONS_LAND; a name repeated within a split gets ' (2)', ' (3)' in component order."""
    regions = pd.read_csv(R + REGIONS_LAND, dtype={'counties': str})
    counties = pd.read_parquet(R + 'data/interim/gis/counties_FL.parquet', columns=['GEOID', 'NAME'])
    county_name = dict(zip(counties.GEOID, counties.NAME))
    names = {}
    seen = {}
    for row in regions.sort_values(['seed', 'component']).itertuples():
        base = '/'.join(county_name.get(g, g) for g in row.counties.split())
        seen[(row.seed, base)] = seen.get((row.seed, base), 0) + 1
        count = seen[(row.seed, base)]
        names[(row.seed, row.component)] = base if count == 1 else f'{base} ({count})'
    return names
