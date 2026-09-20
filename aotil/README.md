# aotil — spatially regularized OODSelect

Implementation of OODSelect (Salaudeen et al., [arXiv:2510.24884](https://arxiv.org/abs/2510.24884))
on the FEMA Zone A / Zone AE label-provenance benchmark, with a graph-Laplacian
smoothness term that turns its scattered subsets into physical regions.

Method spec: `docs/SPATIAL-OODSELECT.md`. Framing: `claude/project-abstract.md`.

---

## Run this first

```bash
./run_aotil_laptop.sh          # ~2 min, no cluster, no GPU, no data needed for stage 1
```

Stage 1 is `aotil/test_aotil.py`: finite-difference checks on every hand-written
gradient, numpy/torch backend agreement, μ behaviour, the controls, the coherence
metrics, and an end-to-end pass on synthetic cells. Stage 2 runs a 24-model cell
on the real Florida parquet to prove the data path.

**The gradient checks are the point.** Every gradient in `oodselect.py` is derived
by hand. A wrong one does not crash — it quietly optimises a different objective
and returns a subset that looks like a result. If you change the objective, change
those checks in the same edit.

## Then the cluster

```bash
./upload_to_cluster.sh && ./upload_data_to_cluster.sh
LIMIT=1 ./submit_aotil_all.sh --date $(date +%F)     # watch one job first
./submit_aotil_all.sh --date $(date +%F)             # then the rest
./download_from_cluster.sh $(date +%F) --cells
```

One job per **cell** = (state, protocol, train-domain, tile-split seed). Every
output, cell and log is per-cell, so the jobs are independent.

---

## Where things live on the cluster

Two trees, split by size and by whether a loss costs a number or only compute.

| | path | holds | size |
|---|---|---|---|
| **home** | `/users/$USER/cs371w` | code, `results/<date>/aotil/`, `logs/<date>/` | a few hundred KB of code, a few MB per run |
| **scratch** | `/scratch/$USER/cs371w` | `data/processed/`, `aotil/cells/` | ~220 MB + ~20–40 MB per cell |

Home quota is the binding constraint, and the parquets and the cells are the
only large things involved, so both go to scratch. **Scratch is purged**, which
is survivable exactly because both are reconstructible: the parquets re-upload
from the laptop in minutes, and a cell refits from them. Nothing that is a
result is ever written there — `results/` and `logs/` stay in the home checkout
on purpose, so a purge costs compute and never costs a number.

`run_aotil.sh` resolves this itself: it uses `AOTIL_ARTIFACT_ROOT` if set,
otherwise `/scratch/$USER/cs371w` **when `/scratch` exists**, otherwise
repo-local paths. It checks for `/scratch` rather than trusting `mkdir -p` to
fail, because on a machine with a writable `/` that mkdir succeeds everywhere
and would quietly put the laptop's cells somewhere nobody looks again. The
laptop therefore runs unchanged, out of `data/processed` and
`data/interim/aotil/cells`.

Override any of it:

```bash
AOTIL_ARTIFACT_ROOT=/some/other/scratch sbatch run_aotil.sh --date <D>
DATA_DIR=... CELLS_DIR=... sbatch run_aotil.sh --date <D>
CS371_REMOTE_SCRATCH=/scratch/vmli3/cs371w ./upload_data_to_cluster.sh
```

The sync scripts use the same split: `upload_to_cluster.sh` sends code to home
and creates both trees, `upload_data_to_cluster.sh` sends the parquets to
scratch, and `download_from_cluster.sh <date>` pulls results and logs from home
while `--cells` pulls the cells from scratch. Pull the cells once a run
finishes — you need them to re-sweep or re-characterise without refitting the
model population, and scratch will not keep them.

## What each file does

| file | role |
|---|---|
| `zoo.py` | fits the heterogeneous model population and writes the cell `.npz` — the models, their per-example scores on the ID and OOD halves of the same held-out tiles, and the spatial columns |
| `gate.py` | ID spread, the aggregate ID→OOD correlation, and the paper's model-count convergence rule |
| `graph.py` | spatial adjacency, the Laplacian, and every coherence metric. Runnable self-test |
| `oodselect.py` | the selection: relaxed objective, hand-written gradients, Adam + cosine annealing, restarts, μ term, numpy and torch backends |
| `cells.py` | loading a cell and defining the model splits — in one place, so every stage splits identically |
| `pointcov.py` | **E0**, the go/no-go: per-point covariance and its Moran's I null |
| `analyse.py` | **E1**, the μ sweep with every control at every μ — the core result |
| `transfer.py` | **E2**, held-out-point transfer inside the region |
| `characterise.py` | metadata enrichment of a selected subset |
| `test_aotil.py` | the self-test |

## The order the phases must run in

`zoo` → `gate` → **E0** → E1 → E2. E0 is a gate, not a formality: if the
per-point covariance map is inside its permutation null it is spatial white
noise, no μ > 0 can assemble a region from it, and the sweep will spend a day
confirming that. `STOP_ON_NOGO=1` makes the driver stop there. Read the verdict
line in the log before queueing the rest.

## Where the compute goes

The model population is scikit-learn on CPU and is roughly 95% of the wall clock;
the selection that follows is seconds. The GPU earns its place only because
restarts and μ values batch into a single mat-vec (`DEVICE=cuda`), and because
the XGBoost family can use it. **This workload does not need a B200.** If the GPU
queue is long, `GPUS=0 DEVICE=cpu` costs very little.

Size the population with `gate.py`'s convergence table rather than the paper's
4,200: it reports the smallest population whose correlation is within 1% of the
full one, which is the paper's own stated criterion (about ten models for VLCS,
six for WILDS-Camelyon). Past that point more models buy precision in the
*selection*, not in the correlation.

## Two deviations from the paper, both forced by this data

1. **Continuous scores.** The paper uses binary correctness `Z_ij`. AUC has no
   per-example decomposition, and a 0.5 threshold would mean different things
   across cells whose achieved positive:negative ratios run 0.18 to 1.04. `S_ij`
   is the predicted probability of the observed class; the objective is unchanged.
2. **The spatial term.** `μ · wᵀLw / (S·k)`, normalised so P is an incoherence
   score in [0, ~1] and a given μ means the same thing in every cell. μ = 0 is the
   paper's objective exactly.

## The controls are the result

`random` must stay positive — that is what makes aggregate accuracy-on-the-line an
averaging artifact. `hard` (most-misclassified, built from the **selection models
only**) near zero or positive is what separates a spurious-correlation finding
from "these examples are difficult". `null-select` runs the identical
optimisation against shuffled ID performance **at the same μ**, because the
spatial constraint changes what the optimiser can manufacture from noise; if it
inverts, nothing else in the row means anything. The two family-disjoint model
splits exist because a random split does not make the halves independent when the
population varies mainly along overall quality.

μ is chosen on **purity**, never on test r. Purity is a property of the mask and
the graph — no model enters it — so choosing on it cannot leak the reported
correlation.

## Carried over from the pilot

Under `band`, the inverse line is real (r = −0.80 to −0.90 on held-out models,
most-misclassified positive, null near zero, holds across family-disjoint splits).
Under `shadedX` it is not separable from difficulty, which is why `band` is
primary and `shadedX` is the sensitivity arm. The pilot's cells carried no
lon/lat and cannot be reused; `zoo.SPATIAL_COLS` is the fix.
