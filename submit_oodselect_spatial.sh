#!/bin/bash
# =============================================================================
# submit_oodselect_spatial.sh -- FIRST EXPERIMENT, VERSION 2 OF 2
# The classic OODSelect of submit_oodselect_classic.sh plus the simplest
# spatial-coherence regulariser: a graph-Laplacian smoothness term on the
# relaxed selection weights z = sigmoid(theta),
#
#     loss = [paper's loss] + mu * z' L z / (S * k),      mu = 3, k = 10
#
# L is the Laplacian of the symmetric 10-nearest-neighbour graph on the
# projected OOD point locations, so z'Lz counts cut edges and S*k normalises it
# to an incoherence score in [0, ~1] (graph.py). mu = 3 is the operating point
# from job 562664. mu is FIXED, not tuned: the term can only cost correlation,
# so letting Optuna pick it on validation r would drive it to zero.
#
# Everything else is identical to the classic run -- same worker, same cells,
# same 300 models, same model split, same Optuna seeds per budget, same
# validation / selection / test rule -- so the two results differ by that one
# term and nothing else.
#
# Run from the repo root on the login node, after ./upload_to_cluster.sh:
#   ./submit_oodselect_spatial.sh --date YYYY-MM-DD
#   DRY_RUN=1 ./submit_oodselect_spatial.sh --date YYYY-MM-DD    # print only
# Submitting both with the same --date is intended: the report job writes
# <cell>_comparison.csv once both variants are present.
# Knobs: see oodselect_paper_fanout.sh.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./oodselect_paper_fanout.sh

MU=3
KNN=${KNN:-10}
oodsel_submit "$MU" "$@"
