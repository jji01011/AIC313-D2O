#!/bin/bash
# A piece of a held-out line (runs/e112_h20, 2026-10-15), selected on the held-out train images H (diagnostics/make_holdout.py:
# 20 per category, 3,020, never seen by the run's discriminator). The README allows train images for selection.
#   bash runs/h20_score.sh <run> <kimg>
cd /home/jiin/AIC313-D2O || exit 1
REFS=results/reference_holdout_h20,results/reference_holdout_c5,results/reference_train,results/reference_train_balanced \
    exec bash runs/ref_score.sh results/reference_holdout_h20 results/h20_calibration.log 'e112_h20[a-z0-9_]*' "held-out line" "$1" "$2"
