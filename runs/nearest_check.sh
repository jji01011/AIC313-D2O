#!/bin/bash
# Between two trainings: does the run's generator start to reproduce training images? (train split only, reporting only)
# Written 2026-10-13, when the distribution-matching teacher became one that sits closer to single training images
# (0.40% of its samples within 0.10 pixel RMS of one, against none before) and the user was away for a day.
# Takes the EMA generator of the run's state with output gain 1.10 (the gain the calibration has chosen since 2175 kimg),
# samples 3,020 images and appends two lines to results/nearest_train.log: FID terms and diversity
# (diagnostics/gap_probe.py), distance to the nearest training image (diagnostics/nearest_train.py).
# Nothing reads the log back, and the chain goes on whatever this script returns.
#   bash runs/nearest_check.sh <run> <kimg>
cd /home/jiin/AIC313-D2O || exit 1
export PYTHONPATH=. OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3
PY=/home/jiin/.venvs/d2o/bin/python
W=results/nearest_check
OUT=results/nearest_train.log
state=runs/$1/d2o-state-latest.pt
[ -f "$state" ] || { echo "$(date -u +%FT%TZ) $1 $2: no state file"; exit 1; }
$PY -W ignore calibrate_generator.py --state "$state" --gains 1.1 --fid 0 --out $W.ckpt || exit 1
{
    $PY -W ignore diagnostics/gap_probe.py --ckpt $W.ckpt --only_ckpt 1 --workdir $W 2>&1 | grep -a "nearest_check.ckpt"
    $PY -W ignore diagnostics/nearest_train.py "generator, gain 1.10=$W/student" 2>&1 | grep -a "^generator"
} | sed "s|^|$1 $2 $(date -u +%FT%TZ) |" >> $OUT
rm -f $W.ckpt
tail -n 2 $OUT
