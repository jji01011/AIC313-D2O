#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3
PY=${PY:-python}
T=checkpoints/edm/teacher_unguided-0058720256-0.130.pt
$PY -W ignore diagnostics/teammate_edm2/generate_teacher_samples.py --teacher $T --steps 32 --mean_start 1 --seed 5000 --shards 5 --outdir data/cache/teacher58_ref --low_gb 0.5 || exit 1
$PY -W ignore diagnostics/teammate_edm2/generate_teacher_samples.py --teacher $T --steps 32 --mean_start 1 --seed 1000 --shards 17 --outdir data/cache/teacher58_real --low_gb 0.5
