#!/bin/bash
# Teacher samples for the teacher-centred experiments (2026-10-15): the 58.7M teacher, EMA 0.13, 32 Heun steps from
# mean_image + 80 * noise (its own prior). A reference set (5 shards, seeds 5000+) for selection, never shown to a
# discriminator, then the set a discriminator may take as real images (17 shards, seeds 1000+). Runs next to training.
cd /home/jiin/AIC313-D2O || exit 1
export PYTHONPATH=. OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3
PY=/home/jiin/.venvs/d2o/bin/python
T=checkpoints/edm/teacher_unguided-0058720256-0.130.pt
$PY -W ignore diagnostics/teammate_edm2/generate_teacher_samples.py --teacher $T --steps 32 --mean_start 1 --seed 5000 --shards 5 --outdir data/cache/teacher58_ref --low_gb 0.5 || exit 1
$PY -W ignore diagnostics/teammate_edm2/generate_teacher_samples.py --teacher $T --steps 32 --mean_start 1 --seed 1000 --shards 17 --outdir data/cache/teacher58_real --low_gb 0.5
