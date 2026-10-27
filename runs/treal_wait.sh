#!/bin/bash
# Waits for runs/gen_teacher58.sh to finish, then runs runs/treal_setup.py once (2026-10-15). Start detached.
cd /home/jiin/AIC313-D2O || exit 1
while pgrep -f "bash runs/gen_teacher58.sh" > /dev/null; do sleep 60; done
PYTHONPATH=. OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 /home/jiin/.venvs/d2o/bin/python -W ignore runs/treal_setup.py >> results/treal_setup.out 2>&1
