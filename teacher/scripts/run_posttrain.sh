#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PREP=${PREP:-data/teacher_prep}
TEACHER=${TEACHER:-E112}
RUN=${RUN:-runs/main_$TEACHER}

python -m teacher.posttrain_sweep --prep-dir "$PREP" --run-dir "$RUN" --stage all --out "$RUN/posttrain"
for steps in 32 64; do
    python -m teacher.eval_fid --prep-dir "$PREP" --net "$RUN/posttrain/handoff/teacher_unguided.pt" --seeds 0,1,2 \
        --steps "$steps" --out "$RUN/posttrain/saturation.jsonl"
done
