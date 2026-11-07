#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PREP=${PREP:-data/teacher_prep}
TEACHER=${TEACHER:-E112}
TOTAL_STEPS=${TOTAL_STEPS:?set TOTAL_STEPS to the main run length printed by the benchmark}
FLAGS=${FLAGS:-}
HOURS=${HOURS:-6}
EXTRA_ARMS=${EXTRA_ARMS:-}
HALF_LR_ARM=${HALF_LR_ARM:-0}
HALF_LR_PRESET=${HALF_LR_PRESET:-$TEACHER}

LR=$(python -c "from teacher import presets; print(presets.NETWORK_PRESETS['$TEACHER']['ref_lr'])")
LR2=$(python -c "print(2 * $LR)")
LRH=$(python -c "from teacher import presets; print(0.5 * presets.NETWORK_PRESETS['$HALF_LR_PRESET']['ref_lr'])")

arm() {
    local name=$1 preset=$2; shift 2
    python -m teacher.train --prep-dir "$PREP" --run-dir "runs/pilot_$name" --preset "$preset" --subset pilot \
        --total-steps "$TOTAL_STEPS" --max-hours "$HOURS" --keep-states 1 $FLAGS "$@"
}

arm "${TEACHER}_lr1x" "$TEACHER" --ref-lr "$LR"
arm "${TEACHER}_lr2x" "$TEACHER" --ref-lr "$LR2"
arm DDPMpp128 DDPMpp128
RUNS="runs/pilot_${TEACHER}_lr1x,runs/pilot_${TEACHER}_lr2x,runs/pilot_DDPMpp128"
for a in $EXTRA_ARMS; do
    arm "$a" "$a"
    RUNS="$RUNS,runs/pilot_$a"
done
if [ "$HALF_LR_ARM" = 1 ]; then
    arm "${HALF_LR_PRESET}_lr0.5x" "$HALF_LR_PRESET" --ref-lr "$LRH"
    RUNS="$RUNS,runs/pilot_${HALF_LR_PRESET}_lr0.5x"
fi

python -m teacher.pilot_compare --prep-dir "$PREP" --runs "$RUNS" --out runs/pilot_decision.json
