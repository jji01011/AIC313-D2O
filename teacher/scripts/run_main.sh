#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PREP=${PREP:-data/teacher_prep}
DECISION=${DECISION:-runs/pilot_decision.json}
TEACHER=${TEACHER:-$(python -c "import json; print(json.load(open('$DECISION'))['decision']['winner']['preset'])")}
TOTAL_STEPS=${TOTAL_STEPS:?set TOTAL_STEPS}
FLAGS=${FLAGS:-}
STOP_AT=${STOP_AT:-2026-10-17T23:59+09:00}
RUN=${RUN:-runs/main_$TEACHER}

if ls "$RUN"/training-state-*.pt >/dev/null 2>&1; then
    RESUME=(--resume auto)
else
    if [ "$(python -c "import json; print(int(json.load(open('$DECISION'))['decision']['rerun_half_lr']))")" = 1 ]; then
        echo "$DECISION waits for the 0.5 x alpha_ref arm: run HALF_LR_ARM=1 bash teacher/scripts/run_pilot.sh first" >&2
        exit 1
    fi
    STATE=$(python -c "import json; print(json.load(open('$DECISION'))['decision']['winner']['state'])")
    RESUME=(--resume "$STATE" --allow-data-switch)
    if [ "${PRUNE_PILOT:-0}" = 1 ]; then
        python - "$DECISION" <<'PY'
import glob, json, os, sys
d = json.load(open(sys.argv[1]))
win = os.path.abspath(d['decision']['winner']['run_dir'])
for arm in d['arms']:
    if os.path.abspath(arm['run_dir']) != win:
        files = glob.glob(os.path.join(arm['run_dir'], 'training-state-*.pt')) + glob.glob(os.path.join(arm['run_dir'], 'ema-*.pt'))
        freed = sum(os.path.getsize(p) for p in files)
        for p in files:
            os.remove(p)
        print(f"pruned {arm['run_dir']}: {len(files)} files, {freed / 1e9:.1f} GB")
PY
    fi
fi

python -m teacher.train --prep-dir "$PREP" --run-dir "$RUN" --preset "$TEACHER" --subset main \
    "${RESUME[@]}" --total-steps "$TOTAL_STEPS" --stop-at "$STOP_AT" $FLAGS
python -c "import json; t = json.load(open('$RUN/training_time.json')); print(f\"total training time {t['total_wall_hours']:.2f} h ({t['mimg']:.1f} Mimg, stop: {t['stop_reason']})\")"
