#!/bin/bash
# Unattended GPU schedule (the two trainings do not fit in 20 GB together):
#   1. pause EDM pre-training (it saves its state on SIGTERM)
#   2. first full D2O run from the 1000-kimg EDM snapshot
#   3. resume EDM pre-training to 5000 kimg
# Start detached:  setsid nohup bash runs/chain_d2o_then_edm.sh >> runs/chain.log 2>&1 < /dev/null &
# Stop D2O early but keep the chain going:  kill -TERM $(cat runs/d2o_c96_edm1000/train.pid)
cd /home/jiin/AIC313-D2O || exit 1
PY=/home/jiin/.venvs/d2o/bin/python
EDM=runs/edm_c96/edm-001000.pt
OUT=runs/d2o_c96_edm1000
log() { echo "$(date -u +%FT%TZ) $*"; }

EDM_PID=$(cat runs/edm_c96/train.pid)
if kill -0 "$EDM_PID" 2>/dev/null; then
    log "pausing EDM pre-training (pid $EDM_PID)"
    kill -TERM "$EDM_PID"
    while kill -0 "$EDM_PID" 2>/dev/null; do sleep 5; done
fi

log "multi-step FID of $EDM"
$PY -W ignore eval_edm.py --edm "$EDM" 2>&1 | grep -a "NFE Heun" | tee -a results/edm_fid.log

mkdir -p "$OUT"
log "starting D2O in $OUT"
$PY -W ignore train_d2o.py --edm "$EDM" --outdir "$OUT" \
    --freeze 0 --xflip 1 --ada_target 0.6 --kimg 1000 --snap_kimg 25 >> "$OUT/train.log" 2>&1 &
echo $! > "$OUT/train.pid"
wait $!
log "D2O exited with status $?"

log "resuming EDM pre-training to 5000 kimg"
$PY train_edm.py --outdir ./runs/edm_c96 --resume ./runs/edm_c96/edm-state-latest.pt --kimg 5000 >> runs/edm_c96/train.log 2>&1 &
echo $! > runs/edm_c96/train.pid
wait $!
log "EDM exited with status $?"
