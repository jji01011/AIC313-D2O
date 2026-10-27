#!/bin/bash
# Unattended GPU schedule (one job at a time: two trainings do not fit in 20 GB).
# Runs the D2O jobs listed in runs/queue.txt in order, re-reading the file after every job,
# then resumes EDM pre-training to 5000 kimg.
# Start detached:  setsid nohup bash runs/chain_queue.sh >> runs/chain.log 2>&1 < /dev/null &
# End the current D2O job early (the chain moves on):  kill -TERM $(cat <outdir>/train.pid)
cd /home/jiin/AIC313-D2O || exit 1
PY=/home/jiin/.venvs/d2o/bin/python
BASE="--edm runs/edm_c96/edm-001000.pt --freeze 0 --xflip 1 --ada_target 0.6 --kimg 150 --snap_kimg 25"
log() { echo "$(date -u +%FT%TZ) $*"; }

done_jobs=0
while true; do
    job=$(grep -v '^#' runs/queue.txt | grep -v '^[[:space:]]*$' | sed -n "$((done_jobs + 1))p")
    [ -z "$job" ] && break
    set -- $job
    out=$1; shift
    mkdir -p "$out"
    log "starting D2O in $out: $*"
    $PY -W ignore train_d2o.py $BASE --outdir "$out" "$@" >> "$out/train.log" 2>&1 &
    echo $! > "$out/train.pid"
    wait $!
    log "D2O in $out exited with status $?"
    done_jobs=$((done_jobs + 1))
done

log "queue empty; resuming EDM pre-training to 5000 kimg"
$PY train_edm.py --outdir ./runs/edm_c96 --resume ./runs/edm_c96/edm-state-latest.pt --kimg 5000 >> runs/edm_c96/train.log 2>&1 &
echo $! > runs/edm_c96/train.pid
wait $!
log "EDM exited with status $?"
