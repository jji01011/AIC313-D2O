#!/bin/bash
# Unattended GPU schedule after EDM pre-training (one job at a time: two trainings do not fit in 20 GB).
#   1. wait until EDM pre-training (runs/edm_c96) has ended
#   2. multi-step FID of the last EDM snapshots; the best snapshot by train-reference FID initialises D2O
#   3. the D2O jobs listed in runs/queue_final.txt, in order, re-reading the file after every job
# Start detached:  setsid nohup bash runs/chain_final.sh >> runs/chain.log 2>&1 < /dev/null &
# Starting it again after it died (reboot, kill) is safe: every job whose run directory already holds a state is continued
# from that state, and a job that had reached its --kimg ends at once without writing anything.
# End the current D2O job early (the chain moves on):  kill -TERM $(cat <outdir>/train.pid)
cd /home/jiin/AIC313-D2O || exit 1
PY=/home/jiin/.venvs/d2o/bin/python
log() { echo "$(date -u +%FT%TZ) $*"; }

# One instance only: a second chain would start a second training in the same run directory.
exec 9> runs/chain_final.lock
flock -n 9 || { log "another chain_final.sh is already running; this one exits"; exit 1; }

EDM_PID=$(cat runs/edm_c96/train.pid)
# The command line is checked too: after a reboot the recorded PID can belong to an unrelated process.
while grep -qa train_edm.py "/proc/$EDM_PID/cmdline" 2>/dev/null; do sleep 30; done
[ -f runs/edm_c96/edm-005000.pt ] || log "WARNING: EDM pre-training ended before 5000 kimg; continuing with the snapshots that exist"

for kimg in 004500 005000; do
    snapshot=runs/edm_c96/edm-$kimg.pt
    [ -f "$snapshot" ] || continue
    grep -q "^$snapshot:" results/edm_fid.log && continue
    log "multi-step FID of $snapshot"
    $PY -W ignore eval_edm.py --edm "$snapshot" 2>&1 | grep -a "NFE Heun" | tee -a results/edm_fid.log
done
EDM=$(awk '{print $NF, $1}' results/edm_fid.log | sort -g | head -n 1 | awk '{print $2}' | tr -d ':')
log "D2O initialisation: $EDM"
BASE="--edm $EDM --freeze 0 --xflip 1 --ada_target 0.6 --lr 5e-5 --dlr 1e-4 --snap_kimg 25 --keep_snapshots 0"

done_jobs=0
while true; do
    job=$(grep -v '^#' runs/queue_final.txt | grep -v '^[[:space:]]*$' | sed -n "$((done_jobs + 1))p")
    [ -z "$job" ] && break
    set -- $job
    out=$1; shift
    mkdir -p "$out"
    # Never restart an existing run: a fresh start would replace its best.ckpt and state (train_d2o.py refuses it as well).
    case " $* " in
        *" --resume "*) ;;
        *) [ -f "$out/d2o-state-latest.pt" ] && set -- "$@" --resume "$out/d2o-state-latest.pt" ;;
    esac
    log "starting D2O in $out: $*"
    $PY -W ignore train_d2o.py $BASE --outdir "$out" "$@" >> "$out/train.log" 2>&1 &
    echo $! > "$out/train.pid"
    wait $!
    log "D2O in $out exited with status $?"
    done_jobs=$((done_jobs + 1))
done
log "queue empty"
