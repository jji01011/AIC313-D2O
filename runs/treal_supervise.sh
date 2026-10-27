#!/bin/bash
# Keeps the teacher samples of the teacher-centred line coming and hands the line to the director once they are all there
# (2026-10-15, the user away for two days). cron calls it every 10 minutes; it does nothing once results/treal_setup.done
# exists, and only one copy runs at a time. While shards are missing and runs/gen_teacher58.sh is not running (killed
# with the session that started it, a reboot, too little GPU memory), it starts that script again, detached; the
# generator skips finished shards. When all 22 shards exist it runs runs/treal_setup.py once (reference statistics,
# floor, director entry), which writes results/treal_setup.done on success. Log: results/treal_supervise.log.
# Stop it: touch runs/treal_supervise.stop
cd /home/jiin/AIC313-D2O || exit 1
[ -e results/treal_setup.done ] && exit 0
[ -e runs/treal_supervise.stop ] && exit 0
exec 7> runs/treal_supervise.lock
flock -n 7 || exit 0
log() { echo "$(date -u +%FT%TZ) $*" >> results/treal_supervise.log; }
starts=0
while [ ! -e results/treal_setup.done ] && [ ! -e runs/treal_supervise.stop ]; do
    ref=$(ls data/cache/teacher58_ref/shard-*.pt 2>/dev/null | wc -l)
    real=$(ls data/cache/teacher58_real/shard-*.pt 2>/dev/null | wc -l)
    if [ "$ref" -ge 5 ] && [ "$real" -ge 17 ]; then
        pgrep -f "generate_teacher_samples.py" > /dev/null && { sleep 30; continue; }
        log "all shards there ($ref + $real): runs/treal_setup.py"
        PYTHONPATH=. OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 /home/jiin/.venvs/d2o/bin/python -W ignore runs/treal_setup.py >> results/treal_setup.out 2>&1
        status=$?
        log "runs/treal_setup.py exited with status $status"
        [ -e results/treal_setup.done ] || { log "no results/treal_setup.done: trying again in 10 minutes"; sleep 600; }
        continue
    fi
    if ! pgrep -f "generate_teacher_samples.py" > /dev/null && ! pgrep -f "bash runs/gen_teacher58.sh" > /dev/null; then
        starts=$((starts + 1))
        if [ "$starts" -gt 40 ]; then log "the generator was started 40 times without finishing: giving up (h20 keeps the GPU)"; exit 1; fi
        log "shards $ref + $real of 5 + 17, the generator is not running: starting runs/gen_teacher58.sh (start $starts)"
        setsid nohup bash runs/gen_teacher58.sh >> results/gen_teacher58.out 2>&1 < /dev/null &
        sleep 300
        continue
    fi
    sleep 60
done
