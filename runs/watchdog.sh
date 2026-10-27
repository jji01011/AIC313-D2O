#!/bin/bash
# Keeps the unattended queue alive (written 2026-10-11: the user is away for a day or more and the server may go down).
# cron calls it every 10 minutes (crontab -l). It does three things, each logged in runs/watchdog.log:
#   1. Disk: below 3 GB free it removes the next entries of REMOVABLE in runs/director.py (the director does the same,
#      but only between two training pieces).
#   2. Restart: if runs/chain_final.sh is gone although its queue is not finished (reboot, kill), it starts the chain
#      again. Before that it marks every queue line above the last training line as done: a restarted chain reads the
#      queue from the top, and the old "! python3 runs/director.py" lines would each append a new block.
#   3. Revival: if the chain ended with "queue empty" although nobody asked for that (the director crashed, for
#      example on a full disk, or every run failed twice), it starts the chain the same way, at most MAX_REVIVALS
#      times: the last training line ends at once if its piece was complete, and the director is called again. The
#      director's list of dropped runs is cleared so that they get another try.
# Stop it: touch runs/watchdog.stop (and "crontab -r" removes the cron entry). runs/director.stop stops the queue itself.
# Try it without touching anything: bash runs/watchdog.sh --dry-run
ROOT=${WATCHDOG_ROOT:-/home/jiin/AIC313-D2O}
MAX_REVIVALS=5
cd "$ROOT" || exit 1
DRY=""; [ "$1" = "--dry-run" ] && DRY=1
log() { if [ -n "$DRY" ]; then echo "$(date -u +%FT%TZ) $*"; else echo "$(date -u +%FT%TZ) $*" >> runs/watchdog.log; fi; }
[ -e runs/watchdog.stop ] && exit 0

# One instance only.
exec 8> runs/watchdog.lock
flock -n 8 || exit 0

if [ -z "$DRY" ]; then
    python3 - >> runs/watchdog.log 2>&1 <<'PY'
import sys, time
sys.path.insert(0, "runs")
import director
director.make_room(lambda text: print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), text))
PY
fi

# The chain keeps a lock on runs/chain_final.lock, and so does a training it started: nothing to do while either lives.
if ! flock -n runs/chain_final.lock true; then
    [ -n "$DRY" ] && log "the chain is running: nothing to do"
    exit 0
fi
[ -e runs/director.stop ] && exit 0
if [ -z "$WATCHDOG_ROOT" ] && pgrep -u "$(id -u)" -f "train_d2o.py|train_edm.py" > /dev/null; then  # not in a test sandbox
    log "the chain is not running, but another training is: left alone"
    exit 0
fi
if ! nvidia-smi > /dev/null 2>&1; then
    log "the chain is not running and the GPU does not answer: waiting"
    exit 0
fi

revive=""
if tail -n 1 runs/chain.log | grep -q "queue empty$"; then
    # The director ends the queue on purpose only past its deadline or its number of calls.
    reason=$(python3 runs/director.py --dry-run 2>&1 | tail -n 1)
    case "$reason" in *"past 20"*|*"calls"*|*"director.stop"*) exit 0 ;; esac
    count=$(cat runs/watchdog_revivals 2>/dev/null || echo 0)
    if [ "$count" -ge "$MAX_REVIVALS" ]; then
        [ -e runs/watchdog_gave_up ] || { log "the queue ended again after $count revivals: giving up"; [ -n "$DRY" ] || touch runs/watchdog_gave_up; }
        exit 0
    fi
    revive=$((count + 1))
fi

if [ -n "$DRY" ]; then
    log "would start the chain again${revive:+ (revival $revive: the queue had ended)}; the queue would become:"
fi
python3 - "$DRY" "$revive" <<'PY'
import json, shutil, sys, time
dry, revive = sys.argv[1] == "1", sys.argv[2]
path = "runs/queue_final.txt"
lines = open(path).read().splitlines()
jobs = [index for index, line in enumerate(lines) if line.strip() and not line.startswith("#")]
training = [index for index in jobs if not lines[index].startswith("! ")]
if training:
    for index in jobs:
        if index < training[-1]:
            lines[index] = "#done " + lines[index]
after = [lines[index] for index in jobs if not training or index > training[-1]]
if "! python3 runs/director.py" not in after:  # the block has lost its call of the director: nothing would follow it
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines += [f"# watchdog, {stamp}: no call of the director was left in the queue", "! python3 runs/director.py"]
if dry:
    print("\n".join(line[:150] for line in lines if line.strip() and not line.startswith("#")))
    sys.exit()
shutil.copy(path, path + ".before-restart")
with open(path, "w") as file:
    file.write("\n".join(lines) + "\n")
if revive:
    try:
        with open("runs/director_state.json") as file:
            state = json.load(file)
        state["dropped"], state["retries"] = {}, {}
        with open("runs/director_state.json", "w") as file:
            json.dump(state, file, indent=1)
    except (OSError, ValueError):
        pass
    with open("runs/watchdog_revivals", "w") as file:
        file.write(revive + "\n")
PY
[ -n "$DRY" ] && exit 0
log "the chain was not running${revive:+ and its queue had ended (revival $revive)}: started again"
setsid nohup bash runs/chain_final.sh >> runs/chain.log 2>&1 < /dev/null 8>&- &
