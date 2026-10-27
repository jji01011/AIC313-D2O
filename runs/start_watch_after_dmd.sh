#!/bin/bash
# Starts runs/watch_e112.sh by itself once the chain has moved on from runs/e112_dmd to "e112_fit -> 2000".
# The watcher was stopped on 2026-10-10 17:48 UTC because its calibration (3.8 GB) next to the DMD branch (three
# networks) could go over the 20 GB of the GPU. e112_fit's state rests at 1750 kimg while the branch runs and is first
# rewritten 25 kimg (about 12 minutes) into the next piece, so the watcher still reads the 1750-kimg state.
# Start detached:  setsid nohup bash runs/start_watch_after_dmd.sh >> runs/watch_e112.out 2>&1 < /dev/null &
cd /home/jiin/AIC313-D2O || exit 1
until grep -q "starting D2O in runs/e112_fit: .*--kimg 2000" runs/chain.log; do sleep 15; done
echo "$(date -u +%FT%TZ) start_watch_after_dmd.sh: the chain is on e112_fit -> 2000, starting watch_e112.sh"
setsid nohup bash runs/watch_e112.sh >> runs/watch_e112.out 2>&1 < /dev/null &
