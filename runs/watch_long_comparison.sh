#!/bin/bash
# Scores the long comparison of runs/queue_final.txt (lines 15-24: runs/d2o_one_aug20 and runs/d2o_one_dn30 continued in
# alternating 250-kimg pieces) at every matched point, after output calibration, against the train reference.
# A run's state file rests at the matched kimg while the other run trains its piece; that is when it is scored.
# Results are appended to results/long_comparison.log; the best calibration of every point is kept as <run>/calib-<kimg>.ckpt.
# Start detached:  setsid nohup bash runs/watch_long_comparison.sh >> runs/watch_long_comparison.out 2>&1 < /dev/null &
# Starting it again is safe: points that are already in the log are skipped.
cd /home/jiin/AIC313-D2O || exit 1
PY=/home/jiin/.venvs/d2o/bin/python
OUT=results/long_comparison.log

exec 9> runs/watch_long_comparison.lock
flock -n 9 || { echo "another watch_long_comparison.sh is already running; this one exits"; exit 1; }

# True once the chain has started the piece of run $1 that ends at $2 kimg and that piece has exited with status 0.
# A piece that failed or was killed does not count (2026-10-09 19:09 UTC: five queue lines without --resume failed at
# once, and the state they would have been scored on was still at the earlier kimg).
piece_done() {
    awk -v run="runs/$1" -v target="--kimg $2" '
        index($0, "starting D2O in " run ":") { started = (index($0, target) && substr($0, index($0, target) + length(target), 1) !~ /[0-9]/); next }
        index($0, "D2O in " run " exited") { if (started && $0 ~ /exited with status 0$/) { found = 1; exit } started = 0 }
        END { exit !found }' runs/chain.log
}

# Gains 1.1 and 1.15 were added on 2026-10-09 14:45 UTC (points scored before that have 1.2 and 1.3 only): under the
# teammate-EDM discriminator the generator needs less of the gain (tdisc at 2700 kimg: 17.42 at 1.2, 17.54 at 1.3).
score() {  # run, kimg
    grep -q "^$1 $2 " "$OUT" 2>/dev/null && return
    until piece_done "$1" "$2"; do sleep 30; done
    local tag
    tag=$(printf "%06d" "$2")
    $PY -W ignore calibrate_generator.py --state "runs/$1/d2o-state-latest.pt" --gains=1.1,1.15,1.2,1.3 --smooths=0.2 --seeds 0,1,2 \
        --workdir results/long_comparison_fid_samples --out "runs/$1/calib-$tag.ckpt" 2>&1 \
        | grep -a "train-reference\|kimg\|wrote" | sed "s|^|$1 $2 $(date -u +%FT%TZ) |" >> "$OUT"
}

# Reporting only. The course grades evaluate.py against the validation split (README, "Final Evaluation"), and the same
# README forbids using the validation split for training or for choosing anything. So the choice stays on the train
# reference above, and this function only measures where the chosen point stands on the graded number: when a point has
# just become the lowest train-reference FID of the whole log (the submission candidate), evaluate.py is run three times
# (it does not fix the seed) and one line is appended to results/official_fid.log. Nothing reads those numbers back.
report_official() {  # run, kimg
    local tag ckpt mine best one scores mean
    tag=$(printf "%06d" "$2")
    ckpt="runs/$1/calib-$tag.ckpt"
    [ -f "$ckpt" ] || return
    grep -qF "| $ckpt:" results/official_fid.log && return
    mine=$(grep -a "^$1 $2 .*train-reference FID" "$OUT" | sed 's/.*train-reference FID *\([0-9.]*\).*/\1/' | sort -g | head -n 1)
    best=$(grep -a "train-reference FID" "$OUT" | sed 's/.*train-reference FID *\([0-9.]*\).*/\1/' | sort -g | head -n 1)
    [ -n "$mine" ] && [ "$mine" = "$best" ] || return
    scores=""
    for _ in 1 2 3; do
        one=$($PY -W ignore evaluate.py --model_checkpoint "$ckpt" --evaluate_mode one_nfe --reference_dir ./results/reference \
            --output_dir results/evaluation_candidate 2>&1 | grep -a "one_nfe FID:" | awk '{print $NF}')
        [ -n "$one" ] || { echo "$(date -u +%FT%TZ) official evaluation of $ckpt failed"; return; }
        scores="$scores${scores:+, }$one"
    done
    mean=$(echo "$scores" | tr -d ' ' | awk -F, '{printf "%.2f", ($1 + $2 + $3) / 3}')
    echo "$(date -u +%F) | $ckpt: $(grep -a "^$1 $2 .*wrote" "$OUT" | sed 's/.* with //') (calibrate_generator.py via runs/watch_long_comparison.sh; it became the lowest train-reference FID, so its official score was measured automatically, for reporting only) | 1 | $mine | $scores (three runs, mean $mean)" >> results/official_fid.log
}

point() { score "$1" "$2"; report_official "$1" "$2"; }

# 1700 and 1950 were never scored: the session that wrote this script dropped before starting it (2026-10-08 20:52 UTC),
# and the state files have moved on since. Do not put them back, the state at those points no longer exists.
# The order below is the order of the lines in runs/queue_final.txt: every point is scored right after its piece ends.
for kimg in 2200 2450; do
    point d2o_one_aug20 $kimg
    point d2o_one_dn30 $kimg
done
# Two branches from dn30's 2200-kimg state, compared with d2o_one_dn30 at 2450: samples of the teammate's EDM mixed into
# the real images (tmix), and a lower learning rate (lr 2e-5 / dlr 4e-5).
point d2o_one_dn30_tmix 2450
# The teammate's EDM encoder as the discriminator's feature network, from dn30's 2450-kimg state; compare with dn30 at 2700.
point d2o_one_dn30_tdisc 2700
# The same with the teammate's whole encoder (four stages).
point d2o_one_dn30_tdisc8 2700
point d2o_one_dn30_tdisc 2950
point d2o_one_dn30 2700
# d2o_one_dn30_tdisc8 2950 was taken out of the queue on 2026-10-09 17:20 UTC (tdisc did not improve from 2700 to 2950).
point d2o_one_dn30_lr2e5 2450
for kimg in 2950 3200; do
    point d2o_one_dn30 $kimg
done
echo "$(date -u +%FT%TZ) all points scored"
