#!/bin/bash
# Scores the E112-generator runs of runs/queue_final.txt (runs/e112_live, runs/e112_fit and runs/e112_full, continued in alternating
# pieces) at every matched point, after output calibration, against the train reference. A run's state file rests at the
# matched kimg while the other run trains its piece; that is when it is scored (the same scheme as watch_long_comparison.sh).
# The EDM2 generator takes the gain only: smoothing made it worse (20 kimg of e112_full: 20.1 without, 20.5 with 0.2).
# Results are appended to results/e112_calibration.log. Only runs/e112_live and runs/e112_fit, the forms inside the 100M limit, get their best
# calibration written as <run>/calib-<kimg>.ckpt; runs/e112_full is over the limit and is a reference, not a candidate.
# The official (validation) score is measured only for a point that has just become the lowest of all (report_official).
# Start detached:  setsid nohup bash runs/watch_e112.sh >> runs/watch_e112.out 2>&1 < /dev/null &
# Starting it again is safe: points that are already in the log are skipped.
cd /home/jiin/AIC313-D2O || exit 1
PY=/home/jiin/.venvs/d2o/bin/python
OUT=results/e112_calibration.log

exec 9> runs/watch_e112.lock
flock -n 9 || { echo "another watch_e112.sh is already running; this one exits"; exit 1; }

# True once the chain has started the piece of run $1 that ends at $2 kimg and that piece has exited with status 0.
# A piece that failed or was killed does not count (2026-10-09 19:09 UTC: five queue lines without --resume failed at
# once, and the state they would have been scored on was still at the earlier kimg).
piece_done() {
    awk -v run="runs/$1" -v target="--kimg $2" '
        index($0, "starting D2O in " run ":") { started = (index($0, target) && substr($0, index($0, target) + length(target), 1) !~ /[0-9]/); next }
        index($0, "D2O in " run " exited") { if (started && $0 ~ /exited with status 0$/) { found = 1; exit } started = 0 }
        END { exit !found }' runs/chain.log
}

# Reporting only, as in watch_long_comparison.sh: the choice is made on the train reference above. When a kept point
# has just become the lowest train-reference FID of all calibrated candidates (the runs inside the 100M limit here, and
# the baseline runs in results/long_comparison.log), evaluate.py is run three times (it does not fix the seed) and one
# line is appended to results/official_fid.log. Nothing reads those numbers back.
report_official() {  # run, kimg, checkpoint, its train-reference FID
    local best one scores mean runs note
    runs='^e112_fit \|^e112_live \|^e112_lorag[a-z0-9_]* '  # e112_lorag and the continuations runs/director.py names after it
    note="it became the lowest train-reference FID, so its official score was measured automatically, for reporting only"
    # e112_lora2 and its continuation: the fake score is the frozen teacher plus low-rank branches. Inside the limit if the
    # shared teacher is counted once, as the teacher (the user's working assumption, 2026-10-11); the TA's answer read
    # strictly (generator and fake score fit in 100M together, frozen parameters included) says otherwise. Provisional, so
    # they do not stand in the way of e112_lorag's measurement, and are measured themselves when lowest of all.
    case "$1" in e112_lora2*)
        runs="$runs"'\|^e112_lora2[a-z0-9_]* '
        note="PROVISIONAL CANDIDATE: its fake score is the frozen teacher plus low-rank branches, inside the 100M limit only if the shared teacher is counted once (the user's working assumption, to be confirmed by the TA); the lowest train-reference FID of all runs, official score measured for reporting only" ;;
    esac
    grep -qF "| $3:" results/official_fid.log && return
    best=$( { grep -a "$runs" "$OUT"; cat results/long_comparison.log; } | grep -a "train-reference FID" \
        | sed 's/.*train-reference FID *\([0-9.]*\).*/\1/' | sort -g | head -n 1)
    [ -n "$4" ] && [ "$4" = "$best" ] || return
    scores=""
    for _ in 1 2 3; do
        one=$($PY -W ignore evaluate.py --model_checkpoint "$3" --evaluate_mode one_nfe --reference_dir ./results/reference \
            --output_dir results/evaluation_candidate 2>&1 | grep -a "one_nfe FID:" | awk '{print $NF}')
        [ -n "$one" ] || { echo "$(date -u +%FT%TZ) official evaluation of $3 failed"; return; }
        scores="$scores${scores:+, }$one"
    done
    mean=$(echo "$scores" | tr -d ' ' | awk -F, '{printf "%.2f", ($1 + $2 + $3) / 3}')
    echo "$(date -u +%F) | $3: E112 one-step generator, run $1 at $2 kimg, $(grep -a "^$1 $2 .*wrote" "$OUT" | sed 's/.* with //') (calibrate_generator.py via runs/watch_e112.sh; $note) | 1 | $4 | $scores (three runs, mean $mean)" >> results/official_fid.log
}

score() {  # run, kimg, "keep" to write the calibrated checkpoint
    grep -q "^$1 $2 .*train-reference" "$OUT" 2>/dev/null && return
    if ! piece_done "$1" "$2"; then echo "$(date -u +%FT%TZ) $1 $2 skipped: its piece has not ended normally"; return; fi
    local tag out=() ckpt mine best other
    tag=$(printf "%06d" "$2")
    ckpt="runs/$1/calib-$tag.ckpt"
    [ "$3" = keep ] && out=(--out "$ckpt")
    $PY -W ignore calibrate_generator.py --state "runs/$1/d2o-state-latest.pt" --gains=1.0,1.1,1.2,1.3 --smooths=0 --seeds 0,1,2 \
        --workdir results/e112_calibration_fid_samples "${out[@]}" 2>&1 \
        | grep -a "train-reference\|kimg\|wrote" | sed "s|^|$1 $2 $(date -u +%FT%TZ) |" >> "$OUT"
    [ "$3" = keep ] && [ -f "$ckpt" ] || return
    # Disk: a calibrated checkpoint is 364 MB, so only the best one of a run (lowest train-reference FID) stays. Until
    # 2026-10-12 a checkpoint named in results/official_fid.log was never removed; the user then left the disk to the
    # session ("only the low FIDs need to stay"), and the score itself stays in that log.
    mine=$(grep -a "^$1 $2 .*train-reference FID" "$OUT" | sed 's/.*train-reference FID *\([0-9.]*\).*/\1/' | sort -g | head -n 1)
    best=$(grep -a "^$1 [0-9]* .*train-reference FID" "$OUT" | sed 's/.*train-reference FID *\([0-9.]*\).*/\1/' | sort -g | head -n 1)
    if [ -n "$mine" ] && [ "$mine" = "$best" ]; then
        keep=$(bash runs/lowest_official.sh)  # the lowest official score so far stays (the user, 2026-10-15)
        for other in runs/$1/calib-*.ckpt; do
            [ "$other" = "$ckpt" ] && continue
            [ "$other" = "$keep" ] && continue
            rm -f "$other" && echo "$1 $2 $(date -u +%FT%TZ) removed $other (no longer this run's best)" >> "$OUT"
        done
        report_official "$1" "$2" "$ckpt" "$mine"
    else
        rm -f "$ckpt" && echo "$1 $2 $(date -u +%FT%TZ) removed $ckpt (train-reference FID $mine, this run's best is $best)" >> "$OUT"
    fi
}

# The order below is the order of the lines in runs/queue_final.txt: every point is scored right after its piece ends
# (the state file is read at once; the next piece first writes it 25 kimg later). The over-limit reference e112_full is
# scored at 150 only: its 250-kimg piece is followed by the baseline (13.5 GB on the GPU), and a calibration next to
# that does not fit safely.
score e112_live 150 keep
score e112_fit 150 keep
score e112_full 150
score e112_fit 250 keep
score e112_fit 500 keep
score e112_fit 750 keep
score e112_fit 1000 keep
score e112_fit 1250 keep
# Branch from e112_fit's 1250-kimg state with discriminator noise -3.0 (numbers only, no checkpoint is written).
score e112_dn30 1350
# Branch from the same state with the ECT consistency term (numbers only).
score e112_ect50 1350
score e112_fit 1500 keep
# Fresh start from E112 with paired distillation from the teacher's samples (numbers only).
score e112_kd 150
# Branch from e112_fit's 1500-kimg state with the relativistic loss (numbers only).
score e112_rp 1600
# Fresh start from E112 with the generator's input noise level at 2.5 (numbers only).
score e112_s25 150
# From 2026-10-10 20:40 UTC the queue alternates the DMD branch (runs/e112_dmd, 16 GB on the GPU) with 100-kimg pieces
# of e112_fit (9 to 13 GB). A calibration (3.7 GB) only fits next to the latter, so each group of points below first
# waits for its e112_fit piece to start. The states scored there are at rest: e112_fit's own state is first rewritten
# 25 kimg (12 minutes) into its piece and is read right at the start, the DMD branch's state is not touched meanwhile.
# A DMD point is scored only if its piece ended normally (it is over the parameter limit as it stands, so its
# checkpoint is kept for inspection, not as a candidate; its official score is not measured from here).
window() {  # run, kimg: wait until the chain has started that piece
    until grep -q "starting D2O in runs/$1: .*--kimg $2\$" runs/chain.log; do sleep 15; done
}
# Three official evaluations of a checkpoint (evaluate.py does not fix the seed): prints "s1, s2, s3|mean", or nothing
# if one of them failed. Reporting only, nothing reads the numbers back.
official3() {  # checkpoint
    local one scores=""
    for _ in 1 2 3; do
        one=$($PY -W ignore evaluate.py --model_checkpoint "$1" --evaluate_mode one_nfe --reference_dir ./results/reference \
            --output_dir results/evaluation_candidate 2>&1 | grep -a "one_nfe FID:" | awk '{print $NF}')
        [ -n "$one" ] || { echo "$(date -u +%FT%TZ) official evaluation of $1 failed" >&2; return; }
        scores="$scores${scores:+, }$one"
    done
    echo "$scores|$(echo "$scores" | tr -d ' ' | awk -F, '{printf "%.2f", ($1 + $2 + $3) / 3}')"
}
lowest() {  # log file, line prefix: the lowest train-reference FID among those lines
    grep -a "^$2.*train-reference FID" "$1" | sed 's/.*train-reference FID *\([0-9.]*\).*/\1/' | sort -g | head -n 1
}
# The DMD branch with the E112-sized fake score: an effect test, over the limit if the fake score counts with the
# student. Its official score is measured all the same, to see whether the gain on the train reference carries over;
# only its newest best checkpoint is kept.
dmd_point() {  # kimg
    if ! piece_done e112_dmd "$1"; then echo "$(date -u +%FT%TZ) e112_dmd $1 skipped: its piece has not ended normally"; return; fi
    score e112_dmd "$1" keep
    local ckpt mine result other
    ckpt="runs/e112_dmd/calib-$(printf "%06d" "$1").ckpt"
    [ -f "$ckpt" ] || return
    grep -qF "| $ckpt:" results/official_fid.log && return
    mine=$(lowest "$OUT" "e112_dmd $1 ")
    result=$(official3 "$ckpt")
    [ -n "$result" ] || return
    echo "$(date -u +%F) | $ckpt: E112 one-step generator with the DMD2 term on top of the GAN loss, run e112_dmd at $1 kimg, $(grep -a "^e112_dmd $1 .*wrote" "$OUT" | sed 's/.* with //') (via runs/watch_e112.sh; EFFECT TEST, NOT A CANDIDATE while its E112-sized fake score may count with the student; reporting only) | 1 | $mine | ${result%|*} (three runs, mean ${result#*|})" >> results/official_fid.log
    for other in runs/e112_dmd/calib-*.ckpt; do
        [ "$other" = "$ckpt" ] || { rm -f "$other" && echo "e112_dmd $1 $(date -u +%FT%TZ) removed $other (an earlier effect-test checkpoint)" >> "$OUT"; }
    done
}
# A run of our own generator (the U-Net of d2o/networks.py): gain and smoothing as in watch_long_comparison.sh, logged
# to the same file. dn30_dmd is inside the parameter limit (generator + discriminator + fake score 94.5M, the teacher a
# model of its own), so it is a candidate: its official score is measured when it is the lowest of that log.
old_point() {  # run, kimg
    local log=results/long_comparison.log tag ckpt mine result
    if ! piece_done "$1" "$2"; then echo "$(date -u +%FT%TZ) $1 $2 skipped: its piece has not ended normally"; return; fi
    tag=$(printf "%06d" "$2"); ckpt="runs/$1/calib-$tag.ckpt"
    grep -q "^$1 $2 " "$log" 2>/dev/null || $PY -W ignore calibrate_generator.py --state "runs/$1/d2o-state-latest.pt" --gains=1.1,1.15,1.2,1.3 --smooths=0.2 --seeds 0,1,2 \
        --workdir results/e112_calibration_fid_samples --out "$ckpt" 2>&1 \
        | grep -a "train-reference\|kimg\|wrote" | sed "s|^|$1 $2 $(date -u +%FT%TZ) |" >> "$log"
    [ -f "$ckpt" ] || return
    grep -qF "| $ckpt:" results/official_fid.log && return
    mine=$(lowest "$log" "$1 $2 ")
    [ -n "$mine" ] && [ "$mine" = "$(lowest "$log" "")" ] || return
    result=$(official3 "$ckpt")
    [ -n "$result" ] || return
    echo "$(date -u +%F) | $ckpt: our 34.8M generator with the DMD2 term on top of the GAN loss (teacher E112, fake score our own EDM; generator + discriminator + fake score 94,549,958), run $1 at $2 kimg, $(grep -a "^$1 $2 .*wrote" "$log" | sed 's/.* with //') (calibrate_generator.py via runs/watch_e112.sh; it became the lowest train-reference FID of the runs of our own generator, so its official score was measured, for reporting only) | 1 | $mine | ${result%|*} (three runs, mean ${result#*|})" >> results/official_fid.log
}
# From 2026-10-11 the points are no longer listed here. Each measurement is a line of runs/queue_final.txt
# ("! bash runs/watch_e112.sh dmd_point 2150"), which runs/chain_final.sh executes between two trainings, alone on the
# GPU. Until then a group of points waited for a 100-kimg piece of e112_fit to start and was scored next to it; that
# run has stopped improving, and when a full disk ended its piece the points behind it waited for ever.
# Points scored that way: e112_fit 1750, 1850, 1950; e112_dmd 1850, 2000; dn30_dmd 2550 (its calibration died of the
# full disk and is repeated from the queue).
[ $# -gt 0 ] || { echo "usage: bash runs/watch_e112.sh <score|dmd_point|old_point|report_official> <arguments>"; exit 1; }
"$@"
