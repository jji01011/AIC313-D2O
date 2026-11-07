#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3
PY=${PY:-python}
REF=$1; OUT=$2; LINE=$3; WHAT=$4; run=$5; kimg=$6
REFS=${REFS:-$REF,results/reference_holdout_h20,results/reference_holdout_c5,results/reference_train}
state=runs/$run/d2o-state-latest.pt
[ -f "${REF}_stats.npz" ] || { echo "$(date -u +%FT%TZ) $run $kimg: ${REF}_stats.npz is missing"; exit 1; }
[ -f "$state" ] || { echo "$(date -u +%FT%TZ) $run $kimg: no state file"; exit 1; }
grep -q "^$run $kimg .*train-reference FID" "$OUT" 2>/dev/null && { echo "$run $kimg already scored"; exit 0; }
ckpt=runs/$run/calib-$(printf "%06d" "$kimg").ckpt
$PY -W ignore calibrate_generator.py --state "$state" --gains=1.0,1.1,1.2 --smooths=0 --seeds 0,1,2 \
    --fid_reference "$REF" --workdir results/ref_score_fid_samples --out "$ckpt" 2>&1 \
    | grep -a "train-reference\|kimg\|wrote" | sed "s|^|$run $kimg $(date -u +%FT%TZ) |" >> "$OUT"
[ -f "$ckpt" ] || { echo "$(date -u +%FT%TZ) $run $kimg: calibration wrote nothing"; exit 1; }
$PY -W ignore diagnostics/score_refs.py --ckpts "$ckpt" --refs "$REFS" --log "${OUT%.log}_refs.log" --workdir results/ref_score_refs_samples > /dev/null 2>&1

value() { grep -a "$1.*train-reference FID" "$OUT" | sed 's/.*train-reference FID *\([0-9.]*\).*/\1/' | sort -g | head -n 1; }
mine=$(value "^$run $kimg ")
best_run=$(value "^$run [0-9]* ")
best_all=$(value "^$LINE [0-9]* ")
if [ "$mine" != "$best_run" ]; then
    rm -f "$ckpt" && echo "$run $kimg $(date -u +%FT%TZ) removed $ckpt (FID $mine on $REF, this run's best is $best_run)" >> "$OUT"
    exit 0
fi
keep=$(bash runs/lowest_official.sh)
for other in runs/$run/calib-*.ckpt; do
    [ "$other" = "$keep" ] && continue
    [ "$other" = "$ckpt" ] || { rm -f "$other" && echo "$run $kimg $(date -u +%FT%TZ) removed $other (no longer this run's best)" >> "$OUT"; }
done
[ "$mine" = "$best_all" ] || exit 0
scores=""
for _ in 1 2 3; do
    one=$($PY -W ignore evaluate.py --model_checkpoint "$ckpt" --evaluate_mode one_nfe --reference_dir ./results/reference \
        --output_dir results/evaluation_candidate 2>&1 | grep -a "one_nfe FID:" | awk '{print $NF}')
    [ -n "$one" ] || { echo "$(date -u +%FT%TZ) official evaluation of $ckpt failed"; exit 0; }
    scores="$scores${scores:+, }$one"
done
mean=$(echo "$scores" | tr -d ' ' | awk -F, '{printf "%.2f", ($1 + $2 + $3) / 3}')
echo "$(date -u +%F) | $ckpt: E112 one-step generator, $WHAT $run at $kimg kimg, $(grep -a "^$run $kimg .*wrote" "$OUT" | sed 's/.* with //') (runs/ref_score.sh; it became the lowest FID on $REF of its line, so its official score was measured, for reporting only; the train-reference column holds the FID ON $REF) | 1 | $mine | $scores (three runs, mean $mean)" >> "${OFFICIAL_LOG:-results/official_fid.log}"
