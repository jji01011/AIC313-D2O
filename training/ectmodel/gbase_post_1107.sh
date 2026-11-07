#!/bin/bash
set -u
ROOT=/root/AIC313-Project-FastGen; PY=$ROOT/.venv/bin/python; ECT=$ROOT/ectmodel
PKG=$ROOT/submission_pkg_1107/team_16; C=$ROOT/cand_few_gbase; FIN=$ECT/runs/gbase_final
OUT=$ROOT/onefix/gbase_post_1107.log
export EDM_TEACHER_CODE=/root/refs/junseokl_FastGen/edm_teacher
log() { echo "$(TZ=Asia/Seoul date '+%m/%d %H:%M KST') $*" | tee -a $OUT; }

rescore() {
  local RUN=$1
  until [ -f $RUN/DONE ]; do sleep 60; done
  while pgrep -f "watch_ls.sh $RUN" > /dev/null; do sleep 30; done
  sleep 70
  cd $ECT && SKIP=30 ./watch_ls.sh $RUN >> $RUN/watch.log 2>&1
  log "scored all snapshots of $(basename $RUN)"
}

log "start"
rescore /dev/shm/gbase12_lr1e4
rescore /dev/shm/gbase12_lr3e5

PICK=$($PY - <<'EOF'
import collections, re
acc = collections.defaultdict(list)
for line in open("/root/AIC313-Project-FastGen/onefix/few_knob.txt"):
    f = line.rstrip("\n").split("\t")
    m = re.fullmatch(r"(gbase12_lr\w+?)_(\d+)", f[0])
    if m and len(f) > 3 and f[1].startswith("80,5,1.8,0.4@") and f[3].startswith("HO "):
        acc[(m.group(1), int(m.group(2)))].append(float(f[3].split()[1]))
best = {k: sum(v) / len(v) for k, v in acc.items() if len(v) >= 4}
r, k = min(best, key=best.get)
print(r, k, f"{best[(r, k)]:.3f}")
EOF
)
read RUNNAME KIMG HO4 <<< "$PICK"
log "pick: $RUNNAME ${KIMG}k (4-seed HO $HO4)"
SNAP=/dev/shm/$RUNNAME/ema500_$(printf %06d $KIMG)kimg.ckpt
[ -f $SNAP ] || { log "ERROR: $SNAP missing"; exit 1; }

mkdir -p $FIN
DISK=$FIN/${RUNNAME}_${KIMG}.ckpt
cp $SNAP $DISK
[ "$(md5sum < $SNAP)" == "$(md5sum < $DISK)" ] || { log "ERROR: copy md5 mismatch"; exit 1; }
for R in gbase12_lr1e4 gbase12_lr3e5; do
  mkdir -p $FIN/$R && cp /dev/shm/$R/{log.jsonl,train.log,watch.log} $FIN/$R/
done
log "copied $DISK (md5 $(md5sum < $DISK | cut -c1-32)) and logs"

rm -rf $C
cp -al $PKG $C
rm $C/checkpoints/few_nfe.ckpt && ln $DISK $C/checkpoints/few_nfe.ckpt
ln -s $ROOT/data $C/data
log "built $C"

cd $ROOT
TAG=cand_gbase_${RUNNAME}_${KIMG}
NOVAL=1 timeout 2400 $PY onefix/one_seed_pick.py $C 0 8 $TAG few_nfe 2>&1 | grep "seed" | tee -a $OUT
SEED=$($PY - "$TAG" <<'EOF'
import sys
rows = [l.rstrip("\n").split("\t") for l in open("/root/AIC313-Project-FastGen/onefix/one_seed_pick.txt")]
rows = [r for r in rows if r[0] == sys.argv[1]]
best = min(rows, key=lambda r: float(r[2].split()[1]))
print(best[1].split()[-1])
EOF
)
sed -i "s/^FEW_NOISE_SEED = .*/FEW_NOISE_SEED = $SEED/" $C/model.py
log "seed by HO: $SEED ($(grep '^FEW_NOISE_SEED' $C/model.py))"

mkdir -p /dev/shm/gbase_eval
cd $C && timeout 2400 $PY evaluate.py --model_checkpoint ./checkpoints/few_nfe.ckpt --evaluate_mode few_nfe \
    --reference_dir /dev/shm/gbase_eval/reference --output_dir /dev/shm/gbase_eval/evaluation 2>&1 \
    | grep -E "parameters|FID|Error|Traceback" | tee $ROOT/onefix/gbase_official_val_1107.log | tee -a $OUT
cd $ROOT && timeout 2400 $PY onefix/one_seed_pick.py $C 0 8 ${TAG}_val few_nfe 2>&1 | grep "seed" | tee -a $OUT
rm -rf /dev/shm/gbase_eval/evaluation
log "GBASE_POST_DONE"
