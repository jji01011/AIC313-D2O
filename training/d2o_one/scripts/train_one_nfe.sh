#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."
ROOT=$(cd ../.. && pwd)
export PYTHONPATH=".:$ROOT"
[ -e data ] || ln -s "$ROOT/data" data
PY=${PY:-python}
E112=checkpoints/edm/ema-0020971520-0.100.pt
T58=checkpoints/edm/teacher_unguided-0058720256-0.130.pt
BASE="--freeze 0 --xflip 1 --ada_target 0.6 --lr 5e-5 --dlr 1e-4 --snap_kimg 25 --keep_snapshots 0"
G="--g_edm2 $E112 --freeze_encoder 1 --d_shared 1 --d_width 0.5625 --batch_gpu 16"
LORA="--dmd_ratio 1.0 --dmd_fake_lora 6 --dmd_fake_lora_base generator --dmd_fake_lora_dec 1 --dmd_fake_lr 2e-4 --dmd_from_kimg 2000"
TREAL="--dmd_teacher $T58 --dmd_guidance 1.0 --extra_real data/cache/teacher58_real --extra_real_p 1.0 --fid_reference results/reference_teacher58"

run() { mkdir -p "$1"; $PY -W ignore train_d2o.py $BASE --outdir "$@" 2>&1 | tee -a "$1/train.log"; }

run runs/e112_fit $G --lr 1e-3 --kimg 1950
run runs/e112_lorag $G --d_drop_heads 1 --lr 1e-3 $LORA --dmd_teacher $E112 --dmd_fake_steps 3 \
    --resume runs/e112_fit/d2o-state-latest.pt --kimg 2875
run runs/e112_lorag_f5 $G --d_drop_heads 1 --lr 1e-3 $LORA --dmd_teacher $E112 --dmd_fake_steps 5 \
    --resume runs/e112_lorag/d2o-state-latest.pt --kimg 3425
run runs/e112_lorag_f5_g13 $G --d_drop_heads 1 --lr 1e-3 $LORA --dmd_teacher $E112 --dmd_fake_steps 5 \
    --dmd_guidance 1.3 --dmd_guidance_lo 0.25 --dmd_guidance_hi 4 \
    --resume runs/e112_lorag_f5/d2o-state-latest.pt --kimg 3675
run runs/e112_lorag_f5_g13_lr3b $G --d_drop_heads 1 $LORA --dmd_teacher $E112 --dmd_fake_steps 5 \
    --dmd_guidance 1.3 --dmd_guidance_lo 0.25 --dmd_guidance_hi 4 --lr 3e-4 --dlr 3e-5 \
    --resume runs/e112_lorag_f5_g13/d2o-state-latest.pt --kimg 5125
run runs/e112_treal $G --d_drop_heads 1 $LORA --dmd_fake_steps 5 $TREAL --lr 1e-3 --dlr 1e-4 \
    --resume runs/e112_lorag_f5_g13_lr3b/d2o-state-latest.pt --kimg 5525
run runs/e112_treal_lr3 $G --d_drop_heads 1 $LORA --dmd_fake_steps 5 $TREAL --lr 3e-4 --dlr 3e-5 \
    --resume runs/e112_treal/d2o-state-latest.pt --kimg 5825

$PY -W ignore calibrate_generator.py --state runs/e112_treal_lr3/d2o-state-latest.pt --gains=1.0,1.1,1.2 --smooths=0 \
    --seeds 0,1,2 --fid_reference results/reference_teacher58 --workdir results/calib_samples \
    --out runs/e112_treal_lr3/calib-005825.ckpt
cp runs/e112_treal_lr3/calib-005825.ckpt "$ROOT/checkpoints/one_nfe.ckpt"
