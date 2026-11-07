#!/bin/bash
ROOT=/root/AIC313-Project-FastGen; PY=$ROOT/.venv/bin/python
export EDM_TEACHER_CODE=/root/refs/junseokl_FastGen/edm_teacher
cd $ROOT/ectmodel
for cfg in "gbase12_lr1e4 1e-4" "gbase12_lr3e5 3e-5"; do
  set -- $cfg; RUN=/dev/shm/$1
  mkdir -p $RUN
  SKIP=30 nohup ./watch_ls.sh $RUN > $RUN/watch.log 2>&1 &
  $PY sid_few.py --run_dir $RUN --init $ROOT/cand_few_lhs/checkpoints/few_nfe.ckpt \
      --teacher /root/teacher_e112_handoff/handoff/teacher_ag_main.pt --cfg 1.5 --sigmas 80,3,1.5,0.4 \
      --glr 3e-6 --lr $2 --batch 64 --micro_batch 8 --fake_only_kimg 20 --ema_kimg 20 --snap_kimg 5 --total_kimg 80 \
      --save_state 0 --fake_lora 12 --fake_lora_base generator > $RUN/train.log 2>&1
  touch $RUN/DONE
  wait
done
echo "$(date) GBASE_QUEUE_DONE"
