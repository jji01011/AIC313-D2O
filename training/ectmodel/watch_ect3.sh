#!/bin/bash
cd /workspace/fastgen/ectmodel
RUN=$1; GPU=$2; D=/workspace/runs/$RUN; TSV=$D/fid.tsv
export CUDA_VISIBLE_DEVICES=$GPU
[ -f $TSV ] || printf "kimg\tema\tnfe\tsigmas\tfid\tmetric\n" > $TSV
while true; do
  for ck in $D/ema500_*kimg.ckpt; do
    [ -f "$ck" ] || continue
    k=$((10#$(basename $ck | grep -oP '_\K\d+(?=kimg)')))
    cut -f1 $TSV | grep -qx "$k" && continue
    [ $(( $(date +%s) - $(stat -c %Y $ck) )) -lt 90 ] && continue
    kk=$(printf "%06d" $k)
    for h in 200 500 1500; do
      f=$D/ema${h}_${kk}kimg.ckpt; [ -f $f ] || continue
      sched="80 80,1.5 80,4.0,1.5,0.5"; [ $h = 500 ] && sched="80 80,1.5 80,2.5 80,4.0,1.5 80,2.0,0.8,0.3 80,4.0,1.5,0.5 80,10,2.0,0.5 80,6,2.5,1.0"
      /venv/main/bin/python eval_ect.py $f $sched 2>&1 | grep HELDOUT | awk -v k=$k -v h=$h -F'\t' '{print k"\t"h"\t"$2"\t"$3"\t"$4"\theldout"}' >> $TSV
    done
    f=$D/ema500_${kk}kimg.ckpt
    for s in 80 80,4.0,1.5,0.5; do
      mode=few_nfe; [ $s = 80 ] && mode=one_nfe
      v=$(ECT_SIGMAS=$s /venv/main/bin/python evaluate.py --model_checkpoint $f --evaluate_mode $mode --batch_size 64 --output_dir /tmp/val_$RUN 2>&1 | grep -oP "FID: \K[0-9.]+")
      printf "%s\t500\t%s\t%s\t%s\tval\n" $k $(echo $s | tr ',' '\n' | wc -l) $s ${v:-ERR} >> $TSV
    done
  done
  sleep 300
done
