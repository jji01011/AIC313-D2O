#!/bin/bash
cd /workspace/fastgen/ectmodel
RUN=$1; GPU=$2; shift 2
mkdir -p /workspace/runs/$RUN
CUDA_VISIBLE_DEVICES=$GPU setsid nohup /venv/main/bin/python ect_train.py --run_dir /workspace/runs/$RUN --batch 128 --micro_batch 64 "$@" \
    > /workspace/runs/$RUN/train.log 2>&1 < /dev/null &
sleep 2
setsid nohup bash watch_ect3.sh $RUN $GPU > /workspace/runs/$RUN/watch.log 2>&1 < /dev/null &
echo "started $RUN on GPU $GPU"
