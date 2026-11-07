#!/bin/bash
RUN=$1; NAME=$(basename $RUN); ROOT=/root/AIC313-Project-FastGen; PY=$ROOT/.venv/bin/python
SKIP=${SKIP:-30}
while true; do
  for c in $(ls $RUN/ema500_*kimg.ckpt 2>/dev/null | sort); do
    k=$((10#$(basename $c | grep -oP '_\K\d+(?=kimg)')))
    [ $(( $(date +%s) - $(stat -c %Y $c) )) -lt 60 ] && continue
    if [ $k -le $SKIP ]; then rm -f "$c"; continue; fi
    grep -q "^${NAME}_${k}	" $ROOT/onefix/few_knob.txt 2>/dev/null && continue
    LS=$($PY -c "import torch,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); s=c.get('sigmas'); print(','.join(f'{v:.4g}' for v in s) if s else '')" $c)
    CFGS=("80,5,1.8,0.4@0#all!0/0/0/0")
    [ -f $RUN/EXTRA ] && mapfile -t EX < $RUN/EXTRA && CFGS=("${CFGS[@]}" "${EX[@]}")
    [ -n "$LS" ] && CFGS=("$LS@0#all!0/0/0/0" "${CFGS[@]}")
    NOVAL=1 TORCH_THREADS=8 $PY $ROOT/onefix/few_knob.py ${NAME}_${k} $c "${CFGS[@]}" 2>&1 | grep -v -i warn
    $PY - "$RUN" "$NAME" <<'EOF'
import collections, glob, os, re, sys
run, name = sys.argv[1], sys.argv[2]
acc = collections.defaultdict(list)
for line in open("/root/AIC313-Project-FastGen/onefix/few_knob.txt"):
    f = line.rstrip("\n").split("\t")
    m = re.fullmatch(re.escape(name) + r"_(\d+)", f[0])
    if m and len(f) > 3 and f[3].startswith("HO "):
        acc[(int(m.group(1)), f[1])].append(float(f[3].split()[1]))
best = collections.defaultdict(lambda: 1e9)
for (k, cfg), v in acc.items():
    if len(v) >= 4:
        best[k] = min(best[k], sum(v) / len(v))
keep = sorted(best, key=best.get)[:2]
for ck in glob.glob(f"{run}/ema500_*kimg.ckpt"):
    k = int(re.search(r"_(\d+)kimg", ck).group(1))
    if k in best and k not in keep:
        os.remove(ck); print(f"pruned {k} kimg (HO {best[k]:.3f}); keep {[(x, round(best[x], 3)) for x in keep]}", flush=True)
EOF
  done
  [ -f $RUN/DONE ] && exit 0
  sleep 120
done
