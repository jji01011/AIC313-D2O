#!/bin/bash
# Live state of the one-step D2O work, for a session that starts without context (see STATUS.md).
# Read-only: starts nothing and touches no GPU memory.   Usage: bash runs/status.sh
cd /home/jiin/AIC313-D2O || exit 1
PY=/home/jiin/.venvs/d2o/bin/python

echo "== $(date -u +%FT%TZ)"
echo "== GPU jobs (chain script, trainers, FID evaluations)"
ps -eo pid,etime,args | grep -E 'chain_[a-z0-9_]+\.s[h]|train_d2o\.p[y]|train_edm\.p[y]|eval_edm\.p[y]|evaluate\.p[y]' | grep -v shell-snapshots | cut -c1-400 | grep . || echo "(none running)"
echo "== Claude sessions alive on this machine (more than one: do not start GPU work before sorting out which one acts)"
ps -eo pid,etime,args | grep 'native-binary/claud[e]' | cut -c1-110
echo "== GPU memory"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "== runs/chain.log (last 6 lines)"
tail -n 6 runs/chain.log
echo "== queue: runs/queue_final.txt (the chain counts finished lines in memory, top to bottom)"
grep -v '^#' runs/queue_final.txt | grep -v '^[[:space:]]*$'
echo "== one-step FID per D2O run (train reference, 1 NFE; the official evaluate.py value is about 12 to 14.5 higher)"
$PY - <<'EOF'
import glob, json, os

for path in sorted(glob.glob("runs/d2o_*/log.jsonl"), key=os.path.getmtime):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    fids = [(row["kimg"], row["fid_train_ref"]) for row in rows if "fid_train_ref" in row and row["kimg"] > 0]
    if not fids:
        continue
    best = min(fids, key=lambda item: item[1])
    speed = next((row["img_per_sec"] for row in reversed(rows) if row.get("img_per_sec")), None)
    print(f"{os.path.basename(os.path.dirname(path)):26s} at {rows[-1]['kimg']:6.0f} kimg | best {best[1]:6.1f} @ {best[0]:.0f} kimg"
          f" | last {fids[-1][1]:6.1f} | {speed} img/s")
newest = max(glob.glob("runs/d2o_*/log.jsonl"), key=os.path.getmtime)
rows = [json.loads(line) for line in open(newest) if line.strip()]
curve = [f"{row['kimg']:.0f}:{row['fid_train_ref']:.1f}" for row in rows if "fid_train_ref" in row]
print(f"curve of {os.path.dirname(newest)} (kimg:FID): " + " ".join(curve[-16:]))
EOF
echo "== official FID measurements so far (results/official_fid.log)"
cat results/official_fid.log 2>/dev/null || echo "(none recorded)"
