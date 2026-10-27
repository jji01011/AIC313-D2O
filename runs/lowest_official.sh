#!/bin/bash
# Prints the one-step checkpoint with the lowest official FID (mean of its runs in results/official_fid.log) among the
# files that still exist. The scoring scripts never remove it (the user, 2026-10-15, after the 21.63 checkpoint of
# e112_lorag_f5 at 3325 kimg had been removed for not being its run's lowest train-reference FID).
cd "$(dirname "$0")/.." || exit 1
grep -a '| 1 |' results/official_fid.log | awk -F' \\| ' '{
    m = $5
    if (match(m, /mean [0-9.]+/)) v = substr(m, RSTART + 5, RLENGTH - 5); else { split(m, a, /[ ,(]/); v = a[1] }
    split($2, c, ":"); print v, c[1] }' | sort -g | while read -r value file; do
    [ -f "$file" ] && { echo "$file"; break; }
done
