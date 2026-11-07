#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
grep -a '| 1 |' results/official_fid.log | awk -F' \\| ' '{
    m = $5
    if (match(m, /mean [0-9.]+/)) v = substr(m, RSTART + 5, RLENGTH - 5); else { split(m, a, /[ ,(]/); v = a[1] }
    split($2, c, ":"); print v, c[1] }' | sort -g | while read -r value file; do
    [ -f "$file" ] && { echo "$file"; break; }
done
