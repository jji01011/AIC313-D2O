#!/bin/bash
# Downloads the checkpoints listed in handoff/checkpoints.txt from this repository's GitHub Release and puts each
# at the path the training and scoring scripts expect. The repository is private, so a token with read access is needed:
#   GH_TOKEN=<token> bash handoff/fetch_checkpoints.sh
# Files already present with the right md5 are skipped. teacher58_real.tar is unpacked into data/cache/teacher58_real/.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${REPO:-__REPO__}"
TAG="${TAG:-handoff-2026-10-27}"
: "${GH_TOKEN:?set GH_TOKEN to a GitHub token that can read $REPO}"
api() { curl -fsSL -H "Authorization: Bearer $GH_TOKEN" -H "X-GitHub-Api-Version: 2022-11-28" "$@"; }

release=$(api "https://api.github.com/repos/$REPO/releases/tags/$TAG")
while read -r asset path md5 size; do
    if [ -f "$path" ] && [ "$(md5sum "$path" | cut -d' ' -f1)" = "$md5" ]; then
        echo "ok      $path"; continue
    fi
    id=$(printf '%s' "$release" | python3 -c "import json,sys; print(next(a['id'] for a in json.load(sys.stdin)['assets'] if a['name'] == sys.argv[1]))" "$asset")
    echo "fetch   $path ($((size / 1000000)) MB)"
    mkdir -p "$(dirname "$path")"
    api -H "Accept: application/octet-stream" -o "$path.part" "https://api.github.com/repos/$REPO/releases/assets/$id"
    [ "$(md5sum "$path.part" | cut -d' ' -f1)" = "$md5" ] || { echo "md5 mismatch: $path" >&2; exit 1; }
    mv "$path.part" "$path"
done < handoff/checkpoints.txt

if [ -f data/cache/teacher58_real.tar ] && [ ! -d data/cache/teacher58_real ]; then
    tar -xf data/cache/teacher58_real.tar -C data/cache && echo "unpacked data/cache/teacher58_real/"
fi
