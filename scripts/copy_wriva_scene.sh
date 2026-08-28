#!/usr/bin/env bash
# One-time: copy a WRIVA scene from Mehar → YOUR NAS (/data3/$USER).
# Do not train from Mehar's tree; do not fill /scratch from here for the whole dataset.

set -euo pipefail
SCS_ID="${SCS_ID:-tnishida}"
SCENE="${1:?usage: $0 <scene_name>}"
SRC="${SRC:-/data2/mehark/WRIVA/$SCENE}"
DST="${DST:-/data3/${SCS_ID}/WRIVA/$SCENE}"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: source missing: $SRC"
  exit 1
fi
if [[ "$DST" == /data2/mehark/* ]] || [[ "$DST" == /data/mehark/* ]]; then
  echo "ERROR: refusing to write under Mehar paths"
  exit 1
fi

echo "Copy $SRC → $DST"
mkdir -p "$(dirname "$DST")"
# tar|tar is often faster for many small files (cluster doc tip)
if command -v tar >/dev/null; then
  mkdir -p "$DST"
  (cd "$SRC" && tar cf - .) | (cd "$DST" && tar xf -)
else
  rsync -a "$SRC/" "$DST/"
fi
echo "Done. Next: on a compute node, SCENE=$SCENE ./scripts/run_trinity_stage2_lora.sh"
