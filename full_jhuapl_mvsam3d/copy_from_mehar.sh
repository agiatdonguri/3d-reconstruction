#!/usr/bin/env bash
# One-time copy of the JHU APL scene from Mehar's WRIVA tree to YOUR NAS (/data3/$SCS_ID/WRIVA/<scene>).
# Copies sparse/ (images, COLMAP poses + points, MoGe depth, depth_scale.json) without the 3DGS assets and
# the MapAnything camera set. With COPY_EVAL=1, model/ (the Accenture site model), metrics/ and
# sparse/gsplat/ go to <scene>__mehar_eval/, for comparison only.
#
#   SCENE=<scene> ./copy_from_mehar.sh                    # shows sizes, asks before copying
#   SCENE=<scene> MASKS_SRC=/data2/mehark/…/masks ./copy_from_mehar.sh   # also copy his SAM 3 masks
set -euo pipefail

SCS_ID="${SCS_ID:-tnishida}"
SCENE="${SCENE:?set SCENE=<scene folder name under SRC_ROOT>}"
SRC_ROOT="${SRC_ROOT:-/data2/mehark/WRIVA}"
DST_ROOT="${DST_ROOT:-/data3/${SCS_ID}/WRIVA}"
MASKS_SRC="${MASKS_SRC:-}"
COPY_EVAL="${COPY_EVAL:-0}"
SRC="$SRC_ROOT/$SCENE"
DST="$DST_ROOT/$SCENE"

case "$DST_ROOT" in *mehark*) echo "ERROR: DST_ROOT must be your own path, not $DST_ROOT"; exit 1 ;; esac
[[ -d "$SRC" ]] || { echo "ERROR: $SRC does not exist (ask Mehar for the exact folder)"; exit 1; }

echo "source: $SRC"
du -sh "$SRC"/* "$SRC"/sparse/* 2>/dev/null | sort -h | tail -n 25
echo ""
echo "copy → $DST (excluding model/, metrics/, sparse/gsplat/, sparse/map_anything_rescaled_sparse/)"
[[ "$COPY_EVAL" == 1 ]] && echo "     + model/, metrics/, sparse/gsplat/ → ${DST}__mehar_eval/"
[[ -n "$MASKS_SRC" ]] && echo "     + masks $MASKS_SRC → $DST_ROOT/${SCENE}__masks/"
read -r -p "continue? [y/N] " ok
[[ "$ok" == y || "$ok" == Y ]] || exit 0

mkdir -p "$DST"
rsync -a --info=progress2 --exclude 'model/' --exclude 'metrics/' --exclude 'sparse/gsplat/' \
  --exclude 'sparse/map_anything_rescaled_sparse/' "$SRC/" "$DST/"
if [[ "$COPY_EVAL" == 1 ]]; then
  mkdir -p "${DST}__mehar_eval/sparse"
  for d in model metrics; do [[ -d "$SRC/$d" ]] && rsync -a "$SRC/$d/" "${DST}__mehar_eval/$d/"; done
  [[ -d "$SRC/sparse/gsplat" ]] && rsync -a "$SRC/sparse/gsplat/" "${DST}__mehar_eval/sparse/gsplat/"
fi
if [[ -n "$MASKS_SRC" ]]; then
  rsync -a --info=progress2 "$MASKS_SRC/" "$DST_ROOT/${SCENE}__masks/"
fi
chmod -R u+w "$DST"
echo ""
echo "done: $DST"
echo "next (on a compute node, in tmux): SCENE=$SCENE STAGES=inspect $(dirname "$0")/run_trinity.sh"
