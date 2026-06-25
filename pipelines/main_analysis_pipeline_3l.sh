#!/bin/bash
# Standalone analysis for 3-level fractal latent MDGen outputs.
# Use this when inference has already produced PDB/XTC files in OUT_DIR
# and you just want to (re-)compute the analysis pickle without re-running
# inference. Strips the multi-GPU launcher + checkpoint + sharding from
# main_inference_analysis_pipeline_3l_multigpu.sh, keeping only the
# analyze_peptide_sim + show_result steps.

set -e
time_start=$(date +%s)

# === Edit these three blocks ===
OUT_DIR="/path/to/output/frame_3l"
MDDIR="/path/to/4AA_sims"
NUM_FRAMES=10000
# === end edit ===

if [ ! -d "$OUT_DIR" ]; then
    echo "ERROR: OUT_DIR does not exist: $OUT_DIR" >&2
    exit 1
fi

n_pdb=$(ls "$OUT_DIR"/*.pdb 2>/dev/null | wc -l)
n_xtc=$(ls "$OUT_DIR"/*.xtc 2>/dev/null | wc -l)
echo "=========================================="
echo "Analysis pipeline (3L cascade)"
echo "  pdbdir   = $OUT_DIR"
echo "             $n_pdb PDB files, $n_xtc XTC files"
echo "  mddir    = $MDDIR"
echo "  truncate = $NUM_FRAMES"
echo "=========================================="

pip install -q numpy==1.23.2

python -m scripts.analyze_peptide_sim \
    --mddir "$MDDIR" \
    --pdbdir "$OUT_DIR" \
    --plot --save \
    --num_workers 2 \
    --truncate $NUM_FRAMES

python tools/show_result.py --path "$OUT_DIR/out.pkl"

time_end=$(date +%s)
time_taken=$((time_end - time_start))
echo "=========================================="
echo "Total time taken: $time_taken seconds ($(echo "scale=1;$time_taken/60" | bc) min)"
echo "=========================================="
