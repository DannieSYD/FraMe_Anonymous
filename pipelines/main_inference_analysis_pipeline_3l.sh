#!/bin/bash
#
# Inference + analysis pipeline for the 3-level fractal latent MDGen model.
#
# Usage:
#   1. Edit CHECKPOINT, DATA_DIR, OUT_DIR, GPU below.
#   2. Run from the repo root: `bash pipelines/main_inference_analysis_pipeline_3l.sh`
#
# Pipeline steps:
#   - Switch to numpy 1.21.2 for inference
#   - Run sim_inference_mdgen.py with --model fractal_latent_md_gen_3l (single-pass cascade,
#     no --infer_beta needed)
#   - Switch to numpy 1.23.2 for analysis
#   - Run scripts.analyze_peptide_sim
#   - Print the summary via tools/show_result.py

time_start=$(date +%s)

# === EDIT THESE ===
CHECKPOINT="${CKPT:?set CKPT to your checkpoint}"
DATA_DIR="/path/to/4AA_data/"
OUT_DIR="/path/to/output/frame_3l"
NUM_FRAMES=10000
GPU=0
# ==================

mkdir -p "$OUT_DIR"

echo "=========================================="
echo "[1/3] Running inference (3L cascade, single pass)"
echo "  ckpt:    $CHECKPOINT"
echo "  out:     $OUT_DIR"
echo "  frames:  $NUM_FRAMES"
echo "  GPU:     $GPU"
echo "=========================================="

pip install numpy==1.21.2

python sim_inference_mdgen.py \
  --sim_ckpt "$CHECKPOINT" \
  --data_dir "$DATA_DIR" \
  --split splits/4AA_test.csv \
  --num_rollouts 1 \
  --num_frames $NUM_FRAMES \
  --xtc \
  --out_dir "$OUT_DIR" \
  --suffix _i100 \
  --gpu $GPU \
  --model fractal_latent_md_gen_3l

echo "=========================================="
echo "[2/3] Running analysis (numpy 1.23.2)"
echo "=========================================="

pip install numpy==1.23.2

python -m scripts.analyze_peptide_sim \
  --mddir /path/to/4AA_sims \
  --pdbdir "$OUT_DIR" \
  --plot --save --num_workers 2 --truncate $NUM_FRAMES

echo "=========================================="
echo "[3/3] Summary"
echo "=========================================="

python tools/show_result.py --path "$OUT_DIR/out.pkl"

time_end=$(date +%s)
time_taken=$((time_end - time_start))
echo "=========================================="
echo "Total time: $time_taken seconds"
echo "=========================================="
