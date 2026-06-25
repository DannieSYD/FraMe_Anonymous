#!/bin/bash
#
# Training pipeline for the 3-level fractal latent MDGen model.
#
# Constraints (asserted at model init time):
#   - NUM_FRAMES must be divisible by S_OUTER * S_INNER
#   - S_OUTER and S_INNER must both be even (because pad_size = stride/2)
#   - S_FINE = NUM_FRAMES / (S_OUTER * S_INNER) is derived
#
# Two variants supported via AUX_LOSS_WEIGHT:
#   - 0.0  (default) → pure latent cascade, single MSE loss on L3
#   - >0   (e.g. 0.1) → also supervise L1/L2 with auxiliary velocity heads
#
# Edit the variables below, then run from the repo root: `bash pipelines/main_train_pipeline_3l.sh`

DATA_DIR="/path/to/4AA_data"
NUM_FRAMES=10000
S_OUTER=10                 # stride between L1 (coarsest) and L2 (mid)
S_INNER=10                 # stride between L2 (mid) and L3 (fine); S_FINE = NUM_FRAMES / (S_OUTER*S_INNER) = 100
EMBED_DIM=768              # default 384; doubled for higher capacity
AUX_LOSS_WEIGHT=0.0        # set to e.g. 0.1 to enable the L1/L2 anchor losses

RUN_NAME="frac3l_so${S_OUTER}_si${S_INNER}_ed${EMBED_DIM}_aux${AUX_LOSS_WEIGHT}"
GPU="0"

pip install numpy==1.21.2
python train.py \
  --sim_condition \
  --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv \
  --data_dir $DATA_DIR \
  --model fractal_latent_md_gen_3l \
  --num_frames $NUM_FRAMES \
  --s_outer $S_OUTER --s_inner $S_INNER \
  --embed_dim $EMBED_DIM \
  --aux_loss_weight $AUX_LOSS_WEIGHT \
  --prepend_ipa --abs_pos_emb \
  --crop 4 \
  --ckpt_freq 50 \
  --val_repeat 25 \
  --suffix _i100 \
  --epochs 10000 \
  --batch_size 1 \
  --gpu $GPU \
  --run_name $RUN_NAME \
  --wandb
