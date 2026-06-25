# FRAME: Fractal Generative Model for Molecular Dynamics

FRAME is a fractal latent generative model for molecular dynamics (MD)
trajectories. It learns a multi-level (coarse-to-fine) cascade of
flow-matching models over a latent representation of peptide structure: a
coarse level captures long-timescale motion, while successively finer levels
fill in intermediate frames. This release contains the minimal code to train,
run inference with, and analyze the 3-level FRAME model
(`fractal_latent_md_gen_3l`) on tetrapeptide MD data.

## Environment

The training/inference stack and the analysis stack require **different numpy
versions**. The pipeline scripts under `pipelines/` switch between them
automatically (`pip install numpy==1.21.2` for train/inference,
`pip install numpy==1.23.2` for analysis); if you run steps by hand, switch
versions accordingly or you will hit import errors and silent breakage.

- Python 3.8
- Training / inference: `numpy==1.21.2`, `torch==2.0.0`,
  `pytorch_lightning==2.0.4`, `mdtraj==1.9.9`, `biopython==1.79`, `einops`,
  `torchdiffeq`, `fair-esm`, `dm-tree`, `matplotlib==3.7.2`, `wandb`
- Analysis: `numpy==1.23.2`, `pyemma`, `statsmodels`
- `openfold` is required for the IPA modules and for residue constants. In
  particular, `mdgen/residue_constants.py` loads `stereo_chemical_props.txt`
  from the external `openfold.resources` package — install `openfold` per its
  own instructions; it is **not** vendored here.

Install the core dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

All commands are run from the repository root. Replace the `/path/to/...`
placeholders with your own data, checkpoint, and output directories.

```bash
# 1. Preprocess (10ps tetrapeptide data)
python -m scripts.prep_sims --split splits/4AA.csv --sim_dir /path/to/4AA_sims \
  --outdir /path/to/4AA_data --num_workers $(nproc) --suffix _i100 --stride 100

# 2. Train FRAME 3-level
python train.py --sim_condition --train_split splits/4AA_train.csv --val_split splits/4AA_val.csv \
  --data_dir /path/to/4AA_data --num_frames 10000 --prepend_ipa --abs_pos_emb --crop 4 \
  --suffix _i100 --model fractal_latent_md_gen_3l --run_name frame_3l --gpu 0

# 3. Inference + analysis
python sim_inference_mdgen.py --sim_ckpt /path/to/checkpoints/frame_3l/CKPT.ckpt \
  --data_dir /path/to/4AA_data --split splits/4AA_test.csv --num_rollouts 1 \
  --num_frames 10000 --xtc --out_dir /path/to/output --suffix _i100 --model fractal_latent_md_gen_3l --gpu 0
python -m scripts.analyze_peptide_sim --mddir /path/to/4AA_sims --pdbdir /path/to/output \
  --plot --save --num_workers 20
```

End-to-end wrappers for the three stages are provided in `pipelines/`
(`main_train_pipeline_3l.sh`, `main_inference_analysis_pipeline_3l.sh`,
`main_analysis_pipeline_3l.sh`). Edit the variables at the top of each script,
then run from the repo root, e.g. `bash pipelines/main_train_pipeline_3l.sh`.

## Notes

- `--num_frames` must be divisible by `s_outer * s_inner`; `s_outer` and
  `s_inner` must both be even. The finest stride is derived as
  `num_frames / (s_outer * s_inner)`.
- `--aux_loss_weight 0.0` (default) trains a pure latent cascade with a single
  MSE loss; setting it `> 0` adds auxiliary velocity losses on the coarser
  levels.
- `--suffix _i100` selects the 10ps-interval tetrapeptide preprocessing and
  must match between preprocessing, training, and inference.
- Set `--gpu N` to choose a GPU (it sets `CUDA_VISIBLE_DEVICES` before torch
  is imported).
