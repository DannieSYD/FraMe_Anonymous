from argparse import ArgumentParser
from IPython import get_ipython
import os


def parse_train_args():
    parser = ArgumentParser()

    ## Added settings
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--use_window_attention", action='store_true')
    parser.add_argument("--aug_data", action='store_true')
    parser.add_argument("--tok_version", type=int, default=0)
    parser.add_argument("--model", type=str, default='fractal_latent_md_gen')  # latent_md_gen, 
    parser.add_argument("--num_residues", type=int, default=4)
    parser.add_argument("--label_drop_prob", type=float, default=0.1)
    parser.add_argument("--class_num", type=int, default=1000)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--proj_dropout", type=float, default=0.1)
    parser.add_argument("--guiding_pixel", type=bool, default=False)
    parser.add_argument("--num_conds", type=int, default=5)
    parser.add_argument("--r_weight", type=float, default=1.0)
    parser.add_argument("--latent_dim", type=int, default=21)
    
    ## Fractal settings
    group = parser.add_argument_group("Fractal settings")
    group.add_argument("--fractal", action='store_true', help="Enable fractal two-level structure")
    group.add_argument("--coarse_stride", type=int, default=10, help="Temporal downsampling factor for coarse level")
    group.add_argument("--coarse_embed_dim", type=int, default=None, help="Embedding dimension for coarse level (defaults to embed_dim)")
    group.add_argument("--fine_embed_dim", type=int, default=None, help="Embedding dimension for fine level (defaults to embed_dim)")
    group.add_argument("--share_ipa", action='store_true', help="Share IPA layers between coarse and fine levels")
    group.add_argument("--s_outer", type=int, default=100, help="3L: stride between L1 (coarsest) and L2 (mid)")
    group.add_argument("--s_inner", type=int, default=100, help="3L: stride between L2 (mid) and L3 (fine)")
    group.add_argument("--aux_loss_weight", type=float, default=0.0,
                       help="3L: weight for auxiliary velocity losses at L1/L2 (0 disables them entirely)")
    

    ## Trainer settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--validate", action='store_true', default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    
    ## Epoch settings
    group = parser.add_argument_group("Epoch settings")
    group.add_argument("--epochs", type=int, default=100)
    group.add_argument("--overfit", action='store_true')
    group.add_argument("--overfit_peptide", type=str, default=None)
    group.add_argument("--overfit_frame", action='store_true')
    group.add_argument("--train_batches", type=int, default=None)
    group.add_argument("--val_batches", type=int, default=None)
    group.add_argument("--val_repeat", type=int, default=1)
    group.add_argument("--train_repeat", type=int, default=1,
                       help="Iterate the train dataset N times per epoch. Use "
                            "this for single-trajectory experiments (e.g. CLN025) "
                            "so each epoch sees N random windows from the same "
                            "trajectory instead of just 1.")
    group.add_argument("--inference_batches", type=int, default=0)
    group.add_argument("--batch_size", type=int, default=8)
    group.add_argument("--val_freq", type=int, default=None)
    group.add_argument("--val_epoch_freq", type=int, default=1)
    group.add_argument("--no_validate", action='store_true')
    group.add_argument("--designability_freq", type=int, default=1)

    ## Logging args
    group = parser.add_argument_group("Logging settings")
    group.add_argument("--print_freq", type=int, default=100)
    group.add_argument("--ckpt_freq", type=int, default=1)
    group.add_argument("--wandb", action="store_true")
    group.add_argument("--run_name", type=str, default="default")
    

    ## Optimization settings
    group = parser.add_argument_group("Optimization settings")
    group.add_argument("--accumulate_grad", type=int, default=1)
    group.add_argument("--grad_clip", type=float, default=1.)
    group.add_argument("--check_grad", action='store_true')
    group.add_argument('--grad_checkpointing', action='store_true')
    group.add_argument('--adamW', action='store_true')
    group.add_argument('--ema', action='store_true')
    group.add_argument('--ema_decay', type=float, default=0.999)
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument('--precision', type=str, default='32-true')
    
    ## Training data 
    group = parser.add_argument_group("Training data settings")
    # group.add_argument('--train_split', type=str, default=None, required=True)
    # group.add_argument('--val_split', type=str, default=None, required=True)
    # group.add_argument('--data_dir', type=str, default=None, required=True)
    group.add_argument('--train_split', type=str, default=None)
    group.add_argument('--val_split', type=str, default=None)
    group.add_argument('--data_dir', type=str, default=None)
    group.add_argument('--num_frames', type=int, default=50)
    group.add_argument('--crop', type=int, default=256)
    group.add_argument('--suffix', type=str, default='')
    group.add_argument('--atlas', action='store_true')
    group.add_argument('--copy_frames', action='store_true')
    group.add_argument('--no_pad', action='store_true')
    group.add_argument('--short_md', action='store_true')
    group.add_argument('--start_frame_interval', type=int, default=10)
    group.add_argument('--stride', type=int, default=100)

    ### Masking settings
    group = parser.add_argument_group("Masking settings")
    group.add_argument('--design_key_frames', action='store_true')
    group.add_argument('--no_aa_emb', action='store_true')
    group.add_argument("--no_torsion", action='store_true')
    group.add_argument("--no_design_torsion", action='store_true')
    group.add_argument("--supervise_no_torsions", action='store_true')
    group.add_argument("--supervise_all_torsions", action='store_true')

    ## Ablations settings
    group = parser.add_argument_group("Ablations settings")
    group.add_argument('--no_offsets', action='store_true')
    group.add_argument('--no_frames', action='store_true')
    
    
    ## Model settings
    group = parser.add_argument_group("Model settings")
    group.add_argument('--hyena', action='store_true')
    group.add_argument('--no_rope', action='store_true')
    group.add_argument('--dropout', type=float, default=0.0)
    group.add_argument('--scale_factor', type=float, default=1.0)
    group.add_argument('--interleave_ipa', action='store_true')
    group.add_argument('--prepend_ipa', action='store_true')
    group.add_argument('--oracle', action='store_true')
    group.add_argument('--num_layers', type=int, default=5)
    group.add_argument('--embed_dim', type=int, default=384)
    group.add_argument('--mha_heads', type=int, default=16)
    group.add_argument('--ipa_heads', type=int, default=4)
    # group.add_argument('--ipa_layers', type=int, default=None)
    group.add_argument('--ipa_head_dim', type=int, default=32)
    group.add_argument('--ipa_qk', type=int, default=8)
    group.add_argument('--ipa_v', type=int, default=8)

    group.add_argument('--time_multiplier', type=float, default=100.)
    group.add_argument('--abs_pos_emb', action='store_true')
    group.add_argument('--abs_time_emb', action='store_true')

    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="GVP", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--sampling_method", type=str, default="dopri5", choices=["dopri5", "euler"])
    group.add_argument('--alpha_max', type=float, default=8)
    group.add_argument('--discrete_loss_weight', type=float, default=0.5)
    group.add_argument("--dirichlet_flow_temp", type=float, default=1.0)
    group.add_argument('--allow_nan_cfactor', action='store_true')
    # group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    

    ## video settings
    group = parser.add_argument_group("Video settings")
    group.add_argument('--tps_condition', action='store_true')
    group.add_argument('--design', action='store_true')
    group.add_argument('--design_from_traj', action='store_true')
    group.add_argument('--sim_condition', action='store_true')
    group.add_argument('--inpainting', action='store_true')
    group.add_argument('--dynamic_mpnn', action='store_true')
    group.add_argument('--mpnn', action='store_true')
    group.add_argument('--frame_interval', type=int, default=None)
    group.add_argument('--cond_interval', type=int, default=None) # for superresolution
    
    ipython = get_ipython()
    is_interactive = ipython is not None and 'ipykernel' in str(ipython).lower()

    if is_interactive:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)
    
    return args


