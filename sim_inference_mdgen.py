import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sim_ckpt', type=str, default=None, required=True)
parser.add_argument('--data_dir', type=str, default=None, required=True)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--num_frames', type=int, default=1000)
parser.add_argument('--num_rollouts', type=int, default=100)
parser.add_argument('--no_frames', action='store_true')
parser.add_argument('--tps', action='store_true')
parser.add_argument('--xtc', action='store_true')
parser.add_argument('--out_dir', type=str, default=".")
parser.add_argument('--split', type=str, default='splits/4AA_test.csv')
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--model", type=str, default='fractal_latent_md_gen_3l')
parser.add_argument("--infer_beta", type=float, default=0.5)
parser.add_argument("--shard_index", type=int, default=0,
                    help="Run only peptides whose index mod num_shards == shard_index. "
                         "Use with multi-GPU launchers to parallelise across GPUs.")
parser.add_argument("--num_shards", type=int, default=1,
                    help="Total number of shards across all parallel processes (>=1).")
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import os, torch, mdtraj, tqdm, time
import numpy as np
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.residue_constants import restype_order, restype_atom37_mask
from mdgen.tensor_utils import tensor_tree_map
from mdgen.wrapper import NewMDGenWrapper, FractalMDGenWrapper
from mdgen.utils import atom14_to_pdb
import pandas as pd


os.makedirs(args.out_dir, exist_ok=True)


def get_batch(name, seqres, num_frames):
    arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}.npy', 'r')

    if not args.tps: # else keep all frames
        arr = np.copy(arr[0:1]).astype(np.float32)

    frames = atom14_to_frames(torch.from_numpy(arr))
    seqres = torch.tensor([restype_order[c] for c in seqres])
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres[None])).float()
    L = len(seqres)
    mask = torch.ones(L)
    
    if args.no_frames:
        return {
            'atom37': atom37,
            'seqres': seqres,
            'mask': restype_atom37_mask[seqres],
        }
        
    torsions, torsion_mask = atom37_to_torsions(atom37, seqres[None])
    return {
        'torsions': torsions,
        'torsion_mask': torsion_mask[0],
        'trans': frames._trans,
        'rots': frames._rots._rot_mats,
        'seqres': seqres,
        'mask': mask, # (L,)
    }

def rollout(model, batch):

    #print('Start sim', batch['trans'][0,0,0])
    if args.no_frames:
        
        expanded_batch = {
            'atom37': batch['atom37'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
        }
    else:    
        expanded_batch = {
            'torsions': batch['torsions'].expand(-1, args.num_frames, -1, -1, -1),
            'torsion_mask': batch['torsion_mask'],
            'trans': batch['trans'].expand(-1, args.num_frames, -1, -1),
            'rots': batch['rots'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
        }
    atom14, _ = model.inference(expanded_batch, infer_beta=args.infer_beta)
    new_batch = {**batch}

    if args.no_frames:
        new_batch['atom37'] = torch.from_numpy(
            atom14_to_atom37(atom14[:,-1].cpu(), batch['seqres'][0].cpu())
        ).cuda()[:,None].float()
        
        
        
    else:
        frames = atom14_to_frames(atom14[:,-1])
        new_batch['trans'] = frames._trans[None]
        new_batch['rots'] = frames._rots._rot_mats[None]
        atom37 = atom14_to_atom37(atom14[0,-1].cpu(), batch['seqres'][0].cpu())
        torsions, _ = atom37_to_torsions(atom37, batch['seqres'][0].cpu())
        new_batch['torsions'] = torsions[None, None].cuda()

    return atom14, new_batch
    
    
def do(model, name, seqres):

    item = get_batch(name, seqres, num_frames = model.args.num_frames)
    batch = next(iter(torch.utils.data.DataLoader([item])))

    batch = tensor_tree_map(lambda x: x.cuda(), batch)  
    
    all_atom14 = []
    start = time.time()
    for _ in tqdm.trange(args.num_rollouts):
        atom14, batch = rollout(model, batch)
        # print(atom14[0,0,0,1], atom14[0,-1,0,1])
        all_atom14.append(atom14)

    print(time.time() - start)
    all_atom14 = torch.cat(all_atom14, 1)
    
    path = os.path.join(args.out_dir, f'{name}.pdb')
    atom14_to_pdb(all_atom14[0].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)

    if args.xtc:
        traj = mdtraj.load(path)
        traj.superpose(traj)
        traj.save(os.path.join(args.out_dir, f'{name}.xtc'))
        traj[0].save(os.path.join(args.out_dir, f'{name}.pdb'))

@torch.no_grad()
def main():
    if args.model == 'fractal_latent_md_gen_3l':
        model = FractalMDGenWrapper.load_from_checkpoint(args.sim_ckpt)
    else:
        raise ValueError(f"Model {args.model} not found")

    model.eval().to('cuda')
    
    
    df = pd.read_csv(args.split, index_col='name')
    # Filter once so the counter reflects what will actually run.
    all_names = [n for n in df.index if (not args.pdb_id) or n in args.pdb_id]
    # Stride sharding: each shard takes every Nth peptide. Even split
    # regardless of where heavy peptides land in the list.
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f'invalid shard_index={args.shard_index} num_shards={args.num_shards}')
    names = all_names[args.shard_index::args.num_shards]
    total = len(names)
    bar = '=' * 70
    print(f'shard {args.shard_index}/{args.num_shards} on gpu {args.gpu}: '
          f'{total} of {len(all_names)} peptides', flush=True)
    overall_start = time.time()
    for i, name in enumerate(names, start=1):
        elapsed = time.time() - overall_start
        eta_str = ''
        if i > 1:
            avg = elapsed / (i - 1)
            remaining = avg * (total - i + 1)
            eta_str = (f"  elapsed={elapsed/60:.1f}min"
                       f"  eta={remaining/60:.1f}min"
                       f"  avg={avg:.1f}s/peptide")
        print(f"\n{bar}\n[shard {args.shard_index}/{args.num_shards}] "
              f"[{i:>3}/{total}] peptide={name} "
              f"seqres={df.seqres[name]}{eta_str}\n{bar}", flush=True)
        do(model, name, df.seqres[name])
    total_min = (time.time() - overall_start) / 60
    print(f"\n{bar}\nAll {total} peptides done in {total_min:.1f} min\n{bar}",
          flush=True)
        

main()