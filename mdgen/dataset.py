import torch
from .rigid_utils import Rigid
from .residue_constants import restype_order
import numpy as np
import pandas as pd
from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
       
class MDGenDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, repeat=1):
        super().__init__()
        self.df = pd.read_csv(split, index_col='name')
        self.args = args
        self.repeat = repeat
    def __len__(self):
        if self.args.overfit_peptide:
            return 1000
        return self.repeat * len(self.df)

    def __getitem__(self, idx):
        idx = idx % len(self.df)
        if self.args.overfit:
            idx = 0

        if self.args.overfit_peptide is None:
            name = self.df.index[idx]
            seqres = self.df.seqres[name]
        else:
            name = self.args.overfit_peptide
            seqres = name

        if self.args.atlas:
            i = np.random.randint(1, 4)
            full_name = f"{name}_R{i}"
        elif self.args.aug_data:
            # Random value between: 0, 10, 20, ..., 90
            start_frame = np.random.choice(np.arange(0, self.args.stride, self.args.start_frame_interval))
            full_name = f"{name}_{start_frame}"
        else:
            full_name = name
        # shape of arr: [10000, 4, 14, 3]
        arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')
        if self.args.frame_interval:
            arr = arr[::self.args.frame_interval]

        # randomly select a contiguous subset of frames
        if self.args.num_frames < arr.shape[0]:
            frame_start = np.random.choice(np.arange(arr.shape[0] - self.args.num_frames))
        else:
            frame_start = 0
        if self.args.overfit_frame:
            frame_start = 0
        end = frame_start + self.args.num_frames
        # arr = np.copy(arr[frame_start:end]) * 10 # convert to angstroms
        arr = np.copy(arr[frame_start:end]).astype(np.float32)  # / 10.0 # convert to nm
        if self.args.copy_frames:  # all the same across time axis
            arr[1:] = arr[0]

        # arr should be in ANGSTROMS (A = 10^{-10} meters)
        frames = atom14_to_frames(torch.from_numpy(arr))  # obtain the translation and rotation for each frame
        # converts amino-acid letter codes into integer indices
        seqres = np.array([restype_order[c] for c in seqres])
        # duplicates the sequence across all frames
        aatype = torch.from_numpy(seqres)[None].expand(self.args.num_frames, -1)
        # convert from atom 14 to 37, add place-holders
        atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype)).float()
        
        L = frames.shape[1]
        mask = np.ones(L, dtype=np.float32)  # a basic mask, for all residues
        
        if self.args.no_frames:
            return {
                'name': full_name,
                'frame_start': frame_start,
                'atom37': atom37,
                'seqres': seqres,
                'mask': restype_atom37_mask[seqres], # (L,)
            }
        # shape: [10000, 4, 7, 2]
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        
        torsion_mask = torsion_mask[0]
        
        if self.args.atlas:
            if L > self.args.crop:
                start = np.random.randint(0, L - self.args.crop + 1)
                torsions = torsions[:,start:start+self.args.crop]
                frames = frames[:,start:start+self.args.crop]
                seqres = seqres[start:start+self.args.crop]
                mask = mask[start:start+self.args.crop]
                torsion_mask = torsion_mask[start:start+self.args.crop]
                
            
            elif L < self.args.crop:
                pad = self.args.crop - L
                frames = Rigid.cat([
                    frames, 
                    Rigid.identity((self.args.num_frames, pad), requires_grad=False, fmt='rot_mat')
                ], 1)
                mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
                seqres = np.concatenate([seqres, np.zeros(pad, dtype=int)])
                torsions = torch.cat([torsions, torch.zeros((torsions.shape[0], pad, 7, 2), dtype=torch.float32)], 1)
                torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)])

        return {
            'name': full_name,
            'frame_start': frame_start,
            'torsions': torsions,  # [10000, 4, 7, 2]
            'torsion_mask': torsion_mask,  # [4, 7]
            'trans': frames._trans,
            'rots': frames._rots._rot_mats,
            'seqres': seqres,  # residue type with shape: [4]
            'mask': mask, # (L,)
        }

