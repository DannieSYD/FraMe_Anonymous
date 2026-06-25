import copy
import torch
import torch.nn as nn
import numpy as np
from .latent_model import LatentMDGenModel
from mdgen.rigid_utils import Rigid, Rotation


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class FractalLatent3LMDGenModel(nn.Module):
    """
    Three-level fractal MDGen model with temporal hierarchy.
    T = s_outer * s_inner * s_fine.
      - L1 (coarsest): one trajectory of length s_outer.
      - L2 (mid): s_outer parallel chunks, each length s_inner. Conditioned on L1.
      - L3 (finest): s_outer*s_inner parallel chunks, each length s_fine. Conditioned on L2.
    L1 and L2 emit embed-space features; only L3 produces a physical-space velocity.
    """

    def __init__(self, args, latent_dim):
        super().__init__()
        self.args = args
        assert args.num_frames % (args.s_outer * args.s_inner) == 0, \
            f"num_frames={args.num_frames} not divisible by s_outer*s_inner={args.s_outer*args.s_inner}"
        assert args.s_outer % 2 == 0, f"s_outer must be even, got {args.s_outer}"
        assert args.s_inner % 2 == 0, f"s_inner must be even, got {args.s_inner}"

        self.s_outer = args.s_outer
        self.s_inner = args.s_inner
        self.s_fine = args.num_frames // (args.s_outer * args.s_inner)

        if args.abs_time_emb:
            self.register_buffer('time_embed',
                                 nn.Parameter(torch.zeros(1, args.num_frames, 8), requires_grad=False))
            latent_dim += 8

        self.latent_dim = latent_dim

        # Per-level channel-cat factors (frames pooled into channels per token at each level):
        #   L1: each L1 token covers s_inner * s_fine raw frames.
        #   L2: each L2 token covers s_fine raw frames.
        #   L3: each L3 token is one raw frame.
        args_l1 = copy.deepcopy(args)
        args_l1.cat_factor = self.s_inner * self.s_fine
        args_l1.coarse_in_embed_space = False  # L1 has no coarse input
        args_l1.coarse_stride = self.s_inner * self.s_fine  # unused for padding (L1 has no x_coarse)

        args_l2 = copy.deepcopy(args)
        args_l2.cat_factor = self.s_fine
        args_l2.coarse_in_embed_space = True
        args_l2.coarse_stride = self.s_outer  # padding pad_size = s_outer/2

        args_l3 = copy.deepcopy(args)
        args_l3.cat_factor = 1
        args_l3.coarse_in_embed_space = True
        args_l3.coarse_stride = self.s_inner  # padding pad_size = s_inner/2

        self.l1_model = LatentMDGenModel(args_l1, latent_dim * self.s_inner * self.s_fine, stage='coarse')
        self.l2_model = LatentMDGenModel(args_l2, latent_dim * self.s_fine, stage='coarse')
        self.l3_model = LatentMDGenModel(args_l3, latent_dim, stage='fine')

        # L1 and L2 do not have emb_to_latent in default mode -- strip them so the param
        # count reflects the latent-cascade design.
        if args.aux_loss_weight == 0.0:
            del self.l1_model.emb_to_latent
            del self.l2_model.emb_to_latent

    # -------------- fractalize / de_fractalize --------------

    def fractalize(self, x, mask, t=None, trans=None, rots=None, aatype=None, mode='l1'):
        """
        Modes:
          'l1': [B, T, L, C] -> [B, s_outer, L, s_inner*s_fine*C]
          'l2': [B, T, L, C] -> [B*s_outer, s_inner, L, s_fine*C]
          'l3': [B, T, L, C] -> [B*s_outer*s_inner, s_fine, L, C]
          'l3_args': returns batch-replicated start/end frames, aatype, t for L3.
        """
        B, T, L, C = x.shape
        s_o, s_i, s_f = self.s_outer, self.s_inner, self.s_fine
        assert T == s_o * s_i * s_f, f"T={T} != s_outer*s_inner*s_fine={s_o*s_i*s_f}"

        if mode == 'l1':
            # Group all (s_i*s_f) frames within each outer block and cat channels.
            # [B, T, L, C] -> [B, s_o, s_i*s_f, L, C] -> [B, s_o, L, s_i*s_f, C] -> [B, s_o, L, s_i*s_f*C]
            x = x.reshape(B, s_o, s_i * s_f, L, C)
            x = x.transpose(2, 3)          # [B, s_o, L, s_i*s_f, C]
            x = x.reshape(B, s_o, L, s_i * s_f * C)
            mask_l1 = mask[:, ::(s_i * s_f), :]
            assert x.shape[1] == mask_l1.shape[1]
            return x, mask_l1

        elif mode == 'l2':
            # Group s_f frames within each inner block and cat channels; fold outer into batch.
            # [B, T, L, C] -> [B, s_o, s_i, s_f, L, C] -> [B, s_o, s_i, L, s_f, C]
            #               -> [B, s_o, s_i, L, s_f*C] -> [B*s_o, s_i, L, s_f*C]
            x = x.reshape(B, s_o, s_i, s_f, L, C)
            x = x.transpose(3, 4)          # [B, s_o, s_i, L, s_f, C]
            x = x.reshape(B, s_o, s_i, L, s_f * C)
            x = x.reshape(B * s_o, s_i, L, s_f * C)
            mask_l2 = mask.reshape(B, s_o, s_i, s_f, L)[:, :, :, 0, :].reshape(B * s_o, s_i, L)
            return x, mask_l2

        elif mode == 'l3':
            # Fold all outer*inner blocks into batch; each chunk has s_f frames, no channel-cat.
            # [B, T, L, C] -> [B, s_o*s_i, s_f, L, C] -> [B*s_o*s_i, s_f, L, C]
            x = x.reshape(B, s_o * s_i, s_f, L, C)
            x = x.reshape(B * s_o * s_i, s_f, L, C)
            mask_l3 = mask.reshape(B, s_o * s_i, s_f, L).reshape(B * s_o * s_i, s_f, L)
            return x, mask_l3

        elif mode == 'l3_args':
            assert trans is not None and rots is not None and aatype is not None and t is not None
            trans = trans.reshape(B, s_o * s_i, s_f, L, 3).reshape(B * s_o * s_i, s_f, L, 3)
            rots = rots.reshape(B, s_o * s_i, s_f, L, 3, 3).reshape(B * s_o * s_i, s_f, L, 3, 3)
            rigids = Rigid(trans=trans, rots=Rotation(rot_mats=rots))
            start_frames = rigids[:, 0]
            end_frames = rigids[:, -1]
            aatype = aatype.repeat_interleave(s_o * s_i, dim=0)
            t = t.repeat_interleave(s_o * s_i, dim=0)
            return t, start_frames, end_frames, aatype

        else:
            raise ValueError(f"Invalid fractalize mode: {mode}")

    def de_fractalize(self, x, mode='l1'):
        s_o, s_i, s_f = self.s_outer, self.s_inner, self.s_fine
        if mode == 'l1':
            # Reverse: [B, s_o, L, s_i*s_f*C] -> [B, s_o, L, s_i*s_f, C]
            #        -> [B, s_o, s_i*s_f, L, C] -> [B, s_o*s_i*s_f, L, C]
            B = x.shape[0]
            L = x.shape[2]
            C = x.shape[3] // (s_i * s_f)
            assert x.shape[1] == s_o and x.shape[3] == s_i * s_f * C
            x = x.reshape(B, s_o, L, s_i * s_f, C)
            x = x.transpose(2, 3)          # [B, s_o, s_i*s_f, L, C]
            x = x.reshape(B, s_o * s_i * s_f, L, C)
            return x

        elif mode == 'l2':
            # Reverse: [B*s_o, s_i, L, s_f*C] -> [B, s_o, s_i, L, s_f*C]
            #        -> [B, s_o, s_i, L, s_f, C] -> [B, s_o, s_i, s_f, L, C]
            #        -> [B, s_o*s_i*s_f, L, C]
            BSo = x.shape[0]
            B = BSo // s_o
            L = x.shape[2]
            C = x.shape[3] // s_f
            assert x.shape[1] == s_i and x.shape[3] == s_f * C
            x = x.reshape(B, s_o, s_i, L, s_f, C)
            x = x.transpose(3, 4)          # [B, s_o, s_i, s_f, L, C]
            x = x.reshape(B, s_o * s_i * s_f, L, C)
            return x

        elif mode == 'l3':
            # Reverse: [B*s_o*s_i, s_f, L, C] -> [B, s_o*s_i, s_f, L, C]
            #        -> [B, s_o*s_i*s_f, L, C]
            BSoSi = x.shape[0]
            B = BSoSi // (s_o * s_i)
            L = x.shape[2]
            C = x.shape[3]
            assert x.shape[1] == s_f
            x = x.reshape(B, s_o * s_i, s_f, L, C)
            x = x.reshape(B, s_o * s_i * s_f, L, C)
            return x

        else:
            raise ValueError(f"Invalid de_fractalize mode: {mode}")

    def _forward_l1(self, x, t, mask, start_frames, end_frames, x_cond, x_cond_mask, aatype):
        """L1 forward: returns embed-space features [B, s_outer, L, embed_dim]."""
        x_l1, mask_l1 = self.fractalize(x, mask, mode='l1')
        x_cond_l1, x_cond_mask_l1 = self.fractalize(x_cond, x_cond_mask, mode='l1')
        f1 = self._run_inner_until_features(
            self.l1_model, x_l1, t, mask_l1, start_frames, end_frames,
            x_cond_l1, x_cond_mask_l1, aatype, x_coarse=None, coarse_mask=None
        )
        return f1

    def _forward_l2(self, x, t, mask, start_frames, end_frames, x_cond, x_cond_mask, aatype, f1):
        """L2 forward: conditioned on f1. Returns embed-space features [B*s_outer, s_inner, L, embed_dim]."""
        s_o = self.s_outer
        x_l2, mask_l2 = self.fractalize(x, mask, mode='l2')
        x_cond_l2, x_cond_mask_l2 = self.fractalize(x_cond, x_cond_mask, mode='l2')
        # f1 is the coarse input; mask for f1 is all-ones along its time dim.
        coarse_mask = torch.ones(f1.shape[0], f1.shape[1], f1.shape[2], device=f1.device)
        # Replicate t / aatype across the L2 batch dim (B*s_outer).
        # t and aatype have batch dim B; L2 has batch dim B*s_outer, so replicate.
        t_l2 = t.repeat_interleave(s_o, dim=0)
        aatype_l2 = aatype.repeat_interleave(s_o, dim=0)
        # Replicate start/end frames so each L2 chunk has its own copy.
        sf_l2 = self._repeat_rigids(start_frames, s_o)
        ef_l2 = self._repeat_rigids(end_frames, s_o)
        f_full = self._run_inner_until_features(
            self.l2_model, x_l2, t_l2, mask_l2, sf_l2, ef_l2,
            x_cond_l2, x_cond_mask_l2, aatype_l2, x_coarse=f1, coarse_mask=coarse_mask
        )
        # Slice off the appended coarse-context tokens (s_outer of them, after the s_inner own tokens).
        f2 = f_full[:, :self.s_inner]
        return f2

    @staticmethod
    def _repeat_rigids(rigids, n):
        """Repeat-interleave a Rigid object along its leading batch dim by n."""
        trans = rigids.get_trans().repeat_interleave(n, dim=0)
        rots_mats = rigids.get_rots().get_rot_mats().repeat_interleave(n, dim=0)
        return Rigid(trans=trans, rots=Rotation(rot_mats=rots_mats))

    def _forward_l3(self, x, t, mask, start_frames, end_frames, x_cond, x_cond_mask, aatype,
                    f2, trans, rots):
        """L3 forward: conditioned on f2. Returns velocity prediction [B, T, L, latent_dim]."""
        # Per-chunk start/end frames + replicated aatype/t (for B*s_outer*s_inner chunks).
        t_l3, sf_l3, ef_l3, aatype_l3 = self.fractalize(
            x, mask, t=t, trans=trans, rots=rots, aatype=aatype, mode='l3_args'
        )
        x_l3, mask_l3 = self.fractalize(x, mask, mode='l3')
        x_cond_l3, x_cond_mask_l3 = self.fractalize(x_cond, x_cond_mask, mode='l3')
        coarse_mask = torch.ones(f2.shape[0], f2.shape[1], f2.shape[2], device=f2.device)
        # Run the inner model fully (with emb_to_latent) -- L3 is the only level
        # that has a physical-space output head.
        from .latent_model import grad_checkpoint
        inner = self.l3_model
        x_emb = inner.latent_to_emb(x_l3)
        if inner.args.abs_pos_emb:
            x_emb = x_emb + inner.pos_embed
        x_emb = x_emb + inner.cond_to_emb(x_cond_l3) + inner.mask_to_emb(x_cond_mask_l3)
        f2_proj = inner.latent_to_emb_coarse(f2)
        f2_padded, mask_coarse = inner.padding(f2_proj, coarse_mask)
        x_emb = torch.cat([x_emb, f2_padded], dim=1)
        mask_full = torch.cat([mask_l3, mask_coarse], dim=1)
        t_emb_proj = inner.t_embedder(t_l3 * inner.args.time_multiplier)[:, None]
        if inner.args.prepend_ipa:
            x_emb = x_emb + inner.run_ipa(t_emb_proj[:, 0], mask_full[:, 0], sf_l3, ef_l3, aatype_l3)[:, None]
        for layer in inner.layers:
            x_emb = grad_checkpoint(layer, (x_emb, t_emb_proj, mask_full, sf_l3), inner.args.grad_checkpointing)
        # Slice to L3 own tokens, run emb_to_latent.
        x_emb = x_emb[:, :self.s_fine]
        v3_chunked = inner.emb_to_latent(x_emb, t_emb_proj)  # [B*s_o*s_i, s_fine, L, cond_dim]
        v3 = self.de_fractalize(v3_chunked, mode='l3')
        return v3

    def forward(self, x, t, mask, start_frames=None, end_frames=None,
                x_cond=None, x_cond_mask=None, aatype=None,
                trans=None, rots=None, **kwargs):
        """
        Training-mode forward.
        Returns: v3 [B, T, L, latent_dim] (de-fractalized velocity prediction).
        L3's emb_to_latent already outputs `latent_dim` channels (cond_dim = D
        thanks to the cond_dim accounting in LatentMDGenModel), so no output-side
        slicing is needed.
        """
        if self.args.abs_time_emb:
            time_embed = get_1d_sincos_pos_embed_from_grid(
                self.time_embed.shape[-1], np.arange(self.args.num_frames)
            )
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))
            te = self.time_embed[:, :, None].expand(x.shape[0], -1, x.shape[2], -1)
            x = torch.cat([x, te], dim=-1)
            # NOTE: x_cond is NOT cat'd with time-emb. The inner LatentMDGenModel's
            # cond_to_emb is sized for the no-time-emb channel count (cond_dim) -- this
            # matches the existing 2-level pattern where only `x` carries time-emb.

        f1 = self._forward_l1(x, t, mask, start_frames, end_frames, x_cond, x_cond_mask, aatype)
        f2 = self._forward_l2(x, t, mask, start_frames, end_frames, x_cond, x_cond_mask, aatype, f1=f1)
        v3 = self._forward_l3(x, t, mask, start_frames, end_frames, x_cond, x_cond_mask, aatype,
                               f2=f2, trans=trans, rots=rots)
        if self.args.aux_loss_weight > 0.0:
            # Need t_emb to call emb_to_latent. Each level's t_embedder may have been
            # initialized differently (per-level args copies); recompute per level.
            t_emb_l1 = self.l1_model.t_embedder(t * self.args.time_multiplier)[:, None]
            t_l2 = t.repeat_interleave(self.s_outer, dim=0)
            t_emb_l2 = self.l2_model.t_embedder(t_l2 * self.args.time_multiplier)[:, None]
            v1 = self._aux_velocity_l1(f1, t_emb_l1)
            v2 = self._aux_velocity_l2(f2, t_emb_l2)
            return v3, v1, v2
        return v3

    def _aux_velocity_l1(self, f1, t_emb):
        """Run l1.emb_to_latent on f1 and de-fractalize to [B, T, L, latent_dim]."""
        v1_chunked = self.l1_model.emb_to_latent(f1, t_emb)
        return self.de_fractalize(v1_chunked, mode='l1')

    def _aux_velocity_l2(self, f2, t_emb):
        v2_chunked = self.l2_model.emb_to_latent(f2, t_emb)
        return self.de_fractalize(v2_chunked, mode='l2')

    def forward_inference(self, x, t, mask, start_frames=None, end_frames=None,
                          x_cond=None, x_cond_mask=None, aatype=None,
                          trans=None, rots=None, **kwargs):
        return self.forward(x, t, mask, start_frames, end_frames,
                            x_cond, x_cond_mask, aatype, trans=trans, rots=rots, **kwargs)

    def _run_inner_until_features(self, inner, x, t, mask, start_frames, end_frames,
                                  x_cond, x_cond_mask, aatype, x_coarse, coarse_mask):
        """
        Replicate LatentMDGenModel.forward() but stop before emb_to_latent and
        return the embed-space tensor. Caller is responsible for slicing off any
        appended coarse-context tokens (when x_coarse is not None).
        """
        from .latent_model import grad_checkpoint
        x = inner.latent_to_emb(x)
        if inner.args.abs_pos_emb:
            x = x + inner.pos_embed
        if x_cond is not None:
            x = x + inner.cond_to_emb(x_cond) + inner.mask_to_emb(x_cond_mask)
        if x_coarse is not None:
            x_coarse = inner.latent_to_emb_coarse(x_coarse)
            x_coarse, mask_coarse = inner.padding(x_coarse, coarse_mask)
            x = torch.cat([x, x_coarse], dim=1)
            mask = torch.cat([mask, mask_coarse], dim=1)
        t_emb = inner.t_embedder(t * inner.args.time_multiplier)[:, None]
        if inner.args.prepend_ipa:
            x = x + inner.run_ipa(t_emb[:, 0], mask[:, 0], start_frames, end_frames, aatype)[:, None]
        for layer in inner.layers:
            x = grad_checkpoint(layer, (x, t_emb, mask, start_frames), inner.args.grad_checkpointing)
        return x
