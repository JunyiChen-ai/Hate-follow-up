"""MACIL-SD networks, from MACIL_SD @ c20943f avce_network.py.

Architecture untouched: a linear projection per modality into `hid_dim`, one
shared cross-attention transformer layer applied in both directions, and an
attention-MIL head whose per-frame logit is the sum of the audio and visual
logits. `Single_Model` is upstream's uni-modal network, the self-distillation
partner that `main.py` trains alongside the audio-visual model.

Patches, all mechanical:

  M3  `.cuda()` -> the device of the incoming tensor. Upstream hard-codes cuda
      inside `clas`, so the model cannot forward on cpu.
  M4  `squeeze()` -> `squeeze(-1)` on the per-frame logits. Upstream's bare
      `squeeze()` also removes the batch axis when the batch happens to hold a
      single item, which makes `logits[i]` a scalar and raises inside `topk`.
      For every batch of two or more the two spellings are identical.
  M5  the two input widths come from the caller instead of the literals 1024
      and 128. The values this port passes are 1024 and 128, so nothing moves;
      the argument exists so `Single_Model` can be built over VGGish for the
      audio-only ablation.
"""

import copy

import torch
import torch.nn as nn

from .Transformer import (CrossAttentionBlock, MultiHeadAttention,
                          PositionwiseFeedForward, SelfAttentionBlock,
                          TransformerLayer)


class AVCE_Model(nn.Module):
    def __init__(self, args):
        super(AVCE_Model, self).__init__()
        c = copy.deepcopy
        dropout = args.dropout
        nhead = args.nhead
        hid_dim = args.hid_dim
        ffn_dim = args.ffn_dim
        self.multiheadattn = MultiHeadAttention(nhead, hid_dim)
        self.feedforward = PositionwiseFeedForward(hid_dim, ffn_dim)
        # PORT PATCH (patch M5): widths from args rather than the literals
        # 1024 / 128. args.v_feature_size is 1024 (I3D) and args.a_feature_size
        # is 128 (VGGish), i.e. exactly the upstream literals.
        self.fc_v = nn.Linear(args.v_feature_size, hid_dim)
        self.fc_a = nn.Linear(args.a_feature_size, hid_dim)
        self.cma = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), c(self.feedforward), dropout))
        self.att_mmil = Att_MMIL(hid_dim, args.num_classes)

    def forward(self, f_a, f_v, seq_len):
        f_v, f_a = self.fc_v(f_v), self.fc_a(f_a)
        v_out, a_out = self.cma(f_v, f_a)
        mmil_logits, audio_logits, visual_logits, av_logits = self.att_mmil(a_out, v_out, seq_len)
        # NOTE (not a patch): upstream returns (..., v_out, a_out) and its
        # caller unpacks that pair as (audio_rep, visual_rep), so the two
        # representations arrive at the contrastive loss under each other's
        # names while the logits that select the top-k positions do not. The
        # return order is left exactly as published, because the reported
        # 83.40 AP was obtained with it; macilsd/train.py exposes
        # --fix-rep-swap for anyone who wants to see the corrected pairing, and
        # PATCHES.md patch M13 records the finding.
        return mmil_logits, audio_logits, visual_logits, av_logits, v_out, a_out


class Att_MMIL(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Att_MMIL, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def clas(self, logits, seq_len):
        # PORT PATCH (patch M4): squeeze(-1), not squeeze().
        logits = logits.squeeze(-1)
        # PORT PATCH (patch M3): device from the input, not .cuda().
        instance_logits = torch.zeros(0, device=logits.device)
        for i in range(logits.shape[0]):
            if seq_len is None:
                tmp = torch.mean(logits[i]).view(1)
            else:
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
                tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    def forward(self, a_out, v_out, seq_len):
        # prediction
        x = torch.cat([a_out.unsqueeze(-2), v_out.unsqueeze(-2)], dim=-2)
        frame_prob = self.fc(x)
        av_logits = frame_prob.sum(dim=2)
        a_logits = torch.sigmoid(frame_prob[:, :, 0, :])
        v_logits = torch.sigmoid(frame_prob[:, :, 1, :])
        mmil_logits = self.clas(av_logits, seq_len)
        return mmil_logits, a_logits, v_logits, av_logits


class Single_Model(nn.Module):
    """Upstream's uni-modal network.

    In `main.py` this is the visual-only self-distillation partner, trained at
    lr/5 and mixed into the audio-visual model by the per-epoch EMA. This port
    also instantiates it standalone over VGGish for the audio-only ablation and
    over I3D for the matched visual-only row; see macilsd/train.py.
    """

    def __init__(self, args, n_dim=None):
        super(Single_Model, self).__init__()
        c = copy.deepcopy
        dropout = args.dropout
        nhead = args.nhead
        hid_dim = args.hid_dim
        ffn_dim = args.ffn_dim
        # PORT PATCH (patch M5): n_dim overridable. Upstream always reads
        # args.v_feature_size, which is right for the visual partner and wrong
        # for a VGGish input.
        n_dim = args.v_feature_size if n_dim is None else n_dim
        self.multiheadattn = MultiHeadAttention(nhead, hid_dim)
        self.feedforward = PositionwiseFeedForward(hid_dim, ffn_dim)
        self.fc_v = nn.Linear(n_dim, hid_dim)
        self.cma = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), c(self.feedforward), dropout))
        self.fc = nn.Linear(hid_dim, args.num_classes)

    def clas(self, logits, seq_len):
        # PORT PATCH (patch M4): squeeze(-1), not squeeze().
        logits = logits.squeeze(-1)
        # PORT PATCH (patch M3): device from the input, not .cuda().
        instance_logits = torch.zeros(0, device=logits.device)
        for i in range(logits.shape[0]):
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    def forward(self, f, seq_len):
        f = self.fc_v(f)
        sa = self.cma(f)
        out = self.fc(sa)
        if seq_len is not None:
            out = self.clas(out, seq_len)
        return out
