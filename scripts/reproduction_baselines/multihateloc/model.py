"""MultiHateLoc (WWW'26, arXiv 2512.10408), reimplemented from the paper.

There is no reference implementation: the repository the paper announces,
github.com/mmilabuk/multihateloc, holds a LICENSE file and nothing else. What
the paper fixes is reproduced here exactly; what it leaves open is filled with
the simplest reading that makes the described object run, and every such
choice is marked `INFERRED` at the point it is made and listed in DESIGN.md.

Fixed by the paper and implemented as stated
    per-frame sigmoid probabilities, one stream per modality plus a fused
    stream; top-K multiple-instance learning where K is a proportion of the
    frames (their Table 4 best, K = 3, i.e. the top third); a video-level
    binary cross-entropy on the top-K mean; a temporal smoothness
    regulariser at lambda = 0.1; a cross-modal contrastive term at
    lambda = 0.2; a Dynamic Modality Selection block producing modality
    importance weights; the final frame set as the union of the fused
    branch's top-K frames with the modality branches' top-K frames, gated by
    those weights.

Left open by the paper and inferred here
    branch widths and depth; where the modality weights enter the network;
    the exact contrastive pairing; input normalisation; whether the losses
    cover every branch or only the fused one; the exact form of the weight
    gate in the union rule. See DESIGN.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-7


def masked_mean(x, mask):
    """Mean over the valid frames of each item. x (B, T, D), mask (B, T)."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


def topk_counts(lengths, k_proportion):
    """Frames per video that the MIL pool reads.

    The paper reports K as a proportion, its best setting being K = 3, "the
    top 33% of frames". So the pool size is ceil(T / K), never smaller than
    one frame -- a 2-second video still has a most-hateful second.
    """
    return torch.clamp(torch.ceil(lengths.float() / float(k_proportion)),
                       min=1.0).long()


def topk_mean(prob, mask, counts):
    """Mean of each video's `counts` highest valid frame probabilities."""
    filled = prob.masked_fill(~mask, -1.0)
    order = torch.argsort(filled, dim=1, descending=True)
    ranks = torch.arange(prob.shape[1], device=prob.device)[None, :]
    keep = ranks < counts[:, None]
    picked = torch.gather(prob, 1, order)
    return (picked * keep).sum(1) / counts.clamp(min=1).to(prob.dtype)


def topk_mask(prob, mask, counts):
    """Boolean (B, T): the `counts` highest valid frames of each video."""
    filled = prob.masked_fill(~mask, -1.0)
    order = torch.argsort(filled, dim=1, descending=True)
    ranks = torch.arange(prob.shape[1], device=prob.device)[None, :]
    keep = ranks < counts[:, None]
    out = torch.zeros_like(mask)
    return out.scatter(1, order, keep) & mask


class ModalityBranch(nn.Module):
    """One modality -> a frame embedding and a frame logit.

    INFERRED (widths). The paper names modality-specific branches and a fused
    branch but gives no layer sizes. Two linear layers, 256 then 128, with
    ReLU and dropout, is the smallest stack that is still a branch rather than
    a bare linear probe, and it keeps the model in the 0.3-1 M parameter range
    every weakly-supervised localiser in this table sits in.

    INFERRED (input LayerNorm). The paper states no input normalisation. The
    three feature families arrive on very different scales -- on HateMM the
    mean row norm is 25.4 for ViT-B/16 and 2.9 for VGGish -- so an unnormalised
    concatenation would hand the fused branch a nine-fold scale advantage for
    vision that has nothing to do with what the modalities carry. A LayerNorm
    on each branch input removes the scale difference and adds no capacity.

    INFERRED (no temporal mixing). The branch is applied frame by frame. The
    paper describes per-frame prediction and does not name a temporal encoder
    (no transformer, no temporal convolution); its only temporal coupling is
    the smoothness regulariser. Adding an attention block would be adding a
    component the paper does not have.
    """

    def __init__(self, in_dim, hidden=256, embed=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed), nn.ReLU(inplace=True))
        self.head = nn.Linear(embed, 1)

    def forward(self, x):
        e = self.proj(self.norm(x))
        return e, self.head(e).squeeze(-1)


class DynamicModalitySelection(nn.Module):
    """Per-video importance weights over the modalities.

    INFERRED (form). The paper names the block and uses its output to weight
    the modality branches in the final frame selection, but gives no equation.
    The simplest object that does what is described: pool each modality's
    frame embeddings over the valid frames, score each pooled vector with one
    shared two-layer head, and softmax the three scores. Shared rather than
    per-modality parameters, so the block compares modalities on one scale
    instead of learning three independent biases.

    INFERRED (where the weights enter training). Weights used only at
    inference would receive no gradient and stay at their initialisation. The
    weights therefore also scale each modality's contribution to the fused
    branch, which is the reading that makes "dynamic modality selection"
    select something. They are multiplied by the modality count so that a
    uniform weighting reproduces the plain concatenation exactly.
    """

    def __init__(self, embed=128, n_modalities=3):
        super().__init__()
        self.n_modalities = n_modalities
        self.score = nn.Sequential(
            nn.Linear(embed, embed // 2), nn.Tanh(),
            nn.Linear(embed // 2, 1))

    def forward(self, embeds, mask):
        pooled = [masked_mean(e, mask) for e in embeds]
        s = torch.cat([self.score(p) for p in pooled], dim=1)
        return F.softmax(s, dim=1)


class MultiHateLoc(nn.Module):
    def __init__(self, dims, hidden=256, embed=128, dropout=0.1,
                 k_proportion=3, temperature=0.07):
        super().__init__()
        self.modalities = tuple(dims.keys())
        self.k_proportion = int(k_proportion)
        self.temperature = float(temperature)
        self.branches = nn.ModuleDict(
            {m: ModalityBranch(d, hidden, embed, dropout)
             for m, d in dims.items()})
        self.dms = DynamicModalitySelection(embed, len(dims))
        self.fuse = nn.Sequential(
            nn.Linear(embed * len(dims), hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed), nn.ReLU(inplace=True))
        self.fuse_head = nn.Linear(embed, 1)

    def forward(self, feats, mask):
        embeds, probs = [], {}
        for m in self.modalities:
            e, logit = self.branches[m](feats[m])
            embeds.append(e)
            probs[m] = torch.sigmoid(logit) * mask
        w = self.dms(embeds, mask)                       # (B, M)
        scaled = [e * (w[:, i] * len(self.modalities))[:, None, None]
                  for i, e in enumerate(embeds)]
        ef = self.fuse(torch.cat(scaled, dim=-1))
        probs["fused"] = torch.sigmoid(self.fuse_head(ef).squeeze(-1)) * mask
        return {"probs": probs, "weights": w, "embeds": embeds,
                "fused_embed": ef}

    # ------------------------------------------------------------- losses
    def mil_loss(self, probs, mask, lengths, labels):
        """Video-level BCE on each branch's top-K mean.

        INFERRED (which branches). The paper states one MIL objective and does
        not say whether the modality branches are supervised. They must be:
        without a video-level loss of their own their frame probabilities are
        unconstrained, and the paper reads a top-K set off each of them. Every
        branch therefore gets the same BCE and the four terms are summed.
        """
        counts = topk_counts(lengths, self.k_proportion)
        per_branch = {}
        total = 0.0
        for name, p in probs.items():
            v = topk_mean(p, mask, counts).clamp(EPS, 1.0 - EPS)
            loss = F.binary_cross_entropy(v, labels)
            per_branch[name] = loss
            total = total + loss
        return total, per_branch

    def smoothness_loss(self, probs, mask):
        """Mean squared first difference of the frame probabilities.

        Sultani-style temporal smoothness: hateful content occupies stretches
        of seconds, not isolated frames. Averaged over branches so that the
        published lambda = 0.1 keeps its meaning when four streams contribute.
        """
        pair = mask[:, 1:] & mask[:, :-1]
        denom = pair.sum().clamp(min=1).to(next(self.parameters()).dtype)
        total = 0.0
        for p in probs.values():
            d = (p[:, 1:] - p[:, :-1]) ** 2
            total = total + (d * pair).sum() / denom
        return total / len(probs)

    def contrastive_loss(self, embeds, mask):
        """Cross-modal InfoNCE over the batch.

        INFERRED (pairing). The paper names a cross-modal contrastive loss and
        does not state what is contrasted against what. The reading taken here
        is the standard one: for each unordered pair of modalities, the two
        video-level pooled embeddings of the *same* video are the positive
        pair and every other video in the batch supplies the negatives,
        symmetric in both directions, temperature 0.07, averaged over the
        three pairs. It is label-agnostic, as a cross-modal alignment term
        should be.
        """
        pooled = [F.normalize(masked_mean(e, mask), dim=-1) for e in embeds]
        b = pooled[0].shape[0]
        if b < 2:
            return pooled[0].sum() * 0.0
        target = torch.arange(b, device=pooled[0].device)
        total, n = 0.0, 0
        for i in range(len(pooled)):
            for j in range(i + 1, len(pooled)):
                sim = pooled[i] @ pooled[j].t() / self.temperature
                total = total + 0.5 * (F.cross_entropy(sim, target)
                                       + F.cross_entropy(sim.t(), target))
                n += 1
        return total / n

    # ---------------------------------------------------------- inference
    def video_scores(self, probs, mask, lengths):
        """Video-level probability per branch, the MIL pooling read directly."""
        counts = topk_counts(lengths, self.k_proportion)
        return {name: topk_mean(p, mask, counts) for name, p in probs.items()}

    def union_frames(self, probs, weights, mask, lengths):
        """The paper's final frame set, as a 0/1 array.

        "The union of the fused branch's top-K frames with the
        modality-specific top-K frames weighted by the importance weights."
        A positive scalar cannot change a top-K set within one modality, so
        the weights can only decide *which* modality sets join the union.

        INFERRED (gate). A modality contributes its top-K set when its
        importance weight is at least uniform, 1/M. That is the reading with
        no free threshold in it. The array this returns is binary, so its
        ranking metrics collapse to a single operating point; it is reported
        beside the continuous branches, not instead of them.
        """
        counts = topk_counts(lengths, self.k_proportion)
        out = topk_mask(probs["fused"], mask, counts)
        gate = 1.0 / weights.shape[1]
        for i, m in enumerate(self.modalities):
            sel = topk_mask(probs[m], mask, counts)
            sel = sel & (weights[:, i] >= gate)[:, None]
            out = out | sel
        return out.to(torch.float32) * mask

    def dms_frames(self, probs, weights):
        """Importance-weighted combination of the modality frame scores.

        The union rule above returns a set. This study's evaluator scores
        ranked frame arrays, so the same importance weights are also read as a
        convex combination over the modality probabilities, which keeps the
        ranking the union rule discards. Reported as `score_dms`, and labelled
        as our continuous reading rather than as the paper's output.
        """
        stack = torch.stack([probs[m] for m in self.modalities], dim=1)
        return (stack * weights[:, :, None]).sum(1)
