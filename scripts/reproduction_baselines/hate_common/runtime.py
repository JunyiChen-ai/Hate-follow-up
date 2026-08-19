"""Losses, chunk bookkeeping and CLI pieces shared by the two ports.

CLAS2 and CLASM are byte-identical in VadCLIP @ c41067f src/xd_train.py and
DSANet @ eb335b2 src/xd_train.py, so they live here once rather than twice.
DSANet's own extra losses (CLASM_EVENT, CLASM_BKG, the reconstruction
consistency term) stay in dsanet/train.py, where they belong.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from .data import CORPORA, NUM_CLASSES


# ------------------------------------------------------------------ losses
def CLAS2(logits, labels, lengths, device):
    """Binary MIL loss on the classification branch. Verbatim upstream.

    The bag score is the mean of the top `length / 16 + 1` snippet sigmoids.
    That is a *fraction* of the sequence (the top ~6 %), not an absolute count
    tied to the 16-frame snippet, so it carries over to the 1 fps grid
    unchanged: on a 96 s video it reads the 7 most anomalous seconds, exactly
    as on a 96-snippet XD video it read the 7 most anomalous snippets.
    """
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]],
                            k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    return F.binary_cross_entropy(instance_logits, labels)


def CLASM(logits, labels, lengths, device):
    """Multi-class MIL loss on the alignment branch. Verbatim upstream."""
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]],
                            k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat(
            [instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    return -torch.mean(
        torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


def text_orthogonality_loss(text_features, device):
    """Push each anomalous class prompt off the normal one.

    PORT PATCH (patch T1): upstream divides the accumulated term by the literal
    constant 6, the number of anomalous classes in XD-Violence. This port
    divides by `num_class - 1`, which equals 6 on XD and 1 here. The formula is
    otherwise verbatim; with two classes the sum has a single term, the cosine
    between the "normal content" and "hateful content" embeddings.
    """
    loss = torch.zeros(1).to(device)
    normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    for j in range(1, text_features.shape[0]):
        abnormal = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
        loss = loss + torch.abs(normal @ abnormal)
    return loss / max(text_features.shape[0] - 1, 1)


# -------------------------------------------------- test-time chunk lengths
def chunk_lengths(total_length, maxlen):
    """Per-chunk valid lengths for a `process_split` item. Verbatim upstream.

    Lifted unchanged from xd_test.py / ucf_test.py so that a video longer than
    `maxlen` is scored exactly as upstream scores it: consecutive blocks, the
    tail zero-padded, the padded rows discarded afterwards.
    """
    length = int(total_length)
    n = int(length / maxlen) + 1
    lengths = torch.zeros(n)
    for j in range(n):
        if j == 0 and length < maxlen:
            lengths[j] = length
        elif j == 0 and length > maxlen:
            lengths[j] = maxlen
            length -= maxlen
        elif length > maxlen:
            lengths[j] = maxlen
            length -= maxlen
        else:
            lengths[j] = length
    return lengths.to(int)


def prepare_test_item(visual, total_length, maxlen):
    """Reshape one test item to (n_chunks, maxlen, D) and give its lengths."""
    visual = visual.squeeze(0)
    if int(total_length) < maxlen:
        visual = visual.unsqueeze(0)
    return visual, chunk_lengths(total_length, maxlen)


# ---------------------------------------------------------------- val score
def video_scores_from_logits(logits1, lengths):
    """Per-video MIL bag score, the same aggregate CLAS2 optimises."""
    probs = torch.sigmoid(logits1).reshape(logits1.shape[0], logits1.shape[1])
    out = []
    for i in range(probs.shape[0]):
        n = int(lengths[i])
        tmp, _ = torch.topk(probs[i, 0:n], k=int(n / 16 + 1), largest=True)
        out.append(float(tmp.mean()))
    return out


# --------------------------------------------------------------- utilities
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(requested):
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ABORT: --device cuda requested but no GPU is visible")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def add_common_args(parser):
    parser.add_argument("--corpus", default="hatemm", choices=list(CORPORA))
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", default=234, type=int,
                        help="upstream default")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--limit-videos", default=0, type=int,
                        help="debug only: keep at most N videos per split "
                             "(0 = all). Used by the CPU dry run; never set "
                             "it for a real run")
    return parser


VISUAL_LENGTH = {"hatemm": 256, "hateclipseg": 256}
ATTN_WINDOW = {"hatemm": 64, "hateclipseg": 64}


def default_visual_length(corpus):
    """Per-corpus temporal window, in 1 fps rows (i.e. seconds).

    Upstream uses visual_length 256 with attn_window 64 on both XD-Violence and
    UCF-Crime. Read in seconds those windows are 171 s and 137 s respectively.

    hatemm    median 108 s, p90 254 s, max 5809 s. 256 rows leaves 90 % of the
              corpus untruncated and coincides with the published number, so
              the default is kept at 256 / 64.
    mhclip_*  max 61 s in both languages. 256 rows would make four rows in five
              zero padding, which the temporal transformer attends to (upstream
              passes padding_mask=None during training). 64 / 16 covers every
              video whole and keeps the published 4:1 window ratio.
    hateclipseg
              min 180 s, median 239 s, p90 286 s, max 350 s. Every video is
              minutes long, so the mhclip argument for shrinking the window
              (mostly-padding batches) does not apply and the HateMM setting
              of 256 / 64 is kept, which is also the published number.
    """
    return VISUAL_LENGTH.get(corpus, 64)


def default_attn_window(corpus):
    return ATTN_WINDOW.get(corpus, 16)


def scores_out_path(method, corpus, root=None):
    if root is None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", "..",
                                            "results", "reproduction",
                                            "baselines"))
    return os.path.join(root, method, corpus, "scores.jsonl")
