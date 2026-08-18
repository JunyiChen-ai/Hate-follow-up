#!/usr/bin/env python
"""Dynamic spatiotemporal graph, and the reconstructed graph propagation.

`build_dynamic_graph` ports upstream `uniseg_processor.UniSegProcessor.
build_dynamic_graph` with the paper's Eq. (4) as the authority where the two
disagree.

`graph_propagation` is **not** a port. Upstream imports it --
`from graph_operations import graph_propagation` in `uniseg_processor.py:7` --
but `graph_operations.py` is a byte-identical duplicate of
`video_processing.py` and defines only `process_video`. The function exists
nowhere in the release, so `import main` raises ImportError and the released
event-segmentation pipeline cannot run at all. It is reconstructed here from
Eq. (5) through Eq. (8) of the paper; every inferred choice is listed in
DESIGN_EVENTVAD.md under gap G1.

Data structure
    Upstream builds a `networkx.Graph`. Its edge count is not the kNN fan-out
    it looks like: the top-k is taken *per block pair*, so a node collects
    `init_k` edges against every one of the `ceil(n / 200)` column blocks, or
    `5 * 150 = 750` edges on a 30000-frame video -- 22.5M edges, tens of GB in
    networkx. The same edges as a scipy CSR matrix are about 270 MB, and the
    propagation step becomes one sparse-dense product instead of a Python loop
    over adjacency lists. The edge set and the weights are unchanged; only the
    container is. Upstream's `combined_sim` is symmetric in (i, j) -- each of
    its three terms is -- so networkx's last-write-wins on a repeated
    unordered pair and this module's de-duplication agree by construction.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------- features
def fuse_node_features(clip_feats, flow_feats, cfg):
    """Paper Eq. (1) and Eq. (3): L2-normalise CLIP, then weight and concat.

    Upstream's `extract_features` concatenates `alpha * clip_raw` with
    `(1 - alpha) * flow` and normalises the CLIP half only later, inside the
    similarity computation, so its *node* features carry raw CLIP magnitudes.
    Eq. (1) normalises before anything else and Eq. (3) fuses the normalised
    vector, which matters because the node feature is what Eq. (9)'s
    `||f_{i+1} - f_i||^2` term measures. `cfg.clip_norm_in_nodes` selects.
    """
    clip = np.asarray(clip_feats, dtype=np.float32)
    flow = np.asarray(flow_feats, dtype=np.float32)
    if clip.shape[0] != flow.shape[0]:
        raise ValueError("clip has %d rows, flow has %d"
                         % (clip.shape[0], flow.shape[0]))
    if cfg.clip_norm_in_nodes:
        clip = clip / (np.linalg.norm(clip, axis=1, keepdims=True) + 1e-6)
    fused = np.concatenate(
        [cfg.alpha * clip, (1.0 - cfg.alpha) * flow], axis=1)
    return fused.astype(np.float32)


# ---------------------------------------------------------- dynamic graph
def build_dynamic_graph(clip_feats, flow_feats, fps, cfg):
    """Paper Eq. (4). Returns a symmetric CSR adjacency of edge weights.

        E_ij = [ alpha * cos(clip_i, clip_j)
                 + (1 - alpha) * exp(-||flow_i - flow_j||) ] / (1 + gamma|i-j|)

    with the kNN sparsification upstream applies: within each 200x200 block
    of the similarity matrix, every row keeps its `dynamic_k` largest strictly
    positive entries.
    """
    clip = np.asarray(clip_feats, dtype=np.float32)
    flow = np.asarray(flow_feats, dtype=np.float32)
    n = clip.shape[0]
    if n < 2:
        return sp.csr_matrix((n, n), dtype=np.float32)

    # Eq. (1). Upstream normalises here too, with the same 1e-6 floor.
    clip = clip / (np.linalg.norm(clip, axis=1, keepdims=True) + 1e-6)

    gamma = cfg.gamma_per_frame(fps)
    block = cfg.graph_block_size
    rows, cols, vals = [], [], []

    # Upstream: dynamic_k = max(3, init_k - i // (n // 10)), with i the first
    # frame index of the row block, so the fan-out shrinks from init_k to 3 as
    # the sweep advances. The paper describes no such positional schedule; it
    # is reproduced because it is what the released code does. The guard is
    # ours: n < 10 makes n // 10 zero and upstream divides by it.
    decade = max(1, n // 10)

    for i in range(0, n, block):
        i_end = min(i + block, n)
        clip_i = clip[i:i_end]
        flow_i = flow[i:i_end]
        dynamic_k = max(3, cfg.init_k - (i // decade))
        for j in range(0, n, block):
            j_end = min(j + block, n)
            sim = clip_i @ clip[j:j_end].T
            # ||flow_i - flow_j|| over the full 128-d projection. The
            # projection has orthonormal rows, so this equals the distance
            # between the 2-d mean-flow vectors; smoke_cpu_eventvad.py
            # asserts that identity rather than assuming it here.
            dist = np.sqrt(np.sum(
                (flow_i[:, None, :] - flow[j:j_end][None, :, :]) ** 2, axis=2))
            tdiff = np.abs(np.arange(i, i_end)[:, None] - np.arange(j, j_end))
            combined = (cfg.alpha * sim
                        + (1.0 - cfg.alpha) * np.exp(-dist)) / (1.0 + gamma * tdiff)

            width = j_end - j
            valid_k = min(dynamic_k, width)
            if valid_k <= 0:
                continue
            kth = min(valid_k - 1, width - 1)
            for local_i in range(i_end - i):
                row = combined[local_i]
                top = np.argpartition(-row, kth)[:valid_k]
                gi = i + local_i
                for local_j in top:
                    gj = j + int(local_j)
                    w = row[local_j]
                    if gi != gj and w > 0:
                        rows.append(gi)
                        cols.append(gj)
                        vals.append(w)

    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float32)

    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    vals = np.asarray(vals, dtype=np.float32)
    # Undirected, as networkx.Graph is: mirror, then collapse repeats. The
    # similarity is symmetric so a repeat carries the same value, and `max`
    # picks that value rather than summing it the way COO->CSR would.
    both_r = np.concatenate([rows, cols])
    both_c = np.concatenate([cols, rows])
    both_v = np.concatenate([vals, vals])
    return _dedupe_max(both_r, both_c, both_v, n)


def _dedupe_max(rows, cols, vals, n):
    """CSR from COO triplets, keeping the maximum of any repeated (i, j)."""
    order = np.lexsort((cols, rows))
    r, c, v = rows[order], cols[order], vals[order]
    keep = np.empty(len(r), dtype=bool)
    keep[0] = True
    keep[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
    starts = np.flatnonzero(keep)
    best = np.maximum.reduceat(v, starts)
    adj = sp.csr_matrix((best, (r[starts], c[starts])), shape=(n, n),
                        dtype=np.float32)
    adj.sort_indices()
    return adj


# ----------------------------------------------------- graph propagation
def orthogonal_projections(d, k, seed):
    """Paper Eq. (5): fixed Q, K in R^{d x k} and V in R^{d x d}, orthonormal.

    "QR(.) denotes orthonormal columns via QR decomposition, d = 640 is the
    fused feature dimension, and k = 64 is the projected dimension. These
    fixed orthogonal matrices maximize feature retention."

    The paper fixes no seed. Upstream seeds its own fixed random projection --
    the flow projection in `feature_extractor._init_random_ortho` -- with 42,
    so 42 is the default here and `cfg.ortho_seed` exposes it.
    """
    rng = np.random.RandomState(seed)
    q_mat, _ = np.linalg.qr(rng.randn(d, k))
    k_mat, _ = np.linalg.qr(rng.randn(d, k))
    v_mat, _ = np.linalg.qr(rng.randn(d, d))
    return (q_mat.astype(np.float32), k_mat.astype(np.float32),
            v_mat.astype(np.float32))


def graph_propagation(features, adj, cfg):
    """Reconstruction of the missing `graph_propagation`, Eq. (5) - (8).

        score_ij  = (f_i Q)(f_j K)^T / sqrt(d_a)          over j in N_i
        a_ij      = softmax_j(score_ij)
        f_i       <- f_i + sum_j a_ij * E_(i,j) * (f_j V)
        f_i       <- f_i - mean_k f_k

    Returns the propagated (n, d) feature matrix. `features` is left intact.
    """
    feats = np.asarray(features, dtype=np.float32)
    n, d = feats.shape
    if n == 0:
        return feats.copy()
    q_mat, k_mat, v_mat = orthogonal_projections(d, cfg.ortho_dim,
                                                 cfg.ortho_seed)
    scale = 1.0 / math.sqrt(cfg.ortho_dim)

    adj = adj.tocsr()
    adj.sort_indices()
    indptr, indices, weights = adj.indptr, adj.indices, adj.data

    out = feats.copy()
    for _ in range(cfg.gat_iters):
        proj_q = out @ q_mat                      # (n, k)
        proj_k = out @ k_mat                      # (n, k)
        proj_v = out @ v_mat                      # (n, d)

        if len(indices) == 0:
            attn = np.zeros(0, dtype=np.float32)
        else:
            src = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
            logits = np.einsum("ec,ec->e", proj_q[src], proj_k[indices],
                               dtype=np.float32) * scale
            attn = _segment_softmax(logits, indptr, n)

        if cfg.gat_edge_term == "weight":
            coeff = attn * weights
        elif cfg.gat_edge_term == "indicator":
            coeff = attn
        else:
            raise ValueError("unknown gat_edge_term %r" % (cfg.gat_edge_term,))

        msg_op = sp.csr_matrix((coeff, indices, indptr), shape=(n, n))
        out = out + msg_op.dot(proj_v).astype(np.float32)
        out -= out.mean(axis=0, keepdims=True)          # Eq. (8)
    return out.astype(np.float32)


def _segment_softmax(logits, indptr, n):
    """Softmax within each CSR row; an empty row contributes nothing."""
    counts = np.diff(indptr)
    nonempty = np.flatnonzero(counts > 0)
    out = np.zeros_like(logits)
    if nonempty.size == 0:
        return out
    starts = indptr[nonempty]
    row_max = np.maximum.reduceat(logits, starts)
    src = np.repeat(np.arange(n, dtype=np.int64), counts)
    # Map every edge to the position of its row inside `nonempty`.
    slot = np.zeros(n, dtype=np.int64)
    slot[nonempty] = np.arange(nonempty.size)
    shifted = np.exp(logits - row_max[slot[src]])
    row_sum = np.add.reduceat(shifted, starts)
    return (shifted / row_sum[slot[src]]).astype(np.float32)
