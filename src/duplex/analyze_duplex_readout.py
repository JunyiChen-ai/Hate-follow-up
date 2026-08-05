"""
Duplex readout probe: pre-registered analysis.

Implements `docs/duplex/PREREG_duplex_readout.md` against the extraction output
of `src/duplex/extract_duplex_readout.py`. Nothing here is tuned: the anchor
rule, the direction estimator, the layer-selection procedure and every
threshold are fixed by the pre-registration, and gold labels enter only as
diagnostics and as ground truth for the prediction table.

Usage:
  python src/duplex/analyze_duplex_readout.py --dataset ImpliHateVid
"""

import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "our_method"))

from data_utils import DATASET_ROOTS, load_annotations  # noqa: E402

# Pre-registered constants. None of these is a free parameter.
Z_DISMISSED = -2.944          # sigmoid(z) < 0.05
Z_ASSERTED = +2.944           # sigmoid(z) > 0.95
ANCHOR_K_PRIMARY = 0.05
ANCHOR_K_ROBUST = (0.02, 0.10)
PURITY_ABORT = 0.85
N_FOLDS = 5
N_PERM = 200
N_SPLIT_HALF = 100
P1_AUC_FLOOR = 0.65
P1_MARGIN = 0.10
P1_KILL_MARGIN = 0.03
P2_MARGIN = 0.03
P3_FLOOR = 0.90
P5A_COSINE_FLOOR = 0.8
P6_MARGIN = 0.10
DEGEN_MIN_CHARS = 40
DEGEN_MIN_LATIN_FRAC = 0.5
TRANSCRIPT_LIMIT = 300        # the extraction prompt's truncation
# "<=720p": native frame pixel count at most 1280x720. The post-mortem census
# reports a clean tier at 1280x720 among retained videos and >=2,073,600 pixels
# among the lost ones; the conservative reading of the pre-registration's
# "<=720p stratum" is the pixel cap itself, which also excludes the 808x1440
# tier (1,163,520 px) that sits above 720p but below 1080p.
P720_MAX_PIXELS = 1280 * 720
SEED = 0


# --------------------------------------------------------------------------
# statistics


def auc(scores_pos, scores_neg):
    """Mann-Whitney AUC with mid-ranks for ties. Returns nan if a side is empty."""
    scores_pos = np.asarray(scores_pos, dtype=np.float64)
    scores_neg = np.asarray(scores_neg, dtype=np.float64)
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    allv = np.concatenate([scores_pos, scores_neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[:n_pos].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def direction_from(H, idx_hi, idx_lo):
    """Unit-normalized difference of anchor means at one layer, fp32 math."""
    return unit(H[idx_hi].mean(axis=0) - H[idx_lo].mean(axis=0))


def stratified_folds(n_hi, n_lo, n_folds, rng):
    """Fold assignment balanced across the two anchor sets."""
    hi = rng.permutation(n_hi)
    lo = rng.permutation(n_lo)
    folds = []
    for f in range(n_folds):
        folds.append((set(hi[f::n_folds].tolist()), set(lo[f::n_folds].tolist())))
    return folds


# --------------------------------------------------------------------------
# loading


def load_scores(scores_path):
    """Last row per video_id with a finite z."""
    rows = {}
    with open(scores_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vid, z = r.get("video_id"), r.get("z")
            if vid and isinstance(z, (int, float)) and np.isfinite(z):
                rows[vid] = r
    return rows


def gold_group(vid):
    pre = vid.split("_")[0]
    return pre if pre in ("EX", "IM", "NH") else "OTHER"


def native_pixels(dataset, vid):
    """Native pixel count of frame_000.jpg (the post-mortem's census method)."""
    from PIL import Image
    p = os.path.join(DATASET_ROOTS[dataset], "frames_16", vid, "frame_000.jpg")
    if not os.path.isfile(p):
        return None
    with Image.open(p) as im:
        w, h = im.size
    return int(w) * int(h)


def is_degenerate(transcript):
    t = (transcript or "")[:TRANSCRIPT_LIMIT]
    if len(t) < DEGEN_MIN_CHARS:
        return True
    if len(t) == 0:
        return True
    latin = sum(1 for c in t if ("a" <= c <= "z") or ("A" <= c <= "Z"))
    return (latin / len(t)) < DEGEN_MIN_LATIN_FRAC


# --------------------------------------------------------------------------
# core pipeline at one anchor fraction


def build_anchors(z, k):
    n = len(z)
    n_anchor = int(n * k)
    order = np.argsort(z, kind="mergesort")
    idx_lo = order[:n_anchor]
    idx_hi = order[::-1][:n_anchor]
    return idx_hi, idx_lo, n_anchor


def anchor_purity(idx_hi, idx_lo, groups):
    g = np.asarray(groups)
    top = float(np.mean(np.isin(g[idx_hi], ["EX", "IM"])))
    bot = float(np.mean(g[idx_lo] == "NH"))
    return top, bot


def layer_cv_accuracy(H, idx_hi, idx_lo, rng):
    """5-fold CV accuracy per layer, sign-of-projection about the centroid
    midpoint, trained and tested only inside the anchor union."""
    n_layers = H.shape[1]
    folds = stratified_folds(len(idx_hi), len(idx_lo), N_FOLDS, rng)
    acc = np.zeros((N_FOLDS, n_layers), dtype=np.float64)
    for fi, (te_hi, te_lo) in enumerate(folds):
        tr_hi = idx_hi[[i for i in range(len(idx_hi)) if i not in te_hi]]
        tr_lo = idx_lo[[i for i in range(len(idx_lo)) if i not in te_lo]]
        ev_hi = idx_hi[sorted(te_hi)]
        ev_lo = idx_lo[sorted(te_lo)]
        for L in range(n_layers):
            HL = H[:, L, :]
            m_hi, m_lo = HL[tr_hi].mean(axis=0), HL[tr_lo].mean(axis=0)
            w = unit(m_hi - m_lo)
            thr = 0.5 * (m_hi @ w + m_lo @ w)
            correct = int((HL[ev_hi] @ w > thr).sum()) + int((HL[ev_lo] @ w <= thr).sum())
            acc[fi, L] = correct / (len(ev_hi) + len(ev_lo))
    return acc.mean(axis=0), acc


def probe_scores_all_layers(H, idx_hi, idx_lo):
    """s[:, L] = h_L . w_L for every layer."""
    n_v, n_layers, _ = H.shape
    S = np.zeros((n_v, n_layers), dtype=np.float64)
    W = np.zeros((n_layers, H.shape[2]), dtype=np.float32)
    for L in range(n_layers):
        w = direction_from(H[:, L, :], idx_hi, idx_lo)
        W[L] = w
        S[:, L] = H[:, L, :] @ w
    return S, W


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ImpliHateVid")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res_dir = args.results_dir or os.path.join(
        PROJECT_ROOT, "results", "duplex_readout", args.dataset)
    scores_path = os.path.join(res_dir, "scores.jsonl")
    hidden_dir = os.path.join(res_dir, "hidden")
    out_path = args.out or os.path.join(
        PROJECT_ROOT, "docs", "duplex", "reports",
        f"readout_probe_8b_train.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    report = {"prereg": "docs/duplex/PREREG_duplex_readout.md",
              "dataset": args.dataset, "split": "train",
              "model": "Qwen/Qwen3-VL-8B-Instruct"}

    rows = load_scores(scores_path)
    vids = sorted(rows.keys())
    z = np.array([rows[v]["z"] for v in vids], dtype=np.float64)
    groups = [gold_group(v) for v in vids]
    print(f"loaded {len(vids)} videos with finite z")

    H = np.zeros((len(vids), 37, 4096), dtype=np.float32)
    for i, v in enumerate(vids):
        a = np.load(os.path.join(hidden_dir, f"{v}.npy"))
        if a.shape != (37, 4096):
            raise SystemExit(f"{v}: shape {a.shape}")
        H[i] = a.astype(np.float32)
    if not np.isfinite(H).all():
        raise SystemExit("non-finite hidden states")
    n_layers = H.shape[1]
    report["sample"] = {
        "n_videos": len(vids),
        "n_layers": n_layers,
        "hidden_size": int(H.shape[2]),
        "group_counts": {g: int(sum(1 for x in groups if x == g))
                         for g in ("EX", "IM", "NH", "OTHER")},
        "all_hidden_finite": True,
    }

    g = np.array(groups)
    dismissed = z < Z_DISMISSED
    asserted = z > Z_ASSERTED
    interior = ~dismissed & ~asserted

    # ---- z distribution per gold group
    zdist = {}
    for grp in ("EX", "IM", "NH"):
        m = g == grp
        zdist[grp] = {
            "n": int(m.sum()),
            "median_z": float(np.median(z[m])),
            "mean_z": float(z[m].mean()),
            "frac_dismissed": float(dismissed[m].mean()),
            "frac_interior": float(interior[m].mean()),
            "frac_asserted": float(asserted[m].mean()),
        }
    report["z_distribution_by_group"] = zdist
    report["strata"] = {
        "z_dismissed_threshold": Z_DISMISSED,
        "z_asserted_threshold": Z_ASSERTED,
        "dismissed": {grp: int(((g == grp) & dismissed).sum())
                      for grp in ("EX", "IM", "NH")},
        "interior": {grp: int(((g == grp) & interior).sum())
                     for grp in ("EX", "IM", "NH")},
        "asserted": {grp: int(((g == grp) & asserted).sum())
                     for grp in ("EX", "IM", "NH")},
    }

    # ---- bf16 z-grid tie structure inside the dismissed stratum
    zd = z[dismissed]
    uniq, counts = np.unique(zd, return_counts=True)
    zd_im = z[dismissed & (g == "IM")]
    zd_nh = z[dismissed & (g == "NH")]
    tied_pairs = 0
    for u, c in zip(uniq, counts):
        n_i = int((zd_im == u).sum())
        n_n = int((zd_nh == u).sum())
        tied_pairs += n_i * n_n
    total_pairs = len(zd_im) * len(zd_nh)
    report["tie_diagnostics_dismissed"] = {
        "n_dismissed": int(dismissed.sum()),
        "n_distinct_z": int(len(uniq)),
        "frac_in_tied_groups": float((counts[counts > 1].sum()) / max(len(zd), 1)),
        "largest_tie_group": int(counts.max()) if len(counts) else 0,
        "top_tie_values": [[float(u), int(c)] for u, c in
                           sorted(zip(uniq, counts), key=lambda t: -t[1])[:10]],
        "im_vs_nh_pairs": int(total_pairs),
        "im_vs_nh_tied_pairs": int(tied_pairs),
        "frac_tied_pairs": float(tied_pairs / total_pairs) if total_pairs else float("nan"),
        "note": ("tied pairs contribute 0.5 each to the raw-z AUC; the fraction "
                 "above bounds how far the raw-z baseline is pulled toward 0.5 "
                 "by bf16 logit quantization."),
    }

    # Descriptive context for P1, not an additional arm. The pre-registration
    # framed the dismissed stratum as one where IM and NH "carry indistinguishable
    # near-zero verbalized scores", which held for the retired clipped P(Yes)
    # readout. Under the raw unclipped z the stratum spans a wide range, so z
    # keeps rank information inside it. These numbers are reported so the P1
    # outcome can be read against that premise.
    report["dismissed_stratum_z_spread"] = {
        "note": ("descriptive only; the raw-z readout leaves large dynamic range "
                 "inside the dismissed stratum, unlike the clipped readout the "
                 "pre-registration's framing assumed"),
        "z_min": float(zd.min()), "z_max": float(zd.max()),
        "z_iqr": [float(np.percentile(zd, 25)), float(np.percentile(zd, 75))],
        "by_group": {grp: {"n": int((dismissed & (g == grp)).sum()),
                           "median_z": float(np.median(z[dismissed & (g == grp)]))
                           if (dismissed & (g == grp)).sum() else None}
                     for grp in ("EX", "IM", "NH")},
    }

    rng = np.random.default_rng(SEED)

    # ---- anchors at k = 5%
    idx_hi, idx_lo, n_anchor = build_anchors(z, ANCHOR_K_PRIMARY)
    p_top, p_bot = anchor_purity(idx_hi, idx_lo, groups)
    report["anchors"] = {
        "k": ANCHOR_K_PRIMARY, "n_per_side": int(n_anchor),
        "purity_top_hateful": p_top, "purity_bottom_NH": p_bot,
        "abort_line": PURITY_ABORT,
        "top_composition": {grp: int((g[idx_hi] == grp).sum())
                            for grp in ("EX", "IM", "NH")},
        "bottom_composition": {grp: int((g[idx_lo] == grp).sum())
                               for grp in ("EX", "IM", "NH")},
        "z_range_top": [float(z[idx_hi].min()), float(z[idx_hi].max())],
        "z_range_bottom": [float(z[idx_lo].min()), float(z[idx_lo].max())],
    }
    if p_top < PURITY_ABORT or p_bot < PURITY_ABORT:
        report["diagnostic_abort"] = {
            "triggered": True,
            "reason": (f"anchor purity below {PURITY_ABORT}: top={p_top:.3f} "
                       f"bottom={p_bot:.3f}; run declared invalid for prediction "
                       f"testing per the pre-registration."),
        }
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report["diagnostic_abort"], indent=2))
        return
    report["diagnostic_abort"] = {"triggered": False}

    # ---- label-free layer selection
    cv_mean, cv_folds = layer_cv_accuracy(H, idx_hi, idx_lo, rng)
    primary = int(np.argmax(cv_mean))
    tied = [int(L) for L in np.flatnonzero(cv_mean >= cv_mean[primary] - 1e-12)]
    report["layer_selection"] = {
        "method": "5-fold CV inside the 128 anchors, sign of projection about "
                  "the centroid midpoint, direction refit per fold",
        "primary_layer": primary,
        "tied_layers_at_max": tied,
        "tie_break": ("the anchor sets are linearly separable from layer 19 up, so "
                      "held-out accuracy saturates at 1.0 across many layers. The "
                      "pre-registration says argmax and does not name a tie-break; "
                      "the lowest tied index is taken, which is the choice that "
                      "gives the probe the least post-hoc freedom. The full sweep "
                      "is reported, and the P1 verdict is unchanged at every tied "
                      "layer."),
        "primary_layer_cv_accuracy": float(cv_mean[primary]),
        "cv_accuracy_by_layer": [float(x) for x in cv_mean],
        "cv_accuracy_std_by_layer": [float(x) for x in cv_folds.std(axis=0)],
    }

    # ---- probe scores at every layer
    S, W = probe_scores_all_layers(H, idx_hi, idx_lo)

    def p1_stat(scores):
        return auc(scores[dismissed & (g == "IM")], scores[dismissed & (g == "NH")])

    z_p1 = p1_stat(z)
    probe_p1_by_layer = [p1_stat(S[:, L]) for L in range(n_layers)]
    probe_p1 = probe_p1_by_layer[primary]

    p1_pass = (probe_p1 >= P1_AUC_FLOOR) and (probe_p1 >= z_p1 + P1_MARGIN)
    p1_kill = probe_p1 <= z_p1 + P1_KILL_MARGIN
    report["P1"] = {
        "statement": "dismissed-IM vs dismissed-NH separated by the probe where z cannot",
        "n_IM": int((dismissed & (g == "IM")).sum()),
        "n_NH": int((dismissed & (g == "NH")).sum()),
        "probe_auc_primary_layer": probe_p1,
        "raw_z_auc": z_p1,
        "delta": probe_p1 - z_p1,
        "pass_floor_0.65": bool(probe_p1 >= P1_AUC_FLOOR),
        "pass_margin_+0.10": bool(probe_p1 >= z_p1 + P1_MARGIN),
        "pass": bool(p1_pass),
        "kill_condition_probe_le_z_plus_0.03": bool(p1_kill),
        "probe_auc_by_layer": probe_p1_by_layer,
        "best_layer_any": int(np.nanargmax(probe_p1_by_layer)),
        "best_auc_any_layer": float(np.nanmax(probe_p1_by_layer)),
        "n_layers_passing_floor_and_margin": int(sum(
            1 for a in probe_p1_by_layer
            if a >= P1_AUC_FLOOR and a >= z_p1 + P1_MARGIN)),
    }

    # ---- P2 gain asymmetry, full sample
    def full_auc(scores, pos, neg):
        return auc(scores[g == pos], scores[g == neg])

    z_im_nh = full_auc(z, "IM", "NH")
    z_ex_nh = full_auc(z, "EX", "NH")
    pr_im_nh = full_auc(S[:, primary], "IM", "NH")
    pr_ex_nh = full_auc(S[:, primary], "EX", "NH")
    d_im, d_ex = pr_im_nh - z_im_nh, pr_ex_nh - z_ex_nh
    report["P2"] = {
        "statement": "gain asymmetry: the probe buys more on implicit than explicit",
        "raw_z_auc_IM_vs_NH": z_im_nh, "probe_auc_IM_vs_NH": pr_im_nh,
        "raw_z_auc_EX_vs_NH": z_ex_nh, "probe_auc_EX_vs_NH": pr_ex_nh,
        "delta_IM": d_im, "delta_EX": d_ex,
        "delta_IM_minus_delta_EX": d_im - d_ex,
        "pass": bool(d_im >= d_ex + P2_MARGIN),
        "delta_IM_by_layer": [full_auc(S[:, L], "IM", "NH") - z_im_nh
                              for L in range(n_layers)],
        "delta_EX_by_layer": [full_auc(S[:, L], "EX", "NH") - z_ex_nh
                              for L in range(n_layers)],
    }

    # ---- P3 explicit sanity
    report["P3"] = {
        "statement": "the direction is a hatefulness direction",
        "probe_auc_EX_vs_NH": pr_ex_nh, "floor": P3_FLOOR,
        "pass": bool(pr_ex_nh >= P3_FLOOR),
        "probe_auc_EX_vs_NH_by_layer": [full_auc(S[:, L], "EX", "NH")
                                        for L in range(n_layers)],
    }

    # ---- P4 permuted-anchor placebo
    union = np.concatenate([idx_hi, idx_lo])
    HL = H[:, primary, :]
    perm_rng = np.random.default_rng(SEED + 1)
    perm_stats, perm_max_over_layers = [], []
    for _ in range(N_PERM):
        pm = perm_rng.permutation(len(union))
        fake_hi, fake_lo = union[pm[:n_anchor]], union[pm[n_anchor:]]
        w = direction_from(HL, fake_hi, fake_lo)
        perm_stats.append(p1_stat(HL @ w))
        best = 0.0
        for L in range(n_layers):
            wl = direction_from(H[:, L, :], fake_hi, fake_lo)
            best = max(best, p1_stat(H[:, L, :] @ wl))
        perm_max_over_layers.append(best)
    perm_stats = np.array(perm_stats)
    perm_max_over_layers = np.array(perm_max_over_layers)
    q95 = float(np.percentile(perm_stats, 95))
    report["P4"] = {
        "statement": "the effect is not anchor-set structure",
        "n_permutations": N_PERM,
        "real_probe_p1_auc": probe_p1,
        "permuted_p1_mean": float(perm_stats.mean()),
        "permuted_p1_p95": q95,
        "permuted_p1_max": float(perm_stats.max()),
        "empirical_p_value": float((perm_stats >= probe_p1).mean()),
        "pass": bool(probe_p1 > q95),
        "conservative_variant_max_over_layers": {
            "note": ("placebo given the same layer freedom: per permutation, the "
                     "maximum P1 statistic over all 37 layers. Reported as an "
                     "extra-conservative check; the pre-registered arm fixes the "
                     "primary layer."),
            "p95": float(np.percentile(perm_max_over_layers, 95)),
            "pass": bool(probe_p1 > float(np.percentile(perm_max_over_layers, 95))),
        },
    }

    # ---- P5a split-half stability
    sh_rng = np.random.default_rng(SEED + 2)
    cosines = []
    for _ in range(N_SPLIT_HALF):
        ph, pl = sh_rng.permutation(n_anchor), sh_rng.permutation(n_anchor)
        half = n_anchor // 2
        wa = direction_from(HL, idx_hi[ph[:half]], idx_lo[pl[:half]])
        wb = direction_from(HL, idx_hi[ph[half:]], idx_lo[pl[half:]])
        cosines.append(float(wa @ wb))
    cosines = np.array(cosines)
    report["P5a"] = {
        "statement": "direction stability under anchor resampling",
        "n_resamples": N_SPLIT_HALF, "layer": primary,
        "mean_cosine": float(cosines.mean()), "std_cosine": float(cosines.std()),
        "min_cosine": float(cosines.min()), "floor": P5A_COSINE_FLOOR,
        "pass": bool(cosines.mean() >= P5A_COSINE_FLOOR),
    }

    # ---- P5b resolution stratum
    px = np.array([native_pixels(args.dataset, v) or -1 for v in vids])
    lowres = px <= P720_MAX_PIXELS
    lr = lowres & dismissed
    z_p1_lr = auc(z[lr & (g == "IM")], z[lr & (g == "NH")])
    pr_p1_lr = auc(S[lr & (g == "IM"), primary], S[lr & (g == "NH"), primary])
    report["P5b"] = {
        "statement": "P1 holds inside the <=720p native-resolution stratum",
        "definition": f"frame_000.jpg pixel count <= {P720_MAX_PIXELS} (1280x720)",
        "conservative_choice": ("the pre-registration says '<=720p'; the strictest "
                                "reading is the 921,600-pixel cap, which also drops "
                                "the 808x1440 tier. Counts at the looser <1080p "
                                "reading are reported alongside."),
        "n_lowres_total": int(lowres.sum()),
        "n_highres_total": int((~lowres).sum()),
        "n_lt_1080p_alt": int((px < 1920 * 1080).sum()),
        "n_IM": int((lr & (g == "IM")).sum()), "n_NH": int((lr & (g == "NH")).sum()),
        "probe_auc": pr_p1_lr, "raw_z_auc": z_p1_lr,
        "delta": pr_p1_lr - z_p1_lr,
        "pass": bool(pr_p1_lr >= P1_AUC_FLOOR and pr_p1_lr >= z_p1_lr + P1_MARGIN),
        "resolution_by_group": {
            grp: {"n_lowres": int((lowres & (g == grp)).sum()),
                  "n_highres": int((~lowres & (g == grp)).sum())}
            for grp in ("EX", "IM", "NH")},
    }

    # ---- P6 over-flagging side
    ann = load_annotations(args.dataset)
    degen = np.array([is_degenerate(ann.get(v, {}).get("transcript", "")) for v in vids])
    p6 = {}
    for tag, mask in (("included", asserted), ("excluded", asserted & ~degen)):
        zi = auc(z[mask & (g == "EX")], z[mask & (g == "NH")])
        pi = auc(S[mask & (g == "EX"), primary], S[mask & (g == "NH"), primary])
        p6[tag] = {
            "n_EX": int((mask & (g == "EX")).sum()),
            "n_NH": int((mask & (g == "NH")).sum()),
            "probe_auc": pi, "raw_z_auc": zi, "delta": pi - zi,
            "pass": bool(np.isfinite(pi) and np.isfinite(zi)
                         and pi >= zi + P6_MARGIN),
        }
    report["P6"] = {
        "statement": "internals recognize over-flagged benign content",
        "screen_rule": (f"transcript (first {TRANSCRIPT_LIMIT} chars, as the "
                        f"model saw it) shorter than {DEGEN_MIN_CHARS} chars "
                        f"or under {DEGEN_MIN_LATIN_FRAC:.0%} Latin letters"),
        "n_degenerate_in_asserted": int((asserted & degen).sum()),
        "n_degenerate_overall": int(degen.sum()),
        "arms": p6,
        "pass": bool(p6["included"]["pass"]),
    }

    # ---- per-group median probe score at the primary layer
    report["probe_score_by_group_primary_layer"] = {
        grp: {"n": int((g == grp).sum()),
              "median": float(np.median(S[g == grp, primary])),
              "mean": float(S[g == grp, primary].mean())}
        for grp in ("EX", "IM", "NH")}
    report["probe_score_dismissed_stratum"] = {
        grp: {"n": int((dismissed & (g == grp)).sum()),
              "median": float(np.median(S[dismissed & (g == grp), primary]))
              if (dismissed & (g == grp)).sum() else None}
        for grp in ("EX", "IM", "NH")}

    # ---- robustness: k = 2% and 10%
    robust = {}
    for k in ANCHOR_K_ROBUST:
        hi_k, lo_k, n_k = build_anchors(z, k)
        pt, pb = anchor_purity(hi_k, lo_k, groups)
        cvk, _ = layer_cv_accuracy(H, hi_k, lo_k, np.random.default_rng(SEED + 3))
        Lk = int(np.argmax(cvk))
        wk = direction_from(H[:, Lk, :], hi_k, lo_k)
        sk = H[:, Lk, :] @ wk
        robust[f"k={k}"] = {
            "n_per_side": int(n_k), "purity_top": pt, "purity_bottom": pb,
            "primary_layer": Lk, "cv_accuracy": float(cvk[Lk]),
            "P1_probe_auc": p1_stat(sk), "P1_raw_z_auc": z_p1,
            "P1_pass": bool(p1_stat(sk) >= P1_AUC_FLOOR
                            and p1_stat(sk) >= z_p1 + P1_MARGIN),
            "P3_probe_auc_EX_vs_NH": auc(sk[g == "EX"], sk[g == "NH"]),
            "cosine_with_primary_direction": float(
                unit(direction_from(H[:, Lk, :], idx_hi, idx_lo)) @ wk),
        }
    report["robustness_anchor_k"] = robust

    # ---- kill rule and fingerprint
    p1_any_layer = any(a >= P1_AUC_FLOOR and a >= z_p1 + P1_MARGIN
                       for a in probe_p1_by_layer)
    kill_reasons = []
    if not p1_pass and not p1_any_layer:
        kill_reasons.append("P1 fails at the primary layer and at every layer")
    if not report["P4"]["pass"]:
        kill_reasons.append("P4 fails (anchor-structure artifact)")
    if not report["P2"]["pass"]:
        kill_reasons.append("P2 fails (uniform gain, not a dissociation)")
    report["kill_rule"] = {
        "killed": bool(kill_reasons),
        "reasons": kill_reasons,
        "verdict": "KILL" if kill_reasons else "SURVIVES",
    }

    curve = np.array(probe_p1_by_layer, dtype=np.float64)
    flat = bool(np.nanmax(np.abs(curve - 0.5)) < 0.05)
    if flat:
        fp = ("knowledge absence: dismissed IM is indistinguishable from dismissed "
              "NH at every layer, which argues for a knowledge-side successor "
              "rather than a readout-side one")
    else:
        sep = [L for L, a in enumerate(probe_p1_by_layer)
               if a >= z_p1 + P1_MARGIN and a >= P1_AUC_FLOOR]
        fp = (f"separation present at {len(sep)} layer(s); best layer "
              f"{int(np.nanargmax(curve))} at AUC {float(np.nanmax(curve)):.3f}; "
              f"layers meeting the P1 bar: {sep}")
    # interior IM as the partial-perception contrast
    int_auc = auc(S[interior & (g == "IM"), primary], S[interior & (g == "NH"), primary])
    report["fingerprint"] = {
        "dismissed_IM_vs_NH_curve_flat_at_0.5": flat,
        "max_abs_deviation_from_0.5": float(np.nanmax(np.abs(curve - 0.5))),
        "interior_IM_vs_NH_probe_auc": int_auc,
        "n_interior_IM": int((interior & (g == "IM")).sum()),
        "n_interior_NH": int((interior & (g == "NH")).sum()),
        "description": fp,
    }

    report["prediction_table"] = {
        k: {"pass": bool(report[k]["pass"])} for k in
        ("P1", "P2", "P3", "P4", "P5a", "P5b", "P6")}

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")
    for k in ("P1", "P2", "P3", "P4", "P5a", "P5b", "P6"):
        print(f"  {k}: {'PASS' if report[k]['pass'] else 'FAIL'}")
    print(f"  verdict: {report['kill_rule']['verdict']} {report['kill_rule']['reasons']}")


if __name__ == "__main__":
    main()
