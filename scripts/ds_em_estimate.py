#!/usr/bin/env python3
"""Dawes-Skene EM estimation of per-verifier reliability rho^(k).

Compares estimated rho against label-peeked true accuracy.
Verifiers: holistic models (binarized at 0.5) + judge offline outputs.
"""
import json
import math
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP2")
DATA = ROOT / "datasets"
LABMAP = {
    "MHClip_EN": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    "MHClip_ZH": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    "HateMM": {"Hate": 1, "Non Hate": 0},
    "ImpliHateVid": {"Hateful": 1, "Normal": 0},
}

HOLISTIC = [
    "holistic_2b", "holistic_8b", "holistic_gemma-3-12b-it",
    "holistic_internvl3-14b", "holistic_minicpm-v-26",
    "holistic_pixtral-12b-2409", "holistic_qwen2.5-vl-7b",
]
JUDGES = ["qwen3-vl-8b", "gemma-3-27b-it"]


def load_holistic(ds, model, thr=0.5):
    """video_id -> binary pred from continuous score binarized at thr."""
    fp = ROOT / "results" / model / ds / "test_binary.jsonl"
    if not fp.exists():
        return None
    out = {}
    for line in open(fp):
        r = json.loads(line)
        s = r.get("score")
        if s is None:
            continue
        out[r["video_id"]] = 1 if s >= thr else 0
    return out


def load_judge(ds, judge):
    fp = ROOT / "results" / "boundary_rescue" / ds / f"offline_test_lp_{judge}.jsonl"
    if not fp.exists():
        return None
    out = {}
    for line in open(fp):
        r = json.loads(line)
        try:
            p = int(r.get("pred"))
        except (TypeError, ValueError):
            continue
        out[r["video_id"]] = p
    return out


def load_labels(ds):
    ann = json.load(open(DATA / ds / "annotation(new).json"))
    return {r["Video_ID"]: LABMAP[ds].get(r["Label"], -1) for r in ann}


def build_matrix(ds):
    """Return (videos, verifier_names, R[N,K] binary matrix, labels[N])."""
    labels = load_labels(ds)
    verifiers = []
    name2pred = {}
    for m in HOLISTIC:
        d = load_holistic(ds, m)
        if d is not None and len(d) > 0:
            verifiers.append(m)
            name2pred[m] = d
    for j in JUDGES:
        d = load_judge(ds, j)
        if d is not None and len(d) > 0:
            verifiers.append(j)
            name2pred[j] = d
    common = set(labels.keys())
    for v in verifiers:
        common &= set(name2pred[v].keys())
    common = sorted(common)
    common = [v for v in common if labels[v] in (0, 1)]
    R = [[name2pred[v][vid] for v in verifiers] for vid in common]
    y = [labels[vid] for vid in common]
    return common, verifiers, R, y


def ds_em(R, n_iter=200, tol=1e-6, eps=1e-3):
    """Asymmetric Dawes-Skene EM. R is list of K-length binary vectors.
    Returns (pi, rho_plus, rho_minus, z_post_hate)."""
    N = len(R)
    K = len(R[0])
    # Init z by majority vote
    z = []
    for r in R:
        s = sum(r)
        z.append(1.0 if s > K / 2 else (0.5 if s == K / 2 else 0.0))
    pi = sum(z) / N
    rho_p = [0.7] * K
    rho_n = [0.7] * K
    last_ll = -float("inf")
    for it in range(n_iter):
        # M-step
        sz = sum(z)
        s1z = N - sz
        if sz < eps or s1z < eps:
            sz = max(sz, eps)
            s1z = max(s1z, eps)
        for k in range(K):
            num_p = sum(z[i] * R[i][k] for i in range(N))
            num_n = sum((1 - z[i]) * (1 - R[i][k]) for i in range(N))
            rho_p[k] = max(0.5 + eps, min(1 - eps, num_p / sz))
            rho_n[k] = max(0.5 + eps, min(1 - eps, num_n / s1z))
        pi = max(eps, min(1 - eps, sz / N))
        # E-step + log-likelihood
        ll = 0.0
        new_z = []
        for i in range(N):
            log_pos = math.log(pi)
            log_neg = math.log(1 - pi)
            for k in range(K):
                if R[i][k] == 1:
                    log_pos += math.log(rho_p[k])
                    log_neg += math.log(1 - rho_n[k])
                else:
                    log_pos += math.log(1 - rho_p[k])
                    log_neg += math.log(rho_n[k])
            mx = max(log_pos, log_neg)
            ll += mx + math.log(math.exp(log_pos - mx) + math.exp(log_neg - mx))
            zp = math.exp(log_pos - mx)
            zn = math.exp(log_neg - mx)
            new_z.append(zp / (zp + zn))
        z = new_z
        if abs(ll - last_ll) < tol:
            break
        last_ll = ll
    return pi, rho_p, rho_n, z, it + 1


def acc(yhat, y):
    return sum(1 for a, b in zip(yhat, y) if a == b) / len(y)


def f1m(yhat, y):
    cls = sorted(set(y))
    s = 0.0
    for c in cls:
        tp = sum(1 for a, b in zip(y, yhat) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yhat) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yhat) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        s += 2 * p * r / (p + r) if p + r else 0
    return s / len(cls) if cls else 0


def per_verifier_truth(R, y, verifiers):
    """Label-peeked sensitivity, specificity, accuracy per verifier."""
    rows = []
    K = len(verifiers)
    for k in range(K):
        preds = [R[i][k] for i in range(len(y))]
        # sensitivity = P(pred=1|y=1)
        n_pos = sum(1 for yy in y if yy == 1)
        n_neg = sum(1 for yy in y if yy == 0)
        tp = sum(1 for i in range(len(y)) if y[i] == 1 and R[i][k] == 1)
        tn = sum(1 for i in range(len(y)) if y[i] == 0 and R[i][k] == 0)
        sens = tp / n_pos if n_pos else float("nan")
        spec = tn / n_neg if n_neg else float("nan")
        rows.append((verifiers[k], sens, spec, acc(preds, y)))
    return rows


def main():
    print("=" * 100)
    print("Dawes-Skene EM estimation of verifier reliability")
    print("=" * 100)
    for ds in ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]:
        print(f"\n### {ds}")
        videos, verifiers, R, y = build_matrix(ds)
        N, K = len(videos), len(verifiers)
        print(f"N={N} videos, K={K} verifiers, prevalence(true) = {sum(y)/N:.3f}")
        pi, rho_p, rho_n, z, iters = ds_em(R)
        print(f"EM converged in {iters} iters; pi_estimated={pi:.3f}")
        truth = per_verifier_truth(R, y, verifiers)
        # Label-switching check: if EM-pi is near 1-true_pi, posterior may be flipped
        true_pi = sum(y) / N
        if abs(pi - (1 - true_pi)) < abs(pi - true_pi):
            print(f"  WARN: label-switched (pi_est={pi:.3f}, true={true_pi:.3f}). Flipping interpretation.")
            rho_p, rho_n = rho_n, rho_p
            pi = 1 - pi
            z = [1 - zi for zi in z]
        print(f"\n  {'verifier':30s} {'rho+ est':>9s} {'rho- est':>9s} | {'sens true':>9s} {'spec true':>9s} {'acc true':>9s}")
        print("  " + "-" * 96)
        for k in range(K):
            name, sens, spec, accv = truth[k]
            print(f"  {name:30s} {rho_p[k]:>9.3f} {rho_n[k]:>9.3f} | {sens:>9.3f} {spec:>9.3f} {accv:>9.3f}")
        # Spearman / Pearson correlation between estimated avg rho and true acc
        avg_rho_est = [(rho_p[k] + rho_n[k]) / 2 for k in range(K)]
        true_acc = [truth[k][3] for k in range(K)]
        n = K
        mean_x = sum(avg_rho_est) / n
        mean_y = sum(true_acc) / n
        cov = sum((avg_rho_est[i] - mean_x) * (true_acc[i] - mean_y) for i in range(n))
        var_x = sum((avg_rho_est[i] - mean_x) ** 2 for i in range(n))
        var_y = sum((true_acc[i] - mean_y) ** 2 for i in range(n))
        pearson = cov / (math.sqrt(var_x * var_y) + 1e-12)
        # Rank correlation
        rx = sorted(range(n), key=lambda i: avg_rho_est[i])
        ry = sorted(range(n), key=lambda i: true_acc[i])
        rank_x = [0] * n
        rank_y = [0] * n
        for r, i in enumerate(rx):
            rank_x[i] = r
        for r, i in enumerate(ry):
            rank_y[i] = r
        d2 = sum((rank_x[i] - rank_y[i]) ** 2 for i in range(n))
        spearman = 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else float("nan")
        print(f"\n  Pearson(est_rho_avg, true_acc) = {pearson:+.3f}")
        print(f"  Spearman                       = {spearman:+.3f}")
        # DS-EM consensus accuracy vs majority vote
        ds_pred = [1 if zi >= 0.5 else 0 for zi in z]
        mv_pred = [1 if sum(R[i]) > K / 2 else (0 if sum(R[i]) < K / 2 else 1) for i in range(N)]
        print(f"  DS-EM consensus  acc={acc(ds_pred, y):.3f}  F1={f1m(ds_pred, y):.3f}")
        print(f"  Majority vote    acc={acc(mv_pred, y):.3f}  F1={f1m(mv_pred, y):.3f}")
    print()


if __name__ == "__main__":
    main()
