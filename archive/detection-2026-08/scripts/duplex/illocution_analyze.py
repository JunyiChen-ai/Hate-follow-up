"""Illocution transfer audit: readout fitting and sealed natural evaluation.

Pre-registration: docs/duplex/PREREG_illocution_transfer_audit.md.

CPU only, numpy/scipy only. No model call: the frozen judge's hidden states,
both for the researcher-authored instrument (results/illocution/hidden) and for
the natural corpora (results/testruns/...), are read from disk.

Phase 1 fits the speech-act readout on the instrument alone, over the frozen
grid of layer rows {20,24,27,30,33} and subspace ranks {1,2,4,8,16}, selecting
by leave-family-out AUROC on the assert-vs-attribute contrast with EN and ZH
pooled. The selected readout is written to results/illocution/readout.npz
before any natural file is opened.

Phase 2 applies the frozen readout, one pass, to the two sealed strata:
MHClip-ZH keyword-markup videos and the HateMM code-Q valley false positives
against high-z HateMM Hate videos.

Output: results/illocution/report.json.

Usage:
  python scripts/duplex/illocution_analyze.py
"""

import csv
import hashlib
import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = "/home/jehc223/Hate-follow-up"
ILL = os.path.join(ROOT, "results", "illocution")
BANK_DIR = os.path.join(ROOT, "scripts", "duplex", "illocution_bank")
MANIFEST = os.path.join(ILL, "manifest.jsonl")
INSTR_HIDDEN = os.path.join(ILL, "hidden")
READOUT = os.path.join(ILL, "readout.npz")
OUT = os.path.join(ILL, "report.json")

ZH_ITEMS = os.path.join(ROOT, "results", "ranking_autopsy", "zh", "items.json")
ZH_HIDDEN = os.path.join(ROOT, "results", "testruns", "mhclip_zh", "judge_8b",
                         "hidden")
HM_CODING = os.path.join(ROOT, "results", "hatemm_fp_audit", "coding.tsv")
HM_ALIAS = os.path.join(ROOT, "results", "hatemm_fp_audit", "alias_map.json")
HM_SCORES = os.path.join(ROOT, "results", "testruns", "hatemm", "judge_8b",
                         "scores.jsonl")
HM_HIDDEN = os.path.join(ROOT, "results", "testruns", "hatemm", "judge_8b",
                         "hidden")
EN_PACKET = os.path.join(ROOT, "results", "ranking_autopsy", "en", "packet.json")
EN_HIDDEN = os.path.join(ROOT, "results", "testruns", "mhclip_en", "judge_8b",
                         "hidden")

LAYERS = [20, 24, 27, 30, 33]
RANKS = [1, 2, 4, 8, 16]
TIEBREAK_LAYER = 27
PROBE_LAMBDA = 1.0          # same fixed L2 as readout_bottleneck_killtest.py
WHITEN_SHRINKAGE = 0.1      # fixed before any result was seen; never tuned
HATEMM_VALLEY_Z = -2.3455
BOOT_N = 10000
BOOT_SEED = 20260810
EXPECTED_ROWS, EXPECTED_DIM = 37, 4096

CLAUSE1_FLOOR = 0.90
CLAUSE2_FLOOR = 0.85
CLAUSE2_GAP = 0.05
CLAUSE34_FLOOR = 0.70
CLAUSE34_LCB = 0.55

KEYWORD_MARKUP = '<em class="keyword">'


# ------------------------------------------------------------------ helpers --
def auc(scores, labels):
    """Mann-Whitney ROC-AUC with midranks for ties."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auc_pos_neg(pos, neg):
    return auc(list(pos) + list(neg), [1] * len(pos) + [0] * len(neg))


def standardize_stats(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd <= 0, 1.0, sd)
    return mu, sd


def load_hidden(hidden_dir, ids):
    out = np.empty((len(ids), EXPECTED_ROWS, EXPECTED_DIM), dtype=np.float32)
    for k, v in enumerate(ids):
        p = os.path.join(hidden_dir, v + ".npy")
        a = np.load(p)
        if a.shape != (EXPECTED_ROWS, EXPECTED_DIM):
            raise SystemExit(f"ABORT: {p} shape {a.shape}")
        a = a.astype(np.float32)
        if not np.isfinite(a).all():
            raise SystemExit(f"ABORT: non-finite values in {p}")
        out[k] = a
    return out


def logistic_fit(X, y, lam):
    """L2-penalised logistic regression, unpenalised intercept, L-BFGS."""
    n, d = X.shape
    t = np.where(np.asarray(y) == 1, 1.0, -1.0)

    def obj(theta):
        w, b = theta[:d], theta[d]
        m = t * (X @ w + b)
        loss = np.logaddexp(0.0, -m).sum() + 0.5 * lam * float(w @ w)
        s = -t / (1.0 + np.exp(m))
        g = np.empty(d + 1)
        g[:d] = X.T @ s + lam * w
        g[d] = s.sum()
        return loss, g

    r = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                 options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9})
    return r.x[:d], float(r.x[d]), bool(r.success)


def contrast_directions(Z, y, pair_index, kmax):
    """Directions of the assert-vs-attribute contrast, in the r-dim work space.

    Direction 1 is the class mean-difference direction (attribute minus
    assert), matching the pre-registered rank-1 readout. Directions 2..kmax are
    the leading principal directions of the within-pair difference vectors
    after whitening by the shrunk pooled within-class covariance and deflating
    direction 1, matching the pre-registered rank-k readout.

    Returns an (r, kmax) matrix whose columns already fold in the whitening, so
    features are simply Z @ W.
    """
    r = Z.shape[1]
    y = np.asarray(y)
    mu1 = Z[y == 1].mean(axis=0)
    mu0 = Z[y == 0].mean(axis=0)
    d1_raw = mu1 - mu0
    n1 = np.linalg.norm(d1_raw)
    d1_raw = d1_raw / (n1 if n1 > 0 else 1.0)
    if kmax == 1:
        return d1_raw.reshape(r, 1)

    # pooled within-class covariance with fixed shrinkage toward scaled identity
    C = np.zeros((r, r))
    n = 0
    for c in (0, 1):
        Zc = Z[y == c]
        Zc = Zc - Zc.mean(axis=0)
        C += Zc.T @ Zc
        n += Zc.shape[0] - 1
    C /= max(n, 1)
    tr = float(np.trace(C)) / r
    C = (1.0 - WHITEN_SHRINKAGE) * C + WHITEN_SHRINKAGE * tr * np.eye(r)
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, 1e-10)
    Wh = evecs @ np.diag(evals ** -0.5) @ evecs.T      # symmetric whitener

    Zw = Z @ Wh
    # within-pair difference vectors (attribute minus assert), whitened
    diffs = []
    for idx1, idx0 in pair_index:
        diffs.append(Zw[idx1] - Zw[idx0])
    D = np.asarray(diffs)
    m = D.mean(axis=0)
    nm = np.linalg.norm(m)
    u1 = m / (nm if nm > 0 else 1.0)
    Dd = D - np.outer(D @ u1, u1)
    _, _, Vt = np.linalg.svd(Dd, full_matrices=False)
    U = [u1]
    for j in range(min(kmax - 1, Vt.shape[0])):
        U.append(Vt[j])
    U = np.asarray(U).T                                # (r, k)
    return Wh @ U                                      # fold whitening in


# -------------------------------------------------------------- instrument ---
def load_instrument():
    items = []
    seen = set()
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            items.append(r)
    bank_ids = []
    for lang in ("en", "zh"):
        with open(os.path.join(BANK_DIR, f"{lang}.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    bank_ids.append(json.loads(line)["id"])
    if set(bank_ids) != seen:
        raise SystemExit(f"ABORT: manifest has {len(seen)} ids, bank has "
                         f"{len(bank_ids)}; sets differ")
    items.sort(key=lambda r: bank_ids.index(r["id"]))
    return items


def main():
    report = {
        "protocol": "docs/duplex/PREREG_illocution_transfer_audit.md",
        "compute": "CPU only; frozen hidden states read from disk",
        "frozen_constants": {
            "layers": LAYERS, "ranks": RANKS,
            "logistic_l2_lambda": PROBE_LAMBDA,
            "whitening_shrinkage": WHITEN_SHRINKAGE,
            "standardisation": ("per-dimension mean and standard deviation of "
                                "the pool's own hidden states, no labels; "
                                "instrument standardised by the instrument, "
                                "each natural corpus by itself, as in "
                                "readout_bottleneck_killtest.py"),
            "score_orientation": "higher = ATTRIBUTED (attributed pole positive)",
            "hatemm_valley_z": HATEMM_VALLEY_Z,
            "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED},
            "clause_floors": {"clause1": CLAUSE1_FLOOR, "clause2": CLAUSE2_FLOOR,
                              "clause2_gap": CLAUSE2_GAP,
                              "clause34": CLAUSE34_FLOOR,
                              "clause34_lower_bound": CLAUSE34_LCB},
        },
    }

    # ---------------------------------------------------------- instrument --
    items = load_instrument()
    ids = [r["id"] for r in items]
    H = load_hidden(INSTR_HIDDEN, ids)
    act = np.array([r["speech_act"] for r in items])
    fam = np.array([f"{r['language']}:{r['family']}" for r in items])
    quoted = np.array([bool(r["quoted"]) for r in items])
    lang = np.array([r["language"] for r in items])
    z_instr = np.array([float(r["z"]) for r in items])
    prim = np.where((act == "assert") | (act == "attribute"))[0]
    y_all = (act == "attribute").astype(int)

    pair_of = {}
    for i, r in enumerate(items):
        if r["speech_act"] in ("assert", "attribute"):
            pair_of.setdefault(r["pair_id"], {})[r["speech_act"]] = i
    pairs = [(v["attribute"], v["assert"]) for v in pair_of.values()
             if "attribute" in v and "assert" in v]

    report["instrument"] = {
        "n_items": len(items),
        "n_en": int((lang == "en").sum()), "n_zh": int((lang == "zh").sum()),
        "n_assert": int((act == "assert").sum()),
        "n_attribute": int((act == "attribute").sum()),
        "n_attribute_reject": int((act == "attribute_reject").sum()),
        "n_quoted": int(quoted.sum()),
        "n_families": int(len(set(fam))),
        "n_primary_pairs": len(pairs),
        "z_mean": float(z_instr.mean()), "z_min": float(z_instr.min()),
        "z_max": float(z_instr.max()),
        "bank_sha256": {},
    }
    for l in ("en", "zh"):
        p = os.path.join(BANK_DIR, f"{l}.jsonl")
        report["instrument"]["bank_sha256"][l] = hashlib.sha256(
            open(p, "rb").read()).hexdigest()

    families = sorted(set(fam[prim]))
    kmax = max(RANKS)

    # Work space: unsupervised PCA basis of the standardised instrument states
    # for this layer (label-free, exact for the instrument since n < d).
    grid = {}
    heldout_cache = {}
    for layer in LAYERS:
        X = H[:, layer, :]
        mu, sd = standardize_stats(X)
        Xs = (X - mu) / sd
        cmean = Xs.mean(axis=0)
        Xc = Xs - cmean
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        keep = S > (S[0] * 1e-8)
        V = Vt[keep].T                       # (d, r)
        Zall = Xc @ V                        # (n, r)

        held = {k: np.full(len(items), np.nan) for k in RANKS}
        for f in families:
            te = np.array([i for i in prim if fam[i] == f])
            tr = np.array([i for i in prim if fam[i] != f])
            trpos = {g: j for j, g in enumerate(tr.tolist())}
            tr_pairs = [(trpos[a], trpos[b]) for a, b in pairs
                        if a in trpos and b in trpos]
            Wk = contrast_directions(Zall[tr], y_all[tr], tr_pairs, kmax)
            Ftr_full = Zall[tr] @ Wk
            Fte_full = Zall[te] @ Wk
            for k in RANKS:
                Ftr = Ftr_full[:, :k]
                Fte = Fte_full[:, :k]
                m, s = Ftr.mean(axis=0), np.where(Ftr.std(axis=0) <= 0, 1.0,
                                                  Ftr.std(axis=0))
                w, b, _ = logistic_fit((Ftr - m) / s, y_all[tr], PROBE_LAMBDA)
                held[k][te] = ((Fte - m) / s) @ w + b
        heldout_cache[layer] = held
        for k in RANKS:
            h = held[k][prim]
            yy = y_all[prim]
            row = {
                "lofo_auroc": auc(h, yy),
                "lofo_auroc_en": auc(held[k][prim][lang[prim] == "en"],
                                     yy[lang[prim] == "en"]),
                "lofo_auroc_zh": auc(held[k][prim][lang[prim] == "zh"],
                                     yy[lang[prim] == "zh"]),
                "lofo_auroc_quoted_only": auc(held[k][prim][quoted[prim]],
                                              yy[quoted[prim]]),
                "lofo_auroc_unquoted_only": auc(held[k][prim][~quoted[prim]],
                                                yy[~quoted[prim]]),
            }
            row["quote_reversal_gap"] = abs(row["lofo_auroc_quoted_only"]
                                            - row["lofo_auroc_unquoted_only"])
            grid[f"L{layer}_R{k}"] = row

    report["phase1_grid"] = grid

    # selection: max LOFO AUROC, ties -> smaller rank, then layer 27
    cand = []
    for layer in LAYERS:
        for k in RANKS:
            a = grid[f"L{layer}_R{k}"]["lofo_auroc"]
            cand.append((-a, k, 0 if layer == TIEBREAK_LAYER else 1, layer))
    cand.sort()
    sel_layer, sel_rank = cand[0][3], cand[0][1]
    sel = grid[f"L{sel_layer}_R{sel_rank}"]
    report["selection"] = {"layer": sel_layer, "rank": sel_rank,
                           "rule": "max LOFO AUROC; ties -> smaller rank, then layer 27",
                           **sel}

    # descriptive: attribute_reject cell at the selected cell (held-out scores
    # are unavailable for reject items, which never enter fitting; use the
    # full-instrument readout below).

    # ---- freeze the readout on the full instrument ------------------------
    X = H[:, sel_layer, :]
    mu, sd = standardize_stats(X)
    Xs = (X - mu) / sd
    cmean = Xs.mean(axis=0)
    Xc = Xs - cmean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    keep = S > (S[0] * 1e-8)
    V = Vt[keep].T
    Zall = Xc @ V
    plist = list(prim)
    Wk = contrast_directions(Zall[prim], y_all[prim],
                             [(plist.index(a), plist.index(b)) for a, b in pairs],
                             kmax)[:, :sel_rank]
    F = Zall[prim] @ Wk
    fm, fs = F.mean(axis=0), np.where(F.std(axis=0) <= 0, 1.0, F.std(axis=0))
    wlog, blog, conv = logistic_fit((F - fm) / fs, y_all[prim], PROBE_LAMBDA)
    # collapse to a single d-dim map: score(x) = ((((x-mu)/sd - cmean) @ V @ Wk) - fm)/fs @ w + b
    A = (V @ Wk) / fs                       # (d, k) with feature scaling folded in
    w_full = A @ wlog                       # (d,)
    b_full = float(blog - ((cmean @ (V @ Wk)) / fs) @ wlog - (fm / fs) @ wlog)
    np.savez(READOUT, mu=mu, sd=sd, w=w_full, b=np.array([b_full]),
             layer=np.array([sel_layer]), rank=np.array([sel_rank]),
             W_directions=(V @ Wk), feat_mean=fm, feat_std=fs,
             centre=cmean, logistic_w=wlog, logistic_b=np.array([blog]))
    report["readout_frozen"] = {
        "path": READOUT, "layer": sel_layer, "rank": sel_rank,
        "logistic_converged": conv,
        "in_sample_auroc": auc((Zall[prim] @ Wk - fm) / fs @ wlog + blog,
                               y_all[prim]),
    }

    def score_from_raw(Xraw, mu_, sd_):
        """Frozen readout applied to raw hidden states standardised by mu_/sd_."""
        return ((Xraw - mu_) / sd_) @ w_full + b_full

    # sanity: the collapsed map reproduces the pipeline on the instrument
    chk = score_from_raw(X, mu, sd)[prim]
    ref = ((Zall[prim] @ Wk - fm) / fs) @ wlog + blog
    if not np.allclose(chk, ref, atol=1e-3, rtol=1e-4):
        raise SystemExit(f"ABORT: collapsed readout mismatch "
                         f"{np.abs(chk - ref).max()}")

    s_instr = score_from_raw(X, mu, sd)
    report["instrument_descriptive"] = {
        "attributedness_mean_assert": float(s_instr[act == "assert"].mean()),
        "attributedness_mean_attribute": float(s_instr[act == "attribute"].mean()),
        "attributedness_mean_attribute_reject":
            float(s_instr[act == "attribute_reject"].mean()),
        "auroc_reject_vs_assert": auc_pos_neg(s_instr[act == "attribute_reject"],
                                              s_instr[act == "assert"]),
        "auroc_reject_vs_attribute": auc_pos_neg(s_instr[act == "attribute_reject"],
                                                 s_instr[act == "attribute"]),
        "spearman_score_vs_z": float(spearmanr(s_instr, z_instr)[0]),
        "in_sample_auroc_quoted_only": auc_pos_neg(
            s_instr[(act == "attribute") & quoted],
            s_instr[(act == "assert") & quoted]),
        "in_sample_auroc_unquoted_only": auc_pos_neg(
            s_instr[(act == "attribute") & ~quoted],
            s_instr[(act == "assert") & ~quoted]),
    }

    # ------------------------------------------------- clauses 1 and 2 -----
    c1 = bool(sel["lofo_auroc"] >= CLAUSE1_FLOOR)
    c2 = bool(sel["lofo_auroc_quoted_only"] >= CLAUSE2_FLOOR
              and sel["lofo_auroc_unquoted_only"] >= CLAUSE2_FLOOR
              and sel["quote_reversal_gap"] <= CLAUSE2_GAP)
    report["clause1_instrument_validity"] = {
        "lofo_auroc": sel["lofo_auroc"], "floor": CLAUSE1_FLOOR, "passes": c1}
    report["clause2_format_invariance"] = {
        "lofo_auroc_quoted_only": sel["lofo_auroc_quoted_only"],
        "lofo_auroc_unquoted_only": sel["lofo_auroc_unquoted_only"],
        "quote_reversal_gap": sel["quote_reversal_gap"],
        "floor": CLAUSE2_FLOOR, "gap_ceiling": CLAUSE2_GAP, "passes": c2}

    # ================= PHASE 2: sealed natural strata ======================
    rng = np.random.default_rng(BOOT_SEED)

    natural_cache = {}

    def natural_stack(hidden_dir, pool_ids):
        key = hidden_dir
        if key not in natural_cache:
            natural_cache[key] = load_hidden(hidden_dir, list(pool_ids))
        return natural_cache[key]

    def stratum_eval(name, ids_ment, ids_asrt, hidden_dir, pool_ids, zmap):
        pool = list(pool_ids)
        Hp = natural_stack(hidden_dir, pool)[:, sel_layer, :]
        mu_p, sd_p = standardize_stats(Hp)
        s_pool = score_from_raw(Hp, mu_p, sd_p)
        s_instr_norm = score_from_raw(Hp, mu, sd)   # descriptive variant
        idx = {v: i for i, v in enumerate(pool)}
        ev = ids_ment + ids_asrt
        yy = np.array([1] * len(ids_ment) + [0] * len(ids_asrt))
        s = np.array([s_pool[idx[v]] for v in ev])
        s_alt = np.array([s_instr_norm[idx[v]] for v in ev])
        zz = np.array([zmap[v] for v in ev])

        def resid(sv, zv):
            Amat = np.column_stack([np.ones_like(zv), zv])
            coef, *_ = np.linalg.lstsq(Amat, sv, rcond=None)
            return sv - Amat @ coef, coef

        r, coef = resid(s, zz)
        a_res = auc(r, yy)
        a_raw = auc(s, yy)
        boots = []
        i1 = np.where(yy == 1)[0]
        i0 = np.where(yy == 0)[0]
        for _ in range(BOOT_N):
            b1 = rng.choice(i1, size=len(i1), replace=True)
            b0 = rng.choice(i0, size=len(i0), replace=True)
            bi = np.concatenate([b1, b0])
            rb, _ = resid(s[bi], zz[bi])
            v = auc(rb, yy[bi])
            if v is not None:
                boots.append(v)
        boots = np.asarray(boots)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        r_alt, _ = resid(s_alt, zz)
        passes = bool(a_res >= CLAUSE34_FLOOR and lo > CLAUSE34_LCB
                      and a_res > 0.5)
        return {
            "n_mentioned": len(ids_ment), "n_asserted": len(ids_asrt),
            "n_pool_for_standardisation": len(pool),
            "auroc_residualised": a_res,
            "boot_ci95": [float(lo), float(hi)],
            "boot_mean": float(boots.mean()), "boot_n": int(len(boots)),
            "predicted_sign_holds": bool(a_res > 0.5),
            "auroc_raw_unresidualised": a_raw,
            "ols_intercept": float(coef[0]), "ols_slope_on_z": float(coef[1]),
            "spearman_score_vs_z": float(spearmanr(s, zz)[0]),
            "mean_score_mentioned": float(s[yy == 1].mean()),
            "mean_score_asserted": float(s[yy == 0].mean()),
            "sd_score_mentioned": float(s[yy == 1].std()),
            "sd_score_asserted": float(s[yy == 0].std()),
            "mean_z_mentioned": float(zz[yy == 1].mean()),
            "mean_z_asserted": float(zz[yy == 0].mean()),
            "descriptive_instrument_normalisation": {
                "auroc_residualised": auc(r_alt, yy),
                "auroc_raw": auc(s_alt, yy),
            },
            "passes": passes,
        }

    # ---- ZH stratum -------------------------------------------------------
    zh_items = json.load(open(ZH_ITEMS, encoding="utf-8"))
    zh_pool_all = [x["video_id"] for x in zh_items]
    zmap_zh = {x["video_id"]: float(x["z"]) for x in zh_items}
    kw = [x for x in zh_items if KEYWORD_MARKUP in (x.get("title") or "")]
    zh_ment = [x["video_id"] for x in kw if x["fine"] == "Normal"]
    zh_asrt = [x["video_id"] for x in kw if x["fine"] in ("Hateful", "Offensive")]
    if (len(zh_ment), len(zh_asrt)) != (31, 32):
        raise SystemExit(f"ABORT: ZH strata {len(zh_ment)}/{len(zh_asrt)}, "
                         "expected 31/32")
    zh = stratum_eval("zh", zh_ment, zh_asrt, ZH_HIDDEN, zh_pool_all, zmap_zh)
    zh["definition"] = ("MHClip-ZH test videos whose harvest title carries "
                        "the keyword markup; MENTIONED = Normal, ASSERTED = "
                        "Hateful or Offensive")
    report["clause3_natural_transfer_zh"] = zh

    # ---- HateMM stratum ---------------------------------------------------
    alias = json.load(open(HM_ALIAS))
    qcodes = []
    with open(HM_CODING) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["primary"].strip() == "Q":
                qcodes.append(row["alias"].strip())
    hm_ment = [alias[a] for a in qcodes]
    hm_rows = [json.loads(l) for l in open(HM_SCORES) if l.strip()]
    zmap_hm = {r["video_id"]: float(r["z"]) for r in hm_rows}
    hm_pool_all = [r["video_id"] for r in hm_rows]
    hm_asrt = [r["video_id"] for r in hm_rows
               if r["video_id"].startswith("hate_video_")
               and float(r["z"]) > HATEMM_VALLEY_Z]
    if len(hm_ment) != 11:
        raise SystemExit(f"ABORT: HateMM code-Q count {len(hm_ment)}, expected 11")
    if set(hm_ment) & set(hm_asrt):
        raise SystemExit("ABORT: HateMM strata overlap")
    hm = stratum_eval("hatemm", hm_ment, hm_asrt, HM_HIDDEN, hm_pool_all, zmap_hm)
    hm["definition"] = ("MENTIONED = blind code-Q valley false positives; "
                        "ASSERTED = HateMM test-split Hate-labelled videos "
                        f"(video_id prefix hate_video_) with z > {HATEMM_VALLEY_Z}")
    hm["code_q_aliases"] = qcodes
    report["clause4_natural_transfer_hatemm"] = hm

    # ---- EN descriptive ---------------------------------------------------
    packet = json.load(open(EN_PACKET, encoding="utf-8"))
    amap = {x["alias"]: x for x in packet}
    en_ids = sorted(os.path.splitext(f)[0]
                    for f in os.listdir(EN_HIDDEN) if f.endswith(".npy"))
    Hen = load_hidden(EN_HIDDEN, en_ids)[:, sel_layer, :]
    mu_e, sd_e = standardize_stats(Hen)
    s_en = score_from_raw(Hen, mu_e, sd_e)
    eidx = {v: i for i, v in enumerate(en_ids)}
    en_cases = {}
    for a in ("EN-FN-02", "EN-FP-07", "EN-FP-16"):
        rec = amap.get(a)
        if rec is None:
            en_cases[a] = {"error": "alias not in packet.json"}
            continue
        v = rec["video_id"]
        en_cases[a] = {
            "video_id": v, "z": rec.get("z"), "fine": rec.get("fine"),
            "attributedness": float(s_en[eidx[v]]) if v in eidx else None,
            "percentile_in_en_corpus": (
                float((s_en < s_en[eidx[v]]).mean()) if v in eidx else None),
        }
    report["en_descriptive_cases"] = {
        "n_en_corpus": len(en_ids),
        "corpus_score_mean": float(s_en.mean()),
        "corpus_score_sd": float(s_en.std()),
        "cases": en_cases,
    }

    # ---- verdict ----------------------------------------------------------
    c3, c4 = zh["passes"], hm["passes"]
    report["verdict"] = ("SURVIVES" if (c1 and c2 and c3 and c4) else "DEAD")
    report["verdict_clauses"] = {"clause1": c1, "clause2": c2,
                                 "clause3_zh": c3, "clause4_hatemm": c4}

    # ---- single permitted exploratory full-grid sweep ---------------------
    if not (c3 and c4):
        sweep = {}
        for layer in LAYERS:
            Xg = H[:, layer, :]
            mug, sdg = standardize_stats(Xg)
            Xsg = (Xg - mug) / sdg
            cm = Xsg.mean(axis=0)
            Xcg = Xsg - cm
            _, Sg, Vtg = np.linalg.svd(Xcg, full_matrices=False)
            Vg = Vtg[Sg > Sg[0] * 1e-8].T
            Zg = Xcg @ Vg
            Wg = contrast_directions(Zg[prim], y_all[prim],
                                     [(plist.index(a), plist.index(b))
                                      for a, b in pairs], kmax)
            for k in RANKS:
                Wkk = Wg[:, :k]
                Fg = Zg[prim] @ Wkk
                fmg = Fg.mean(axis=0)
                fsg = np.where(Fg.std(axis=0) <= 0, 1.0, Fg.std(axis=0))
                wl, bl, _ = logistic_fit((Fg - fmg) / fsg, y_all[prim],
                                         PROBE_LAMBDA)
                wf = ((Vg @ Wkk) / fsg) @ wl
                bf = float(bl - ((cm @ (Vg @ Wkk)) / fsg) @ wl - (fmg / fsg) @ wl)

                def sc(Xr, m_, s_):
                    return ((Xr - m_) / s_) @ wf + bf

                cell = {}
                for nm, ment, asrt, hdir, pool, zm in (
                        ("zh", zh_ment, zh_asrt, ZH_HIDDEN, zh_pool_all, zmap_zh),
                        ("hatemm", hm_ment, hm_asrt, HM_HIDDEN, hm_pool_all,
                         zmap_hm)):
                    Hp = natural_stack(hdir, pool)[:, layer, :]
                    m_, s_ = standardize_stats(Hp)
                    sp = sc(Hp, m_, s_)
                    ix = {v: i for i, v in enumerate(pool)}
                    ev = ment + asrt
                    yy = np.array([1] * len(ment) + [0] * len(asrt))
                    sv = np.array([sp[ix[v]] for v in ev])
                    zv = np.array([zm[v] for v in ev])
                    Am = np.column_stack([np.ones_like(zv), zv])
                    co, *_ = np.linalg.lstsq(Am, sv, rcond=None)
                    cell[nm] = {"auroc_residualised": auc(sv - Am @ co, yy),
                                "auroc_raw": auc(sv, yy)}
                sweep[f"L{layer}_R{k}"] = cell
        report["exploratory_full_grid_natural_sweep"] = {
            "status": "EXPLORATORY, never promoted; the single sweep the "
                      "pre-registration permits after a failed selected cell",
            "cells": sweep,
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    # -------------------------------------------------------- compact print --
    print(f"bank sha256 en={report['instrument']['bank_sha256']['en']}")
    print(f"bank sha256 zh={report['instrument']['bank_sha256']['zh']}")
    print(f"instrument n={len(items)} families={report['instrument']['n_families']} "
          f"pairs={len(pairs)}")
    print("\nPhase-1 grid, LOFO AUROC (attribute positive):")
    print("layer " + "".join(f"{('R' + str(k)):>10}" for k in RANKS))
    for layer in LAYERS:
        print(f"{layer:>5} " + "".join(
            f"{grid[f'L{layer}_R{k}']['lofo_auroc']:>10.4f}" for k in RANKS))
    print(f"\nselected: layer {sel_layer} rank {sel_rank} "
          f"LOFO={sel['lofo_auroc']:.4f}")
    print(f"clause1 {c1}  LOFO={sel['lofo_auroc']:.4f} (>= {CLAUSE1_FLOOR})")
    print(f"clause2 {c2}  quoted={sel['lofo_auroc_quoted_only']:.4f} "
          f"unquoted={sel['lofo_auroc_unquoted_only']:.4f} "
          f"gap={sel['quote_reversal_gap']:.4f}")
    print(f"clause3 {c3}  ZH resid AUROC={zh['auroc_residualised']:.4f} "
          f"CI={zh['boot_ci95']} raw={zh['auroc_raw_unresidualised']:.4f}")
    print(f"clause4 {c4}  HateMM resid AUROC={hm['auroc_residualised']:.4f} "
          f"CI={hm['boot_ci95']} raw={hm['auroc_raw_unresidualised']:.4f}")
    print(f"\nverdict: {report['verdict']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
