#!/usr/bin/env python3
"""Ablation gate table and DeHate / main-corpus checks for one decoder and one Integrator (README §4).

Frame score of a video v (every row, every corpus, the same constants):
    s_v(t) = K_v + lam * r_v(t)
    K_v    = c_w * mean_w a(w) + a * c_v * z_video + b * c_t * z_scope
    r_v    = centred rank of the decoder's cell posterior (the Temporal Decoder's within-video order)
a(w) = max(z_visual, z_speech) (the window read as stored, `z`); z_scope = the scope (target-group) read, one isolated
Yes/No branch on the Reader's prefix (experiments/20260927_dvd, `z_target`). Integrator forms: `raw` (c_w = c_v =
c_t = 1), `sd` (each term divided by its corpus standard deviation over the full method's videos) and `cdf` (each
term replaced by the corpus empirical distribution function of the full method's videos, so it lies in [0, 1] like
the rank term); all transforms are computed from the reads only. The current method is `--form raw --a 1 --b 0 --lam 1` and reproduces its metrics.json exactly
(`check`).

Decoder: one or more components (COMPONENTS, extra twolevel_r2.py flags on top of the current flags). With several,
the cell posteriors of the components are averaged (`--combine prob` or `logit`) before the centred rank.
Every decode a row needs is looked up in EXISTING or run once into runs/20261114_exp_gates/gate_table/decodes/.

Rows: the 14 rows of the appendix remove-one table plus `w/o Scope Read`, `w/o Duration Posterior` and
`w/o EM Evidence` (listed apart, not counted in the 14; README §9).
`w/ ASR Windows` keeps its original composition: ASR windows cannot enter the time level, so both sides use the
SPVL composition's within-video term (experiments/20260910_spvl/compose.py, rank residual) instead of the decoder,
with the Integrator term of this run on each side. Reader rows use their own reads for z_video and the window reads;
`w/o Global Transcript` and `w/o Global Frames` use the scope read on their own prefix (Slurm 2300); the other reader
rows keep the prefix of the full method and use its scope read.

`--family ns` (README §14): the method without the verdict turn. The full method's reads are the `no_stance` reads;
reader rows use the `ns_<arm>` reads of experiments/20261101_reader_ablation (README §8 there); `w/o Verdict Turn` is
replaced by the apart row `w/ Verdict Turn` (the reads with the verdict turn); with `--decoder em0` the row
`w/ Linear Evidence` is listed apart (without EM it is a slope change, README §13.2). `--skip-missing` leaves out rows
whose reads are not on disk yet and names them under the table.

Metrics: the canonical evaluator's functions (src/eval/evaluate.py `pooled`, `within_video_macro`) on the test split.
Paired bootstrap over videos (videos common to the row and the full method, 1000 draws, seed 0) for every cell.
No label enters any score; labels are read only for the metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.evaluate import pooled, within_video_macro  # noqa: E402

PY = "/home/jehc223/miniconda3/envs/HateVideo/bin/python"
OUT = ROOT / "runs/20261114_exp_gates/gate_table"
DATASETS = ("HateMM", "HateClipSeg", "DeHate")
PART = {"HateMM": "main", "HateClipSeg": "main", "DeHate": "dehate"}
PART_DS = {"main": ["HateMM", "HateClipSeg"], "dehate": ["DeHate"]}
BASE_FLAGS = "--noleak --transform nscore --key raw --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2"
CELL, FPS = 4.0, 4.0

RA = "runs/20261101_reader_ablation"
PR = "runs/20261101_paper_redecode"
DS_ = "runs/20261114_exp_gates/decoder_scan"
READS = {   # read set -> part -> reads directory (predictions.jsonl with z_video and the window reads)
    "full": {"main": "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned", "dehate": "runs/20261031_window_evidence/DeHate/adopted"},
    "rs_old": {"main": "runs/20260926_glr/base_gridA", "dehate": "runs/20261031_window_evidence/DeHate/r6_replay"},
    **{a: {"main": f"{RA}/merged/{a}", "dehate": f"{RA}/DeHate/{a}"}
       for a in ("no_transcript", "no_prefix_frames", "no_isolation", "joint", "no_stance", "asr_windows")},
    # no-verdict-turn family (README §14): read_arms.py ns_<arm>, NAME=ns_<corpus>; ns_merged = HateMM + HateClipSeg
    **{f"ns_{a}": {"main": f"{RA}/ns_merged/ns_{a}", "dehate": f"{RA}/ns_DeHate/ns_{a}"}
       for a in ("no_transcript", "no_prefix_frames", "no_isolation", "joint", "asr_windows", "no_window_evidence")},
}
FULL = "full"       # read set of the full method; `--family ns` sets it to "no_stance"
SCOPE = {   # scope read per prefix
    "full": {"main": "runs/20260927_dvd/reads_main/reads.jsonl", "dehate": "runs/20260927_dvd/reads_dehate/reads.jsonl"},
    **{a: {p: f"runs/20261114_exp_gates/scope_reads/{a}_{p}/reads.jsonl" for p in ("main", "dehate")}
       for a in ("no_transcript", "no_prefix_frames")},
}
COMPONENTS = {   # decoder component -> extra twolevel_r2.py flags
    "m2": "", "kap0.3": "--kappa 0.3", "kap0.5": "--kappa 0.5", "kap0.7": "--kappa 0.7", "fixed80": "--bma-fixed 80",
    "plog": "--bma-prior log", "lin": "--evidence linear", "learned": "--bma-prior learned", "ar1": "--ktemper ar1",
    "em0": "--em-iters 0",      # README §11: the no-EM decoder as the full method's decoder (brief 5 K2)
}
ROW_FLAGS = {"nocoupling": "--nocoupling", "sharedchain": "--sharedchain", "fixed80": "--bma-fixed 80",
             "lin": "--evidence linear", "wprior": "--bma-weights prior", "em0": "--em-iters 0",
             "em300": "--em-iters 300"}     # em300 overrides the em0 component (argparse: last value wins)
# (component, read set, row flag) -> existing decode directory per part (same reads, same flags)
EXISTING = {
    ("m2", "full", ""): {"main": f"{PR}/main/m2", "dehate": f"{PR}/dehate/m2"},
    ("m2", "rs_old", ""): {"main": f"{PR}/main/rs_old", "dehate": f"{PR}/dehate/rs_old"},
    **{("m2", "full", f): {"main": f"{PR}/main/{d}", "dehate": f"{PR}/dehate/{d}"}
       for f, d in (("nocoupling", "nocoupling"), ("sharedchain", "sharedchain"), ("fixed80", "fixed80"),
                    ("lin", "lin_bma_nscore"))},
    **{("m2", a, ""): {"main": f"{RA}/decoded/{a}_main", "dehate": f"{RA}/decoded/{a}_DeHate"}
       for a in ("no_transcript", "no_prefix_frames", "no_isolation", "joint", "no_stance")},
    **{(c, "full", ""): {"main": f"{DS_}/main/{t}", "dehate": f"{DS_}/dehate/{t}"}
       for c, t in (("kap0.3", "kap0.3"), ("kap0.5", "kap0.5"), ("kap0.7", "kap0.7"), ("plog", "plog"),
                    ("learned", "learned"), ("ar1", "ar1"))},
    ("fixed80", "full", ""): {"main": f"{PR}/main/fixed80", "dehate": f"{PR}/dehate/fixed80"},
    ("lin", "full", ""): {"main": f"{PR}/main/lin_bma_nscore", "dehate": f"{PR}/dehate/lin_bma_nscore"},
    **{("m2", "full", f): {p: f"runs/20261114_exp_gates/decoder_d/{p}/{f}" for p in ("main", "dehate")}
       for f in ("wprior", "em0")},     # README §9
    # README §11, decoder em0: the full decode is decoder_d em0; `w/ EM Evidence` = the current m2 decode; linear
    # evidence has no EM (twolevel_r2.py skips em() under --evidence linear), so its decode is unchanged.
    ("em0", "full", ""): {p: f"runs/20261114_exp_gates/decoder_d/{p}/em0" for p in ("main", "dehate")},
    ("em0", "full", "em300"): {"main": f"{PR}/main/m2", "dehate": f"{PR}/dehate/m2"},
    ("em0", "full", "lin"): {"main": f"{PR}/main/lin_bma_nscore", "dehate": f"{PR}/dehate/lin_bma_nscore"},
}
COMPOSE = {"fixed": {"main": f"{RA}/compose/fixed_main", "dehate": f"{RA}/compose/fixed_DeHate"},
           "asr": {"main": f"{RA}/compose/asr_main", "dehate": f"{RA}/compose/asr_DeHate"}}
COMPOSE_NS = {"fixed": {"main": f"{RA}/compose/ns_fixed_main", "dehate": f"{RA}/compose/ns_fixed_DeHate"},
              "asr": {"main": f"{RA}/compose/ns_asr_main", "dehate": f"{RA}/compose/ns_asr_DeHate"}}
# name, group, read set, scope prefix, decoder row flag, Integrator change
ROWS = [
    ("w/o Global Transcript", "Contextual Reader", "no_transcript", "no_transcript", "", None),
    ("w/o Global Frames", "Contextual Reader", "no_prefix_frames", "no_prefix_frames", "", None),
    ("w/o Query Isolation", "Contextual Reader", "no_isolation", "full", "", None),
    ("w/ Joint Query", "Contextual Reader", "joint", "full", "", None),
    ("w/o Verdict Turn", "Contextual Reader", "no_stance", "full", "", None),
    ("w/o Window Evidence", "Contextual Reader", "rs_old", "full", "", None),
    ("w/ ASR Windows", "Contextual Reader", "asr_windows", "full", "compose", None),
    ("w/o Temporal Coupling", "Temporal Decoder", "full", "full", "nocoupling", None),
    ("w/ Shared Chain", "Temporal Decoder", "full", "full", "sharedchain", None),
    ("w/ Fixed Mean Duration", "Temporal Decoder", "full", "full", "fixed80", None),
    ("w/ Linear Evidence", "Temporal Decoder", "full", "full", "lin", None),
    ("w/o Video-Level Term", "Hierarchical Integrator", "full", "full", "", "nokey"),
    ("w/o Window Aggregate", "Hierarchical Integrator", "full", "full", "", "no_window"),
    ("w/ Mean over Modalities", "Hierarchical Integrator", "full", "full", "", "mean_mod"),
    ("w/o Scope Read", "extra (not counted)", "full", "full", "", "no_scope"),
    ("w/o Duration Posterior", "extra (not counted)", "full", "full", "wprior", None),
    ("w/o EM Evidence", "extra (not counted)", "full", "full", "em0", None),
]
# With --decoder em0 the apart row `w/o EM Evidence` is replaced by `w/ EM Evidence` (README §11).
ROWS_EM0 = [r for r in ROWS if r[0] != "w/o EM Evidence"] + [
    ("w/ EM Evidence", "extra (not counted)", "full", "full", "em300", None)]


def ns_rows(rows, em0):
    """Rows of the no-verdict-turn family (README §14)."""
    out = []
    for name, grp, rs, sp, rflag, integ in rows:
        if name == "w/o Verdict Turn":
            out.append(("w/ Verdict Turn", "extra (not counted)", "full", sp, rflag, integ))
            continue
        if em0 and name == "w/ Linear Evidence":
            name, grp = "w/ Linear Evidence (slope 1)", "extra (not counted)"
        rs = {"full": "no_stance", "rs_old": "ns_no_window_evidence"}.get(rs, "ns_" + rs)
        out.append((name, grp, rs, sp, rflag, integ))
    return out


NAIVE = "runs/20261113_lf_baselines/naive/vlm/{ds}/predictions.jsonl"
GATE1 = "runs/20261113_lf_baselines/gate/gate.json"     # best published label-free cell per corpus and metric
CURRENT = {"HateMM": f"{PR}/main/m2", "HateClipSeg": f"{PR}/main/m2", "DeHate": f"{PR}/dehate/m2"}


def jl(path):
    with open(ROOT / path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def by_video(path, ds):
    return {r["video_id"]: r for r in jl(path) if r["dataset"] == ds and not r.get("error")}


def gt(ds):
    z = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): np.asarray(y, np.int8) for v, s, y in zip(z["video_ids"], z["split"], z["y4"]) if str(s) == "test"}


ORD = {"main": "runs/20261114_exp_gates/ordinal_reads/ord_full_main/reads.jsonl",
       "dehate": "runs/20261114_exp_gates/ordinal_reads/ord_full_dehate/reads.jsonl"}


def ordinal_terms(m):
    """(z_ord, logit(s_raw), z_mean) from the ten ordinal margins (README §10): p = sigmoid(m), non-increasing
    projection over tau (pool adjacent violators), s = mean; z_ord = logit(clip(s, 1e-4, 1 - 1e-4))."""
    from sklearn.isotonic import IsotonicRegression
    m = np.asarray(m, float)
    p = 1.0 / (1.0 + np.exp(-m))
    q = IsotonicRegression(increasing=False).fit_transform(np.arange(len(p)), p)
    lg = lambda x: float(np.log(x) - np.log1p(-x))
    return lg(np.clip(q.mean(), 1e-4, 1 - 1e-4)), lg(np.clip(p.mean(), 1e-4, 1 - 1e-4)), float(m.mean())


def z_ord(m):
    return ordinal_terms(m)[0]


def centered_rank(v):     # experiments/20260926_twolevel/twolevel.py centered_rank (copied; no cross-experiment import)
    v = np.asarray(v, float)
    if len(v) <= 1 or np.ptp(v) <= 1e-12:
        return np.zeros_like(v)
    r = (rankdata(v, method="average") - 0.5) / len(v) - 0.5
    return r - r.mean()


def to_frames(cell_vals, L):  # twolevel.py to_frames (copied)
    centers = (np.arange(L) + 0.5) / FPS
    idx = np.clip((centers // CELL).astype(int), 0, len(cell_vals) - 1)
    return np.asarray(cell_vals, float)[idx]


# ------------------------------------------------------------------------------------------------ decodes

def slug(comp, reads, rflag):
    return "__".join(x for x in (comp, reads, rflag) if x)


def decode_dir(comp, reads, rflag, part):
    key = (comp, reads, rflag)
    if key in EXISTING:
        return EXISTING[key][part]
    return str(OUT.relative_to(ROOT) / "decodes" / part / slug(comp, reads, rflag))


def run_decode(comp, reads, rflag, part):
    d = ROOT / decode_dir(comp, reads, rflag, part)
    if (d / "predictions.jsonl").exists() and (d / "metrics.json").exists():
        return str(d), "exists"
    flags = " ".join(x for x in (BASE_FLAGS, COMPONENTS[comp], ROW_FLAGS.get(rflag, "")) if x)
    cmd = [PY, "experiments/20260926_twolevel/twolevel_r2.py", *flags.split(), "--run", READS[reads][part],
           "--datasets", *PART_DS[part], "--out-root", str(d.parent), "--tag", d.name]
    d.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    with open(str(d) + ".out", "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"FAILED decode {d} (log {d}.out)")
    return str(d), "decoded"


def needed_decodes(comps):
    out = set()
    for comp in comps:
        for name, grp, reads, scope, rflag, integ in ROWS:
            if rflag == "compose":
                continue
            for part in sorted({PART[ds] for ds in DATASETS}):
                out.add((comp, reads, "" if integ else rflag, part))
    return sorted(out)


# ------------------------------------------------------------------------------------------------ rows

class Store:
    def __init__(self, comps, combine, taxonomy=False, scope_src="protected"):
        self.comps, self.combine, self.taxonomy, self.scope_src = comps, combine, taxonomy, scope_src
        self.gt = {ds: gt(ds) for ds in DATASETS}
        self._c = {}

    def get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def reads(self, rs, ds):
        return self.get(("reads", rs, ds), lambda: by_video(f"{READS[rs][PART[ds]]}/predictions.jsonl", ds))

    def scope(self, sp, ds):
        """video -> scope read. Default: the protected-group read on prefix `sp`. taxonomy: HateClipSeg's own-taxonomy
        read (README §6) on HateClipSeg. ordinal: z_ord of the video-level ordinal read (README §10), full prefix only,
        so the two ablated-prefix rows borrow the full-prefix value."""
        if self.scope_src == "ordinal":
            path = ORD[PART[ds]]
            return self.get(("scope", "ord", ds), lambda: {v: z_ord(r["m_ord"]) for v, r in by_video(path, ds).items()})
        if self.taxonomy and ds == "HateClipSeg":
            path = f"runs/20261114_exp_gates/scope_reads/tax_{sp}_hcs/reads.jsonl"
        else:
            path = SCOPE[sp][PART[ds]]
        return self.get(("scope", sp, ds, self.taxonomy), lambda: {v: r["z_target"] for v, r in by_video(path, ds).items()})

    def rank(self, rs, rflag, ds):
        """video -> centred rank of the frame curve (decoder, or the composition for ASR windows)."""
        def build():
            if rflag == "compose":
                side = "asr" if rs.endswith("asr_windows") else "fixed"
                comp = COMPOSE_NS if FULL == "no_stance" else COMPOSE
                P = by_video(f"{comp[side][PART[ds]]}/predictions_izv_plus_mean_rrank.jsonl", ds)
                return {v: np.asarray(r["score_curve"], float) - np.mean(r["score_curve"]) for v, r in P.items()}
            per = []
            for comp in self.comps:
                P = by_video(decode_dir(comp, rs, rflag, PART[ds]) + "/predictions.jsonl", ds)
                per.append({v: (np.asarray(r["extra"]["cell_prob"], float), len(r["score_curve"])) for v, r in P.items()})
            vids = set.intersection(*(set(p) for p in per))
            out = {}
            for v in vids:
                ps = [np.clip(p[v][0], 1e-12, 1 - 1e-12) for p in per]
                if self.combine == "logit":
                    lg = np.mean([np.log(p) - np.log1p(-p) for p in ps], 0)
                else:
                    p = np.clip(np.mean(ps, 0), 1e-12, 1 - 1e-12); lg = np.log(p) - np.log1p(-p)
                out[v] = centered_rank(to_frames(lg, per[0][v][1]))
            return out
        return self.get(("rank", rs, rflag, ds), build)

    def terms(self, rs, sp, ds):
        """video -> (z_video, max-window mean, mean-over-modalities window mean, scope read)."""
        def build():
            R, S = self.reads(rs, ds), self.scope(sp, ds)
            out = {}
            for v, r in R.items():
                ws = r["extra"]["windows"]
                mw = float(np.mean([w["z"] for w in ws])) if ws else 0.0
                mm = [np.mean([w[k] for k in ("z_visual", "z_speech") if k in w]) if ("z_visual" in w or "z_speech" in w)
                      else w["z"] for w in ws]
                out[v] = (float(r["extra"]["z_video"]), mw, float(np.mean(mm)) if ws else 0.0, S.get(v, np.nan),
                          np.array([w["z"] for w in ws], float), np.array(mm, float))
            return out
        return self.get(("terms", rs, sp, ds), build)


def scales(store, ds, form):
    """Per-term transforms (window aggregate, z_video, scope read), fitted on the full method's reads of the corpus
    (no label): raw = identity; sd = divided by the corpus s.d.; cdf = the corpus empirical distribution function
    (fraction of the full method's videos below the value), so every term lies in [0, 1] like the rank term."""
    if form == "raw":
        return (lambda x: x,) * 3
    T = np.array([t[:4] for t in store.terms(FULL, "full", ds).values()], float)   # (zv, mw, mm, zt)
    cols = (T[:, 1], T[:, 0], T[:, 3][np.isfinite(T[:, 3])])
    if form == "sd":
        return tuple((lambda x, c=1.0 / c_.std(): c * x) for c_ in cols)
    return tuple((lambda x, c=np.sort(c_): np.searchsorted(c, x, side="right") / len(c)) for c_ in cols)


FORMS = ("F1", "F2", "F3", "F4", "F5")


def f_sigmas(store, ds):
    """sigma(mw), sigma(zT), sigma(A) over all videos of the corpus, from the full method's reads (no label); the
    same constants are used for every row (README §7)."""
    T = store.terms(FULL, "full", ds)
    mw = np.array([t[1] for t in T.values()]); zt = np.array([t[3] for t in T.values()])
    A = np.array([np.mean(np.minimum(t[4], t[3])) if len(t[4]) else 0.0 for t in T.values()])
    return mw.std(), zt.std(), A.std()


def f_key(form, zv, mw, mm, zt, wz, wm, integ, sig):
    """K_v of forms F1-F5 (main-session brief 2026-11-15) and their four Integrator rows."""
    s_mw, s_zt, s_A = sig
    if integ == "nokey":
        return 0.0
    if form == "F1":
        if integ == "no_window":
            return 0.5 * zt
        if integ == "no_scope":
            return mw + 0.5 * zv
        return (mm if integ == "mean_mod" else mw) + 0.5 * zv + 0.5 * zt
    if form in ("F2", "F4"):
        if integ == "no_window":
            return zt / s_zt
        if integ == "no_scope":
            return mw / s_mw
        return (mm if integ == "mean_mod" else mw) / s_mw + zt / s_zt
    # F3 / F5: A = mean over windows of min(z_w, zT)
    if integ == "no_window":
        return zt / s_zt
    if integ == "no_scope":
        return mw / s_mw
    w = wm if integ == "mean_mod" else wz
    A = float(np.mean(np.minimum(w, zt))) if len(w) else 0.0
    return A / s_A + zt / s_zt


def row_scores(store, ds, row, cfg):
    name, grp, rs, sp, rflag, integ = row
    if cfg["form"] in FORMS:
        rk = store.rank(rs, "" if integ else rflag, ds)
        T = store.terms(rs, sp, ds)
        sig = f_sigmas(store, ds)
        out = {}
        for v, r in rk.items():
            if v not in T:
                continue
            zv, mw, mm, zt, wz, wm = T[v]
            if integ not in ("nokey", "no_scope") and not np.isfinite(zt):
                raise RuntimeError(f"missing scope read {ds} {v} ({sp})")
            out[v] = f_key(cfg["form"], zv, mw, mm, zt, wz, wm, integ, sig) + cfg["lam"] * r
        return out
    rk = store.rank(rs, "" if integ else rflag, ds)
    T = store.terms(rs, sp, ds)
    cw, cv, ct = scales(store, ds, cfg["form"])
    a, b, lam = cfg["a"], cfg["b"], cfg["lam"]
    out = {}
    for v, r in rk.items():
        if v not in T:
            continue
        zv, mw, mm, zt = T[v][:4]
        if b and not np.isfinite(zt):
            raise RuntimeError(f"missing scope read {ds} {v} ({sp})")
        if integ == "nokey":
            K = 0.0
        else:
            K = (0.0 if integ == "no_window" else cw(mm if integ == "mean_mod" else mw)) + a * cv(zv) + \
                (0.0 if integ == "no_scope" or not b else b * ct(zt))
        out[v] = K + lam * r
    return out


def metrics(y, s):
    p, w = pooled(y, s), within_video_macro(y, s)
    return np.array([p["frame_ROC_AUC"], p["frame_PR_AUC"], w["within_video_macro_ROC_AUC"]])


# ------------------------------------------------------------------------------------------------ bootstrap

class Boot:
    """Paired bootstrap over videos with fast weighted AUC / AP (sklearn definitions, ties grouped)."""

    def __init__(self, y, sa, sb, vids, n=1000, seed=0):
        self.vids = vids
        self.n, self.rng = n, np.random.default_rng(seed)
        ys, ida, sas, sbs = [], [], [], []
        self.w_ok = []
        for i, v in enumerate(vids):
            m = min(len(y[v]), len(sa[v]), len(sb[v]))
            ys.append(y[v][:m]); sas.append(sa[v][:m]); sbs.append(sb[v][:m]); ida.append(np.full(m, i))
        self.y = np.concatenate(ys).astype(float); self.vid = np.concatenate(ida)
        self.prep = [self._prep(np.concatenate(sas)), self._prep(np.concatenate(sbs))]
        self.win = [np.array([self._auc1(yy, ss) for yy, ss in zip(ys, S)]) for S in (sas, sbs)]

    @staticmethod
    def _auc1(y, s):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan

    def _prep(self, s):
        order = np.argsort(-s, kind="mergesort")
        ss = s[order]
        grp = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])])
        g = np.empty_like(grp); g[order] = grp
        return g, grp[-1] + 1

    def _pooled(self, prep, wf):
        g, ng = prep
        tp = np.bincount(g, wf * self.y, ng); fp = np.bincount(g, wf * (1 - self.y), ng)
        P, N = tp.sum(), fp.sum()
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        auc = (np.sum(tp * (N - cfp)) + 0.5 * np.sum(tp * fp)) / (P * N)       # P(score_pos > score_neg) + ties/2
        prec = ctp / np.maximum(ctp + cfp, 1e-300)
        ap = np.sum(tp / P * prec)
        return auc, ap

    def run(self):
        out = []
        nv = len(self.vids)
        for _ in range(self.n):
            c = np.bincount(self.rng.integers(0, nv, nv), minlength=nv).astype(float)
            wf = c[self.vid]
            if (wf * self.y).sum() == 0 or (wf * (1 - self.y)).sum() == 0:
                continue
            d = []
            for k in range(2):
                auc, ap = self._pooled(self.prep[k], wf)
                ok = np.isfinite(self.win[k]) & (c > 0)
                wm = np.sum(c[ok] * self.win[k][ok]) / c[ok].sum()
                d.append((auc, ap, wm))
            out.append(np.array(d[0]) - np.array(d[1]))
        return np.percentile(np.array(out), [2.5, 97.5], axis=0)


# ------------------------------------------------------------------------------------------------ main

def config_from(args):
    return {"form": args.form, "a": args.a, "b": args.b, "lam": args.lam}


def evaluate(store, cfg, nboot):
    full_row = ("full", "", FULL, "full", "", None)
    res = {}
    for ds in DATASETS:
        y = store.gt[ds]
        sf = row_scores(store, ds, full_row, cfg)
        mf = metrics(y, sf)
        res[ds] = {"full": {"m": mf.tolist()}}
        for row in ROWS:
            name = row[0]
            if row[4] == "compose":        # both sides under the composition
                base = row_scores(store, ds, ("", "", FULL, "full", "compose", None), cfg)
            else:
                base = sf
            s = row_scores(store, ds, row, cfg)
            m, mb = metrics(y, s), metrics(y, base)
            vids = sorted(set(s) & set(base) & set(y))
            ci = Boot(y, s, base, vids, n=nboot).run() if nboot else np.full((2, 3), np.nan)
            res[ds][name] = {"m": m.tolist(), "base": mb.tolist(), "d": (m - mb).tolist(), "ci": ci.tolist()}
    return res


def checks(store, cfg, res):
    out = {}
    for ds in DATASETS:
        y = store.gt[ds]
        mf = np.array(res[ds]["full"]["m"])
        N = {v: np.asarray(r["score_curve"], float) for v, r in by_video(NAIVE.format(ds=ds), ds).items()}
        mn = metrics(y, N)
        C = {v: np.asarray(r["score_curve"], float) for v, r in by_video(CURRENT[ds] + "/predictions.jsonl", ds).items()}
        mc = metrics(y, C)
        out[ds] = {"naive": mn.tolist(), "current": mc.tolist(), "vs_naive": (mf - mn).tolist(),
                   "vs_current": (mf - mc).tolist()}
    if "DeHate" not in DATASETS:
        return out
    g = json.load(open(ROOT / GATE1))
    for sect in ("A", "C"):          # A: all published rows; C: peer-reviewed only
        best = {(c["dataset"], c["metric"]): (max(c["best"], c["naive"]) / 100, c["row"] if c["best"] >= c["naive"]
                                              else "naive Qwen3-VL-8B") for c in g[sect]["cells"]}
        mf = res["DeHate"]["full"]["m"]
        out["DeHate"]["vs_best_" + sect] = [(mf[k] - best[("DeHate", m)][0], best[("DeHate", m)][1])
                                            for k, m in enumerate(("pooled ROC", "pooled PR"))]
    return out


def render(res, chk, cfg, title):
    shown = {"form": cfg["form"], "lam": cfg["lam"]} if cfg["form"] in FORMS else cfg
    L = [f"## {title}", "", f"Integrator {shown}; decoder {res['_decoder']}. Cells: change against the full method in "
         "points (ROC / PR / within); `!` = not below (> −0.05); `~` = 95 % paired bootstrap interval contains 0; "
         "`=` = equal by construction (exempt).", ""]
    L.append("| row | " + " | ".join(f"{ds} ROC / PR / W" for ds in DATASETS) + " | not below |")
    L.append("|---|---|---|---|---|")
    L.append("| full (absolute) | " + " | ".join(" / ".join(f"{100 * x:.1f}" for x in res[ds]["full"]["m"])
                                         for ds in DATASETS) + " | |")
    total, extra, rows_ok, total_rw = 0, {}, 0, 0
    for row in ROWS:
        name, grp, _, _, rflag, integ = row
        cells, nb, nb_rw = [], 0, 0
        for ds in DATASETS:
            r = res[ds][name]
            parts = []
            for k in range(3):
                if integ and k == 2:
                    parts.append("="); continue
                d = 100 * r["d"][k]
                lo, hi = (100 * r["ci"][0][k], 100 * r["ci"][1][k])
                flag = ""
                if d > -0.05:
                    flag += "!"; nb += 1; nb_rw += k != 1
                if np.isfinite(lo) and lo <= 0 <= hi:
                    flag += "~"
                parts.append(f"{d:+.1f}{flag}")
            cells.append(" / ".join(parts))
        if grp.startswith("extra"):
            extra[name] = nb
        else:
            total += nb; rows_ok += nb == 0; total_rw += nb_rw
        L.append(f"| {name}{' (compose, both sides)' if rflag == 'compose' else ''} | " + " | ".join(cells) + f" | {nb} |")
    if res.get("_note"):
        L += ["", res["_note"]]
    n_rows = sum(not r[1].startswith("extra") for r in ROWS)
    L += ["", f"Cells not below the full method, {n_rows} rows: **{total}** (rows with every cell below: {rows_ok} of {n_rows}; "
          f"pooled ROC and within columns only, as printed in the paper's appendix table: {total_rw}); "
          + "; ".join(f"`{k}`: {v}" for k, v in extra.items()) + ".", ""]
    cur = "Full − current method: " + "; ".join(
        f"{ds} " + " / ".join(f"{100 * x:+.2f}" for x in chk[ds]["vs_current"]) for ds in DATASETS)
    if "DeHate" not in DATASETS:
        L.append(cur + " (limits on HateMM and HateClipSeg: −0.5 pooled, −1.0 within).")
        return "\n".join(L) + "\n", total
    vb = lambda sect: " / ".join(f"{100 * d:+.1f} ({r})" for d, r in chk["DeHate"]["vs_best_" + sect])
    L.append("Gate 2 checks (points; need ≥ +2 each on DeHate): full − naive Qwen3-VL-8B row ROC / PR = " +
             " / ".join(f"{100 * x:+.1f}" for x in chk["DeHate"]["vs_naive"][:2]) +
             f"; full − best label-free row, all published rows: {vb('A')}; peer-reviewed only: {vb('C')}. "
             "Full − current method: " +
             "; ".join(f"{ds} " + " / ".join(f"{100 * x:+.2f}" for x in chk[ds]["vs_current"])
                       for ds in ("HateMM", "HateClipSeg")) + " (limits −0.5 pooled, −1.0 within).")
    return "\n".join(L) + "\n", total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoder", default="m2", help="component, or comma list of components (averaged)")
    ap.add_argument("--combine", choices=("prob", "logit"), default="prob")
    ap.add_argument("--form", choices=("raw", "sd", "cdf") + FORMS, default="raw",
                    help="F1-F5: fixed forms of README §7 (a, b ignored; lam from --lam)")
    ap.add_argument("--a", type=float, default=1.0, help="weight of z_video")
    ap.add_argument("--b", type=float, default=0.0, help="weight of the scope read")
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the centred rank")
    ap.add_argument("--hcs-taxonomy", action="store_true", help="HateClipSeg scope read = the taxonomy read (README §6)")
    ap.add_argument("--scope", choices=("protected", "ordinal"), default="protected",
                    help="ordinal: the scope term is z_ord (README §10); ablated-prefix rows borrow the full-prefix z_ord")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--name", default=None)
    ap.add_argument("--check", action="store_true", help="plumbing: current method must equal its metrics.json")
    ap.add_argument("--family", choices=("current", "ns"), default="current",
                    help="ns: the method without the verdict turn (README §14)")
    ap.add_argument("--skip-missing", action="store_true", help="leave out rows whose reads are not on disk yet")
    ap.add_argument("--part", choices=("all", "main"), default="all", help="main: HateMM and HateClipSeg only")
    args = ap.parse_args()
    comps = args.decoder.split(",")
    global ROWS, FULL, DATASETS
    if args.part == "main":
        DATASETS = ("HateMM", "HateClipSeg")
    parts = sorted({PART[ds] for ds in DATASETS})
    if comps == ["em0"]:
        ROWS = ROWS_EM0
    skipped = []
    if args.family == "ns":
        FULL = "no_stance"
        ROWS = ns_rows(ROWS, comps == ["em0"])
    if args.skip_missing:
        def have(r):
            if r[4] == "compose":
                comp = COMPOSE_NS if FULL == "no_stance" else COMPOSE
                return all((ROOT / comp[sd][pt] / "predictions_izv_plus_mean_rrank.jsonl").exists()
                           for sd in ("fixed", "asr") for pt in parts)
            return all((ROOT / READS[r[2]][pt] / "predictions.jsonl").exists() for pt in parts)
        skipped = [r[0] for r in ROWS if not have(r)]
        ROWS = [r for r in ROWS if have(r)]
    todo = needed_decodes(comps)
    with ThreadPoolExecutor(args.jobs) as ex:
        for d, st in ex.map(lambda t: run_decode(*t), todo):
            if st == "decoded":
                print("decoded", d, flush=True)
    store = Store(comps, args.combine, taxonomy=args.form in ("F4", "F5") or args.hcs_taxonomy, scope_src=args.scope)
    cfg = config_from(args)
    if args.check:
        for ds in DATASETS:
            s = row_scores(store, ds, ("full", "", "full", "full", "", None), {"form": "raw", "a": 1, "b": 0, "lam": 1})
            m = metrics(store.gt[ds], s)
            ref = [r for r in json.load(open(ROOT / CURRENT[ds] / "metrics.json"))["per_dataset"] if r["dataset"] == ds][0]
            mr = np.array([ref["frame_ROC_AUC"], ref["frame_PR_AUC"], ref["within_video_macro_ROC_AUC"]])
            print(ds, "max |diff| vs current metrics.json", float(np.max(np.abs(m - mr))))
        return
    res = evaluate(store, cfg, args.boot)
    res["_decoder"] = f"{args.decoder} ({args.combine})" if len(comps) > 1 else args.decoder
    notes = []
    if args.scope == "ordinal":
        notes.append("Scope term = z_ord (README §10). `w/o Global Transcript` and `w/o Global Frames` use the full-prefix "
                     "z_ord (borrowed; no ordinal read on the ablated prefixes).")
    if args.family == "ns":
        notes.append("Family ns (README §14): no verdict turn in the full method (reads `no_stance`) or in any reader "
                     "row (`ns_<arm>` reads).")
    if skipped:
        notes.append("Rows left out (reads not on disk yet): " + ", ".join(skipped) + ".")
    if store.taxonomy:
        notes.append("HateClipSeg scope read = the taxonomy read on each row's own prefix (README §6); HateMM and DeHate "
                     "keep the protected-group read.")
    res["_note"] = " ".join(notes)
    chk = checks(store, cfg, res)
    name = args.name or (("ns__" if args.family == "ns" else "") + (
        f"{args.decoder.replace(',', '+')}_{args.combine}__{args.form}_l{args.lam:g}" if args.form in FORMS
        else f"{args.decoder.replace(',', '+')}_{args.combine}__{args.form}_a{args.a:g}_b{args.b:g}_l{args.lam:g}"))
    md, total = render(res, chk, cfg, name)
    d = OUT / "tables"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.md").write_text(md)
    (d / f"{name}.json").write_text(json.dumps({"config": cfg, "decoder": res["_decoder"], "res": res,
                                                "checks": chk, "not_below_14": total}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
