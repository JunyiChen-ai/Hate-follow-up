"""Test-fit GMM band selection: same Bayes-rate rule, but GMM fit on
TEST scores (label-free) instead of train. This is still label-free
because GMM fitting doesn't use labels.
"""
import json, os, sys, numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
sys.path.insert(0, os.path.join(_HERE, "..", "naive_baseline"))

from quick_eval_all import load_scores_file  # noqa
from data_utils import SKIP_VIDEOS  # noqa

ROOT = "/data/jehc223/EMNLP2"
OUT = os.path.join(ROOT, "results", "boundary_rescue")
DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]

def to_logit(s, eps=1e-6):
    s = np.clip(np.asarray(s, dtype=float), eps, 1-eps)
    return np.log(s/(1-s))

def gmm_fit(s):
    z = to_logit(s).reshape(-1,1)
    g = GaussianMixture(n_components=2, random_state=42, max_iter=200).fit(z)
    return g, int(np.argmax(g.means_.flatten()))

def gmm_bayes_err(g):
    m = g.means_.flatten(); sd = np.sqrt(g.covariances_.flatten()); w = g.weights_.flatten()
    sp = 6*sd.max(); lo = m.min()-sp; hi = m.max()+sp
    z = np.linspace(lo, hi, 8192)
    p0 = w[0]*norm.pdf(z, m[0], sd[0]); p1 = w[1]*norm.pdf(z, m[1], sd[1])
    return float(np.trapz(np.minimum(p0, p1), z))

v2b = json.load(open(os.path.join(OUT, "v2_baseline.json")))
for ds in DATASETS:
    info = v2b[ds]
    test_path = info["test_score_path"]
    test_scores_dict = load_scores_file(test_path)
    ids = list(test_scores_dict.keys())
    scores = np.array([test_scores_dict[v] for v in ids], dtype=float)
    g, hi = gmm_fit(scores)
    eb = gmm_bayes_err(g)
    z = to_logit(scores).reshape(-1,1)
    post = g.predict_proba(z)[:, hi]
    err = np.minimum(post, 1-post)
    skip = SKIP_VIDEOS.get(ds, set())
    base_rows = [json.loads(l) for l in open(os.path.join(OUT, ds, "baseline_preds_v2.jsonl"))]
    id_to_base = {r['video_id']: r for r in base_rows}
    out = []
    for i, v in enumerate(ids):
        if v in skip or v not in id_to_base:
            continue
        if err[i] > eb:
            r = id_to_base[v]
            out.append({
                'video_id': v,
                'score': float(scores[i]),
                'posterior_hi': float(post[i]),
                'err_contribution': float(err[i]),
                'threshold': info['threshold'],
                'pred_baseline': int(r['pred_baseline']),
                'side': 'above' if r['pred_baseline']==1 else 'below',
                'fit_source': 'test',
            })
    out_path = os.path.join(OUT, ds, "candidates_bayes_band_testfit.jsonl")
    with open(out_path, "w") as f:
        for c in out: f.write(json.dumps(c, ensure_ascii=False)+"\n")
    print(f"{ds}: E_bayes(test-fit)={eb:.4f} |band|={len(out)}/{len(ids)} → {out_path}")
