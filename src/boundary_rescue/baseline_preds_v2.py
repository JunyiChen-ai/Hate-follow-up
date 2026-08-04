"""
Task 0 (v2): Mixed TR/TF baseline for the v2 boundary-rescue pipeline.

Per-dataset threshold protocol (asymmetric by user directive
2026-04-14: HateMM has no 2B train scores, so it stays on TF for
this pilot; EN/ZH switch to TR):

  MHClip_EN  : TR-Otsu      (fit on train 2B scores, 550 rows)
  MHClip_ZH  : TR-GMM       (fit on train 2B scores, 579 rows)
  HateMM     : TF-li_lee    (fit on test 2B scores, 215 rows)

Apply each threshold to the test 2B scores, write
`results/boundary_rescue/{dataset}/baseline_preds_v2.jsonl`, pin the
strict-beat targets to `v2_baseline.json`.

2026-04-19: generalised to accept --model-tag <slug> (read scores from
`results/holistic_<slug>/` instead of `results/holistic_2b/`) and
--criterion {otsu,gmm,li_lee,protocol} (override the per-dataset
protocol criterion). Default behaviour (no flags) is byte-identical
to the 2B pinned run.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
sys.path.insert(0, os.path.join(_HERE, "..", "naive_baseline"))

from thresholds import otsu_threshold, gmm_threshold, li_lee_threshold  # noqa: E402
from quick_eval_all import load_scores_file  # noqa: E402
from data_utils import load_annotations, SKIP_VIDEOS  # noqa: E402
from eval_generative_predictions import collapse_label  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "boundary_rescue")

# Per-dataset threshold protocol.
#   "fit_source": where the threshold criterion is fit (train or test)
#   "criterion":  name + callable
PROTOCOL = {
    "MHClip_EN": {
        "fit_source": "train",
        "criterion_name": "otsu",
        "criterion_fn": otsu_threshold,
        "protocol": "TR-Otsu",
    },
    "MHClip_ZH": {
        "fit_source": "train",
        "criterion_name": "gmm",
        "criterion_fn": gmm_threshold,
        "protocol": "TR-GMM",
    },
    "HateMM": {
        "fit_source": "test",
        "criterion_name": "li_lee",
        "criterion_fn": li_lee_threshold,
        "protocol": "TF-li_lee",
    },
    "ImpliHateVid": {
        "fit_source": "train",
        "criterion_name": "gmm",
        "criterion_fn": gmm_threshold,
        "protocol": "TR-GMM",
    },
}

def _train_score_path(slug, dataset):
    base = os.path.join(PROJECT_ROOT, "results", f"holistic_{slug}", dataset, "train_binary.jsonl")
    return base


def _test_score_path(slug, dataset):
    # The 2B ZH test file has a historical _prerepro suffix; other slugs use
    # plain test_binary.jsonl.
    if slug == "2b" and dataset == "MHClip_ZH":
        return os.path.join(PROJECT_ROOT, "results", "holistic_2b",
                            "MHClip_ZH", "test_binary.jsonl.prerepro_20260413")
    return os.path.join(PROJECT_ROOT, "results", f"holistic_{slug}", dataset, "test_binary.jsonl")


def _build_score_files(slug):
    return (
        # train score files (HateMM absent)
        {ds: _train_score_path(slug, ds)
         for ds in ["MHClip_EN", "MHClip_ZH", "ImpliHateVid"]},
        # test score files (all 4)
        {ds: _test_score_path(slug, ds)
         for ds in ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]},
    )


# Default: legacy 2B paths (backwards compatible)
TRAIN_SCORE_FILES, TEST_SCORE_FILES = _build_score_files("2b")


def dataset_metrics(video_ids, preds, dataset):
    ann = load_annotations(dataset)
    skip = SKIP_VIDEOS.get(dataset, set())
    ys_true, ys_pred = [], []
    for vid, p in zip(video_ids, preds):
        if vid in skip:
            continue
        if vid not in ann:
            continue
        y = collapse_label(dataset, ann[vid]["label"])
        ys_true.append(y)
        ys_pred.append(int(p))
    n = len(ys_true)
    tp = sum(1 for p, y in zip(ys_pred, ys_true) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(ys_pred, ys_true) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(ys_pred, ys_true) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(ys_pred, ys_true) if p == 0 and y == 0)
    acc = (tp + tn) / n if n else 0.0
    p_pos = tp / (tp + fp) if (tp + fp) else 0.0
    r_pos = tp / (tp + fn) if (tp + fn) else 0.0
    f_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) else 0.0
    p_neg = tn / (tn + fn) if (tn + fn) else 0.0
    r_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) else 0.0
    mf = (f_pos + f_neg) / 2
    return {"n": n, "acc": acc, "mf1": mf, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _ordered_test_ids(test_path):
    """Preserve file order for downstream determinism."""
    order, seen = [], set()
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            vid = r.get("video_id")
            if vid is None or vid in seen:
                continue
            seen.add(vid)
            order.append(vid)
    return order


def process(dataset, slug="2b", criterion_override=None, suffix=""):
    """Run the baseline threshold pipeline for one dataset.

    slug              : stage-1 MLLM tag (e.g. '2b', 'qwen2.5-vl-7b').
    criterion_override: one of 'otsu','gmm','li_lee' to override the
                        per-dataset protocol criterion. None = protocol.
    suffix            : appended to output filename
                        (`baseline_preds_v2{suffix}.jsonl`).
    """
    global TRAIN_SCORE_FILES, TEST_SCORE_FILES
    TRAIN_SCORE_FILES, TEST_SCORE_FILES = _build_score_files(slug)

    proto = dict(PROTOCOL[dataset])
    if criterion_override:
        from thresholds import otsu_threshold, gmm_threshold, li_lee_threshold
        fn_map = {"otsu": otsu_threshold, "gmm": gmm_threshold,
                  "li_lee": li_lee_threshold}
        proto["criterion_name"] = criterion_override
        proto["criterion_fn"] = fn_map[criterion_override]
        proto["protocol"] = f"{'TF' if proto['fit_source']=='test' else 'TR'}-{criterion_override}"

    test_path = TEST_SCORE_FILES[dataset]
    if not os.path.isfile(test_path):
        raise FileNotFoundError(test_path)
    test_score_dict = load_scores_file(test_path)
    test_ids = [v for v in _ordered_test_ids(test_path) if v in test_score_dict]
    test_scores = np.array([test_score_dict[v] for v in test_ids], dtype=float)

    if proto["fit_source"] == "train":
        train_path = TRAIN_SCORE_FILES[dataset]
        if not os.path.isfile(train_path):
            raise FileNotFoundError(
                f"{dataset} threshold protocol is {proto['protocol']} but "
                f"train score file does not exist: {train_path}"
            )
        train_score_dict = load_scores_file(train_path)
        train_scores = np.array(list(train_score_dict.values()), dtype=float)
        threshold = proto["criterion_fn"](train_scores)
        n_fit = len(train_scores)
        fit_path = train_path
    else:
        threshold = proto["criterion_fn"](test_scores)
        n_fit = len(test_scores)
        fit_path = test_path

    preds = (test_scores >= threshold).astype(int)
    met = dataset_metrics(test_ids, preds, dataset)

    out_dir = os.path.join(OUT_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"baseline_preds_v2{suffix}.jsonl")
    with open(out_path, "w") as f:
        for vid, s, p in zip(test_ids, test_scores, preds):
            f.write(
                json.dumps(
                    {
                        "video_id": vid,
                        "score": float(s),
                        "threshold": float(threshold),
                        "threshold_protocol": proto["protocol"],
                        "criterion": proto["criterion_name"],
                        "fit_source": proto["fit_source"],
                        "pred_baseline": int(p),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "dataset": dataset,
        "protocol": proto["protocol"],
        "criterion": proto["criterion_name"],
        "fit_source": proto["fit_source"],
        "fit_path": fit_path,
        "n_fit": n_fit,
        "threshold": float(threshold),
        "n_test_records": len(test_ids),
        "metrics": met,
        "out_path": out_path,
        "test_score_path": test_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="2b",
                    help="stage-1 MLLM slug (folder = holistic_<slug>/). "
                         "Default '2b' reproduces the pinned baseline.")
    ap.add_argument("--criterion", default="protocol",
                    choices=["protocol", "otsu", "gmm", "li_lee"],
                    help="Override per-dataset criterion. 'protocol' keeps "
                         "the V1 mapping (EN=otsu, ZH=gmm, HM=li_lee, IH=gmm).")
    args = ap.parse_args()

    slug = args.model_tag
    crit = None if args.criterion == "protocol" else args.criterion
    # suffix: default (slug=2b, protocol) produces baseline_preds_v2.jsonl
    if slug == "2b" and crit is None:
        suffix = ""
    else:
        suffix = f"_{slug}_" + (crit or "protocol")

    # Rebuild the module-level file dicts for this invocation
    global TRAIN_SCORE_FILES, TEST_SCORE_FILES
    TRAIN_SCORE_FILES, TEST_SCORE_FILES = _build_score_files(slug)

    rows = []
    all_datasets = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
    for ds in all_datasets:
        proto = PROTOCOL[ds]
        if proto["fit_source"] == "train" and not os.path.isfile(TRAIN_SCORE_FILES[ds]):
            print(f"[skip] {ds}: train score file not yet available "
                  f"({TRAIN_SCORE_FILES[ds]})")
            continue
        if not os.path.isfile(TEST_SCORE_FILES[ds]):
            print(f"[skip] {ds}: test score file not yet available "
                  f"({TEST_SCORE_FILES[ds]})")
            continue
        rows.append(process(ds, slug=slug, criterion_override=crit, suffix=suffix))

    # Print sanity table
    print()
    print(
        f"{'dataset':<10} {'protocol':<12} {'fit_n':>6} {'t':>8} "
        f"{'N_eval':>6} {'ACC':>8} {'mF1':>8}  "
        f"{'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}"
    )
    print("-" * 90)
    for r in rows:
        m = r["metrics"]
        print(
            f"{r['dataset']:<10} {r['protocol']:<12} {r['n_fit']:>6d} "
            f"{r['threshold']:>8.4f} {m['n']:>6d} "
            f"{m['acc']:>8.4f} {m['mf1']:>8.4f}  "
            f"{m['tp']:>3d} {m['fp']:>3d} {m['fn']:>3d} {m['tn']:>3d}"
        )
    print()

    # Pin v2 baseline targets (suffix keeps non-default runs separate)
    out_path = os.path.join(OUT_ROOT, f"v2_baseline{suffix}.json")
    pinned = {}
    for r in rows:
        m = r["metrics"]
        pinned[r["dataset"]] = {
            "acc": m["acc"],
            "mf1": m["mf1"],
            "threshold": r["threshold"],
            "protocol": r["protocol"],
            "criterion": r["criterion"],
            "fit_source": r["fit_source"],
            "fit_path": r["fit_path"],
            "n_fit": r["n_fit"],
            "n_eval": m["n"],
            "test_score_path": r["test_score_path"],
        }
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pinned, f, indent=2)
    print(f"Pinned v2 baseline targets → {out_path}")
    for ds, d in pinned.items():
        print(
            f"  {ds:<10} {d['protocol']:<12} acc={d['acc']:.4f} "
            f"mf1={d['mf1']:.4f}  n_eval={d['n_eval']}"
        )


if __name__ == "__main__":
    main()
