"""Registry of the weakly supervised baselines for oracle test selection (2026-10-09).

A candidate is one (checkpoint, output branch) pair of one seed's training. Checkpoint tags: eNNN / eNN = the
weights after that epoch of the retrained run (oracle_test_selection/<method>/<DS>/seed<k>/epochs/), "current" = the
current run's selected checkpoint (runs/20261008_baselines/<method>/<DS>/seed<k>/, read only).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / "weaksup_common"))
import common as C  # noqa: E402  weaksup_common/common.py

_spec = importlib.util.spec_from_file_location("detwin_common", BASE / "detwin" / "common.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)  # detwin/common.py (window grid of SAGE / CLARA); its RUNS is not used here

CUR = C.REPO / "runs" / "20261008_baselines"
ORACLE = CUR / "oracle_test_selection"
DATASETS = ("HateMM", "HateClipSeg", "DeHate")
SEEDS = (2025, 234, 3407)
METRICS = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")

# kind: frame = 1 fps score files (scores.jsonl format), window = 8-s window scores
# branches: every output branch the method's inference writes; declared = the branch of the current table row;
# current_name = method name in the current run's metrics.json
METHODS = {
    "mil_bert": {"label": "BERT + MIL (text)", "kind": "frame", "branches": ["score_mil"], "declared": "score_mil",
                 "current_name": "mil_bert", "rule": "epoch with the best val video AP"},
    "mil_wav2vec2": {"label": "wav2vec2 + MIL (audio)", "kind": "frame", "branches": ["score_mil"],
                     "declared": "score_mil", "current_name": "mil_wav2vec2", "rule": "epoch with the best val video AP"},
    "mil_clip": {"label": "CLIP + MIL (visual)", "kind": "frame", "branches": ["score_mil"], "declared": "score_mil",
                 "current_name": "mil_clip", "rule": "epoch with the best val video AP"},
    "vadclip": {"label": "VadCLIP", "kind": "frame", "branches": ["score_align", "score_mlp"],
                "declared": "score_align", "current_name": "vadclip", "rule": "last of 10 epochs"},
    "dsanet": {"label": "DSANet", "kind": "frame", "branches": ["score_mlp", "score_refined", "score_align"],
               "declared": "score_mlp", "current_name": "dsanet", "rule": "last of 10 epochs"},
    "avadclip": {"label": "AVadCLIP (audio-visual)", "kind": "frame", "branches": ["score_align", "score_mlp"],
                 "declared": "score_align", "current_name": "avadclip", "rule": "last of 10 epochs"},
    "multihateloc": {"label": "MultiHateLoc (reimpl.)", "kind": "frame",
                     "branches": ["score_fused", "score_visual", "score_audio", "score_text", "score_dms",
                                  "score_union"],
                     "declared": "score_fused", "current_name": "multihateloc", "rule": "epoch with the best val video AP"},
    "sage": {"label": "SAGE (8-s windows)", "kind": "window", "branches": ["fusion", "text", "audio", "vision"],
             "declared": "fusion", "current_name": "SAGE_win8", "rule": "authors' rule: best val macro-F1"},
    "clara": {"label": "CLARA (8-s windows)", "kind": "window", "branches": ["prob"], "declared": "prob",
              "current_name": "CLARA_win8", "rule": "authors' rule: early stopping on val accuracy, best checkpoint"},
}


def seed_dir(method, ds, seed):
    return ORACLE / method / ds / f"seed{seed}"


def current_seed_dir(method, ds, seed):
    return CUR / method / ds / f"seed{seed}"


def current_frame_scores(method, ds, seed):
    """The current run's 1 fps scores file of this seed (all branches it holds)."""
    corpus = C.CORPUS[ds]
    if method == "multihateloc":
        if ds == "DeHate":
            return CUR / "multihateloc" / "DeHate" / "source_lab2" / "final" / f"seed_{seed}" / "dehate" / "scores.jsonl"
        return current_seed_dir(method, ds, seed) / corpus / "scores.jsonl"
    return current_seed_dir(method, ds, seed) / "scores.jsonl"
