"""Masked packed locator (Arm M) on HateClipSeg, frame-level scored.

A sibling of `scripts/duplex/masked_parallel_isolation_mhclip.py` rather than a
fourth corpus inside it, for the same reason `build_gt_arrays_hateclipseg.py`
is a sibling of `build_gt_arrays.py`: HateClipSeg carries a second gold array
(the hateful-strict collapse) that the other corpora do not have, and the
driver has to report both. The mechanism is imported, not copied -- `Judge`,
the block mask, the branch-local positions, the cohort builder and the frame
map all come from the MultiHateClip module, so a HateClipSeg number and a
MultiHateClip number are produced by the same code.

The prompt is this corpus's own frozen convention, not a new one. HateClipSeg
was first scored chunk-by-chunk by `scripts/duplex/isolated_chunk_diag.py`,
whose rules block, question and system message are the frozen judge's, and
whose sha256s are recorded in that module's FROZEN_TEXT_SHA. That template is
imported here and the assembled user text is asserted byte-identical to it
before the model is loaded, so this run cannot silently drift onto a different
prompt than the one the corpus was first read with.

Differences from the MultiHateClip run, and only these:

  * Corpus. Cohort = the 79 videos of the frozen gold array
    (results/reproduction/gt/hateclipseg_test.npz), which is exactly the frozen
    test split. Chunks come from
    results/interleaved_timeline/hateclipseg/timestamped_chunks.jsonl, which
    covers all 394 corpus videos, so there is no late-ASR second source to
    merge.
  * Endpoint. Frame-level through `scripts/duplex/frame_eval_common.py` against
    the PRIMARY gold (offensive union, 52.6 % positive), plus the same
    evaluation against the hateful-strict gold (21.4 % positive) as a
    sensitivity row. The primary is primary; the strict row is reported, never
    substituted.
  * Fidelity. The spot check is three videos re-scored with genuine isolated
    calls, and it is advisory: if it misses Spearman 0.99 the run does not
    stop, it escalates to the full-cohort comparison, which scores every chunk
    both ways and reports both endpoints. `--sequential-reference` forces that
    comparison regardless. The argument for why the spot bar is only a proxy is
    in the MultiHateClip module and is not restated.

Output: results/reproduction/ours/hateclipseg/{per_chunk.jsonl,
frame_eval.json, prompt_fingerprints.json, STATUS, DONE}. No transcript text is
written to any output.
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
from scipy import stats

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "duplex"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "our_method"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "reproduction_baselines"))

from score_duplex_probe import YOUTUBE_RULES  # noqa: E402
import isolated_chunk_diag as icd  # noqa: E402
import masked_parallel_isolation_mhclip as mpi  # noqa: E402
from hate_common import data as hdata  # noqa: E402
import frame_eval_common as fec  # noqa: E402
from eval_baseline_scores import evaluate_scores, format_report  # noqa: E402

CORPUS = "hateclipseg"
CHUNKS_JSONL = os.path.join(PROJECT_ROOT, "results", "interleaved_timeline",
                            CORPUS, "timestamped_chunks.jsonl")
GT_ROOT = os.path.join(PROJECT_ROOT, "results", "reproduction", "gt")
STRICT_NPZ = os.path.join(GT_ROOT, "hateclipseg_test_hateful_strict.npz")

# Data registration, not a new convention: HateClipSeg takes the frozen judge's
# YOUTUBE_RULES, which is what isolated_chunk_diag.py scored this corpus with.
# `mpi.corpus_prompt` then assembles RULES_LEAD_IN + rules exactly as it does
# for MultiHateClip EN, and `Judge.fingerprints` reports the rules source.
mpi.CORPUS_RULES.setdefault(CORPUS, YOUTUBE_RULES)


def assert_frozen_prompt():
    """The assembled user text must be isolated_chunk_diag's, byte for byte."""
    rules_block, user_text = mpi.corpus_prompt(CORPUS)
    for probe in ("<CHUNK>", "abc", ""):
        if user_text(probe) != icd.user_text(probe):
            raise AssertionError(
                "hateclipseg template drifted from isolated_chunk_diag's "
                "frozen prompt")
    got = {"rules_block": mpi.sha(rules_block),
           "question": mpi.sha(icd.QUESTION),
           "system_message": icd.FROZEN_TEXT_SHA["system_message"],
           "user_text_template_notitle": mpi.sha(user_text("<CHUNK>"))}
    for key, value in got.items():
        if icd.FROZEN_TEXT_SHA[key] != value:
            raise AssertionError("frozen sha256 mismatch on %r" % key)
    return got


def load_chunk_records(counts):
    """One source for this corpus; every video in it is a distinct record."""
    recs = {}
    if not os.path.exists(CHUNKS_JSONL):
        raise SystemExit("missing chunk source %s" % CHUNKS_JSONL)
    with open(CHUNKS_JSONL) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            vid = rec["video_id"]
            if vid in recs:
                raise SystemExit("duplicate chunk record for %s" % vid)
            recs[vid] = rec
    counts["chunk_source_records"] = {
        os.path.relpath(CHUNKS_JSONL, PROJECT_ROOT): len(recs)}
    return recs


def strict_gold(gt):
    """The hateful-strict arrays, restricted to the primary cohort."""
    with np.load(STRICT_NPZ) as z:
        strict = {k: np.asarray(z[k]) for k in z.files}
    if set(strict) != set(gt):
        raise SystemExit("hateful-strict gold covers a different cohort")
    for vid in gt:
        if strict[vid].shape != gt[vid].shape:
            raise SystemExit("hateful-strict gold length differs for %s" % vid)
    return strict


def run(args):
    out_dir = os.path.join(args.out_root, CORPUS)
    os.makedirs(out_dir, exist_ok=True)
    per_chunk_path = os.path.join(out_dir, "per_chunk.jsonl")
    status_path = os.path.join(out_dir, "STATUS")

    def status(text):
        with open(status_path, "w") as fh:
            fh.write("%s  %s\n" % (time.strftime("%F %T"), text))

    status("cohort")
    counts = {k: 0 for k in (
        "videos_no_chunks", "videos_unusable_spans", "videos_all_chunks_empty",
        "chunks_total", "chunks_empty_text", "chunks_scored",
        "videos_scored_all_floor", "frames_total", "frames_covered",
        "frames_uncovered")}
    counts["videos_all_floor_ids"] = []

    frozen = assert_frozen_prompt()
    logging.info("frozen prompt sha256 verified against isolated_chunk_diag: "
                 "%s" % json.dumps(frozen, indent=2))

    gt = hdata.gt_arrays(CORPUS, "test")
    gt_strict = strict_gold(gt)
    labels = hdata.load_labels(CORPUS)
    hate_ids = {v for v in gt if labels.get(v) == 1}
    # A video the strict collapse leaves with no positive frame is not a
    # "hateful" video for the strict macro, so the strict restriction is read
    # off the strict array rather than reused from the primary label.
    hate_ids_strict = {v for v in gt_strict if int(gt_strict[v].sum()) > 0}
    chunk_recs = load_chunk_records(counts)
    counts["gold_videos"] = len(gt)
    counts["gold_videos_hate"] = len(hate_ids)
    counts["gold_videos_hate_strict"] = len(hate_ids_strict)
    counts["gold_videos_without_chunk_record"] = len(set(gt) - set(chunk_recs))

    items, spans_by_video = mpi.build_items(CORPUS, gt, chunk_recs, counts)
    by_video = {}
    for item in items:
        by_video.setdefault(item["video_id"], []).append(item)
    for vid in by_video:
        by_video[vid].sort(key=lambda r: r["chunk_index"])
    video_ids = sorted(by_video)
    if args.limit_videos:
        video_ids = video_ids[:args.limit_videos]
    logging.info("%s: %d gold videos (%d hateful, %d hateful-strict), %d with "
                 "scorable chunks, %d chunks"
                 % (CORPUS, len(gt), len(hate_ids), len(hate_ids_strict),
                    len(video_ids), len(items)))

    status("load model")
    judge = mpi.Judge(CORPUS)
    fingerprints = judge.fingerprints()
    with open(os.path.join(out_dir, "prompt_fingerprints.json"), "w") as fh:
        json.dump(fingerprints, fh, indent=2)
    logging.info("prompt fingerprints: %s" % json.dumps(fingerprints, indent=2))

    # ---- spot fidelity, advisory. The videos with the most chunks, so the
    # block mask is actually exercised.
    status("spot fidelity")
    spot_ids = sorted(video_ids,
                      key=lambda v: (-len(by_video[v]), v))[:mpi.SPOT_VIDEOS]
    logging.info("spot videos %s (chunks %s)"
                 % (spot_ids, [len(by_video[v]) for v in spot_ids]))
    plumbing = mpi.verify_mask_plumbing(
        judge, [it["text"] for it in by_video[spot_ids[0]]][:4])
    spot_masked, spot_seq, spot_seq2d = [], [], []
    for vid in spot_ids:
        texts = [it["text"] for it in by_video[vid]]
        zs, _ = judge.packed_forward(judge.branches(texts))
        spot_masked.extend(zs)
        for text in texts:
            spot_seq.append(judge.score_sequential(text, "4d")[0])
            spot_seq2d.append(judge.score_sequential(text, "2d")[0])
    zm_spot = np.array(spot_masked)
    rho_spot = float(stats.spearmanr(spot_masked, spot_seq)[0])
    d_spot = float(np.max(np.abs(zm_spot - np.array(spot_seq))))
    identical = bool(np.array_equal(zm_spot, np.array(spot_seq)))
    rho_2d = float(stats.spearmanr(spot_masked, spot_seq2d)[0])
    d_2d = float(np.max(np.abs(zm_spot - np.array(spot_seq2d))))
    logging.info("%s spot fidelity: %d chunks | vs matched-kernel sequential: "
                 "Spearman %.6f, max |Dz| %.4f, bit-identical %s | vs 2-D-mask "
                 "sequential: Spearman %.6f, max |Dz| %.4f"
                 % (CORPUS, len(spot_masked), rho_spot, d_spot, identical,
                    rho_2d, d_2d))
    spot_passed = bool(rho_spot >= mpi.FIDELITY_SPEARMAN_BAR)
    want_reference = args.sequential_reference or not spot_passed
    if not spot_passed:
        logging.warning(
            "%s spot Spearman %.4f < %.2f. Escalating to the full-cohort "
            "sequential reference, which scores every chunk both ways; that "
            "comparison is the measurement that decides."
            % (CORPUS, rho_spot, mpi.FIDELITY_SPEARMAN_BAR))

    # ---- full cohort, Arm M, one packed forward per video
    status("forward 0/%d" % len(video_ids))
    rows = []
    t0 = time.time()
    tokens_masked = 0
    with open(per_chunk_path, "w") as fout:
        for i, vid in enumerate(video_ids):
            its = by_video[vid]
            zs, ntok = judge.packed_forward(
                judge.branches([it["text"] for it in its]))
            tokens_masked += ntok
            zref = ([judge.score_sequential(it["text"], "4d")[0] for it in its]
                    if want_reference else [None] * len(its))
            for item, z, zr in zip(its, zs, zref):
                rec = {"video_id": vid,
                       "video_label": "hate" if vid in hate_ids else "non_hate",
                       "chunk_index": item["chunk_index"],
                       "span": item["span"],
                       "chunk_tokens": item["chunk_tokens"],
                       "z_masked": z}
                if zr is not None:
                    rec["z_sequential"] = zr
                rows.append(rec)
                fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if i % 10 == 0:
                status("forward %d/%d  %.1f s" % (i, len(video_ids),
                                                  time.time() - t0))
                logging.info("  %d/%d  %.1f s" % (i, len(video_ids),
                                                  time.time() - t0))
    wall = time.time() - t0
    logging.info("%s: %d videos, %d chunks, %.1f s (%.3f s/video), %d packed "
                 "tokens" % (CORPUS, len(video_ids), len(rows), wall,
                             wall / max(1, len(video_ids)), tokens_masked))

    # ---- frame-level evaluation against both frozen gold arrays
    status("evaluate")
    scores = mpi.frame_scores(gt, rows, spans_by_video, counts)
    res = evaluate_scores(scores, gt, hate_ids)
    res_strict = evaluate_scores(scores, gt_strict, hate_ids_strict)

    ref_block = None
    if want_reference:
        zm = np.array([r["z_masked"] for r in rows], dtype=float)
        zr = np.array([r["z_sequential"] for r in rows], dtype=float)
        ref_scores = mpi.frame_scores(gt, rows, spans_by_video, counts,
                                      key="z_sequential")
        ref_res = evaluate_scores(ref_scores, gt, hate_ids)
        ref_block = {
            "note": "one isolated call per chunk, same attention path as the "
                    "packed arm; this is the reference Arm M reproduces",
            "n_chunks": len(rows),
            "spearman_masked_vs_sequential": float(stats.spearmanr(zm, zr)[0]),
            "pearson_masked_vs_sequential": float(stats.pearsonr(zm, zr)[0]),
            "max_abs_delta_z": float(np.max(np.abs(zm - zr))),
            "mean_abs_delta_z": float(np.mean(np.abs(zm - zr))),
            "frac_chunks_identical": float(np.mean(zm == zr)),
            "n_unique_z_sequential": int(len(np.unique(zr))),
            "frame_level": ref_res,
            "endpoint_delta_masked_minus_sequential": {
                "roc_auc": mpi._sub(res["roc_auc"], ref_res["roc_auc"]),
                "pr_auc": mpi._sub(res["pr_auc"], ref_res["pr_auc"]),
                "within_hate_macro_auc": mpi._sub(
                    res["per_video"]["macro_auc"],
                    ref_res["per_video"]["macro_auc"]),
            },
        }
        logging.info("%s sequential reference: Spearman %.6f, Pearson %.6f, "
                     "max |Dz| %.4f, endpoint delta ROC %+.5f PR %+.5f"
                     % (CORPUS, ref_block["spearman_masked_vs_sequential"],
                        ref_block["pearson_masked_vs_sequential"],
                        ref_block["max_abs_delta_z"],
                        ref_block["endpoint_delta_masked_minus_sequential"]["roc_auc"],
                        ref_block["endpoint_delta_masked_minus_sequential"]["pr_auc"]))

    # Video-level readout: max chunk z per video against the video label.
    z_by_video = {}
    for r in rows:
        z_by_video.setdefault(r["video_id"], []).append(r["z_masked"])
    vmax = {v: (max(z_by_video[v]) if v in z_by_video
                else counts["uncovered_floor"]) for v in sorted(gt)}
    video_auc = fec.rank_auc([vmax[v] for v in vmax if v in hate_ids],
                             [vmax[v] for v in vmax if v not in hate_ids])
    video_auc_strict = fec.rank_auc(
        [vmax[v] for v in vmax if v in hate_ids_strict],
        [vmax[v] for v in vmax if v not in hate_ids_strict])

    payload = {
        "method": "ours_zero_label_locator_arm_M",
        "corpus": CORPUS,
        "split": "test",
        "model": mpi.MODEL,
        "arm": "masked packed, one forward per video, branch-local positions",
        "mechanism_reference": {
            "script": "scripts/duplex/masked_parallel_isolation_pilot.py",
            "commit": "aa403cb",
            "note": "masked-vs-sequential equivalence and the causal-arm "
                    "collapse were established on HateMM and are not rerun",
        },
        "prompt_reference": {
            "script": "scripts/duplex/isolated_chunk_diag.py",
            "note": "this corpus's frozen prompt; the assembled user text is "
                    "asserted byte-identical to it before the model loads",
            "frozen_text_sha256": icd.FROZEN_TEXT_SHA,
        },
        "prompt_fingerprints": fingerprints,
        "counts": counts,
        "spot_fidelity": {
            "videos": spot_ids,
            "chunks": len(spot_masked),
            "reference": "sequential isolated calls, one per chunk, on the "
                         "same attention path the packed arm uses",
            "spearman_masked_vs_sequential": rho_spot,
            "max_abs_delta_z": d_spot,
            "bit_identical": identical,
            "bar": mpi.FIDELITY_SPEARMAN_BAR,
            "passed": spot_passed,
            "advisory": True,
            "against_2d_mask_reference": {
                "note": "the tokenizer's all-ones 2-D attention_mask sends HF "
                        "to a different SDPA kernel; recorded so the gap is "
                        "on the record, not used for the decision",
                "spearman": rho_2d,
                "max_abs_delta_z": d_2d,
            },
        },
        "mask_plumbing_max_abs_logit_delta": plumbing,
        "runtime": {"videos": len(video_ids), "chunks": len(rows),
                    "seconds": wall, "packed_tokens": tokens_masked,
                    "mllm_calls_per_video": 1},
        "frame_level": res,
        "frame_level_hateful_strict": {
            "note": "SENSITIVITY, not the primary: same videos, same frames, "
                    "same scores, gold recollapsed to the hateful dimension "
                    "alone. Reported alongside the primary, never instead.",
            "gold": "results/reproduction/gt/hateclipseg_test_hateful_strict.npz",
            "macro_restricted_to": "videos with at least one strict-positive "
                                   "frame",
            "result": res_strict,
        },
        "sequential_reference": ref_block,
        "video_level": {"max_z_auc": video_auc,
                        "n_hate": len(hate_ids),
                        "n_non_hate": len(gt) - len(hate_ids),
                        "max_z_auc_hateful_strict": video_auc_strict,
                        "n_hate_strict": len(hate_ids_strict)},
    }
    with open(os.path.join(out_dir, "frame_eval.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    print(format_report(res, "%s / test / ours (z_masked, PRIMARY gold)"
                        % CORPUS))
    print("  video-level max-z AUC   %s"
          % ("n/a" if video_auc is None else "%.4f" % video_auc))
    print("")
    print(format_report(res_strict,
                        "%s / test / ours (z_masked, hateful-strict gold)"
                        % CORPUS))
    print("")
    status("DONE")
    with open(os.path.join(out_dir, "DONE"), "w") as fh:
        fh.write("%s\n" % time.strftime("%F %T"))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=mpi.OUT_ROOT)
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--sequential-reference", action="store_true",
                    help="force the full-cohort comparison even if the spot "
                         "check clears the bar")
    ap.add_argument("--check-prompt-only", action="store_true",
                    help="assert the frozen prompt identity and exit; no "
                         "model is loaded")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    if args.check_prompt_only:
        print(json.dumps(assert_frozen_prompt(), indent=2))
        return
    run(args)


if __name__ == "__main__":
    main()
