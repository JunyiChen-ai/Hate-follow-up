"""E3 zero-shot control --- run a verifier through the documented Fig.
resolver_prompt pass (generic HATEMM_DEF) on EN / ZH / HateMM.

Uses the hardened shared runner (scripts/_e3_gpu_common.py): fail-loud
verification, per-video crash quarantine to a persistent crash-skip file,
bounded engine re-init, fsync per write, truncation retry. Engine args are
byte-identical to judge_offline.py (the known-good config).

Provenance note (from the E3a spot-check): the generic HATEMM_DEF prompt does
NOT reproduce the submitted Table-1 zero-shot cells (q32 EN generic-def = 77.6 /
66.4 vs Table-1 75.2 / 64.9, which is the per-dataset MHCLIP_DEF value). The
submitted Table-1 zero-shot block has mixed provenance and no single prompt
reproduces it. This script is therefore for the camera-ready single-protocol
regeneration of the zero-shot block, NOT a Table-1 reproduction. For the
reviewer-facing 72B control (cpjh-C1 / 7rqV-C3), the single-pass 72B row is
already available from the existing offline_test verdicts via
rebuttal_e3_eval.py --- no GPU run needed.

Output: results/rebuttal/E3_72b_zeroshot/zeroshot_cot_<tag>_<ds>.jsonl
Crash quarantine: results/rebuttal/E3_72b_zeroshot/crash_skip_<tag>.txt
Exit code: nonzero if any requested video is left unwritten or crashed (so a
failure is never masked). Resubmit to resume; quarantined videos are skipped.

Run (GPU sbatch, env SafetyContradiction):
  python scripts/rebuttal_e3_zeroshot72b.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP2")
OUT_DIR = ROOT / "results" / "rebuttal" / "E3_72b_zeroshot"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))

import judge_offline as jo  # noqa: E402
from data_utils import SKIP_VIDEOS as DATA_SKIP, load_annotations, load_clean_split_ids  # noqa: E402
from _e3_gpu_common import MODEL_ID, GpuRunner, done_ids, read_id_file  # noqa: E402

DATASETS_DEFAULT = ["MHClip_EN", "MHClip_ZH", "HateMM"]  # IH excluded (offline_test_ih)


def build_work(tag, datasets):
    work = []
    crash_prev = read_id_file(OUT_DIR / f"crash_skip_{tag}.txt")
    for ds in datasets:
        roster = load_clean_split_ids(ds, "test")
        blocked = jo.SKIP_VIDEOS | DATA_SKIP.get(ds, set()) | crash_prev
        out_path = OUT_DIR / f"zeroshot_cot_{tag}_{ds}.jsonl"
        done = done_ids(out_path)
        ann = load_annotations(ds)
        seen = set()
        for v in roster:
            if v in blocked or v in done or v in seen:
                continue
            seen.add(v)
            if ann.get(v) is not None:
                work.append((ds, v, ann[v], out_path))
    return work


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen2.5-vl-72b-awq")
    parser.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="1 gives exact crash attribution; raise for speed "
                             "once a clean run is confirmed.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler()])
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    tags = [t.strip() for t in args.models.split(",") if t.strip()]

    def prompt_of(ds, ann):
        title = ann.get("title", "") or ""
        transcript = (ann.get("transcript", "") or "")[:300]
        return jo.PROMPT_TEMPLATE.format(
            title=title, transcript=transcript, definition=jo.HATEMM_DEF)

    failed = []
    for tag in tags:
        if tag not in MODEL_ID:
            logging.error(f"unknown tag {tag}; known={list(MODEL_ID)}")
            failed.append((tag, "unknown-tag"))
            continue
        work = build_work(tag, datasets)
        logging.info(f"=== {tag} on {datasets}: {len(work)} videos to write ===")
        if not work:
            continue
        runner = GpuRunner(MODEL_ID[tag], args.gpu_mem, args.max_model_len)
        stats = runner.run(
            work, prompt_of,
            extra_of=lambda ds, vid, t=tag: {
                "prompt": "resolver_prompt_generic_HATEMM_DEF", "verifier_tag": t},
            batch_size=args.batch_size, max_tokens=args.max_tokens,
            crash_skip_path=OUT_DIR / f"crash_skip_{tag}.txt")
        logging.info(f"[{tag}] written={stats['n_written']} "
                     f"crashed={stats['crashed']} unwritten={stats['unwritten']}")
        if stats["unwritten"] or stats["crashed"]:
            failed.append((tag, stats))

    if failed:
        logging.error("E3 zero-shot INCOMPLETE (fail-loud):")
        for tag, s in failed:
            logging.error(f"  {tag}: {s}")
        logging.error("Resubmit to resume; quarantined videos in crash_skip_<tag>.txt "
                      "are skipped. If a video keeps crashing, it stays quarantined "
                      "(like judge_offline SKIP_VIDEOS) and the pipeline tolerates a "
                      "missing verdict.")
        sys.exit(1)
    logging.info("E3 zero-shot complete: all requested videos written.")


if __name__ == "__main__":
    main()
