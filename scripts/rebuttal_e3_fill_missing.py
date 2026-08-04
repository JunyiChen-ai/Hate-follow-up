"""E3 fill --- score the videos missing from the frozen offline verdict files.

For each requested verifier, find every clean-test video absent from the frozen
offline verdict file (and not on any skip list) and score ONLY those with the
exact prompt/definition/params of the original offline run:
  - EN / ZH / HateMM: PROMPT_TEMPLATE + per-dataset definition (MHCLIP_DEF / HATEMM_DEF)
  - ImpliHateVid: PROMPT_TEMPLATE_IH + IH_DEF (the frozen IH files are ih-prompt)

Uses the hardened shared runner (scripts/_e3_gpu_common.py): fail-loud
verification, per-video crash quarantine, bounded engine re-init, fsync per
write, truncation retry. Engine args are byte-identical to judge_offline.py.

Fills go to results/rebuttal/E3_72b_zeroshot/fills_<tag>_<ds>.jsonl; the frozen
files are never modified. Exit code is nonzero if any targeted video is left
unwritten or crashed (no silent success). Resubmit to resume.

Note: the reviewer-facing 72B single-pass control (cpjh-C1 / 7rqV-C3) is already
available from the existing offline verdicts without these fills; the fills only
close the last few coverage holes (HateMM 1, ImpliHateVid a handful) for tidier
denominators.

Run (GPU sbatch, env SafetyContradiction):
  python scripts/rebuttal_e3_fill_missing.py \
      --models qwen2.5-vl-72b-awq,qwen2.5-vl-32b-awq,gemma-3-27b-it,internvl35-8b,qwen3-vl-8b,gemma-3-12b-it
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/EMNLP3")
OUT_DIR = ROOT / "results" / "rebuttal" / "E3_72b_zeroshot"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))

import judge_offline as jo  # noqa: E402
from data_utils import SKIP_VIDEOS as DATA_SKIP, load_annotations, load_clean_split_ids  # noqa: E402
from grid_eval_all import judge_path, ld_jsonl  # noqa: E402
from _e3_gpu_common import MODEL_ID, GpuRunner, done_ids, read_id_file  # noqa: E402

ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
MODELS_DEFAULT = ("qwen2.5-vl-72b-awq,qwen2.5-vl-32b-awq,gemma-3-27b-it,"
                  "internvl35-8b,qwen3-vl-8b,gemma-3-12b-it")


def frozen_vids(tag, ds):
    p = judge_path(tag, ds)
    if p is None or not p.exists():
        return set()
    return {r["video_id"] for r in ld_jsonl(p)}


def missing_for(tag, ds, crash_prev):
    roster = load_clean_split_ids(ds, "test")
    have = frozen_vids(tag, ds)
    blocked = jo.SKIP_VIDEOS | DATA_SKIP.get(ds, set()) | crash_prev
    seen, out = set(), []
    for v in roster:
        if v in have or v in blocked or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def build_work(tag):
    work = []
    crash_prev = read_id_file(OUT_DIR / f"crash_skip_{tag}.txt")
    for ds in ALL_DATASETS:
        out_path = OUT_DIR / f"fills_{tag}_{ds}.jsonl"
        done = done_ids(out_path)
        miss = missing_for(tag, ds, crash_prev)
        ann = load_annotations(ds)
        for v in miss:
            if v in done or ann.get(v) is None:
                continue
            work.append((ds, v, ann[v], out_path))
    return work


def prompt_of(ds, ann):
    title = ann.get("title", "") or ""
    transcript = (ann.get("transcript", "") or "")[:300]
    if ds == "ImpliHateVid":
        tpl = jo.PROMPT_TEMPLATE_IH
        definition = jo.get_definition(ds, ih_prompt=True)
    else:
        tpl = jo.PROMPT_TEMPLATE
        definition = jo.get_definition(ds, ih_prompt=False)
    return tpl.format(title=title, transcript=transcript, definition=definition)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=MODELS_DEFAULT,
                        help="comma-separated tags, processed in order (72B first).")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--gpu-mem", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler()])
    tags = [t.strip() for t in args.models.split(",") if t.strip()]

    failed = []
    for tag in tags:
        if tag not in MODEL_ID:
            logging.error(f"unknown tag {tag}; known={list(MODEL_ID)}")
            failed.append((tag, "unknown-tag"))
            continue
        work = build_work(tag)
        logging.info(f"=== {tag}: {len(work)} missing videos to fill ===")
        if not work:
            continue
        runner = GpuRunner(MODEL_ID[tag], args.gpu_mem, args.max_model_len)
        stats = runner.run(
            work, prompt_of,
            extra_of=lambda ds, vid, t=tag: {
                "fill_source": "rebuttal_e3", "verifier_tag": t,
                "ih_prompt": ds == "ImpliHateVid"},
            batch_size=args.batch_size, max_tokens=args.max_tokens,
            crash_skip_path=OUT_DIR / f"crash_skip_{tag}.txt")
        logging.info(f"[{tag}] written={stats['n_written']} "
                     f"crashed={stats['crashed']} unwritten={stats['unwritten']}")
        if stats["unwritten"] or stats["crashed"]:
            failed.append((tag, stats))

    if failed:
        logging.error("E3 fill INCOMPLETE (fail-loud):")
        for tag, s in failed:
            logging.error(f"  {tag}: {s}")
        logging.error("Resubmit to resume; quarantined videos in crash_skip_<tag>.txt "
                      "are skipped and the pipeline tolerates a missing verdict.")
        sys.exit(1)
    logging.info("E3 fill complete: all targeted videos written.")


if __name__ == "__main__":
    main()
