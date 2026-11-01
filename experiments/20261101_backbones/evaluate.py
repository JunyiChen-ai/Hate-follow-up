#!/usr/bin/env python3
"""Decode, compose and tabulate the backbone study (README §2). CPU; no label enters any score (the evaluator reads GT).

For every backbone whose two corpora have reads (runs/20261101_backbones/<tag>/<ds>/<arm>/predictions.jsonl):
  merged/<arm>/predictions.jsonl            both corpora in one file (EM and normal scores are per corpus)
  Verdict      compose.py merged/full  --intercept spvl --residual none        -> merged/full/metrics_ispvl_rnone.json
  Per window   compose.py merged/alone --intercept none --residual rank        -> merged/alone/metrics_inone_rrank.json
  ours         twolevel_r2.py (current flags) on merged/full                   -> decoded/<tag>/metrics.json
  reader comp. compose.py merged/<full|arm> --intercept zv_plus_mean --residual rank -> metrics_izv_plus_mean_rrank.json
Writes runs/20261101_backbones/table.md and table.json. Old values: the paper's tables, transcribed below
(paper/overleaf/sections/experiments.tex tab:mllms, appendix.tex tab:mllm_ablation, as of 2026-11-01).
Usage: python experiments/20261101_backbones/evaluate.py [--tags ...] [--skip-existing]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs/20261101_backbones"
DS = ["HateMM", "HateClipSeg"]
TAGS = ["q3vl-2b", "q3vl-4b", "q3vl-8b", "q3vl-32b", "q25vl-7b", "internvl35-8b", "llava-ov-7b", "gemma3-12b"]
NAMES = {"q3vl-2b": "Qwen3-VL-2B", "q3vl-4b": "Qwen3-VL-4B", "q3vl-8b": "Qwen3-VL-8B", "q3vl-32b": "Qwen3-VL-32B", "q25vl-7b": "Qwen2.5-VL-7B",
         "internvl35-8b": "InternVL3.5-8B", "llava-ov-7b": "LLaVA-OneVision-7B", "gemma3-12b": "Gemma-3-12B"}
ARMS = [("alone", "alone"), ("no_transcript", "-ctx"), ("no_prefix_frames", "-frames"), ("asr_windows", "ASR"), ("no_stance", "-stance"), ("joint", "joint")]
DECODE = ["--noleak", "--transform", "nscore", "--key", "raw", "--duration", "bma", "--bma-prior", "length", "--min-windows", "2", "--bma-grid", "6", "--arm", "m2"]
# paper tab:mllms, (verdict, per window, ours) x (HateMM, HateClipSeg), each pooled ROC / pooled PR / within
OLD_MAIN = {
    "q3vl-2b": [(.847, .571, .500), (.544, .271, .628), (.854, .629, .738), (.613, .585, .500), (.525, .494, .542), (.623, .614, .575)],
    "q3vl-4b": [(.871, .627, .500), (.549, .275, .631), (.883, .660, .751), (.683, .652, .500), (.538, .500, .568), (.699, .672, .618)],
    "q3vl-8b": [(.882, .646, .500), (.550, .271, .628), (.894, .691, .764), (.674, .617, .500), (.544, .505, .576), (.716, .672, .637)],
    "q3vl-32b": [(.871, .598, .500), (.547, .270, .619), (.891, .674, .773), (.693, .639, .500), (.544, .507, .571), (.719, .690, .634)],
    "q25vl-7b": [(.871, .639, .500), (.549, .272, .641), (.887, .671, .756), (.690, .647, .500), (.540, .502, .566), (.699, .668, .588)],
    "internvl35-8b": [(.877, .626, .500), (.543, .271, .618), (.888, .673, .739), (.654, .625, .500), (.538, .500, .575), (.678, .653, .632)],
    "llava-ov-7b": [(.878, .641, .500), (.548, .274, .632), (.881, .656, .733), (.682, .649, .500), (.531, .502, .545), (.679, .666, .613)],
    "gemma3-12b": [(.831, .497, .500), (.551, .271, .640), (.863, .629, .781), (.661, .627, .500), (.533, .501, .554), (.720, .696, .625)],
}
# paper tab:mllm_ablation, within arm - full: alone, -ctx, -frames, ASR, -stance, joint; HateMM then HateClipSeg
OLD_ABL = {
    "q3vl-2b": [-.027, -.006, -.020, -.062, -.014, +.013, -.024, -.026, -.005, -.039, +.006, -.014],
    "q3vl-4b": [-.044, -.027, -.050, -.084, -.025, -.005, -.013, -.010, -.012, -.042, +.006, -.002],
    "q3vl-8b": [-.068, -.073, -.056, -.112, -.012, +.003, -.026, -.028, -.017, -.073, -.014, -.029],
    "q3vl-32b": [-.082, -.053, -.040, -.104, -.009, -.004, -.031, -.012, -.026, -.077, -.005, -.008],
    "q25vl-7b": [-.038, +.000, -.027, -.078, -.006, -.009, -.001, +.007, -.011, -.047, +.001, -.011],
    "internvl35-8b": [-.042, +.006, -.029, -.076, +.013, +.002, -.029, +.005, -.045, -.084, -.008, -.040],
    "llava-ov-7b": [-.013, +.005, +.010, -.053, -.021, +.002, -.020, -.002, -.010, -.037, -.028, +.008],
    "gemma3-12b": [-.027, -.016, -.014, -.081, -.019, +.006, -.022, -.007, +.001, -.056, -.021, -.007],
}
METRICS = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")
# Qwen3-VL-8B A1 / A2 / A3 / A6: the reader-ablation runs are the same computation (same functions in the same order;
# the smoke records of both experiments agree to 0.0 on all four videos, README §5), so they are used instead of a
# second run. A4 differs (frame cap here), so this study runs its own q3vl-8b A4.
EXTERNAL = {("q3vl-8b", arm): ROOT / "runs/20261101_reader_ablation/{ds}" / arm / "predictions.jsonl"
            for arm in ("no_transcript", "no_prefix_frames", "no_stance", "joint")}


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=ROOT)


def merge(tag, arm):
    srcs = [R / tag / ds / arm / "predictions.jsonl" for ds in DS]
    if (tag, arm) in EXTERNAL and not all(p.exists() for p in srcs):
        srcs = [Path(str(EXTERNAL[(tag, arm)]).format(ds=ds)) for ds in DS]
    if not all(p.exists() for p in srcs):
        return None
    rows = [r for p in srcs for r in map(json.loads, open(p)) if r["dataset"] in DS]
    n = {ds: sum(r["dataset"] == ds for r in rows) for ds in DS}
    if n != {"HateMM": 215, "HateClipSeg": 118} or any(r.get("error") for r in rows):
        print(f"skip {tag}/{arm}: incomplete {n}")
        return None
    out = R / tag / "merged" / arm
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "SOURCES.txt").write_text("\n".join(str(p.relative_to(ROOT)) for p in srcs) + "\n")
    return out


def load(path):
    if not path.exists():
        return None
    return {p["dataset"]: tuple(p[m] for m in METRICS) for p in json.load(open(path))["per_dataset"]}


def compose(d, intercept, residual, skip):
    tag = f"i{intercept}_r{residual}"
    m = d / f"metrics_{tag}.json"
    if not (skip and m.exists()):
        run([sys.executable, "experiments/20260910_spvl/compose.py", "--run-dir", d, "--intercept", intercept, "--residual", residual, "--datasets", *DS])
    return m


def consistency(tag):
    """Max |difference| between reads of the same video from two runs: the smoke vs the full run (every tag), and for
    q3vl-8b the full run vs the stored adopted reads on every video."""
    out = {}
    for rp in sorted((R / "smoke" / tag / "records" / "main").glob("*/*.json")):
        q = R / tag / rp.parent.name / "records" / "main" / rp.parent.name / rp.name
        if not q.exists():
            continue
        a, b = json.loads(rp.read_text()), json.loads(q.read_text())
        d = abs(a["global_margin"] - b["global_margin"])
        for x, y in zip(a["traces"], b["traces"]):
            for k in ("z_visual", "z_speech", "z_alone"):
                if x.get(k) is not None and y.get(k) is not None:
                    d = max(d, abs(x[k] - y[k]))
        out["smoke_vs_run_max"] = max(out.get("smoke_vs_run_max", 0.0), d)
        out["smoke_vs_run_videos"] = out.get("smoke_vs_run_videos", 0) + 1
    if tag == "q3vl-8b" and (R / tag / "merged" / "full" / "predictions.jsonl").exists():
        ref = {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(ROOT / "runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned/predictions.jsonl"))}
        d, n = 0.0, 0
        for r in map(json.loads, open(R / tag / "merged" / "full" / "predictions.jsonl")):
            g = ref[(r["dataset"], r["video_id"])]
            d = max(d, abs(g["extra"]["z_video"] - r["extra"]["z_video"]))
            for x, y in zip(g["extra"]["windows"], r["extra"]["windows"]):
                for k in ("z_visual", "z_speech"):
                    if (k in x) != (k in y):
                        d = float("inf")
                    elif k in x:
                        d = max(d, abs(x[k] - y[k]))
            n += 1
        out.update(adopted_vs_run_max=d, adopted_vs_run_videos=n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=TAGS)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    table = {}
    for tag in a.tags:
        t = table.setdefault(tag, {})
        full = merge(tag, "full")
        if full is None:
            continue
        t["verdict"] = str(compose(full, "spvl", "none", a.skip_existing).relative_to(ROOT))
        t["full_reader_comp"] = str(compose(full, "zv_plus_mean", "rank", a.skip_existing).relative_to(ROOT))
        dec = R / "decoded" / tag / "metrics.json"
        if not (a.skip_existing and dec.exists()):
            run([sys.executable, "-u", "experiments/20260926_twolevel/twolevel_r2.py", "--run", full, "--datasets", *DS, *DECODE,
                 "--out-root", R / "decoded", "--tag", tag])
        t["ours"] = str(dec.relative_to(ROOT))
        alone = merge(tag, "alone")
        if alone is not None:
            t["per_window"] = str(compose(alone, "none", "rank", a.skip_existing).relative_to(ROOT))
        for arm, _ in ARMS[1:]:
            d = merge(tag, arm)
            if d is not None:
                t[arm] = str(compose(d, "zv_plus_mean", "rank", a.skip_existing).relative_to(ROOT))
    for tag in a.tags:
        table.setdefault(tag, {})["consistency"] = consistency(tag)
    # tables
    f3 = lambda x: f"{x:.3f}".replace("0.", ".", 1) if x >= 0 else f"{x:.3f}".replace("-0.", "-.", 1)  # noqa: E731
    lines = ["## Main table (pooled ROC / pooled PR / within); old = paper tab:mllms, new = this run\n",
             "| backbone | corpus | Verdict old | Verdict new | Per window old | Per window new | ours old | ours new |", "|---|---|---|---|---|---|---|---|"]
    out = {}
    for tag in TAGS:
        t = table.get(tag, {})
        got = {k: load(ROOT / t[k]) for k in ("verdict", "per_window", "ours") if k in t}
        for di, ds in enumerate(DS):
            old = OLD_MAIN[tag][3 * di:3 * di + 3]
            cells = []
            for k, o in zip(("verdict", "per_window", "ours"), old):
                new = got[k].get(ds) if got.get(k) else None
                cells += [" / ".join(f3(x) for x in o), " / ".join(f3(x) for x in new) if new else "—"]
                out.setdefault(tag, {}).setdefault(ds, {})[k] = dict(old=o, new=new, metrics=t.get(k))
            lines.append(f"| {NAMES[tag]} | {ds} | " + " | ".join(cells) + " |")
    lines += ["\n## Appendix table (within, arm − full, reader composition: verdict + mean window read + centred rank); old = paper tab:mllm_ablation\n",
              "| backbone | corpus | " + " | ".join(f"{n} old / new" for _, n in ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
    for tag in TAGS:
        t = table.get(tag, {})
        base = load(ROOT / t["full_reader_comp"]) if "full_reader_comp" in t else None
        for di, ds in enumerate(DS):
            cells = []
            for ai, (arm, name) in enumerate(ARMS):
                old = OLD_ABL[tag][6 * di + ai]
                key = "per_window" if arm == "alone" else arm
                m = load(ROOT / t[key]) if key in t else None
                new = m[ds][2] - base[ds][2] if (m and base) else None
                cells.append(f"{old:+.3f} / " + (f"{new:+.3f}" if new is not None else "—"))
                out.setdefault(tag, {}).setdefault(ds, {})[f"abl_{arm}"] = dict(old=old, new=new, metrics=t.get(key), base=t.get("full_reader_comp"))
            lines.append(f"| {NAMES[tag]} | {ds} | " + " | ".join(cells) + " |")
    lines += ["\n## Consistency (max |difference| of reads, nats)\n"]
    for tag in TAGS:
        if table.get(tag, {}).get("consistency"):
            lines.append(f"- {NAMES[tag]}: {table[tag]['consistency']}")
            out.setdefault(tag, {})["consistency"] = table[tag]["consistency"]
    text = "\n".join(lines) + "\n"
    (R / "table.md").write_text(text)
    (R / "table.json").write_text(json.dumps(out, indent=1) + "\n")
    print(text)


if __name__ == "__main__":
    main()
