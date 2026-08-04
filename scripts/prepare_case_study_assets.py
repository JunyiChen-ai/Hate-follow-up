#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/data/jehc223/home/tmp/matplotlib")

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path("/data/jehc223/EMNLP2")
OUT_DIR = ROOT / "paper" / "analysis" / "case_study_assets"

sys.path.insert(0, str(ROOT / "results" / "paper_stage2_experiments"))

from build_experiments import EvalCache, MAIN_ORDER, ent, logit, sigmoid  # noqa: E402


JUDGE_LABELS = {
    "gemma-3-27b-it": "Gemma-3-27B",
    "qwen2.5-vl-32b-awq": "Qwen2.5-VL-32B-AWQ",
    "qwen2.5-vl-72b-awq": "Qwen2.5-VL-72B-AWQ",
}

CASES = [
    {
        "group": "success",
        "case_id": "S1_early_stop",
        "dataset": "MHClip_EN",
        "video_id": "tOsD1F--EEo",
        "case_type": "Early stop",
        "note": "Mapper and first verifier agree on a normal decision, so the case is finalized early.",
    },
    {
        "group": "success",
        "case_id": "S2_conflict_resolution",
        "dataset": "HateMM",
        "video_id": "non_hate_video_458",
        "case_type": "Conflict resolution",
        "note": "The mapper is initially correct, but conflicting verifier evidence temporarily moves the posterior to the hateful side before later evidence restores the normal decision and exits the boundary region.",
    },
    {
        "group": "success",
        "case_id": "S3_false_positive_correction",
        "dataset": "HateMM",
        "video_id": "non_hate_video_53",
        "case_type": "False-positive correction",
        "note": "The Boundary Mapper predicts hateful, while consistent verifier evidence corrects the case to normal and exits the boundary region.",
    },
    {
        "group": "failure",
        "case_id": "F1_missed_hateful_speech",
        "dataset": "HateMM",
        "video_id": "hate_video_149",
        "case_type": "Failed false-negative correction",
        "note": "The first verifier moves the case toward hateful, but subsequent evidence pulls it back to normal.",
    },
    {
        "group": "failure",
        "case_id": "F2_contextual_gendered_language",
        "dataset": "MHClip_EN",
        "video_id": "R3Xt1__7TwQ",
        "case_type": "Failed false-negative correction",
        "note": "The case requires contextual interpretation, and the verifier sequence remains on the normal side.",
    },
    {
        "group": "failure",
        "case_id": "F3_coded_satirical_wording",
        "dataset": "MHClip_ZH",
        "video_id": "BV1kT411t7ax",
        "case_type": "Failed false-negative correction",
        "note": "Coded or satirical wording is not resolved by later verifier evidence.",
    },
]


FRAME_ROOTS = {
    "HateMM": [
        Path("/data/jehc223/HateMM/frames_16"),
        Path("/data/jehc223/HateMM/frames"),
        Path("/data/jehc223/HateMM/quad"),
    ],
    "ImpliHateVid": [
        Path("/data/jehc223/ImpliHateVid/frames_16"),
        Path("/data/jehc223/ImpliHateVid/frames_32"),
    ],
    "MHClip_EN": [
        Path("/data/jehc223/Multihateclip/English/frames_16"),
        Path("/data/jehc223/Multihateclip/English/frames"),
        ROOT / ".cache" / "quad_resized",
    ],
    "MHClip_ZH": [
        Path("/data/jehc223/Multihateclip/Chinese/frames_16"),
        Path("/data/jehc223/Multihateclip/Chinese/frames"),
        ROOT / ".cache" / "quad_resized",
    ],
}


def numeric_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**9, path.name)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_path(dataset: str, judge: str) -> Path:
    base = ROOT / "results" / "boundary_rescue" / dataset
    if dataset == "ImpliHateVid":
        ih_path = base / f"offline_test_ih_{judge}.jsonl"
        if ih_path.exists():
            return ih_path
    return base / f"offline_test_{judge}.jsonl"


def load_judge_row(dataset: str, video_id: str, judge: str) -> dict | None:
    path = judge_path(dataset, judge)
    if not path.exists():
        return None
    for row in load_jsonl(path):
        if row.get("video_id") == video_id:
            return row
    return None


def label_name(value: int | None) -> str:
    if value == 1:
        return "Hateful"
    if value == 0:
        return "Normal"
    return "Unknown"


def safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def select_source_frames(dataset: str, video_id: str) -> list[Path]:
    for root in FRAME_ROOTS[dataset]:
        folder = root / video_id
        if not folder.exists():
            continue
        files = sorted(
            [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
            key=numeric_key,
        )
        if len(files) >= 3:
            idx = [
                max(round((len(files) - 1) * 0.25), 0),
                max(round((len(files) - 1) * 0.50), 0),
                max(round((len(files) - 1) * 0.75), 0),
            ]
            return [files[i] for i in idx]
    return []


def copy_frame(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def make_contact_sheet(frame_paths: list[Path], dst: Path, tile_width: int = 640) -> None:
    if not frame_paths:
        return
    gap = 12
    tiles = []
    for i, path in enumerate(frame_paths):
        with Image.open(path) as img:
            img = img.convert("RGB")
            scale = tile_width / img.width
            tile = img.resize((tile_width, max(1, round(img.height * scale))), Image.Resampling.LANCZOS)
            tiles.append(tile)
    height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (tile_width * len(tiles) + gap * (len(tiles) - 1), height), "white")
    for i, tile in enumerate(tiles):
        top = (height - tile.height) // 2
        sheet.paste(tile, (i * (tile_width + gap), top))
    sheet.save(dst, quality=94)


def plot_trace(case: dict, trace: list[dict], hbar: float, rho: float, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    x = [0 if t["step"] == "mapper" else MAIN_ORDER.index(t["step"]) + 1 for t in trace]
    y = [float(t["posterior_hateful"]) for t in trace]
    colors = ["#606C76"] + ["#2F6FBB" for _ in trace[1:]]

    fig, ax = plt.subplots(figsize=(3.35, 1.75), dpi=300)
    ax.axhspan(1 - rho, rho, color="#E8EDF5", alpha=0.9, zorder=0)
    ax.axhline(0.5, color="#6F7782", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    ax.plot(x, y, color="#234F8E", linewidth=1.8, zorder=3)
    ax.scatter(x, y, s=36, color=colors, edgecolor="white", linewidth=0.9, zorder=4)

    for xi, yi in zip(x, y):
        ax.text(
            xi,
            min(0.95, yi + 0.052),
            f"{yi:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#1F2933",
        )

    ax.text(
        2.55,
        1 - rho + 0.045,
        "Boundary Region",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#596575",
    )
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.2, 3.2)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Mapper", "V1", "V2", "V3"])
    ax.set_ylabel("Posterior probability", fontsize=8.3)
    ax.tick_params(axis="both", labelsize=7.8, width=0.8, length=3.2)
    ax.grid(axis="y", color="#D6DADF", linewidth=0.5, alpha=0.9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(dst)
    fig.savefig(dst.with_suffix(".png"))
    plt.close(fig)


def build_case(cache: EvalCache, case: dict) -> dict:
    dataset = case["dataset"]
    video_id = case["video_id"]
    labels = cache.load_labels_for(dataset)
    base = cache.load_base("2b", dataset)
    band = cache.load_band("2b", dataset)
    row = band[video_id]
    hbar = cache.hbar[("2b", dataset)]
    rho = cache.rhod[("2b", dataset)]
    lam = math.log(rho / (1.0 - rho))

    ell = logit(float(row.get("posterior_hi", 0.5)))
    trace = [{
        "step": "mapper",
        "step_label": "Mapper",
        "posterior_hateful": sigmoid(ell),
        "entropy": ent(sigmoid(ell)),
        "prediction": label_name(1 if sigmoid(ell) >= 0.5 else 0),
    }]
    raw_outputs = []
    used_calls = 0
    stopped_by_boundary = False

    for judge in MAIN_ORDER:
        judge_row = load_judge_row(dataset, video_id, judge)
        pred = judge_row.get("pred") if judge_row else None
        used_in_trace = not stopped_by_boundary
        if used_in_trace:
            used_calls += 1
            if pred in (0, 1):
                ell += (2 * int(pred) - 1) * lam
            posterior = sigmoid(ell)
            trace.append({
                "step": judge,
                "step_label": JUDGE_LABELS[judge].replace("Qwen2.5-VL-", "Qwen"),
                "posterior_hateful": posterior,
                "entropy": ent(posterior),
                "prediction": label_name(1 if posterior >= 0.5 else 0),
                "verifier_prediction": label_name(int(pred)) if pred in (0, 1) else "Unknown",
            })
            if ent(posterior) <= hbar:
                stopped_by_boundary = True

        raw_outputs.append({
            "model": JUDGE_LABELS[judge],
            "model_id": judge,
            "used_in_trace": used_in_trace,
            "prediction": label_name(int(pred)) if pred in (0, 1) else "Unknown",
            "verdict": safe_text(judge_row.get("verdict") if judge_row else ""),
            "rationale": safe_text(judge_row.get("rationale") if judge_row else ""),
            "raw_output": safe_text(judge_row.get("raw_response") if judge_row else ""),
        })

    final_p = trace[-1]["posterior_hateful"]
    final_pred = 1 if final_p >= 0.5 else 0
    gt = labels[video_id]

    group_dir = OUT_DIR / case["group"] / case["case_id"]
    frame_dir = group_dir / "frames"
    plot_path = group_dir / "posterior_trace.pdf"
    source_frames = select_source_frames(dataset, video_id)
    exported_frames = []
    for i, src in enumerate(source_frames, start=1):
        dst = frame_dir / f"frame_{i}{src.suffix.lower()}"
        copy_frame(src, dst)
        exported_frames.append(dst)
    make_contact_sheet(exported_frames, group_dir / "frames_contact_sheet.jpg")
    plot_trace(case, trace, hbar, rho, plot_path)

    return {
        **case,
        "ground_truth": label_name(gt),
        "ground_truth_value": gt,
        "mapper_prediction": label_name(base[video_id]),
        "mapper_prediction_value": int(base[video_id]),
        "mapper_posterior_hateful": float(row.get("posterior_hi", 0.5)),
        "boundary_region": {
            "lower": 1 - rho,
            "upper": rho,
            "mean_entropy_threshold": hbar,
        },
        "final_result": label_name(final_pred),
        "final_result_value": final_pred,
        "final_posterior_hateful": final_p,
        "is_correct": final_pred == gt,
        "used_verifier_calls": used_calls,
        "stopped_by_boundary": stopped_by_boundary,
        "posterior_trace": trace,
        "raw_outputs": raw_outputs,
        "assets": {
            "posterior_pdf": str(plot_path.relative_to(ROOT)),
            "posterior_png": str(plot_path.with_suffix(".png").relative_to(ROOT)),
            "frames": [str(p.relative_to(ROOT)) for p in exported_frames],
            "frames_contact_sheet": str((group_dir / "frames_contact_sheet.jpg").relative_to(ROOT)) if exported_frames else "",
        },
        "source_frames": [str(p) for p in source_frames],
    }


def write_case_files(case_data: dict) -> None:
    group_dir = OUT_DIR / case_data["group"] / case_data["case_id"]
    group_dir.mkdir(parents=True, exist_ok=True)
    with (group_dir / "case.json").open("w") as f:
        json.dump(case_data, f, indent=2, ensure_ascii=False)
    with (group_dir / "raw_outputs.md").open("w") as f:
        f.write(f"# {case_data['case_id']} ({case_data['group']})\n\n")
        f.write(f"- Dataset: {case_data['dataset']}\n")
        f.write(f"- Video ID: `{case_data['video_id']}`\n")
        f.write(f"- Case type: {case_data['case_type']}\n")
        f.write(f"- Ground truth: {case_data['ground_truth']}\n")
        f.write(f"- Mapper: {case_data['mapper_prediction']} ({case_data['mapper_posterior_hateful']:.3f})\n")
        f.write(f"- Final result: {case_data['final_result']} ({case_data['final_posterior_hateful']:.3f})\n")
        f.write(f"- Used verifier calls: {case_data['used_verifier_calls']}\n\n")
        f.write("## Posterior Trace\n\n")
        for step in case_data["posterior_trace"]:
            f.write(
                f"- {step['step_label']}: P(hateful)={step['posterior_hateful']:.3f}, "
                f"entropy={step['entropy']:.3f}, prediction={step['prediction']}\n"
            )
        f.write("\n## Raw MLLM Outputs\n\n")
        for output in case_data["raw_outputs"]:
            marker = "used" if output["used_in_trace"] else "not used after stopping"
            f.write(f"### {output['model']} ({marker})\n\n")
            f.write(f"- Prediction: {output['prediction']}\n")
            if output["verdict"]:
                f.write(f"- Verdict: {output['verdict']}\n")
            if output["rationale"]:
                f.write(f"- Rationale: {output['rationale']}\n\n")
            if output["raw_output"]:
                f.write("```text\n")
                f.write(output["raw_output"].strip())
                f.write("\n```\n\n")


def write_manifest(cases: list[dict]) -> None:
    with (OUT_DIR / "manifest.json").open("w") as f:
        json.dump({"cases": cases}, f, indent=2, ensure_ascii=False)
    with (OUT_DIR / "manifest.md").open("w") as f:
        f.write("# Case Study Assets\n\n")
        for group in ["success", "failure"]:
            f.write(f"## {group.title()} Cases\n\n")
            for case in [c for c in cases if c["group"] == group]:
                f.write(
                    f"- **{case['case_id']}** | {case['dataset']} / `{case['video_id']}` | "
                    f"{case['case_type']} | GT: {case['ground_truth']} | "
                    f"Mapper: {case['mapper_prediction']} | Final: {case['final_result']} | "
                    f"Calls: {case['used_verifier_calls']} | Correct: {case['is_correct']}\n"
                )
            f.write("\n")


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = EvalCache()
    cases = []
    for case in CASES:
        data = build_case(cache, case)
        write_case_files(data)
        cases.append(data)
    write_manifest(cases)
    print(f"Wrote {len(cases)} cases to {OUT_DIR}")


if __name__ == "__main__":
    main()
