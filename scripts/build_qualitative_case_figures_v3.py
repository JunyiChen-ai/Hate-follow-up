#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/data/jehc223/EMNLP3")
DATA_ROOT = ROOT / "datasets"
CASE_CSV = ROOT / "paper" / "analysis" / "qualitative_cases.csv"
FIG_DIR = ROOT / "paper" / "figures"
OUT_ROOT = ROOT / "results" / "boundary_rescue"

W = 1600
BG = (255, 255, 255)
PANEL_BG = (252, 253, 254)
LINE = (198, 206, 216)
GRID = (226, 231, 238)
TEXT = (30, 36, 45)
MUTED = (92, 101, 115)
QUOTE_BG = (248, 250, 253)
HIGHLIGHT = (255, 237, 158)

NORMAL = (45, 122, 90)
HATEFUL = (180, 72, 68)
UNKNOWN = (116, 124, 136)
SUCCESS_ACCENT = (34, 111, 132)
ERROR_ACCENT = (154, 88, 50)

DS_ROOT = {
    "HateMM": DATA_ROOT / "HateMM",
    "MHClip_EN": DATA_ROOT / "MHClip_EN",
    "MHClip_ZH": DATA_ROOT / "MHClip_ZH",
    "ImpliHateVid": DATA_ROOT / "ImpliHateVid",
}

LABEL = {"0": "Normal", "1": "Hateful", 0: "Normal", 1: "Hateful"}
VALUE_COLOR = {"Normal": NORMAL, "Hateful": HATEFUL, "Unavailable": UNKNOWN}
VERIFIER_ORDER = ["gemma-3-27b-it", "qwen2.5-vl-32b-awq", "internvl35-8b"]
VERIFIER_NAME = {
    "gemma-3-27b-it": "Gemma-3-27B",
    "qwen2.5-vl-32b-awq": "Qwen2.5-VL-32B-AWQ",
    "internvl35-8b": "InternVL3.5-8B",
}
IH_FILE_CANDIDATES = {
    "gemma-3-27b-it": ["offline_test_ih_gemma-3-27b-it.jsonl"],
    "qwen2.5-vl-32b-awq": ["offline_test_ih_qwen2.5-vl-32b-awq.jsonl"],
    "internvl35-8b": ["offline_test_ih_internvl35-8b.jsonl", "offline_test_ih_internvl3_5-8b.jsonl"],
}

SUCCESS = [
    {
        "panel": "A",
        "dataset": "HateMM",
        "video_id": "non_hate_video_53",
        "case": "False-positive correction",
        "frames": [1, 6, 11],
        "highlights": ["does not contain direct attacks", "no explicit language or imagery"],
    },
    {
        "panel": "B",
        "dataset": "HateMM",
        "video_id": "hate_video_45",
        "case": "False-negative correction",
        "frames": [1, 6, 11],
        "highlights": ["targets individuals based on their race", "directly demeans and stereotypes"],
    },
    {
        "panel": "C",
        "dataset": "MHClip_EN",
        "video_id": "tOsD1F--EEo",
        "case": "Correct early stop",
        "frames": [1, 6, 11],
        "highlights": ["no observable mockery", "personal reflection and commentary"],
    },
]

ERROR = [
    {
        "panel": "A",
        "dataset": "HateMM",
        "video_id": "hate_video_149",
        "case": "Low-salience textual evidence",
        "frames": [0, 4, 9],
        "highlights": ["antisemitic remarks", "no evidence of mocking", "No valid verifier output"],
    },
    {
        "panel": "B",
        "dataset": "MHClip_EN",
        "video_id": "R3Xt1__7TwQ",
        "case": "Contextualized gendered language",
        "frames": [0, 4, 9],
        "highlights": ["sexually suggestive", "lighthearted conversation", "humorous podcast discussion"],
    },
    {
        "panel": "C",
        "dataset": "MHClip_ZH",
        "video_id": "BV1kT411t7ax",
        "case": "Coded or satirical wording",
        "frames": [0, 4, 9],
        "highlights": ["exploitative and objectifying context", "metaphorical or satirical reference", "no evidence of mocking"],
    },
]


def font_path(*candidates: str) -> str:
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError(candidates[0])


LATIN = font_path("/usr/share/fonts/urw-base35/NimbusSans-Regular.otf", "/data/jehc223/home/miniconda3/fonts/DejaVuSans.ttf")
LATIN_BOLD = font_path("/usr/share/fonts/urw-base35/NimbusSans-Bold.otf", LATIN)
CJK = font_path("/usr/share/fonts/google-droid/DroidSansFallback.ttf", LATIN)


def font(size: int, bold: bool = False, cjk: bool = False) -> ImageFont.FreeTypeFont:
    if cjk and not bold:
        return ImageFont.truetype(CJK, size)
    return ImageFont.truetype(LATIN_BOLD if bold else LATIN, size)


F = {
    "panel": font(32, True),
    "case": font(30, True),
    "label": font(22, True),
    "small": font(20),
    "small_b": font(20, True),
    "tiny": font(17),
    "quote": font(20, cjk=True),
    "value": font(19, True),
}


def load_cases() -> dict[tuple[str, str], dict[str, str]]:
    out = {}
    with open(CASE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["dataset"], row["video_id"])] = row
    return out


def ld_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_path(judge: str, ds: str) -> Path | None:
    if ds == "ImpliHateVid":
        for name in IH_FILE_CANDIDATES[judge]:
            p = OUT_ROOT / ds / name
            if p.exists():
                return p
        return None
    p = OUT_ROOT / ds / f"offline_test_{judge}.jsonl"
    return p if p.exists() else None


def load_verifier_rows() -> dict[tuple[str, str, str], dict]:
    rows = {}
    for ds in {s["dataset"] for s in SUCCESS + ERROR}:
        for judge in VERIFIER_ORDER:
            p = judge_path(judge, ds)
            if p is None:
                continue
            for row in ld_jsonl(p):
                rows[(ds, row.get("video_id"), judge)] = row
    return rows


def sanitize(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split())


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_px: int) -> list[str]:
    words = sanitize(text).split()
    lines: list[str] = []
    cur = ""
    for word in words:
        cand = word if not cur else f"{cur} {word}"
        if text_size(draw, cand, fnt)[0] <= max_px:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy,
    text: str,
    fnt: ImageFont.FreeTypeFont,
    max_px: int,
    max_lines: int,
    fill=TEXT,
    highlights: list[str] | None = None,
    line_gap=4,
):
    x, y = xy
    lines = wrap_text(draw, text, fnt, max_px)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,;:") + " ..."
    highlights = [h.lower() for h in (highlights or [])]
    lh = text_size(draw, "Ag", fnt)[1] + line_gap
    for i, line in enumerate(lines):
        yy = y + i * lh
        if any(h and h in line.lower() for h in highlights):
            draw.rounded_rectangle((x - 3, yy - 2, x + min(max_px, text_size(draw, line, fnt)[0]) + 7, yy + lh - 1), 3, fill=HIGHLIGHT)
        draw.text((x, yy), line, font=fnt, fill=fill)
    return y + len(lines) * lh


def value_badge(draw: ImageDraw.ImageDraw, xy, value: str):
    x, y = xy
    color = VALUE_COLOR[value]
    w, h = text_size(draw, value, F["value"])
    draw.rounded_rectangle((x, y, x + w + 20, y + 29), 8, fill=color)
    draw.text((x + 10, y + 5), value, font=F["value"], fill=(255, 255, 255))
    return x + w + 20


def frame_files(dataset: str, video_id: str) -> list[Path]:
    for sub in ("frames_16", "frames"):
        d = DS_ROOT[dataset] / sub / video_id
        files = sorted(d.glob("*.jpg")) if d.exists() else []
        if files:
            return files
    raise FileNotFoundError(f"No frames for {dataset}/{video_id}")


def crop_fill(im: Image.Image, size: tuple[int, int]):
    tw, th = size
    im = im.convert("RGB")
    sw, sh = im.size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    return im.crop(((nw - tw) // 2, (nh - th) // 2, (nw + tw) // 2, (nh + th) // 2))


def make_strip(dataset: str, video_id: str, idxs: list[int], size: tuple[int, int]):
    files = frame_files(dataset, video_id)
    w, h = size
    gap = 5
    tw = (w - 2 * gap) // 3
    out = Image.new("RGB", size, (232, 237, 243))
    for i, idx in enumerate(idxs):
        frame = crop_fill(Image.open(files[min(idx, len(files) - 1)]), (tw, h))
        out.paste(frame, (i * (tw + gap), 0))
    return out


def paste_image(base: Image.Image, im: Image.Image, xy):
    base.paste(im, xy)


def verifier_label(pred) -> str:
    if pred == 0:
        return "Normal"
    if pred == 1:
        return "Hateful"
    return "Unavailable"


def verifier_text(row: dict | None) -> str:
    if not row or row.get("pred") not in (0, 1):
        return "No valid verifier output was returned for this call."
    return sanitize(row.get("rationale") or row.get("raw_response") or "")


def draw_trace_table(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, row: dict):
    draw.text((x, y), "Decision trace", font=F["label"], fill=TEXT)
    y += 34
    trace = [
        ("Ground truth", LABEL[row["label"]]),
        ("Boundary Mapper", LABEL[row["stage1_pred"]]),
        ("Adaptive Resolver", LABEL[row["final_pred"]]),
    ]
    rh = 45
    draw.rectangle((x, y, x + w, y + rh * len(trace)), outline=GRID, width=1)
    for i, (name, value) in enumerate(trace):
        yy = y + i * rh
        if i:
            draw.line((x, yy, x + w, yy), fill=GRID, width=1)
        draw.text((x + 10, yy + 12), name, font=F["small_b"], fill=TEXT)
        value_badge(draw, (x + w - 105, yy + 8), value)


def draw_verifier_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    call_idx: int,
    judge: str,
    row: dict | None,
    highlights: list[str],
    max_lines: int,
):
    pred = verifier_label(row.get("pred") if row else None)
    color = VALUE_COLOR[pred]
    draw.rectangle((x, y, x + w, y + h), fill=QUOTE_BG, outline=GRID, width=1)
    draw.rectangle((x, y, x + 5, y + h), fill=color)
    draw.text((x + 13, y + 9), f"Call {call_idx}: {VERIFIER_NAME[judge]}", font=F["small_b"], fill=TEXT)
    value_badge(draw, (x + w - 112, y + 7), pred)
    draw_wrapped(
        draw,
        (x + 13, y + 43),
        verifier_text(row),
        F["quote"],
        max_px=w - 28,
        max_lines=max_lines,
        highlights=highlights,
        line_gap=3,
    )


def draw_case_panel(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    y: int,
    h: int,
    spec: dict,
    case_row: dict,
    verifier_rows: dict[tuple[str, str, str], dict],
    accent: tuple[int, int, int],
):
    x0 = 26
    pw = W - 2 * x0
    draw.rectangle((x0, y, x0 + pw, y + h), fill=PANEL_BG, outline=LINE, width=1)
    draw.rectangle((x0, y, x0 + pw, y + 7), fill=accent)

    header_y = y + 20
    draw.text((x0 + 18, header_y), f"{spec['panel']}. {spec['case']}", font=F["case"], fill=TEXT)
    meta = f"{spec['dataset'].replace('_', '-')}  |  video id: {spec['video_id']}"
    draw.text((x0 + 18, header_y + 36), meta, font=F["small"], fill=MUTED)

    content_y = y + 88
    frame_w, frame_h = 375, 190
    strip = make_strip(spec["dataset"], spec["video_id"], spec["frames"], (frame_w, frame_h))
    paste_image(img, strip, (x0 + 18, content_y))
    draw.rectangle((x0 + 18, content_y, x0 + 18 + frame_w, content_y + frame_h), outline=GRID, width=1)
    draw.text((x0 + 18, content_y + frame_h + 10), "Sampled video frames", font=F["tiny"], fill=MUTED)

    trace_x = x0 + 405
    trace_w = 330
    draw_trace_table(draw, trace_x, content_y, trace_w, case_row)

    right_x = x0 + 762
    right_w = x0 + pw - right_x - 18
    draw.text((right_x, header_y + 8), "Verifier rationales (raw excerpts)", font=F["label"], fill=TEXT)

    calls = int(case_row["calls"])
    gap = 9
    box_y = content_y
    box_h = (y + h - 22 - box_y - gap * (calls - 1)) // calls
    max_lines = max(2, (box_h - 50) // (text_size(draw, "Ag", F["quote"])[1] + 3))
    for i, judge in enumerate(VERIFIER_ORDER[:calls]):
        row = verifier_rows.get((spec["dataset"], spec["video_id"], judge))
        draw_verifier_box(
            draw,
            right_x,
            box_y + i * (box_h + gap),
            right_w,
            box_h,
            i + 1,
            judge,
            row,
            spec["highlights"],
            max_lines=max_lines,
        )


def save(img: Image.Image, stem: str):
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    img.save(png, quality=96)
    img.convert("RGB").save(pdf, "PDF", resolution=300)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def build_figure(stem: str, specs: list[dict], accent, panel_h: int):
    cases = load_cases()
    verifier_rows = load_verifier_rows()
    top = 24
    gap = 16
    h = top * 2 + len(specs) * panel_h + (len(specs) - 1) * gap
    img = Image.new("RGB", (W, h), BG)
    draw = ImageDraw.Draw(img)
    for i, spec in enumerate(specs):
        key = (spec["dataset"], spec["video_id"])
        draw_case_panel(img, draw, top + i * (panel_h + gap), panel_h, spec, cases[key], verifier_rows, accent)
    save(img, stem)


def audit_inputs():
    cases = load_cases()
    verifier_rows = load_verifier_rows()
    missing = []
    for spec in SUCCESS + ERROR:
        key = (spec["dataset"], spec["video_id"])
        if key not in cases:
            missing.append(f"case metadata missing: {key}")
            continue
        try:
            frame_files(*key)
        except FileNotFoundError as exc:
            missing.append(str(exc))
    if missing:
        raise RuntimeError("\n".join(missing))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    audit_inputs()
    build_figure("qualitative_success_cases", SUCCESS, SUCCESS_ACCENT, panel_h=425)
    build_figure("qualitative_error_patterns", ERROR, ERROR_ACCENT, panel_h=455)


if __name__ == "__main__":
    main()
