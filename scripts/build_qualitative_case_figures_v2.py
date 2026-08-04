#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/data/jehc223/EMNLP3")
DATA_ROOT = ROOT / "datasets"
CASE_CSV = ROOT / "paper" / "analysis" / "qualitative_cases.csv"
FIG_DIR = ROOT / "paper" / "figures"

W = 2400
BG = (249, 250, 252)
CARD = (255, 255, 255)
BORDER = (222, 226, 233)
TEXT = (28, 34, 43)
MUTED = (92, 101, 116)
BLUE = (42, 105, 176)
GREEN = (70, 128, 82)
RED = (177, 69, 61)
AMBER = (184, 124, 28)
PURPLE = (132, 92, 166)
INK = (52, 63, 80)

LABEL = {0: "N", 1: "H"}
DS_ROOT = {
    "HateMM": DATA_ROOT / "HateMM",
    "MHClip_EN": DATA_ROOT / "MHClip_EN",
    "MHClip_ZH": DATA_ROOT / "MHClip_ZH",
    "ImpliHateVid": DATA_ROOT / "ImpliHateVid",
}


SUCCESS = [
    {
        "dataset": "HateMM",
        "video_id": "non_hate_video_53",
        "kind": "False-positive rescue",
        "short": "Generic disorder is not hate.",
        "detail": "The verifier panel rejects a high-risk visual scene because no protected group is targeted.",
        "accent": BLUE,
        "frames": [1, 6, 11],
    },
    {
        "dataset": "HateMM",
        "video_id": "hate_video_45",
        "kind": "False-negative rescue",
        "short": "Targeted abuse is recovered.",
        "detail": "Verifier rationales identify hateful framing that the boundary score initially underweights.",
        "accent": RED,
        "frames": [1, 6, 11],
    },
    {
        "dataset": "MHClip_EN",
        "video_id": "tOsD1F--EEo",
        "kind": "Early confirmation",
        "short": "Uncertain but already correct.",
        "detail": "One verifier confirms a normal identity discussion, so no extra verifier calls are needed.",
        "accent": GREEN,
        "frames": [1, 6, 11],
    },
]

FAILURE = [
    {
        "dataset": "HateMM",
        "video_id": "hate_video_149",
        "kind": "Low-salience text",
        "detail": "The harmful cue is in small on-screen chat text, which later verifiers inconsistently recover.",
        "accent": BLUE,
        "frames": [0, 4, 9],
    },
    {
        "dataset": "ImpliHateVid",
        "video_id": "EX_448",
        "kind": "Target-scope ambiguity",
        "detail": "Offensive language appears in a violent encounter, but whether it targets a protected group remains ambiguous.",
        "accent": AMBER,
        "frames": [0, 4, 9],
    },
    {
        "dataset": "MHClip_ZH",
        "video_id": "BV1kT411t7ax",
        "kind": "Coded or satirical wording",
        "detail": "The hateful signal is carried by slang and metaphor, causing the resolver to read the clip as commentary.",
        "accent": RED,
        "frames": [0, 4, 9],
    },
]

TAXONOMY = [
    ("Low-salience / missing text", 3, BLUE),
    ("Coded or implicit cue", 3, AMBER),
    ("Target-scope ambiguity", 2, GREEN),
    ("Noisy multimodal context", 2, PURPLE),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = f"/usr/share/fonts/urw-base35/NimbusSans-{'Bold' if bold else 'Regular'}.otf"
    return ImageFont.truetype(path, size)


F = {
    "title": font(58, True),
    "subtitle": font(32),
    "section": font(39, True),
    "body": font(34),
    "body_b": font(34, True),
    "small": font(28),
    "small_b": font(28, True),
    "tiny": font(23),
    "pill": font(26, True),
}


def load_cases() -> dict[tuple[str, str], dict[str, str]]:
    out = {}
    with open(CASE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["dataset"], row["video_id"])] = row
    return out


def load_band(dataset: str) -> dict[str, dict]:
    path = ROOT / "results" / "boundary_rescue" / dataset / "candidates_entropy_band_2b.jsonl"
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["video_id"]] = row
    return out


def rounded(draw: ImageDraw.ImageDraw, xy, radius: int, fill, outline=None, width: int = 1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy, s: str, fnt, fill=TEXT, wrap=None, spacing=7):
    if wrap is not None:
        s = "\n".join(textwrap.wrap(s, width=wrap, break_long_words=False))
    draw.multiline_text(xy, s, font=fnt, fill=fill, spacing=spacing)


def label_pill(draw: ImageDraw.ImageDraw, xy, s: str, color):
    x, y = xy
    box = draw.textbbox((0, 0), s, font=F["pill"])
    w, h = box[2] - box[0] + 32, box[3] - box[1] + 20
    rounded(draw, (x, y, x + w, y + h), 17, color)
    draw.text((x + 16, y + 8), s, font=F["pill"], fill=(255, 255, 255))
    return x + w, y + h


def arrow(draw: ImageDraw.ImageDraw, x1: int, y: int, x2: int):
    c = (128, 137, 149)
    draw.line((x1, y, x2, y), fill=c, width=4)
    draw.polygon([(x2, y), (x2 - 16, y - 9), (x2 - 16, y + 9)], fill=c)


def decision_row(draw: ImageDraw.ImageDraw, x: int, y: int, row: dict[str, str], compact=False):
    lab = int(row["label"])
    s1 = int(row["stage1_pred"])
    final = int(row["final_pred"])
    x1, _ = label_pill(draw, (x, y), f"GT {LABEL[lab]}", INK)
    arrow(draw, x1 + 18, y + 26, x1 + 62)
    x2, _ = label_pill(draw, (x1 + 80, y), f"S1 {LABEL[s1]}", GREEN if s1 == lab else RED)
    arrow(draw, x2 + 18, y + 26, x2 + 62)
    label_pill(draw, (x2 + 80, y), f"TRIAGE {LABEL[final]}", GREEN if final == lab else RED)


def frame_files(dataset: str, video_id: str) -> list[Path]:
    root = DS_ROOT[dataset]
    for sub in ("frames_16", "frames"):
        d = root / sub / video_id
        files = sorted(d.glob("*.jpg")) if d.exists() else []
        if files:
            return files
    raise FileNotFoundError(f"No frames for {dataset}/{video_id}")


def crop_fill(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    tw, th = size
    im = im.convert("RGB")
    sw, sh = im.size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    return im.crop(((nw - tw) // 2, (nh - th) // 2, (nw + tw) // 2, (nh + th) // 2))


def strip(dataset: str, video_id: str, idxs: list[int], size: tuple[int, int]) -> Image.Image:
    files = frame_files(dataset, video_id)
    w, h = size
    gap = 8
    tw = (w - gap * 2) // 3
    out = Image.new("RGB", (w, h), (238, 241, 245))
    for i, idx in enumerate(idxs):
        frame = crop_fill(Image.open(files[min(idx, len(files) - 1)]), (tw, h))
        out.paste(frame, (i * (tw + gap), 0))
    return out


def paste_round(base: Image.Image, im: Image.Image, xy: tuple[int, int], r=24):
    mask = Image.new("L", im.size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((0, 0, im.size[0], im.size[1]), radius=r, fill=255)
    base.paste(im, xy, mask)


def band_label(dataset: str, vid: str, bands: dict[str, dict[str, dict]]) -> str:
    b = bands[dataset][vid]
    return "entropy band" if b.get("in_band") else "outside band"


def save(img: Image.Image, stem: str):
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    img.save(png, quality=96)
    img.convert("RGB").save(pdf, "PDF", resolution=300.0)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def success_figure():
    cases = load_cases()
    bands = {ds: load_band(ds) for ds in {c["dataset"] for c in SUCCESS}}
    H = 1320
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((82, 50), "Why TRIAGE Works", font=F["title"], fill=TEXT)
    d.text((82, 122), "Successful corrections expose three roles of the adaptive boundary resolver.", font=F["subtitle"], fill=MUTED)

    margin = 72
    gap = 34
    cw = (W - margin * 2 - gap * 2) // 3
    top = 205
    ch = 990
    for i, spec in enumerate(SUCCESS):
        x = margin + i * (cw + gap)
        y = top
        rounded(d, (x, y, x + cw, y + ch), 34, CARD, BORDER, 2)
        d.rectangle((x, y, x + cw, y + 12), fill=spec["accent"])
        d.text((x + 34, y + 42), spec["kind"], font=F["section"], fill=TEXT)
        d.text((x + 34, y + 94), f"{spec['dataset'].replace('_', '-')} | {spec['video_id']}", font=F["small"], fill=MUTED)

        frames = strip(spec["dataset"], spec["video_id"], spec["frames"], (cw - 68, 335))
        paste_round(img, frames, (x + 34, y + 155), 26)

        row = cases[(spec["dataset"], spec["video_id"])]
        decision_row(d, x + 34, y + 535, row)
        calls = row["calls"]
        d.text((x + 34, y + 620), f"{band_label(spec['dataset'], spec['video_id'], bands)} | {calls} verifier call(s)", font=F["small_b"], fill=spec["accent"])
        text(d, (x + 34, y + 684), spec["short"], F["body_b"], fill=TEXT, wrap=25, spacing=8)
        text(d, (x + 34, y + 770), spec["detail"], F["body"], fill=TEXT, wrap=30, spacing=8)

    save(img, "qualitative_success_cases_v2")


def error_figure():
    cases = load_cases()
    bands = {ds: load_band(ds) for ds in {c["dataset"] for c in FAILURE}}
    H = 1390
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((82, 50), "Error Pattern Analysis", font=F["title"], fill=TEXT)
    d.text((82, 122), "Remaining errors concentrate where key evidence is visually subtle or semantically underspecified.", font=F["subtitle"], fill=MUTED)

    # Taxonomy: one clean band, no extra grid.
    x0, y0 = 80, 205
    rounded(d, (x0, y0, W - 80, y0 + 245), 34, CARD, BORDER, 2)
    d.text((x0 + 36, y0 + 32), "Failure taxonomy", font=F["section"], fill=TEXT)
    d.text((x0 + 36, y0 + 82), "Manual coding over 10 sampled final errors", font=F["small"], fill=MUTED)
    bx = x0 + 560
    by = y0 + 52
    max_c = max(v for _, v, _ in TAXONOMY)
    for i, (name, count, color) in enumerate(TAXONOMY):
        yy = by + i * 44
        d.text((bx, yy - 4), name, font=F["small"], fill=TEXT)
        bar_x = bx + 470
        bar_w = int(330 * count / max_c)
        rounded(d, (bar_x, yy, bar_x + bar_w, yy + 28), 12, color)
        d.text((bar_x + bar_w + 18, yy - 2), str(count), font=F["small_b"], fill=TEXT)

    margin = 72
    gap = 34
    cw = (W - margin * 2 - gap * 2) // 3
    top = 520
    ch = 775
    for i, spec in enumerate(FAILURE):
        x = margin + i * (cw + gap)
        y = top
        rounded(d, (x, y, x + cw, y + ch), 34, CARD, BORDER, 2)
        d.rectangle((x, y, x + cw, y + 12), fill=spec["accent"])
        d.text((x + 34, y + 42), spec["kind"], font=F["section"], fill=TEXT)
        d.text((x + 34, y + 88), f"{spec['dataset'].replace('_', '-')} | {spec['video_id']}", font=F["small"], fill=MUTED)
        frames = strip(spec["dataset"], spec["video_id"], spec["frames"], (cw - 68, 285))
        paste_round(img, frames, (x + 34, y + 140), 24)
        row = cases[(spec["dataset"], spec["video_id"])]
        decision_row(d, x + 34, y + 460, row)
        d.text((x + 34, y + 540), band_label(spec["dataset"], spec["video_id"], bands), font=F["small_b"], fill=spec["accent"])
        text(d, (x + 34, y + 595), spec["detail"], F["body"], fill=TEXT, wrap=29, spacing=8)

    save(img, "qualitative_error_patterns_v2")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    success_figure()
    error_figure()


if __name__ == "__main__":
    main()
