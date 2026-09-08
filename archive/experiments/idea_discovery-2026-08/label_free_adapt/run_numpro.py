#!/usr/bin/env python3
"""A01 NumPro transplant on a sanitized manifest using local Qwen3-VL."""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .mechanisms import numbered_timeline_prompt, parse_numbered_intervals
from .schema import Prediction, append_jsonl, intervals_to_curve


def sample_numbered_frames(video_path: str, nframes: int) -> tuple[list[Image.Image], list[float]]:
    raw_frames, indices, fps = None, None, None
    try:
        import decord
        reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
        if len(reader) == 0:
            raise ValueError("empty video")
        indices = np.linspace(0, len(reader) - 1, min(nframes, len(reader))).round().astype(int)
        fps = float(reader.get_avg_fps())
        raw_frames = [reader[int(i)].asnumpy() for i in indices]
    except Exception as decord_error:
        import cv2
        capture = cv2.VideoCapture(video_path)
        count, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))
        if not capture.isOpened() or count <= 0 or fps <= 0:
            capture.release()
            raise ValueError(f"video decode failed: {decord_error}")
        indices = np.linspace(0, count - 1, min(nframes, count)).round().astype(int)
        raw_frames = []
        opencv_error = None
        for i in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = capture.read()
            if not ok:
                opencv_error = ValueError(f"OpenCV failed at frame {i}")
                break
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        if opencv_error is not None:
            # Some AV1 files are visible to ffprobe but unsupported by the
            # Decord/OpenCV builds.  FFmpeg's software decoder is the final,
            # deterministic fallback; seek by time and return the same PIL path.
            duration = count / fps
            sample_count = min(nframes, max(1, count))
            # Exclude the nominal container endpoint: several web videos have
            # a shorter decodable stream than CAP_PROP_FRAME_COUNT advertises.
            times = np.arange(sample_count, dtype=float) * duration / sample_count
            decoded = []
            for timestamp in times:
                command = [
                    "/usr/bin/ffmpeg", "-v", "error", "-ss", f"{timestamp:.6f}",
                    "-i", video_path, "-frames:v", "1", "-f", "image2pipe",
                    "-vcodec", "png", "pipe:1",
                ]
                result = subprocess.run(command, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, check=False)
                if result.returncode or not result.stdout:
                    detail = result.stderr.decode("utf-8", errors="replace")[-300:]
                    raise ValueError(
                        f"all decoders failed at {timestamp:.3f}s: {detail}; "
                        f"Decord: {decord_error}; OpenCV: {opencv_error}")
                decoded.append(np.asarray(Image.open(io.BytesIO(result.stdout)).convert("RGB")))
            raw_frames = decoded
            indices = np.asarray(times) * fps
    frames, times = [], []
    for display_id, (source_id, raw_frame) in enumerate(zip(indices, raw_frames)):
        image = Image.fromarray(raw_frame).convert("RGB")
        image.thumbnail((448, 448))
        draw = ImageDraw.Draw(image)
        label = f"FRAME {display_id}"
        box = draw.textbbox((0, 0), label, font=ImageFont.load_default())
        draw.rectangle((0, 0, box[2] + 8, box[3] + 8), fill="black")
        draw.text((4, 4), label, fill="white", font=ImageFont.load_default())
        frames.append(image)
        times.append(float(source_id) / fps)
    return frames, times


def load_done(path: Path) -> set[tuple]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get("error"):
                    continue
                done.add((row["dataset"], row["video_id"]))
                done.add((row["dataset"], row["video_id"], row.get("method")))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--nframes", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20250819)
    ap.add_argument("--adapter", choices=("numpro", "seq2time"), default="numpro")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.manual_seed(args.seed)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=args.max_pixels, local_files_only=True)
    rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    done = load_done(Path(args.out))
    todo = [x for x in rows if (x["dataset"], x["video_id"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    for row in todo:
        started = time.time()
        method = "A01_NumPro" if args.adapter == "numpro" else "A11_Seq2TimeTransplant"
        prediction = Prediction(method, row["dataset"], row["video_id"],
                                float(row["duration"]), seed=args.seed)
        try:
            frames, times = sample_numbered_frames(row["video_path"], args.nframes)
            if args.adapter == "numpro":
                prompt = numbered_timeline_prompt(range(len(frames)), prediction.duration,
                                                  transcript=row.get("transcript", ""))
            else:
                positions = [round(i * 9999 / max(1, len(frames) - 1))
                             for i in range(len(frames))]
                prompt = (
                    "The images are ordered video frames. Their Seq2Time-style relative positions "
                    f"are {positions}, on a 0000..9999 timeline. Locate every interval containing "
                    "hateful or demeaning content targeting a person or protected group. Use visual "
                    "and spoken evidence without assuming the entire video is hateful. Return strict "
                    "JSON only: {\"intervals\":[{\"start_pos\":int,\"end_pos\":int,"
                    "\"confidence\":float}],\"evidence\":{}}. Return an empty intervals list only "
                    "when the evidence does not support such content. Transcript: "
                    + row.get("transcript", "")
                )
            # NumPro treats the video as numbered manga panels. Multi-image input
            # avoids inventing an internal fps for an already sampled frame list.
            content = [{"type": "image", "image": image} for image in frames]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            inputs = processor(text=[text], images=frames, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                        do_sample=False)
            generated = processor.batch_decode(
                output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0].strip()
            if args.adapter == "numpro":
                intervals, evidence = parse_numbered_intervals(
                    generated, times, prediction.duration)
            else:
                match = __import__("re").search(r"\{.*\}", generated, __import__("re").S)
                obj = json.loads(match.group(0)) if match else {"intervals": []}
                values = []
                for item in obj.get("intervals", []):
                    start = float(item["start_pos"]) / 9999 * prediction.duration
                    end = float(item["end_pos"]) / 9999 * prediction.duration
                    values.append([start, end, float(item.get("confidence", 1.0))])
                from .schema import normalize_intervals
                intervals = normalize_intervals(values, prediction.duration)
                evidence = obj.get("evidence", {})
            prediction.intervals = intervals
            prediction.score_curve = intervals_to_curve(intervals, prediction.duration)
            prediction.modality_evidence = evidence
            prediction.raw = {"response": generated, "frame_times": times,
                              "elapsed_seconds": round(time.time() - started, 3)}
        except Exception as exc:  # preserve failure coverage instead of dropping a video
            prediction.error = f"{type(exc).__name__}: {exc}"
            prediction.raw = {"elapsed_seconds": round(time.time() - started, 3)}
        append_jsonl(Path(args.out), prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
