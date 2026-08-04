"""Shared hardened vLLM runner for the E3 GPU scripts.

Encapsulates the judge_offline-identical engine config plus the resilience the
first GPU jobs lacked:
  - fail-loud: run() returns exact (written, crashed, unwritten) sets; callers
    exit nonzero unless every non-crashed video was written (the earlier scripts
    swallowed an engine death and exited 0, masking the failure and letting the
    && chain continue).
  - crash isolation: a batch that raises is retried one video at a time; a video
    that still raises is recorded to a persistent crash-skip file and excluded on
    resubmission (the vLLM 0.11.0 AWQ "illegal memory access" is unrecoverable
    in-process, so the offending video is quarantined, matching judge_offline's
    SKIP_VIDEOS philosophy).
  - bounded re-init: after an engine death the engine is torn down and
    re-initialized (up to max_reinits) so one bad video does not abort the whole
    run.
  - fsync per write; truncation detection + one retry at doubled max_tokens.

Engine args are byte-identical to src/boundary_rescue/judge_offline.py (the
known-good config that produced the offline_test files).
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = Path("/data/jehc223/EMNLP2")
sys.path.insert(0, str(ROOT / "src" / "boundary_rescue"))
sys.path.insert(0, str(ROOT / "src" / "our_method"))
sys.path.insert(0, str(ROOT / "src" / "naive_baseline"))

import judge_offline as jo  # noqa: E402  (prompts, parse, frame resolver)
from data_utils import get_media_path  # noqa: E402

MODEL_ID = {
    "qwen2.5-vl-72b-awq": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
    "qwen2.5-vl-32b-awq": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "internvl35-8b": "OpenGVLab/InternVL3_5-8B",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
}


def read_id_file(path: Path) -> set:
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
            if r.get("video_id"):
                out.add(r["video_id"])
        except Exception:
            pass
    return out


class GpuRunner:
    def __init__(self, model_id: str, gpu_mem: float, max_model_len: int,
                 max_reinits: int = 3):
        self.model_id = model_id
        self.gpu_mem = gpu_mem
        self.max_model_len = max_model_len
        self.max_reinits = max_reinits
        self.reinits = 0
        self.llm = None
        self.processor = None
        self.use_generate = jo.is_qwen25(model_id)

    # ---- engine lifecycle (judge_offline-identical args) ----
    def init(self):
        from vllm import LLM
        is_qwen = "qwen" in self.model_id.lower()
        kwargs = dict(
            model=self.model_id, trust_remote_code=True,
            gpu_memory_utilization=self.gpu_mem, max_model_len=self.max_model_len,
            limit_mm_per_prompt={"video": 1, "image": jo.NUM_FRAMES},
            enforce_eager=True,
        )
        if is_qwen:
            kwargs["mm_processor_kwargs"] = {"max_pixels": 100352}
        if not self.use_generate:
            kwargs["allowed_local_media_path"] = "/data/jehc223"
        logging.info(f"[{self.model_id}] init vLLM "
                     f"(path={'generate' if self.use_generate else 'chat'})")
        self.llm = LLM(**kwargs)
        if self.use_generate:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True)

    def teardown(self):
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment, destroy_model_parallel)
            self.llm = None
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception as e:  # noqa: BLE001
            logging.warning(f"teardown warning: {e}")
        try:
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            gc.collect()

    def reinit(self) -> bool:
        """Tear down and re-init once. Returns False if budget exhausted."""
        self.teardown()
        if self.reinits >= self.max_reinits:
            return False
        self.reinits += 1
        logging.warning(f"[{self.model_id}] re-init {self.reinits}/{self.max_reinits}")
        self.init()
        return True

    # ---- input construction (judge_offline-identical) ----
    def make_input(self, ds, vid, ann, prompt_text):
        media = get_media_path(vid, ds)
        if media is None:
            return None
        media_path, media_type = media
        frames = jo._resolve_frames(media_path, media_type, ds, vid)
        if not frames:
            return None
        if self.use_generate:
            from PIL import Image, ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            pil = [Image.open(p).convert("RGB") for p in frames]
            content = [{"type": "image"} for _ in pil]
            content.append({"type": "text", "text": prompt_text})
            msgs = [{"role": "system", "content": jo.SYSTEM_MSG},
                    {"role": "user", "content": content}]
            prompt_str = self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
            return ({"prompt": prompt_str, "multi_modal_data": {"image": pil}},
                    len(frames), media_type)
        if media_type == "video":
            media_content = [{"type": "video_url",
                              "video_url": {"url": f"file://{media_path}"}}]
        else:
            media_content = [{"type": "image_url", "image_url": {"url": f"file://{p}"}}
                             for p in frames]
        content = media_content + [{"type": "text", "text": prompt_text}]
        msgs = [{"role": "system", "content": jo.SYSTEM_MSG},
                {"role": "user", "content": content}]
        return (msgs, len(frames), media_type)

    def _gen(self, inputs, max_tokens):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0, max_tokens=max_tokens)
        if self.use_generate:
            return self.llm.generate(inputs, sampling_params=sp)
        return self.llm.chat(messages=inputs, sampling_params=sp)

    def _write(self, out_path, ds, vid, ann, out, n_frames, media_type,
               max_tokens, extra):
        text = out.outputs[0].text if out is not None and out.outputs else ""
        finish = out.outputs[0].finish_reason if out is not None and out.outputs else None
        verdict, pred = jo.parse_verdict(text)
        if pred == -1 and finish == "length" and out is not None:
            # truncation: one retry at doubled budget (single item)
            inp = self.make_input(ds, vid, ann, extra["_prompt_text"])
            if inp is not None:
                o2 = self._gen([inp[0]], max_tokens * 2)[0]
                text = o2.outputs[0].text if o2.outputs else text
                verdict, pred = jo.parse_verdict(text)
        if pred == -1:
            logging.warning(f"  {ds}/{vid}: unparseable, not writing (retry on resume)")
            return False
        raw = ann.get("label", "")
        label_int = 1 if (isinstance(raw, str) and raw.lower() in ("hateful", "offensive")) else 0
        rec = {
            "video_id": vid, "dataset": ds, "pred": pred, "label": label_int,
            "verdict": verdict, "rationale": jo._extract_rationale(text),
            "raw_response": text, "num_frames_requested": jo.NUM_FRAMES,
            "num_frames_resolved": n_frames,
            "media_input": "video" if (not self.use_generate and media_type == "video") else "frames",
        }
        rec.update({k: v for k, v in extra.items() if not k.startswith("_")})
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return True

    def run(self, work, prompt_of, extra_of, batch_size, max_tokens,
            crash_skip_path: Path) -> dict:
        """work: list of (ds, vid, ann, out_path). prompt_of(ds, ann)->str.
        extra_of(ds, vid)->dict of extra record fields. Returns stats dict."""
        crash_prev = read_id_file(crash_skip_path)
        pending = [w for w in work if w[1] not in crash_prev]
        todo_vids = {w[1] for w in pending}
        n_written = 0
        crashed = []
        self.init()
        t0 = time.time()
        i = 0
        while i < len(pending):
            batch = pending[i:i + batch_size]
            inputs, metas = [], []
            for ds, vid, ann, out_path in batch:
                ptext = prompt_of(ds, ann)
                built = self.make_input(ds, vid, ann, ptext)
                if built is None:
                    logging.warning(f"  {ds}/{vid}: no media/frames, skipping")
                    continue
                inp, nf, mt = built
                ex = dict(extra_of(ds, vid)); ex["_prompt_text"] = ptext
                inputs.append(inp)
                metas.append((ds, vid, ann, out_path, nf, mt, ex))
            if inputs:
                try:
                    outs = self._gen(inputs, max_tokens)
                    for meta, out in zip(metas, outs):
                        ds, vid, ann, out_path, nf, mt, ex = meta
                        if self._write(out_path, ds, vid, ann, out, nf, mt, max_tokens, ex):
                            n_written += 1
                except BaseException as e:  # noqa: BLE001  (engine likely dead)
                    logging.error(f"  batch failed ({str(e)[:160]}); isolating videos")
                    if not self.reinit():
                        crashed.extend(m[1] for m in metas)
                        _append_ids(crash_skip_path, [m[1] for m in metas])
                        break
                    # re-process this batch one video at a time to attribute
                    for meta in metas:
                        ds, vid, ann, out_path, nf, mt, ex = meta
                        built = self.make_input(ds, vid, ann, ex["_prompt_text"])
                        if built is None:
                            continue
                        inp1, nf1, mt1 = built
                        try:
                            o1 = self._gen([inp1], max_tokens)[0]
                            if self._write(out_path, ds, vid, ann, o1, nf1, mt1, max_tokens, ex):
                                n_written += 1
                        except BaseException as e2:  # noqa: BLE001
                            logging.error(f"  {ds}/{vid}: CRASHES engine ({str(e2)[:120]}); "
                                          f"quarantining")
                            crashed.append(vid)
                            _append_ids(crash_skip_path, [vid])
                            if not self.reinit():
                                logging.error("  re-init budget exhausted; stopping")
                                i = len(pending)
                                break
            i += len(batch)
            el = time.time() - t0
            logging.info(f"  [{self.model_id}] written={n_written} "
                         f"crashed={len(crashed)} {n_written / el:.3f} vid/s")
        self.teardown()
        done = _done_union(work)
        unwritten = sorted(todo_vids - done - set(crashed))
        return {"n_written": n_written, "crashed": sorted(set(crashed)),
                "unwritten": unwritten, "n_todo": len(todo_vids)}


def _append_ids(path: Path, vids) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for v in vids:
            f.write(v + "\n")
        f.flush()
        os.fsync(f.fileno())


def _done_union(work) -> set:
    """Union of written video_ids across all distinct out_paths in `work`."""
    paths = {w[3] for w in work}
    out = set()
    for p in paths:
        out |= done_ids(Path(p))
    return out
