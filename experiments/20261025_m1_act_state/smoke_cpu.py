#!/usr/bin/env python3
"""Candidate 50: CPU smoke test of everything that runs without a GPU. Run with the HateVLM python (transformers
5.15.1, the r6 reader's version); the decoder dry run uses --cpu-python (HateVideo, as for every r6 decode).

Parts:
 1 measure.py argument parsing and the stage-0 / stage-1 processing and donor plans (--plan-only).
 2 Mask spans on real tokenizations (processor only, no weights): prefix + verdict span + token regions for the first
   videos of each corpus; keep vectors of every arm; the A1 / all_local difference is exactly the verdict span; the
   kept-token counts reproduce Grounder's all_local records; the prefix length reproduces the r6 reads; the verdict
   span positions from get_rope_index are plain text positions; donor and own span tokens are identical.
 3 act_state plumbing with stand-ins: the attention pre-hook injects the right mask at all 36 layers (keyword and
   positional paths); swapped_cache replaces exactly the span; rotate_states equals the model's own RoPE formula.
 4 stage0.py dry run on Grounder's archived reads (all_local = all_local, A1 := base, A1_other := verdict_only).
 5 stage1.py dry run on Grounder's archived reads (base, all_local, A1 := late, A1_other := shifted), decoded with
   the r6 decoder; the base replica must reproduce runs/20260926_twolevel/final_rawkey exactly. The gate verdicts of a
   dry run mean nothing. Rule 10: this reads test GT through the evaluator on reads whose numbers are already in the
   Grounder README.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
G = ROOT / "runs/20261002_m1_grounder/r1_full"


def section(name):
    print(f"\n==== {name}", flush=True)


def part1(py):
    section("1 measure.py arguments and plans")
    subprocess.run([py, str(HERE / "measure.py"), "--help"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    for args in (["--datasets", "HateMM", "HateClipSeg", "DeHate", "--first", "20", "--arms", "all_local", "A1", "A1_other"],
                 ["--datasets", "HateMM", "HateClipSeg"], ["--datasets", "DeHate", "--arms", "all_local", "A1"]):
        out = subprocess.run([py, str(HERE / "measure.py"), "--plan-only", *args], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout
        body = out.rsplit("PLAN_DONE", 1)[0]
        p = json.loads(body)
        for ds, d in p.items():
            assert d["order"] == list(reversed(d["todo"]))
            for v, dn in d["donors"].items():
                assert dn != v and (dn not in d["sign"] or d["sign"][dn] == d["sign"][v]), (ds, v, dn)
            print(f"  {' '.join(args):70s} {ds:12s} todo {len(d['todo']):4d}  yes/no {d['n_yes']}/{d['n_no']}  "
                  f"donor passes {d['donor_passes']}")
    try:
        subprocess.run([py, str(HERE / "measure.py")], cwd=ROOT, check=True, capture_output=True)
        raise AssertionError("measure.py without --run-name should fail")
    except subprocess.CalledProcessError:
        print("  measure.py without --run-name: argparse error as intended")


def processor_judge():
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from src.mllm_judge import FAMILY_IMAGE_KW, MODEL, Judge
    import torch
    j = Judge.__new__(Judge)      # processor-only judge: no weights; prompt / tokenisation methods unchanged
    j.model_id = MODEL
    j.processor = AutoProcessor.from_pretrained(MODEL)
    j.tok = j.processor.tokenizer
    j.family = "qwen3_vl"
    j.img_kw = dict(FAMILY_IMAGE_KW[j.family])
    j.list_content = False
    cfg = AutoConfig.from_pretrained(MODEL)
    j.image_token_id = cfg.image_token_id
    j.device = torch.device("cpu")
    j.dtype = torch.bfloat16
    try:
        j.render([{"role": "user", "content": [{"type": "text", "text": "a"}]},
                  {"role": "user", "content": [{"type": "text", "text": "b"}]}], True)
        j.same_turn = False
    except Exception:
        j.same_turn = True
    j.loose_stance_seam = False

    class Inner:                 # only what get_rope_index needs (config + its own helper)
        get_rope_index = Qwen3VLModel.get_rope_index
        get_vision_position_ids = Qwen3VLModel.get_vision_position_ids
        rope_deltas = None

        def __init__(self, config):
            self.config = config

    class Outer:
        pass
    j.model = Outer()
    j.model.model = Inner(cfg)
    return j, cfg


def part2(videos_per_corpus):
    section("2 mask spans on real tokenisations (processor only)")
    from act_state import ARMS, span_positions, window_keep
    from measure import GROUNDER_ALL_LOCAL, R6_READS, corpus_rows, jl
    from src.mllm_judge import VIDEO_QUESTION
    from src.video_inputs import fixed_windows, frame_paths, load_asr
    from src.window_token_regions import token_regions
    j, _ = processor_judge()
    span_text = {}
    for ds in ("HateMM", "HateClipSeg", "DeHate"):
        rows = corpus_rows(ds)
        r6 = {k[1]: r for k, r in jl(R6_READS[ds]).items() if k[0] == ds}
        gal = {k[1]: r for k, r in jl(GROUNDER_ALL_LOCAL[ds]).items() if k[0] == ds} if ds in GROUNDER_ALL_LOCAL else {}
        asr = load_asr(ds)
        for vid in sorted(rows)[:videos_per_corpus]:
            t = time.time()
            row = rows[vid]
            segs = asr.get(vid, [])
            frames = frame_paths(ds, vid, 20, "k20")
            wins = fixed_windows(float(row["duration"]), 8)
            msgs, files = j.prefix_messages(frames, segs)
            prefix, enc = j.encode_prefix(msgs, files)
            regions = token_regions(j, prefix, enc, frames, segs, wins)
            P = regions["prefix_len"]
            stance = "Yes" if float(r6[vid]["extra"]["z_video"]) > 0 else "No"
            b0, b0text = j.branch_ids(msgs, VIDEO_QUESTION)
            a0, a0text = j.answer_ids(msgs, VIDEO_QUESTION, stance)
            span = list(b0) + list(a0)
            ncache = P + len(span)
            assert P == r6[vid]["extra"]["prefix_tokens"], (ds, vid, P, r6[vid]["extra"]["prefix_tokens"])
            assert j.tok.decode(span) == b0text + a0text
            span_text.setdefault(stance, (b0text + a0text, span))
            assert span_text[stance][1] == span, "verdict span tokens differ between videos with the same stance"
            assert b0text.lstrip().startswith("<|im_start|>user\nDoes this video contain") and a0text.startswith(stance)
            keeps = {arm: [window_keep(regions, i, ncache, arm) for i in range(len(wins))] for arm in ARMS}
            for i in range(len(wins)):
                assert keeps["base"][i].all()
                assert not keeps["all_local"][i][P:].any() and keeps["A1"][i][P:].all()
                assert np.array_equal(keeps["A1"][i], keeps["A1_other"][i])
                diff = np.flatnonzero(keeps["A1"][i] != keeps["all_local"][i])
                assert np.array_equal(diff, np.arange(P, ncache)), "A1 and all_local must differ exactly on the span"
                assert np.array_equal(keeps["all_local"][i][:P], regions["scaffolding"] | regions["local"][i])
            if gal:
                g = gal[vid]
                assert g["extra"]["cached_tokens"] == ncache and g["extra"]["prefix_tokens"] == P
                for i, w in enumerate(g["extra"]["windows"]):
                    m = regions["local"][i]
                    assert int((m & regions["visual"]).sum()) == w["kept_visual_tokens"], (ds, vid, i)
                    assert int((m & regions["speech"]).sum()) == w["kept_speech_tokens"], (ds, vid, i)
            pos = span_positions(j, enc, span)           # asserts plain text positions on all three axes
            vis_kept = [int((regions["local"][i] & regions["visual"]).sum()) for i in range(len(wins))]
            sp_kept = [int((regions["local"][i] & regions["speech"]).sum()) for i in range(len(wins))]
            print(f"  {ds:12s} {vid:18s} windows {len(wins):3d}  prefix {P:5d}  span {len(span)} tokens at positions "
                  f"{int(pos[0, 0, 0])}-{int(pos[0, 0, -1])} (delta {int(pos[0, 0, 0]) - P})  scaffold "
                  f"{int(regions['scaffolding'].sum())}  kept visual per window {vis_kept[:6]}  speech {sp_kept[:6]}  "
                  f"{'grounder-match' if gal else 'no grounder ref'}  {time.time() - t:.1f}s", flush=True)
    for stance, (text, ids) in span_text.items():
        print(f"  verdict span for stance {stance} ({len(ids)} tokens): {text!r}")


def part3():
    section("3 act_state plumbing with stand-ins")
    import torch
    from transformers import AutoConfig
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb
    from act_state import QueryAccess, donor_map, rotate_states, rotation_check, swapped_cache, window_keep
    from src.mllm_judge import MODEL
    cfg = AutoConfig.from_pretrained(MODEL)
    nl, kvh, hd = cfg.text_config.num_hidden_layers, cfg.text_config.num_key_value_heads, cfg.text_config.head_dim

    class Attn(torch.nn.Module):
        def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kw):
            self.seen = attention_mask
            return hidden_states

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

    class FJ:
        family = "qwen3_vl"
        device = torch.device("cpu")
        dtype = torch.bfloat16

        class model:
            class model:
                class language_model:
                    layers = torch.nn.ModuleList([Layer() for _ in range(nl)])
    qa = QueryAccess(FJ)
    P, span, q = 50, 7, 5
    n = P + span
    regions = {"prefix_len": P, "scaffolding": np.r_[np.ones(10, bool), np.zeros(P - 10, bool)],
               "local": [np.r_[np.zeros(20, bool), np.ones(8, bool), np.zeros(P - 28, bool)]]}
    x = torch.zeros(1, q, 4)
    for arm, restricted in (("base", False), ("all_local", True), ("A1", True), ("A1_other", True)):
        keep = window_keep(regions, 0, n, arm)
        for positional in (False, True):
            with qa.restrict(n, q, keep, restricted):
                for layer in FJ.model.model.language_model.layers:
                    if positional:
                        layer.self_attn(x, None, None)
                    else:
                        layer.self_attn(x, None, attention_mask=None)
            for layer in FJ.model.model.language_model.layers:
                m = layer.self_attn.seen
                assert m.shape == (1, 1, q, n + q) and m.dtype == torch.bfloat16
                visible = (m[0, 0] == 0).numpy()
                for r in range(q):
                    assert np.array_equal(visible[r, :n], keep), (arm, r)
                    assert visible[r, n:n + r + 1].all() and not visible[r, n + r + 1:].any()
        print(f"  mask {arm:9s}: injected at {nl} layers (keyword and positional), visible prefix columns "
              f"{int(keep[:P].sum())}/{P}, span columns {int(keep[P:].sum())}/{span}")
    qa.close()
    # swapped_cache replaces exactly the span and leaves the source cache untouched
    torch.manual_seed(0)
    cache = DynamicCache()
    for layer in range(nl):
        cache.update(torch.randn(1, kvh, n, hd).bfloat16(), torch.randn(1, kvh, n, hd).bfloat16(), layer)
    before = [(lay.keys.clone(), lay.values.clone()) for lay in cache.layers]
    new = [(torch.randn(1, kvh, span, hd).bfloat16(), torch.randn(1, kvh, span, hd).bfloat16()) for _ in range(nl)]
    sw = swapped_cache(cache, P, new)
    for lay, lay0, (k0, v0), (k, v) in zip(sw.layers, cache.layers, before, new):
        assert torch.equal(lay0.keys, k0) and torch.equal(lay0.values, v0)
        assert torch.equal(lay.keys[:, :, :P], k0[:, :, :P]) and torch.equal(lay.values[:, :, :P], v0[:, :, :P])
        assert torch.equal(lay.keys[:, :, P:], k) and torch.equal(lay.values[:, :, P:], v)
    print("  swapped_cache: span replaced at every layer, prefix and source cache unchanged")
    # rotate_states equals the model's RoPE (apply_rotary_pos_emb with the model's rotary module)
    rotary = Qwen3VLTextRotaryEmbedding(cfg.text_config)
    pre = [(torch.randn(1, kvh, span, hd).bfloat16() * 3, torch.randn(1, kvh, span, hd).bfloat16()) for _ in range(3)]
    pos = (torch.arange(span) + 1234)[None, None].expand(3, 1, -1)
    rot = rotate_states(pre, pos, rotary)
    for (k, v), (rk, rv) in zip(pre, rot):
        cos, sin = rotary(k, pos)
        ref, _ = apply_rotary_pos_emb(k, k, cos, sin)
        assert torch.equal(rk, ref) and rv is v
    other = rotate_states(pre, pos + 100, rotary)
    assert not torch.equal(other[0][0], rot[0][0])
    c2 = DynamicCache()
    for layer, (rk, rv) in enumerate(rot):
        c2.update(torch.cat([torch.zeros(1, kvh, P, hd).bfloat16(), rk], 2), torch.cat([torch.zeros(1, kvh, P, hd).bfloat16(), rv], 2), layer)
    dk, dkr, dv = rotation_check(c2, P, rot)
    assert dk == dkr == dv == 0
    # relative-position property that makes the re-rotation meaningful: q.k depends only on the position difference
    qv = torch.randn(1, kvh, 1, hd).float()
    qpos = pos[:, :, -1:] + 3
    k0 = pre[0][0].float()
    cq, sq = rotary(qv, qpos)
    ck, sk = rotary(k0, pos)
    a, _ = apply_rotary_pos_emb(qv, qv, cq, sq)
    _, b = apply_rotary_pos_emb(k0, k0, ck, sk)
    cq2, sq2 = rotary(qv, qpos + 500)
    ck2, sk2 = rotary(k0, pos + 500)
    a2, _ = apply_rotary_pos_emb(qv, qv, cq2, sq2)
    _, b2 = apply_rotary_pos_emb(k0, k0, ck2, sk2)
    rel = float(((a @ b.transpose(-1, -2)) - (a2 @ b2.transpose(-1, -2))).abs().max())
    print(f"  rotate_states == model RoPE formula exactly; rotation_check zero on a self-consistent cache; "
          f"q.k shift invariance |d| {rel:.2e}")
    # donor map: cyclic next same-sign id
    sign = {"a": True, "b": False, "c": True, "d": True, "e": False}
    dm = donor_map(sign, list(sign))
    assert dm == {"a": "c", "c": "d", "d": "a", "b": "e", "e": "b"}, dm
    assert donor_map({"a": True, "b": False}, ["a", "b"]) == {"a": None, "b": None}
    print("  donor_map: cyclic next same-sign video; None for a one-video sign group")


def part4(py, out):
    section("4 stage0.py dry run on Grounder reads (stand-ins: A1 := base, A1_other := verdict_only)")
    subprocess.run([py, str(HERE / "stage0.py"), "--datasets", "HateMM", "HateClipSeg", "--first", "20",
                    "--arm-map", f"all_local={G / 'all_local'}", f"A1={G / 'base'}", f"A1_other={G / 'verdict_only'}",
                    "--out", str(out / "stage0_dry.json")], cwd=ROOT, check=True)
    d = json.load(open(out / "stage0_dry.json"))
    assert d["GT_read"] is False and set(d["datasets"]) == {"HateMM", "HateClipSeg"}


def part5(py, out, jobs):
    section("5 stage1.py dry run on Grounder reads (stand-ins: A1 := late, A1_other := shifted)")
    t = time.time()
    subprocess.run([py, str(HERE / "stage1.py"), "--stage", "1", "--decoder-python", py, "--jobs", str(jobs),
                    "--arm-map", f"base={G / 'base'}", f"all_local={G / 'all_local'}", f"A1={G / 'late'}",
                    f"A1_other={G / 'shifted'}", "--out", str(out / "stage1_dry")], cwd=ROOT, check=True)
    s = json.load(open(out / "stage1_dry/summary.json"))
    for ds, r in s["datasets"].items():
        dz = r["base_replica_decoded_abs_diff_vs_r6"]
        assert max(dz.values()) == 0.0, (ds, dz)
        ref = {"HateMM": (.705164, .750171), "HateClipSeg": (.588674, .637249)}[ds]   # Grounder README table
        print(f"  {ds}: base replica decoded == final_rawkey exactly; within (key is constant within a video, so "
              f"raw key = Grounder's calib-key value): all_local {r['final']['all_local']['within_video_macro_ROC_AUC']:.6f} "
              f"(README {ref[0]}), A1:=late {r['final']['A1']['within_video_macro_ROC_AUC']:.6f} (README {ref[1]})")
    print(f"  stage-1 dry run took {time.time() - t:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="directory for dry-run outputs (not under runs/)")
    ap.add_argument("--cpu-python", default=str(Path.home() / "miniconda3/envs/HateVideo/bin/python"))
    ap.add_argument("--videos", type=int, default=3, help="videos per corpus in part 2")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--parts", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if 1 in a.parts:
        part1(sys.executable)
    if 2 in a.parts:
        part2(a.videos)
    if 3 in a.parts:
        part3()
    if 4 in a.parts:
        part4(a.cpu_python, a.out)
    if 5 in a.parts:
        part5(a.cpu_python, a.out, a.jobs)
    print("\nSMOKE_CPU_DONE", flush=True)


if __name__ == "__main__":
    main()
