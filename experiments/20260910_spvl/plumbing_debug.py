#!/usr/bin/env python3
"""Isolate position-id vs mask discrepancies on one video (prints max |delta logit| per arm)."""
import sys, json, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spvl

ROOT = spvl.ROOT
rows = spvl.load_manifest(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl", ["HateMM"])
row = rows[0]; ds, vid, dur = row["dataset"], row["video_id"], float(row["duration"])
segs = spvl.load_asr(ds).get(vid, [])
judge = spvl.Judge(mask_kind="bool")
frames = spvl.frame_paths(ds, vid, 20)
msgs, image_files = judge.prefix_messages(frames, segs, True, True)
prefix_text, enc = judge.encode_prefix(msgs, image_files)
prefix_ids = enc["input_ids"][0].tolist()
bids, btext = judge.branch_ids(spvl.VIDEO_QUESTION)
judge.seam_check_tokens(msgs, image_files, prefix_ids, spvl.VIDEO_QUESTION, bids)
ids = torch.tensor([prefix_ids + bids], device=judge.device)
pv = enc["pixel_values"].to(judge.device, judge.dtype); grid = enc["image_grid_thw"].to(judge.device)
mm_full = judge.mm_types(enc, ids.cpu())
with torch.no_grad():
    # a: model computes everything itself
    a = judge.model(input_ids=ids, pixel_values=pv, image_grid_thw=grid, mm_token_type_ids=mm_full,
                    use_cache=False, logits_to_keep=1).logits[0, -1].float()
    # model's own positions for the full sequence
    pos_model = judge.model.model.compute_3d_position_ids(input_ids=ids, image_grid_thw=grid, video_grid_thw=None,
                                                          inputs_embeds=None, attention_mask=torch.ones_like(ids),
                                                          past_key_values=None, mm_token_type_ids=mm_full)
    pos_p = judge.prefix_positions(enc)
    pos_mine = judge.packed_positions(pos_p, [len(bids)], sequential=True)
    print("pos_model shape", tuple(pos_model.shape), "pos_mine shape", tuple(pos_mine.shape))
    pm = pos_model[-3:] if pos_model.shape[0] == 4 else pos_model
    print("positions equal:", bool((pm.to(pos_mine.device) == pos_mine).all()),
          "max abs diff", int((pm.to(pos_mine.device) - pos_mine).abs().max()))
    # b: my positions, no mask
    b = judge.model(input_ids=ids, pixel_values=pv, image_grid_thw=grid, position_ids=pos_mine,
                    use_cache=False, logits_to_keep=1).logits[0, -1].float()
    # b4: model positions as 4-row tensor, no mask
    b4 = judge.model(input_ids=ids, pixel_values=pv, image_grid_thw=grid, position_ids=pos_model,
                     use_cache=False, logits_to_keep=1).logits[0, -1].float()
    T = ids.shape[1]
    for kind in ("bool", "additive"):
        m = spvl.to_mask(spvl.causal_allow(T), judge.dtype, judge.device, kind)
        c = judge.model(input_ids=ids, pixel_values=pv, image_grid_thw=grid, position_ids=pos_mine,
                        attention_mask=m, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        print(f"causal {kind} mask + my pos vs a: max|d|={float((c-a).abs().max()):.4f}  dz={judge.margin(c)-judge.margin(a):+.4f}")
        c4 = judge.model(input_ids=ids, pixel_values=pv, image_grid_thw=grid, position_ids=pos_model,
                         attention_mask=m, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        print(f"causal {kind} mask + model pos(4-row) vs a: max|d|={float((c4-a).abs().max()):.4f}")
        # 2D all-ones mask (model builds its own causal)
    d2 = judge.model(input_ids=ids, pixel_values=pv, image_grid_thw=grid, position_ids=pos_mine,
                     attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits[0, -1].float()
    print(f"my pos, no mask vs a: max|d|={float((b-a).abs().max()):.4f}  dz={judge.margin(b)-judge.margin(a):+.4f}")
    print(f"model pos 4-row, no mask vs a: max|d|={float((b4-a).abs().max()):.4f}")
    print(f"my pos, 2D ones mask vs a: max|d|={float((d2-a).abs().max()):.4f}")
    # repeat a to see nondeterminism
    a2 = judge.model(input_ids=ids, pixel_values=pv, image_grid_thw=grid, mm_token_type_ids=mm_full,
                     use_cache=False, logits_to_keep=1).logits[0, -1].float()
    print(f"a vs a (repeat): max|d|={float((a2-a).abs().max()):.4f}")
    print("attn impl:", judge.model.config._attn_implementation, "T", T)

# ---- second part: is the 4D-mask discrepancy a kernel effect, and is packing exact under the same kernel?
from torch.nn.attention import sdpa_kernel, SDPBackend
wins = spvl.fixed_windows(dur, 8.0)
qs = [spvl.window_question(i, len(wins), a_, b_, spvl.window_text(segs, a_, b_)) for i, (a_, b_) in enumerate(wins)][:5]
branches = [judge.branch_ids(q)[0] for q in qs]
with torch.no_grad():
    m = spvl.to_mask(spvl.causal_allow(T), judge.dtype, judge.device, "bool")
    for backend, name in ((SDPBackend.MATH, "math"), (SDPBackend.EFFICIENT_ATTENTION, "efficient")):
        try:
            with sdpa_kernel(backend):
                x = judge.model(input_ids=ids, pixel_values=pv, image_grid_thw=grid, mm_token_type_ids=mm_full,
                                use_cache=False, logits_to_keep=1).logits[0, -1].float()
                y = judge.model(input_ids=ids, pixel_values=pv, image_grid_thw=grid, position_ids=pos_mine,
                                attention_mask=m, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            print(f"[{name}] no-mask vs a(default): {float((x-a).abs().max()):.4f}; 4D-causal vs no-mask same kernel: {float((y-x).abs().max()):.4f}")
        except Exception as exc:
            print(f"[{name}] failed: {type(exc).__name__}: {str(exc)[:120]}")
    # packing exactness under the same explicit-mask kernel: plain(prefix+branch_i) with 4D causal vs packed block
    plain = []
    for b in branches:
        ids_i = torch.tensor([prefix_ids + b], device=judge.device)
        mi = spvl.to_mask(spvl.causal_allow(ids_i.shape[1]), judge.dtype, judge.device, "bool")
        pi = judge.packed_positions(pos_p, [len(b)], sequential=True)
        plain.append(judge.margin(judge.model(input_ids=ids_i, pixel_values=pv, image_grid_thw=grid, position_ids=pi,
                                              attention_mask=mi, use_cache=False, logits_to_keep=1).logits[0, -1]))
    packed, _ = judge.packed_forward(enc, pos_p, branches, arm="block")
    nomask = [judge.plain_forward(msgs, image_files, q)[0] for q in qs]
    print("windows plain(4D causal):", [round(v, 3) for v in plain])
    print("windows packed(block)   :", [round(v, 3) for v in packed])
    print("windows plain(no mask)  :", [round(v, 3) for v in nomask])
    print("max|packed - plain4D| =", max(abs(x - y) for x, y in zip(packed, plain)),
          " max|packed - nomask| =", max(abs(x - y) for x, y in zip(packed, nomask)))
