"""MACIL dense inference on frozen train IDs without temporal localization GT."""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from macilsd.dataset import MacilTestDataset, usable_ids
from macilsd.train import build_models
from relation_v9.train_timeline import hatemm_train_timeline


def _to_seconds(value, index_map):
    return np.asarray(value, np.float64)[np.asarray(index_map)]


def infer_hatemm_train(args, model_path):
    ids, lengths, _ = hatemm_train_timeline()
    usable = usable_ids("hatemm", ids)
    if set(usable) != set(ids):
        raise RuntimeError("MACIL features do not cover frozen HateMM train")
    dataset = MacilTestDataset("hatemm", ids, args.max_seqlen, args.grid, "av")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    model, _ = build_models(args)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device).eval()
    out_path = os.path.join(args.out_dir, "scores.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)
    seen = set()
    with torch.no_grad(), open(out_path, "w") as handle:
        for f_v, f_a, index_map, n_seconds, vid in loader:
            vid = vid[0]; n_seconds = int(n_seconds)
            if vid in seen or n_seconds != lengths[vid]:
                raise RuntimeError(f"MACIL label-free timeline mismatch: {vid}")
            seen.add(vid); index_map = index_map[0].numpy()
            out = model(f_a[0].to(args.device), f_v[0].to(args.device), seq_len=None)
            _, audio, visual, av, _, _ = out
            branches = {"score_av": _to_seconds(torch.sigmoid(av.squeeze(-1)).mean(0).cpu(), index_map),
                        "score_audio": _to_seconds(audio.squeeze(-1).mean(0).cpu(), index_map),
                        "score_visual": _to_seconds(visual.squeeze(-1).mean(0).cpu(), index_map)}
            if any(len(value) != lengths[vid] or not np.isfinite(value).all()
                   for value in branches.values()):
                raise RuntimeError(f"MACIL invalid label-free output: {vid}")
            handle.write(json.dumps({"video_id": vid, "n_frames": lengths[vid],
                                     **{k: [round(float(x), 6) for x in v]
                                        for k, v in branches.items()}}) + "\n")
    if seen != set(ids):
        raise RuntimeError("MACIL incomplete frozen train output")
    return out_path
