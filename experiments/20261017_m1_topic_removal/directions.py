#!/usr/bin/env python3
"""Candidate 46 stage 1 (README §2): the topic direction of Qwen3-VL-8B from the synthetic bank.
Each bank sentence is placed in the r6 speech-branch frame (text-only prefix with the sentence as the transcript, then the
speech question) and run once; at the embedding output and every decoder-layer output the residual is recorded at
(a) the sentence tokens inside the question, averaged ("body"), and (b) the last position ("last").
Direction per layer = normalised mean(protected) - mean(non-protected) over the even-indexed carriers; the odd-indexed
carriers give the validity AUC per layer (D1). No dataset text, no label file. GPU inside Slurm only.
Output: data/topic_direction/qwen3vl8b/{directions_body.pt, directions_last.pt, bank.json, PROVENANCE.md},
        runs/20261017_m1_topic_removal/directions/{run.log, validity.json, config.json}.
"""
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from topic_bank import CARRIERS, NON_PROTECTED, PROTECTED, sentences  # noqa: E402
from src.mllm_judge import MODEL, Judge, yesno_question  # noqa: E402

OUT = ROOT / "runs/20261017_m1_topic_removal/directions"
DATA = ROOT / "data/topic_direction/qwen3vl8b"
D1_GATE = .95


class Recorder:
    """Forward hooks on the embedding output and every decoder layer output; stores the residual at given positions."""

    def __init__(self, j):
        self.lm = j.model.model.language_model
        self.points = [self.lm.embed_tokens] + list(self.lm.layers)
        self.body_idx = None
        self.body, self.last = [], []
        self.hooks = [m.register_forward_hook(self._hook) for m in self.points]

    def _hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        assert h.ndim == 3 and h.shape[0] == 1, h.shape
        self.body.append(h[0, self.body_idx].float().mean(0).cpu())
        self.last.append(h[0, -1].float().cpu())

    def reset(self, body_idx):
        self.body_idx, self.body, self.last = body_idx, [], []

    def remove(self):
        for h in self.hooks:
            h.remove()


def frame(j, text):
    """Rendered text-only prompt (README §2) and the token indices of the sentence inside the question."""
    msgs, _ = j.prefix_messages([], [(0.0, 8.0, text)], with_frames=False)
    q = yesno_question(0, 1, 0.0, 8.0, text, "speech")
    full = j.render(j.conv(msgs, q), add_generation_prompt=True)
    start = full.rindex(text)                     # the copy inside the question, not the transcript copy
    end = start + len(text)
    enc = j.tok(full, add_special_tokens=False, return_offsets_mapping=True)
    idx = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < end]
    assert idx, (text, start)
    return full, enc["input_ids"], idx


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "run.log"), logging.StreamHandler()])
    logging.info("host %s", socket.gethostname())
    (OUT / "run.pid").write_text(str(os.getpid()))
    bank = sentences()
    import transformers
    cfg = dict(host=socket.gethostname(), date=time.strftime("%Y-%m-%d"), model=MODEL, sentences=len(bank), carriers=len(CARRIERS),
               protected=PROTECTED, non_protected=NON_PROTECTED, fit="even-indexed carriers", test="odd-indexed carriers", d1_gate=D1_GATE,
               torch=torch.__version__, transformers=transformers.__version__, source="experiments/20261017_m1_topic_removal/directions.py; 2026-10-17")
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    torch.manual_seed(0)
    j = Judge(MODEL)
    rec = Recorder(j)
    L = len(rec.points)
    body = torch.zeros(len(bank), L, j.model.config.text_config.hidden_size)
    last = torch.zeros_like(body)
    t0 = time.perf_counter()
    for k, s in enumerate(bank):
        full, ids, idx = frame(j, s["text"])
        rec.reset(idx)
        out = j.model.model(input_ids=torch.tensor([ids], device=j.device), use_cache=False)
        del out
        assert len(rec.body) == L, len(rec.body)
        body[k] = torch.stack(rec.body)
        last[k] = torch.stack(rec.last)
        if k % 200 == 0:
            logging.info("%d/%d %.0fs  body tokens %d  %s", k, len(bank), time.perf_counter() - t0, len(idx), s["text"][:60])
    rec.remove()
    fit = torch.tensor([s["split"] == "fit" for s in bank])
    prot = torch.tensor([s["protected"] == 1 for s in bank])
    result, validity = {}, {}
    for name, X in (("body", body), ("last", last)):
        d = X[fit & prot].mean(0) - X[fit & ~prot].mean(0)
        d = d / d.norm(dim=-1, keepdim=True)
        torch.save(d.contiguous(), DATA / f"directions_{name}.pt")
        proj = (X * d[None]).sum(-1)                           # (sentences, layers)
        aucs = [float(roc_auc_score(prot[~fit].numpy(), proj[~fit, l].numpy())) for l in range(L)]
        aucs_fit = [float(roc_auc_score(prot[fit].numpy(), proj[fit, l].numpy())) for l in range(L)]
        best = int(np.argmax(aucs))
        validity[name] = dict(auc_test_per_layer=aucs, auc_fit_per_layer=aucs_fit, best_layer=best, best_auc_test=aucs[best],
                              PASS=bool(aucs[best] >= D1_GATE), mean_proj_test={"protected": float(proj[~fit & prot].mean()),
                                                                                "non_protected": float(proj[~fit & ~prot].mean())})
        result[name] = d
        logging.info("%s: best layer %d test AUC %.4f (layers 0..%d: %s)", name, best, aucs[best], L - 1, " ".join(f"{a:.2f}" for a in aucs))
    validity["D1"] = dict(gate=D1_GATE, primary="body", PASS=validity["body"]["PASS"], layers=L, seconds=time.perf_counter() - t0)
    (OUT / "validity.json").write_text(json.dumps(validity, indent=2) + "\n")
    (DATA / "bank.json").write_text(json.dumps(dict(carriers=CARRIERS, protected=PROTECTED, non_protected=NON_PROTECTED, n=len(bank)), indent=2) + "\n")
    (DATA / "PROVENANCE.md").write_text(
        "# data/topic_direction/qwen3vl8b\n\n"
        f"Generated {time.strftime('%Y-%m-%d')} on {socket.gethostname()} by experiments/20261017_m1_topic_removal/directions.py "
        f"(candidate 46, README §2). Model {MODEL} (bf16, sdpa). Input: the synthetic bank in bank.json "
        f"({len(bank)} sentences = 40 carriers x 32 group terms; no dataset text, no labels). directions_body.pt / directions_last.pt: "
        f"float32 tensors ({L} x hidden), one unit vector per residual point (embedding output + 36 decoder-layer outputs), "
        "= normalised mean(protected) - mean(non-protected) over the even-indexed carriers, residual averaged over the sentence "
        "tokens inside the speech question (body) or taken at the last position (last). Validity on the odd-indexed carriers: "
        f"runs/20261017_m1_topic_removal/directions/validity.json.\n")
    logging.info("D1 %s; C46_DIR_DONE", "PASS" if validity["D1"]["PASS"] else "FAIL")


if __name__ == "__main__":
    main()
