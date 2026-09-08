#!/usr/bin/env python3
"""Compute per-sample TokenSAR (Duan et al. ACL 2024) uncertainty score
over boundary-rescue judge outputs.

Adapted from:
  external_repos/SAR/src/get_tokenwise_importance.py
  external_repos/SAR/src/compute_uncertainty.py::token_sar

TokenSAR formula (per generation):
    token_wise_entropy[i] = -log p(token_i)                    # NLL of chosen token
    importance[i]         = 1 - cross_encoder_sim(q + gen,
                                                  q + gen_with_token_i_removed)
    TokenSAR = Σ_i (importance[i] / Σ importance) * token_wise_entropy[i]

Lower TokenSAR = model is confident on semantically important tokens.
Higher TokenSAR = model is uncertain on content-carrying tokens.

Input:  JSONL with per-record `full_logprobs` (list of {tid, tok, lp})
        produced by judge_offline.py --save-full-logprobs
Output: JSONL with per-record added fields:
          - tokensar:  float, TokenSAR score for the whole generation
          - tokensar_rationale_only: float, TokenSAR restricted to rationale span
          - tokensar_n_tokens: int
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(format="[%(asctime)s] %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)


def _lazy_load_encoder(model_name: str, device: str):
    """Load sentence-transformers CrossEncoder (STS regression).
    SAR default: cross-encoder/stsb-roberta-large. Distilroberta smaller fallback.
    """
    from sentence_transformers import CrossEncoder
    logging.info(f"Loading cross-encoder {model_name} on {device}")
    model = CrossEncoder(model_name=model_name, num_labels=1, device=device)
    return model


def _reassemble_text(tokens):
    """Concat decoded token strings to reconstruct the generation.
    Uses the already-decoded `tok` field from extract_full_logprobs."""
    return "".join(t["tok"] for t in tokens)


def _span_rationale_and_verdict(tokens):
    """Identify rationale span (until 'verdict' or 'final_answer' keyword).
    Returns (start_idx, end_idx_exclusive) over the generated tokens."""
    text = _reassemble_text(tokens)
    low = text.lower()
    cut = len(text)
    for kw in ("verdict:", "final_answer:", "final answer:", "answer:"):
        p = low.find(kw)
        if p > 0 and p < cut:
            cut = p
    # walk tokens to find the cut character offset
    if cut >= len(text):
        return 0, len(tokens)
    acc = 0
    for i, tok in enumerate(tokens):
        acc += len(tok["tok"])
        if acc > cut:
            return 0, i  # this token straddles the cut; exclude it
    return 0, len(tokens)


def compute_token_importance_batched(
    encoder,
    question: str,
    tokens: list[dict],
    batch_size: int = 64,
) -> np.ndarray:
    """For each token i, compute 1 - sim(q+gen, q+gen_without_token_i)
    via cross-encoder.

    SAR's exact recipe: gen.replace(tokenizer.decode(token), ''). We
    already store decoded token strings in `tok`, so we do the equivalent
    string-level replace on the FIRST occurrence to avoid mass deletion
    when tokens repeat. (SAR uses .replace which does replace-all; we
    follow that faithfully to match the reference.)
    """
    full = _reassemble_text(tokens)
    pairs = []
    for t in tokens:
        tok_str = t["tok"]
        if not tok_str:
            removed = full
        else:
            removed = full.replace(tok_str, "")
        pairs.append([question + full, question + removed])
    sims = encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    sims = np.asarray(sims, dtype=np.float32).reshape(-1)
    importance = 1.0 - sims
    importance = np.clip(importance, 0.0, None)
    return importance


def token_sar(nll: np.ndarray, importance: np.ndarray) -> float:
    """Weighted sum of NLLs, weights ∝ importance."""
    tot = float(importance.sum())
    if tot <= 0.0 or len(nll) == 0:
        return 0.0
    w = importance / tot
    return float(np.sum(w * nll))


def _nll_from_full_logprobs(tokens: list[dict]) -> np.ndarray:
    nll = []
    for t in tokens:
        lp = t.get("lp")
        if lp is None:
            # missing logprob → treat as no info (0 contribution)
            nll.append(0.0)
        else:
            nll.append(-float(lp))
    return np.asarray(nll, dtype=np.float32)


def process_file(
    in_path: Path,
    out_path: Path,
    question_prefix: str,
    encoder_name: str,
    device: str,
    batch_size: int,
    limit: int | None = None,
):
    if not in_path.exists():
        logging.warning(f"missing input: {in_path}")
        return 0
    encoder = _lazy_load_encoder(encoder_name, device)
    n = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            toks = rec.get("full_logprobs")
            if not toks:
                rec["tokensar"] = None
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            # strip trailing EOS-like empty strings and bad entries
            toks = [t for t in toks if isinstance(t, dict) and "tok" in t]
            if not toks:
                rec["tokensar"] = None
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            nll = _nll_from_full_logprobs(toks)
            imp = compute_token_importance_batched(
                encoder, question_prefix, toks, batch_size=batch_size
            )
            rec["tokensar"] = token_sar(nll, imp)
            rec["tokensar_n_tokens"] = int(len(toks))
            s, e = _span_rationale_and_verdict(toks)
            if e > s:
                rec["tokensar_rationale_only"] = token_sar(nll[s:e], imp[s:e])
            else:
                rec["tokensar_rationale_only"] = rec["tokensar"]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            if n % 20 == 0:
                logging.info(f"  {in_path.name}: {n} scored")
            if limit is not None and n >= limit:
                break
    logging.info(f"Wrote {n} records → {out_path}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--encoder",
                    default="cross-encoder/stsb-roberta-large",
                    help="SAR default is cross-encoder/stsb-roberta-large")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--question-prefix",
                    default=("Is the following video hateful? Consider targeted "
                             "groups and hostile framing. "),
                    help="Prepended to gen / removed-gen in cross-encoder "
                         "similarity — mirrors SAR's 'question + generation'")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    process_file(
        args.in_path, args.out_path,
        question_prefix=args.question_prefix,
        encoder_name=args.encoder,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
