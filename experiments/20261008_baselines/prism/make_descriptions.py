#!/usr/bin/env python3
"""Write PRISM's text description pools for hate from the policy text only (no video, transcript or label is read).

PRISM (ICML 2026) builds one anomaly axis per anomaly class from about 20 short descriptions of that class, and a
normal centre from about 20 normal descriptions (`semantic axis/xd_descriptions .json` in the release: 6 classes x
20 sentences + 20 "Generic" normal scenes). The release does not say how its sentences were written. Here they are
written by an instruction-tuned model from the policy wording alone, greedy decoding, one call per pool:

  classes  = the 9 rules of hate_query.md section 1 + the short positive query of section 2 (10 classes);
             each pool = the rule / query text verbatim + 20 generated descriptions of video content under it;
  normal   = "normal content" (hate_query.md section 2) verbatim + 20 generated descriptions of ordinary videos.

Output: experiments/20261008_baselines/prism/hate_descriptions.json, in the release's XD layout
(prompt_config.class_overrides, prompt_config.scenario_normals.Generic, content.anomalies), plus the raw generations.
Run once on a GPU node: python make_descriptions.py --model <Qwen3-VL-8B-Instruct snapshot>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hate_descriptions.json")
N = 20

RULES = (
    "Encourage violence against groups based on protected status",
    "Incite hatred against groups based on protected status",
    "Dehumanize groups by comparing to non-human entities",
    "Praise or glorify violence against protected groups",
    "Use slurs and stereotypes that incite hatred (including through humor, irony, or coded language)",
    "Claim groups are inferior based on protected status",
    "Promote hateful supremacism",
    "Make conspiratorial claims about groups based on protected status",
    "Deny or minimize major violent events",
)
POSITIVE = "hateful content that attacks or demeans people based on protected attributes"
NORMAL = "normal content"

CLASS_PROMPT = (
    "A video platform's hate speech policy prohibits content that does the following:\n\"{rule}\"\n\n"
    "Write {n} different one-sentence descriptions of what a short video segment that violates this rule could "
    "show or contain: what is seen on screen, what is said or sung, or what text appears. Make the descriptions "
    "concrete and varied (different formats such as speeches, memes, vlogs, music videos, animations, news-style "
    "clips, livestreams). Refer to targeted groups only generically (for example \"a religious group\" or \"an "
    "ethnic minority\"); do not name real people, groups or events. Keep each sentence under 25 words.\n"
    "Output exactly {n} lines, numbered 1 to {n}, and nothing else.")
NORMAL_PROMPT = (
    "A video platform's hate speech policy prohibits content that does any of the following:\n{rules}\n\n"
    "Write {n} different one-sentence descriptions of ordinary short video segments that do not violate any of "
    "these rules: what is seen on screen, what is said or sung, or what text appears. Make them concrete and "
    "varied (vlogs, tutorials, news reports, gaming, music, sports, comedy, interviews, animations, livestreams). "
    "Do not name real people. Keep each sentence under 25 words.\n"
    "Output exactly {n} lines, numbered 1 to {n}, and nothing else.")


def parse_lines(text, n):
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)[.)]\s*(.+?)\s*$", line)
        if m:
            out.append(m.group(2).strip().strip('"'))
    return out[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    def generate(prompt):
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                          return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    raw, classes = {}, {}
    for i, rule in enumerate(RULES + (POSITIVE,), 1):
        code = "R%d" % i if i <= len(RULES) else "Q"
        text = generate(CLASS_PROMPT.format(rule=rule, n=N))
        lines = parse_lines(text, N)
        if len(lines) < N:
            raise SystemExit("class %s: only %d lines parsed:\n%s" % (code, len(lines), text))
        raw[code] = text
        classes[code] = (rule, [rule] + lines)
        print(code, rule, "->", len(lines), flush=True)
    rules_block = "\n".join("%d. %s" % (i, r) for i, r in enumerate(RULES, 1))
    text = generate(NORMAL_PROMPT.format(rules=rules_block, n=N))
    normals = parse_lines(text, N)
    if len(normals) < N:
        raise SystemExit("normal: only %d lines parsed:\n%s" % (len(normals), text))
    raw["normal"] = text
    doc = {
        "project": "PRISM description pools for hateful video (label-free baseline, 2026-10-08)",
        "description": "Generated from the policy text only (hate_query.md); no video, transcript or label used.",
        "generator": {"model": args.model, "decoding": "greedy", "max_new_tokens": args.max_new_tokens,
                      "class_prompt": CLASS_PROMPT, "normal_prompt": NORMAL_PROMPT, "host": socket.gethostname(),
                      "date": time.strftime("%Y-%m-%d")},
        "prompt_config": {"class_overrides": {code: rule for code, (rule, _) in classes.items()},
                          "scenario_normals": {"Generic": [NORMAL] + normals}},
        "content": {"anomalies": {rule: descs for code, (rule, descs) in classes.items()}},
        "raw_generations": raw,
    }
    with open(OUT, "w") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
