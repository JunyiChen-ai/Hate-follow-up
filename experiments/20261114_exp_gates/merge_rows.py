#!/usr/bin/env python3
"""Merged ablation rows (README §15). CPU; imports gate_table.py (same directory) and uses its decodes, Integrator,
evaluator calls and paired bootstrap unchanged. Base: verdict turn kept, decoder em0, Integrator F3 or F5.

Rows evaluated: the 13 base rows, then the merged candidates of README §15 (two or three parts changed at once,
compared with the same full method). `--set b|c` writes the final table of a merge set (README §15.2): the base rows
with each merged row in the place of the rows it replaces.

    python experiments/20261114_exp_gates/merge_rows.py --form F5 --name merge__em0__F5
    python experiments/20261114_exp_gates/merge_rows.py --form F5 --set b --name final_b__em0__F5
"""
from __future__ import annotations

import argparse
import json

import gate_table as G

R = "Contextual Reader"
BASE = [r for r in G.ROWS_EM0 if not r[1].startswith("extra") and r[0] != "w/ Linear Evidence"]
# name, group, read set, scope prefix, decoder row flag, Integrator change
CANDIDATES = [
    ("w/o Global Transcript + Verdict Turn", "merged", "ns_no_transcript", "no_transcript", "", None),
    ("w/o Global Frames + Verdict Turn", "merged", "ns_no_prefix_frames", "no_prefix_frames", "", None),
    ("w/o Query Isolation + Verdict Turn", "merged", "ns_no_isolation", "full", "", None),
    ("w/ Joint Query, w/o Verdict Turn", "merged", "ns_joint", "full", "", None),
    ("w/o Window Evidence + Verdict Turn", "merged", "ns_no_window_evidence", "full", "", None),
    ("w/o Window Evidence, w/ Mean over Modalities", "merged", "rs_old", "full", "", "mean_mod"),
]
# stage 2 (README §15): reader arms that change two reader parts at once; read sets registered when on disk
STAGE2 = {
    "joint__no_isolation": "w/ Joint Query, w/o Query Isolation",
    "joint__no_window_evidence": "w/ Joint Query, w/o Window Evidence",
    "no_isolation__no_window_evidence": "w/o Query Isolation + Window Evidence",
    "joint__no_isolation__no_window_evidence": "w/ Joint Query, w/o Query Isolation + Window Evidence",
}
for arm, label in STAGE2.items():      # experiments/20261101_reader_ablation README §9: NAME=mg_<corpus>
    G.READS[arm] = {"main": f"{G.RA}/mg_merged/{arm}", "dehate": f"{G.RA}/mg_DeHate/{arm}"}
    if all((G.ROOT / G.READS[arm][p] / "predictions.jsonl").exists() for p in ("main", "dehate")):
        CANDIDATES.append((label, "merged", arm, "full", "", None))

# Merge sets of README §15.2: first replaced row (the merged row takes its place) -> merged row; rows dropped.
# b: the fallback set declared in §15.1 before the stage-2 reads; every merged row stays inside one module.
#    `w/ Mean over Modalities` is dropped because `w/o Window Aggregate` removes the whole window term, the maximum
#    over the two modalities included.
# c: found after the stage-2 tables (not declared before); one merge fewer, but it pairs a Reader input with an
#    Integrator choice (the pair §15.1 set aside under rule 2).
T_V = "w/o Global Transcript + Verdict Turn"
SETS = {
    "b": {"place": {"w/o Global Transcript": T_V,                       # four merges, 13 -> 9 rows
                    "w/o Query Isolation": "w/ Joint Query, w/o Query Isolation + Window Evidence"},
          "drop": {"w/o Verdict Turn", "w/ Joint Query", "w/o Window Evidence", "w/ Mean over Modalities"}},
    "c": {"place": {"w/o Global Transcript": T_V,                       # three merges, 13 -> 10 rows
                    "w/o Query Isolation": "w/ Joint Query, w/o Query Isolation",
                    "w/o Window Evidence": "w/o Window Evidence, w/ Mean over Modalities"},
          "drop": {"w/o Verdict Turn", "w/ Joint Query", "w/ Mean over Modalities"}},
}


def set_rows(name):
    cand = {r[0]: r for r in CANDIDATES}
    out = []
    for r in BASE:
        if r[0] in SETS[name]["place"]:
            out.append(cand[SETS[name]["place"][r[0]]])
        elif r[0] not in SETS[name]["drop"]:
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--form", choices=("F2", "F3", "F5"), default="F5")
    ap.add_argument("--set", choices=sorted(SETS), help="final table of this merge set instead of all candidates")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    G.ROWS = set_rows(args.set) if args.set else BASE + CANDIDATES
    G.FULL = "full"
    for d, st in map(lambda t: G.run_decode(*t), G.needed_decodes(["em0"])):
        if st == "decoded":
            print("decoded", d, flush=True)
    store = G.Store(["em0"], "prob", taxonomy=args.form == "F5")
    cfg = {"form": args.form, "a": 1.0, "b": 0.0, "lam": 1.0}
    res = G.evaluate(store, cfg, args.boot)
    res["_decoder"] = "em0"
    res["_note"] = ((f"README §15.2: final table of merge set {args.set} (verdict turn kept, no EM); a merged row "
                     "changes its parts at once and takes the place of their rows." if args.set else
                     "README §15: base rows (verdict turn kept, no EM) followed by merged candidates (group `merged`: "
                     "two or three parts changed at once, compared with the same full method).") +
                    (" HateClipSeg scope read = the taxonomy read." if store.taxonomy else ""))
    chk = G.checks(store, cfg, res)
    md, total = G.render(res, chk, cfg, args.name)
    d = G.OUT / "tables"
    (d / f"{args.name}.md").write_text(md)
    (d / f"{args.name}.json").write_text(json.dumps({"config": cfg, "decoder": "em0", "res": res, "checks": chk,
                                                     "rows": [list(r) for r in G.ROWS]}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
