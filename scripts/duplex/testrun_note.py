"""Render docs/duplex/TEST_RUNS_NOTE.md from the per-dataset test-run JSONs.

Reads whichever of `docs/duplex/reports/test_c2_<slug>_<arm>.json` exist and
writes the prose note. Datasets arrive at different times, so the note is
regenerated from scratch on every call and simply grows a row as each finishes.
Statistics only; nothing here reaches back into transcripts or video ids.

Usage:
  python scripts/duplex/testrun_note.py docs/duplex/TEST_RUNS_NOTE.md
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
REPORTS = os.path.join(ROOT, "docs", "duplex", "reports")

DATASETS = [("implihatevid", "ImpliHateVid"), ("hatemm", "HateMM"),
            ("mhclip_en", "MHClip-EN"), ("mhclip_zh", "MHClip-ZH")]
ARMS = [("8b", "Qwen3-VL-8B"), ("2b", "Qwen3-VL-2B")]

# Train-split context, read off the committed full-corpus C2 reports. Quoted so
# the test row can be compared against the train row it was frozen on, never
# pooled with it.
TRAIN_CONTEXT = os.path.join(REPORTS, "c2_fullcorpus_{arm}.json")


def load(slug, arm):
    p = os.path.join(REPORTS, f"test_c2_{slug}_{arm}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def f(x, nd=4):
    if x is None:
        return "--"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "docs", "duplex", "TEST_RUNS_NOTE.md")
    reps = {(s, a): load(s, a) for s, _ in DATASETS for a, _ in ARMS}
    have = {k: v for k, v in reps.items() if v}
    if not have:
        raise SystemExit("no test_c2_*.json reports on disk; nothing to render")

    any_rep = next(iter(have.values()))
    L = []
    W = L.append

    W("# Held-out test note: the channel-restoration method on four benchmarks")
    W("")
    W("Measurement, not a kill-test. No pre-registration governs these runs and "
      "none is claimed. Each benchmark is measured on its own `test_clean` "
      "split under the configuration frozen during the train-split work; "
      "nothing was refitted for the test data, and train and test scores are "
      "never pooled into one distribution.")
    W("")
    W("## What was run")
    W("")
    W("One condition per dataset per model size: source media pulled from B2 "
      "and byte-verified, audio re-transcribed with Whisper large-v3 on cuda, "
      "the frozen degeneracy gate applied, the gated fresh transcript fed "
      "uncapped to a single judge call, and a label-free KDE-valley threshold "
      "computed on that run's own raw-z distribution. One MLLM call per video "
      "per arm. The 2B arm is the boundary-condition contrast, not a second "
      "method.")
    W("")
    W("Every component is inherited rather than chosen here:")
    W("")
    for k, v in (get(any_rep, "config_provenance", default={}) or {}).items():
        if k == "model":
            continue
        W(f"- **{k}** — {v}")
    W("")
    sc = get(any_rep, "threshold_recipe_selfcheck", default={})
    if sc.get("recomputed_valley") is not None:
        W(f"The threshold recipe was self-checked before use: applied to the "
          f"ImpliHateVid C0 scores it returns {f(sc.get('recomputed_valley'), 6)} "
          f"against the frozen {f(sc.get('frozen_valley'), 6)} in "
          f"`rawz_detector_train.json` (match: {sc.get('matches')}).")
        W("")

    # ---------------------------------------------------------- restoration
    W("## Restoration coverage")
    W("")
    W("How much of the judge input the channel-restoration stage actually "
      "replaced. A low fraction here means the method degenerated to its "
      "fallback branch and the numbers below measure the judge and the "
      "threshold alone.")
    W("")
    W("| dataset | n | with source media | usable audio | fresh transcripts | "
      "gate accepted | gate rejected | no fresh pass | restoration fraction |")
    W("|---|---|---|---|---|---|---|---|---|")
    for slug, name in DATASETS:
        r = reps.get((slug, "8b")) or reps.get((slug, "2b"))
        if not r:
            continue
        rc = get(r, "restoration_coverage", default={})
        g = get(rc, "gate_outcomes", default={})
        W(f"| {name} | {get(r, 'coverage', 'n_scored')} | "
          f"{rc.get('n_with_source_media')} | {rc.get('n_with_usable_audio')} | "
          f"{rc.get('n_fresh_transcripts')} | {g.get('accepted')} | "
          f"{g.get('rejected')} | {g.get('no_fresh_pass')} | "
          f"{f(rc.get('restoration_fraction'))} |")
    W("")

    # ------------------------------------------------------------- headline
    W("## Headline")
    W("")
    W("AUC is hateful vs normal on the raw z. macro-F1 and accuracy are at the "
      "label-free valley computed on that arm's own test distribution; the "
      "oracle column is the F1-maximizing threshold chosen against gold labels "
      "and is a diagnostic ceiling, never part of the method.")
    W("")
    W("| dataset | model | n | prev. | AUC | AUC 95% CI | modes | valley | "
      "macro-F1 (label-free) | acc | FN | FP | oracle macro-F1 | gap |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for slug, name in DATASETS:
        for arm, mname in ARMS:
            r = reps.get((slug, arm))
            if not r:
                continue
            cov = get(r, "coverage", default={})
            a = get(r, "auc", default={})
            ci = a.get("boot95") or [None, None]
            tl = get(r, "threshold_label_free", default={})
            m = get(r, "operating_points", "METHOD_self_computed_valley",
                    default={})
            og = get(r, "oracle_gap", default={})
            W(f"| {name} | {mname} | {cov.get('n_scored')} | "
              f"{f(cov.get('prevalence_hateful'), 3)} | "
              f"{f(a.get('hateful_vs_normal'))} | "
              f"[{f(ci[0], 3)}, {f(ci[1], 3)}] | "
              f"{tl.get('n_grid_local_maxima')} | {f(tl.get('value'), 3)} | "
              f"{f(m.get('macro_f1'))} | {f(m.get('accuracy'))} | "
              f"{m.get('fn', '--')} | {m.get('fp', '--')} | "
              f"{f(og.get('macro_f1_oracle'))} | {f(og.get('gap'))} |")
    W("")

    # -------------------------------------------------------- valley stability
    W("### Valley stability")
    W("")
    W("1000-resample bootstrap of the valley location, plus the relative "
      "trough depth: the fraction by which the valley's density falls below "
      "the lower of the two modes bracketing it. 1.0 is a clean split, 0 is a "
      "valley sitting exactly at mode height, and a dash means the recipe "
      "found fewer than two modes and returned no threshold at all.")
    W("")
    W("| dataset | model | trough depth | valley 95% CI | sd | "
      "resamples with no valley |")
    W("|---|---|---|---|---|---|")
    for slug, name in DATASETS:
        for arm, mname in ARMS:
            r = reps.get((slug, arm))
            if not r:
                continue
            tl = get(r, "threshold_label_free", default={})
            sb = get(tl, "stability_bootstrap", default={})
            ci = sb.get("ci95") or sb.get("valley_ci95") or [None, None]
            nb = sb.get("n_resamples", sb.get("n_boot", 1000))
            W(f"| {name} | {mname} | {f(tl.get('relative_trough_depth'), 3)} | "
              f"[{f(ci[0], 2)}, {f(ci[1], 2)}] | {f(sb.get('sd'), 2)} | "
              f"{sb.get('n_without_valley', '--')}/{nb} |")
    W("")

    # ------------------------------------------------ implicitness anatomy
    ihv = [(a, reps[("implihatevid", a)]) for a, _ in ARMS
           if reps.get(("implihatevid", a))]
    if ihv:
        W("### ImpliHateVid: where the misses sit")
        W("")
        W("The gold EX/IM/NH prefixes are diagnostic ground truth for this "
          "table only; no scoring component reads them. The "
          "channel-restoration story predicts the residual misses concentrate "
          "in the implicit stratum.")
        W("")
        W("| model | AUC EX vs NH | AUC IM vs NH | FN EX | FN IM | "
          "miss rate EX | miss rate IM | share of FN that is IM |")
        W("|---|---|---|---|---|---|---|---|")
        for arm, r in ihv:
            mname = dict(ARMS)[arm]
            sg = get(r, "auc", "by_implicitness_subgroup", default={})
            fs = get(r, "error_anatomy_at_method_threshold", "false_negatives",
                     "by_implicitness_subgroup", default={})
            W(f"| {mname} | {f(sg.get('EX_vs_NH'))} | {f(sg.get('IM_vs_NH'))} | "
              f"{fs.get('EX', '--')} | {fs.get('IM', '--')} | "
              f"{f(fs.get('miss_rate_EX'), 3)} | {f(fs.get('miss_rate_IM'), 3)} | "
              f"{f(fs.get('share_of_FN_that_is_IM'), 3)} |")
        W("")

    # -------------------------------------------------------------- context
    W("## Context")
    W("")
    W("Two reference points, neither of them a like-for-like comparison.")
    W("")
    W("**Supervised ceiling on ImpliHateVid.** IARE (SIGIR 2026) reports F1 "
      "91.75 on this same test split. It is trained on the split's labels; the "
      "method here sees none, at train time or at threshold time. The gap "
      "between the two is the price of dropping supervision, not a defect to "
      "be closed by tuning.")
    W("")
    rows = []
    for arm, mname in ARMS:
        p = TRAIN_CONTEXT.format(arm=arm)
        if not os.path.exists(p):
            continue
        t = json.load(open(p))
        rows.append((
            mname,
            get(t, "auc", "C2_full_corpus", "hateful_vs_NH"),
            get(t, "operating_points", "METHOD_self_computed_valley", "macro_f1"),
            get(t, "oracle_gap", "macro_f1_oracle"),
            get(reps.get(("implihatevid", arm)) or {},
                "auc", "hateful_vs_normal"),
            get(reps.get(("implihatevid", arm)) or {},
                "operating_points", "METHOD_self_computed_valley", "macro_f1"),
        ))
    if rows:
        W("**The ImpliHateVid train row this was frozen on.** The train "
          "numbers come from the full-corpus C2 run over 1283 `train_clean` "
          "videos. They are quoted side by side, never pooled: each split's "
          "threshold is computed on its own distribution.")
        W("")
        W("| model | train AUC | train macro-F1 | train oracle macro-F1 | "
          "test AUC | test macro-F1 |")
        W("|---|---|---|---|---|---|")
        for mname, tauc, tf1, torc, eauc, ef1 in rows:
            W(f"| {mname} | {f(tauc)} | {f(tf1)} | {f(torc)} | "
              f"{f(eauc)} | {f(ef1)} |")
        W("")

    # -------------------------------------------------------------- caveats
    W("## What these numbers do and do not settle")
    W("")
    W("- The threshold is transductive: it reads the unlabeled test scores as "
      "a batch. That is legitimate for a label-free method and is how the "
      "train runs computed it too, but it is not an online per-video decision "
      "rule, and a deployment that scores one video at a time would need the "
      "threshold carried over rather than recomputed.")
    W("- A single arm per dataset per model. No repeated sampling, no "
      "ensembling; the run is one forward pass per video.")
    W("- The oracle column bounds how much of the remaining error is a "
      "threshold problem rather than a ranking problem. Where the gap is near "
      "zero, the label-free recipe has already found what the score supports "
      "and further threshold work is wasted effort.")
    W("- MHClip is 3-class collapsed to binary with `Offensive` mapping to 1. "
      "The `Hateful` and `Offensive` subgroup AUCs in the JSON say how much of "
      "the binary number rests on each.")
    W("- The surface-cue regex family used in the error anatomy is English. On "
      "MHClip-ZH it under-fires and those rates are not interpretable.")
    W("")
    W("Per-arm JSON, with the full distributions, gate statistics and "
      "starvation diagnostics: `docs/duplex/reports/test_c2_<slug>_<arm>.json`.")
    W("")

    with open(dest, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {dest} ({len(have)} arms present of "
          f"{len(DATASETS) * len(ARMS)})")


if __name__ == "__main__":
    main()
