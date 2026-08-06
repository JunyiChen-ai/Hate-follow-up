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

    # ------------------------------------------------ MHClip-ZH diagnostic
    diag_path = os.path.join(REPORTS, "test_zh_anomaly_diag.json")
    if os.path.exists(diag_path):
        dg = json.load(open(diag_path))
        rc, ver = dg.get("root_cause", {}), dg.get("verdict", {})
        cells = get(dg, "four_cell_table", "cells", default={})
        eff = dg.get("restoration_effect", {})
        fz = dg.get("forced_zh_arm", {})

        W(f"## MHClip-ZH anomaly diagnostic ({dg.get('date')})")
        W("")
        W("The first MHClip-ZH test run reported "
          f"{f(get(rc, 'superseded_numbers', '8b_auc_at_n13'), 3)} for the 8B "
          f"against {f(get(rc, 'superseded_numbers', '2b_auc_at_n13'), 4)} for "
          "the 2B, the only scale inversion across the four benchmarks, with "
          "restoration coverage at 1.0. Two explanations were on the table: the "
          "fresh Whisper Chinese transcripts had hurt the larger model, or the "
          "test split was simply harder. Neither is what happened.")
        W("")
        W(f"**Root cause.** {rc.get('what')} {rc.get('mechanism')} "
          f"{rc.get('blast_radius')} A scan of every frame of every "
          "`test_clean` video across all four benchmarks found exactly one "
          "unreadable JPEG, which is why only this dataset was affected. The "
          "frame was re-decoded from the local source mp4 at the index the "
          "original extractor used; the same code path reproduces the intact "
          "neighbour frame at 37.61 dB PSNR, so the frame numbering agrees. "
          "The run was then repeated to full coverage and the reports above "
          "are the corrected ones.")
        W("")
        W("**The four cells.** Every test cell is the same 149 videos, 45 "
          "hateful and 104 normal. The train row is 579 videos. Train under "
          "fresh Whisper was never measured: the train-split source media was "
          "not pulled to this machine, so the ASR route was never exercised "
          "there.")
        W("")
        W("| split | transcript the judge read | 8B AUC | 8B 95% CI | 2B AUC |")
        W("|---|---|---|---|---|")
        for key, split, tname in [
                ("train_x_dataset_transcript", "train", "dataset"),
                ("train_x_fresh_whisper", "train", "fresh Whisper"),
                ("test_x_dataset_transcript", "test", "dataset"),
                ("test_x_fresh_whisper", "test", "fresh Whisper (auto language)"),
                ("test_x_fresh_whisper_language_forced_zh", "test",
                 "fresh Whisper (language forced zh)")]:
            c = cells.get(key) or {}
            a8, a2 = c.get("8b"), c.get("2b")
            ci = (a8 or {}).get("boot95") or [None, None]
            W(f"| {split} | {tname} | "
              f"{f((a8 or {}).get('auc')) if a8 else 'not measured'} | "
              f"{('[' + f(ci[0], 3) + ', ' + f(ci[1], 3) + ']') if a8 else '--'} | "
              f"{f((a2 or {}).get('auc')) if a2 else '--'} |")
        W("")
        d8 = eff.get("8b_fresh_minus_dataset") or {}
        d2 = eff.get("2b_fresh_minus_dataset") or {}
        sp = eff.get("split_effect_8b_dataset_transcript") or {}
        W("**Did restoration hurt?** No, not measurably. On the same 149 "
          f"videos the fresh transcript moves the 8B AUC by "
          f"{f(d8.get('delta_auc'))}, paired bootstrap "
          f"[{f((d8.get('boot95_paired') or [None, None])[0])}, "
          f"{f((d8.get('boot95_paired') or [None, None])[1])}], straddling "
          f"zero; the 2B moves {f(d2.get('delta_auc'))}. The 8B raw z ranks "
          "the two arms at Spearman "
          f"{f(get(dg, 'paired_control_vs_c2', '8b', 'spearman_z_c2_vs_ctrl'), 3)} "
          "even though the median normalized edit distance between the two "
          "transcripts is "
          f"{f(get(dg, 'transcript_forensics', 'degeneracy_and_gate', 'all', 'edit_norm_vs_dataset', 'median'), 2)}. "
          "The text changes substantially; what the judge does with it does not.")
        W("")
        W("**Is the test split different?** No. Holding the transcript fixed "
          f"at the dataset one, the 8B scores {f(sp.get('train_auc'))} on train "
          f"and {f(sp.get('test_auc'))} on test, a difference of "
          f"{f(sp.get('test_minus_train'))} with heavily overlapping intervals.")
        W("")
        tr = fz.get("trigger", {})
        ft = fz.get("effect_on_the_transcript", {})
        dfz = eff.get("8b_forcedzh_minus_autolang") or {}
        W("**Language mismatch.** Whisper's own per-chunk language vote came "
          "back empty for all 149 videos, so language was read off the Unicode "
          "script profile instead: "
          f"{tr.get('observed_non_han_dominant')} of 149 auto-detected "
          f"transcripts ({f(tr.get('observed_frac'), 3)}) are not "
          "Han-dominant, which fired the pre-registered trigger for a "
          f"forced-zh arm. Forcing zh rescues "
          f"{ft.get('n_of_those_now_han_dominant')} of those "
          f"{ft.get('n_auto_non_han_dominant')} and lifts the median Han "
          "fraction to 1.0, and moves the 8B AUC by "
          f"{f(dfz.get('delta_auc'))}, "
          f"[{f((dfz.get('boot95_paired') or [None, None])[0])}, "
          f"{f((dfz.get('boot95_paired') or [None, None])[1])}]. The mechanism "
          "is real and the consequence is nil, so automatic detection stays: "
          "forcing a language would buy nothing and would oblige the method to "
          "make a correct per-corpus language decision on every new corpus.")
        W("")
        W("**What this says about the method on non-English corpora.** "
          "Channel restoration is neutral on MHClip-ZH, not harmful and not "
          "helpful. The starvation premise it runs on is weak here: the "
          "dataset transcript already has a median of "
          f"{f(get(dg, 'transcript_forensics', 'length', 'all', 'dataset_chars', 'median'), 0)} "
          "characters and only 1.3 percent of videos exceed the 300-character "
          "window the old clipped instrument could see, so there is little "
          "starvation left to relieve. Report it as a neutral result on this "
          "corpus rather than as evidence against the mechanism, and do not "
          "tune the ASR to chase it.")
        W("")
        W("**Infrastructure.** One unreadable frame silently cost 91 percent "
          "of a held-out measurement, and the run still reported `DONE` with a "
          "plausible-looking AUC on the surviving 13 videos. The judge should "
          "record and skip an unreadable frame rather than abort, and the "
          "analysis should refuse to write a report when coverage falls far "
          "below the split size instead of quietly reporting the subset.")
        W("")
        W("Full statistics, including the transcript forensics and the "
          "score-movement anatomy: "
          "`docs/duplex/reports/test_zh_anomaly_diag.json`.")
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
