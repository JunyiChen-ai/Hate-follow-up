"""Render docs/duplex/CROSSBENCH_NOTE.md from the per-dataset report JSONs.

Reads whichever of docs/duplex/reports/crossbench_<slug>_<arm>.json exist and
writes the prose note: configuration provenance, the headline table, the
starvation diagnostics, and the caveats. Statistics only; nothing here reaches
back into transcripts or video ids.

Usage:
  python scripts/duplex/crossbench_note.py docs/duplex/CROSSBENCH_NOTE.md
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
REPORTS = os.path.join(ROOT, "docs", "duplex", "reports")

DATASETS = [("hatemm", "HateMM"), ("mhclip_en", "MHClip-EN"),
            ("mhclip_zh", "MHClip-ZH")]
ARMS = [("8b", "Qwen3-VL-8B"), ("2b", "Qwen3-VL-2B")]


def load(slug, arm):
    p = os.path.join(REPORTS, f"crossbench_{slug}_{arm}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def fmt(x, nd=4):
    if x is None:
        return "--"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "docs", "duplex", "CROSSBENCH_NOTE.md")
    reports = {(s, a): load(s, a) for s, _ in DATASETS for a, _ in ARMS}
    have = {k: v for k, v in reports.items() if v}
    if not have:
        raise SystemExit("no crossbench reports on disk; nothing to render")

    any_rep = next(iter(have.values()))
    restored = {s: (reports.get((s, "8b")) or {}).get(
        "restoration_coverage", {}).get("restoration_fraction")
        for s, _ in DATASETS}
    any_restoration = any(bool(x) for x in restored.values())

    L = []
    A = L.append
    A("# Cross-benchmark note: the channel-restoration method on HateMM, "
      "MHClip-EN and MHClip-ZH")
    A("")
    A("Measurement, not a kill-test. No pre-registration governs this run and "
      "none is claimed. Each benchmark is measured on its own `train_clean` "
      "split; no test split of any dataset was read.")
    A("")

    A("## What was run")
    A("")
    A("One condition per dataset per model size, the method as it stands after "
      "the ImpliHateVid full-corpus run: the gated fresh Whisper large-v3 "
      "transcript fed uncapped to a single judge call, with the dataset "
      "transcript as the gate's fallback, and a label-free KDE-valley threshold "
      "computed on that run's own raw-z distribution. One MLLM call per video. "
      "The 2B arm is the boundary-condition contrast, not a second method.")
    A("")
    A("Every component is inherited rather than chosen here:")
    A("")
    prov = any_rep["config_provenance"]
    for k in ("audio", "asr", "gate", "judge", "threshold_recipe"):
        A(f"- **{k}** — {prov[k]}")
    A("")
    sc = any_rep.get("threshold_recipe_selfcheck", {})
    A(f"The threshold recipe was self-checked before use: applied to the "
      f"ImpliHateVid C0 scores it returns {fmt(sc.get('recomputed_valley'), 6)} "
      f"against the frozen {fmt(sc.get('frozen_valley'), 6)} in "
      f"`rawz_detector_train.json` (match: {sc.get('matches')}).")
    A("")

    # ---------------- the coverage caveat, if the ASR route never ran --------
    if not any_restoration:
        A("## The restoration stage did not run on any of the three benchmarks")
        A("")
        A("This is the first thing to read, and it bounds everything below. The "
          "channel-restoration component needs the source video to re-transcribe. "
          "The three benchmarks are present on this machine only as the "
          "frames-plus-annotation payloads; an exhaustive listing of the project "
          "B2 bucket (211,566 objects, single bucket, all prefixes) found source "
          "media for ImpliHateVid alone. What exists for HateMM and MHClip is "
          "pooled Whisper *encoder embeddings* from a different project, not "
          "text, and not usable as judge input. The source cluster that holds "
          "the mp4s is not reachable from this machine by key-based ssh.")
        A("")
        A("So on all three benchmarks the gate took its `no_fresh_pass` branch "
          "for every video and the judge read the dataset transcript, uncapped. "
          "The numbers below therefore measure the judge and the label-free "
          "threshold travelling across benchmarks, at zero restoration coverage. "
          "They do not test whether channel starvation is an ImpliHateVid quirk "
          "or a general property — that question stays open until the source "
          "videos are available.")
        A("")
        A("Everything is staged for that run. `crossbench_audio.py` and "
          "`crossbench_asr.py` take an `--mp4-dir`, the driver is idempotent, "
          "and pointing it at the videos re-runs stages A-C and re-scores into a "
          "fresh judge directory keyed by an input fingerprint.")
        A("")

    # ---------------------------- headline table ----------------------------
    A("## Headline")
    A("")
    A("AUC is hateful vs normal on the raw z. macro-F1 is at the label-free "
      "valley computed on that arm's own distribution; the oracle column is the "
      "F1-maximizing threshold chosen against gold labels and is a diagnostic "
      "ceiling, never part of the method.")
    A("")
    A("| dataset | model | n | prev. | AUC | AUC 95% CI | modes | valley | "
      "macro-F1 (label-free) | FN | FP | oracle macro-F1 | gap |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for slug, name in DATASETS:
        for arm, mname in ARMS:
            r = reports.get((slug, arm))
            if not r:
                continue
            cov, au = r["coverage"], r.get("auc", {})
            th = r.get("threshold_label_free", {})
            op = r.get("operating_points", {}).get("METHOD_self_computed_valley")
            og = r.get("oracle_gap", {})
            ci = au.get("boot95") or [None, None]
            A(f"| {name} | {mname} | {cov['n_scored']} | "
              f"{fmt(cov.get('prevalence_hateful'), 3)} | "
              f"{fmt(au.get('hateful_vs_normal'))} | "
              f"[{fmt(ci[0], 3)}, {fmt(ci[1], 3)}] | "
              f"{th.get('n_grid_local_maxima', '--')} | "
              f"{fmt(th.get('value'), 3)} | "
              f"{fmt(op['macro_f1']) if op else '--'} | "
              f"{op['fn'] if op else '--'} | {op['fp'] if op else '--'} | "
              f"{fmt(og.get('macro_f1_oracle'))} | {fmt(og.get('gap'))} |")
    A("")

    A("### Valley stability")
    A("")
    A("1000-resample bootstrap of the valley location, plus the relative trough "
      "depth: the fraction by which the valley's density falls below the lower "
      "of the two modes bracketing it. 1.0 is a clean split, 0 is a valley "
      "sitting exactly at mode height, and a dash means the recipe found fewer "
      "than two modes and so returned no threshold at all.")
    A("")
    A("| dataset | model | trough depth | valley 95% CI | sd | resamples with "
      "no valley |")
    A("|---|---|---|---|---|---|")
    for slug, name in DATASETS:
        for arm, mname in ARMS:
            r = reports.get((slug, arm))
            if not r:
                continue
            th = r.get("threshold_label_free", {})
            bs = th.get("stability_bootstrap", {})
            ci = bs.get("ci95") or [None, None]
            A(f"| {name} | {mname} | {fmt(th.get('relative_trough_depth'), 3)} | "
              f"[{fmt(ci[0], 2)}, {fmt(ci[1], 2)}] | {fmt(bs.get('sd'), 2)} | "
              f"{bs.get('n_without_valley', '--')}/{bs.get('n_boot', '--')} |")
    A("")

    # ------------------------ starvation diagnostics ------------------------
    A("## Starvation diagnostics")
    A("")
    A("How much text the judge had on each benchmark, before any restoration. "
      "The 300-character column is the old clipped instrument's visible window, "
      "so it bounds how much of each benchmark's transcript that instrument "
      "could never see.")
    A("")
    A("| dataset | median dataset-transcript chars | hateful | normal | empty | "
      "> 300 chars | restoration coverage |")
    A("|---|---|---|---|---|---|---|")
    for slug, name in DATASETS:
        r = reports.get((slug, "8b")) or reports.get((slug, "2b"))
        if not r:
            continue
        d = r["starvation_diagnostics"]["dataset_transcript_chars"]
        rc = r["restoration_coverage"]
        A(f"| {name} | {fmt(d['all']['median'], 0)} | "
          f"{fmt(d['hateful']['median'], 0)} | {fmt(d['normal']['median'], 0)} | "
          f"{d['n_empty']} | {d['n_over_300_chars']} "
          f"({fmt(d['frac_over_300_chars'], 3)}) | "
          f"{fmt(rc.get('restoration_fraction'), 3)} |")
    A("")
    for slug, name in DATASETS:
        r = reports.get((slug, "8b")) or reports.get((slug, "2b"))
        if not r:
            continue
        sd = r["starvation_diagnostics"]
        if sd.get("edit_norm_fresh_vs_dataset"):
            e = sd["edit_norm_fresh_vs_dataset"]
            A(f"- **{name}** — median normalized edit distance between the fresh "
              f"and the dataset transcript: {fmt(e['all']['median'], 3)} overall, "
              f"{fmt(e['hateful']['median'], 3)} on the hateful half, "
              f"{fmt(e['normal']['median'], 3)} on the normal half.")
        else:
            A(f"- **{name}** — no fresh transcript exists, so the fresh-vs-dataset "
              f"edit distance and the gate rejection rates are undefined for this "
              f"run.")
    A("")

    # ------------------------------ references ------------------------------
    A("## Prior-project reference points")
    A("")
    ref = any_rep.get("reference_points_prior_project")
    if ref:
        A(ref["caveat"])
        A("")
        A("| dataset | prior method | variant | accuracy / macro-F1 (test split) |")
        A("|---|---|---|---|")
        for slug, name in DATASETS:
            r = reports.get((slug, "8b")) or reports.get((slug, "2b"))
            rp = (r or {}).get("reference_points_prior_project") or {}
            for m, row in sorted((rp.get("best_variant_per_method") or {}).items()):
                A(f"| {name} | {m} | {row['variant']} | "
                  f"{row['acc_over_macro_f1']} |")
        A("")
        A("No published SOTA numbers are quoted: the repo's own baseline briefs "
          "carry reproductions rather than reported figures, and no web search "
          "was performed.")
    else:
        A("None available in the repo docs.")
    A("")

    A("## Caveats")
    A("")
    A("- Measurement on `train_clean` only. No test split of any dataset was "
      "read, so nothing here is a benchmark result and nothing here is "
      "comparable to a published test number.")
    A("- Labels are used for evaluation alone. The threshold is computed from "
      "the score distribution with no labels; the oracle threshold is reported "
      "as a gap and never used.")
    A("- MHClip is 3-class and `Offensive` maps to 1, so the binary task is "
      "Hateful+Offensive vs Normal. That is a coarser positive class than "
      "HateMM's, and the prevalences differ across the three benchmarks, so the "
      "macro-F1 column is not comparable row to row without that in mind.")
    A("- The surface-cue regex family is English. On MHClip-ZH it under-fires "
      "and the cue rates in that report are not interpretable.")
    if not any_restoration:
        A("- **The restoration stage never ran.** Zero restoration coverage on "
          "all three benchmarks, for want of source media. The cross-benchmark "
          "generalization question the run was launched to answer is not "
          "answered by it.")
    A("")

    with open(dest, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {dest} ({len(L)} lines) from {len(have)} report(s)")


if __name__ == "__main__":
    main()
