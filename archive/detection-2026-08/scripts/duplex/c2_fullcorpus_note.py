"""Render docs/duplex/C2_FULLCORPUS_NOTE.md from the report JSON.

Statistics only. No transcript text and no individual video ids: everything
written here is read out of the report, which carries neither.
"""

import datetime
import json
import sys


def pct(x):
    return "n/a" if x is None else f"{x:.4f}"


def main():
    rep_path, dest = sys.argv[1], sys.argv[2]
    rep2b_path = sys.argv[3] if len(sys.argv) > 3 else None
    r = json.load(open(rep_path))
    r2b = None
    if rep2b_path:
        try:
            r2b = json.load(open(rep2b_path))
        except Exception:
            r2b = None

    thr = r["threshold_label_free"]
    m = r["operating_points"].get("METHOD_self_computed_valley")
    c0 = r["c0_baseline_operating_point"]
    diag = r["operating_points"]["diagnostic_frozen_C0_valley"]
    orc = r["operating_points"]["diagnostic_oracle_f1max_NOT_label_free"]
    auc = r["auc"]
    ea = r.get("error_anatomy_at_method_threshold", {})
    fn, fp = ea.get("false_negatives", {}), ea.get("false_positives", {})
    g = r["gate_stats"]
    det = r["determinism_vs_killtest_c2_8b"]
    sc = r["c0_recipe_selfcheck"]
    boot = thr.get("stability_bootstrap", {})
    c0boot = sc.get("c0_stability_bootstrap", {})

    L = []
    A = L.append
    A("# Full-Corpus C2 Measurement")
    A("")
    A(f"**Date**: {datetime.date.today().isoformat()}. "
      "**Dataset**: ImpliHateVid `train_clean`, 1283 videos. The test split is "
      "not touched.")
    A("**Model**: Qwen3-VL-8B-Instruct, bf16, one forward pass per video.")
    A("")
    A("This is method-development measurement, not a kill-test. No "
      "pre-registration governs it and none is claimed. The channel-restoration "
      "kill-test scored the C2 condition on the 662 dismissed videos only; the "
      "flip statistics it reported were conditioned on a stratum that was itself "
      "selected by the C0 scores. Extending C2 to all 1283 videos gives the "
      "first measurement of the methodized pipeline that is not conditioned on "
      "the baseline it is being compared against.")
    A("")
    A("## What the pipeline is")
    A("")
    A("Three stages, one MLLM call per video.")
    A("")
    A("1. The original mp4 audio is re-transcribed with Whisper large-v3.")
    A("2. The fresh transcript passes the frozen degeneracy gate and is fed to "
      "the judge uncapped.")
    A("3. The judge emits a raw unclipped `z = logsumexp(Yes ids) - "
      "logsumexp(No ids)`, and the decision threshold is the valley of a kernel "
      "density estimate over the 1283 scores. No labels enter any of the three.")
    A("")
    A("## Configuration provenance")
    A("")
    A("Nothing here was chosen for this run. Every component is inherited "
      "unmodified from an earlier frozen artifact.")
    A("")
    A("| Component | Source | Status |")
    A("|---|---|---|")
    A("| Audio and VAD | `scripts/duplex/channel_restoration_audio.py` | "
      "unmodified |")
    A("| Whisper ASR | `scripts/duplex/channel_restoration_asr.py` | unmodified |")
    A("| Degeneracy gate | `scripts/duplex/channel_restoration_gate.py` | "
      "unmodified |")
    A("| Judge and readout | `src/duplex/extract_duplex_readout.py` | unmodified |")
    A("| Threshold recipe | `docs/duplex/reports/rawz_detector_train.json`, "
      "estimator `b_kde_valley` | reimplemented, verified |")
    A("")
    A("The ASR configuration is the pre-registered one: `large-v3` through the "
      "`transformers` pipeline, fp16 on CUDA with the device asserted at load, "
      "language auto-detected, 30-second chunked long-form decoding, and a "
      "30-minute audio cap applied when the wav is extracted. The gate is the "
      "pre-registered conjunction: clause-level repetition collapse applied to "
      "every fresh transcript, then rejection if and only if the silero-VAD "
      "speech fraction is below 0.05 **and** the gzip ratio of the raw fresh "
      "text exceeds 7.")
    A("")
    A("The threshold recipe was validated before use. Applied to the C0 scores "
      f"it returns {sc['recomputed_valley']}, against the frozen "
      f"{sc['frozen_valley']} recorded in `rawz_detector_train.json` "
      f"(match: {sc['matches']}).")
    A("")
    A("## Headline numbers")
    A("")
    A("| | C0 baseline | C2 full corpus |")
    A("|---|---|---|")
    A(f"| AUC, hateful vs NH | {pct(auc['C0_baseline']['hateful_vs_NH'])} | "
      f"**{pct(auc['C2_full_corpus']['hateful_vs_NH'])}** |")
    A(f"| AUC, IM vs NH | {pct(auc['C0_baseline']['IM_vs_NH'])} | "
      f"{pct(auc['C2_full_corpus']['IM_vs_NH'])} |")
    A(f"| AUC, EX vs NH | {pct(auc['C0_baseline']['EX_vs_NH'])} | "
      f"{pct(auc['C2_full_corpus']['EX_vs_NH'])} |")
    if m:
        A(f"| Label-free macro-F1 | {pct(c0['macro_f1'])} | "
          f"**{pct(m['macro_f1'])}** |")
        A(f"| Label-free threshold | {c0['threshold']} | {m['threshold']} |")
        A(f"| FN / FP at that threshold | {c0['fn']} / {c0['fp']} | "
          f"{m['fn']} / {m['fp']} |")
    A("")
    A(f"The C2 AUC carries a bootstrap 95% interval of "
      f"{auc['C2_hateful_vs_NH_boot95']}.")
    A("")
    A("## The label-free threshold")
    A("")
    if thr.get("value") is None:
        A("**The recipe found no valley.** The KDE over the C2 scores does not "
          "present two modes, so the threshold component has no output on this "
          "distribution and needs redesign before it can be carried into a "
          "method.")
    else:
        A(f"The valley computed on the real full-corpus C2 distribution sits at "
          f"**{thr['value']:.4f}**, with Scott bandwidth "
          f"{thr['bandwidth_scott']:.4f} and modes at "
          f"{[round(x, 3) for x in thr['mode_locations']]}. It lies "
          f"{thr['distance_from_frozen_C0_valley']:+.4f} from the frozen C0 "
          f"valley of {sc['frozen_valley']:.4f}.")
        A("")
        A(f"Trough depth is {thr['relative_trough_depth']:.4f}: the valley "
          f"density is {thr['valley_over_lower_mode']:.4f} of the lower "
          "bracketing mode. Under 1000 bootstrap resamples of the score set the "
          f"valley has standard deviation {boot.get('sd', float('nan')):.4f} and "
          f"a 95% interval of "
          f"{[round(x, 3) for x in boot.get('ci95', [])]}, against the C0 "
          f"valley's own {c0boot.get('sd', float('nan')):.4f} and "
          f"{[round(x, 3) for x in c0boot.get('ci95', [])]}. "
          f"{boot.get('n_without_valley', 0)} of the resamples produced no "
          "valley at all.")
    A("")
    A("Two further thresholds are reported in the JSON as diagnostics and are "
      "not part of the method. Carrying the frozen C0 valley over unchanged "
      f"gives macro-F1 {pct(diag['macro_f1'])}. The F1-maximizing threshold "
      f"chosen against gold labels gives {pct(orc['macro_f1'])} at "
      f"{orc['threshold']}, so the label-free choice costs "
      f"{r.get('oracle_gap', {}).get('gap', float('nan')):.4f} macro-F1 against "
      "an oracle that is not available to a label-free method.")
    A("")
    A("## Error anatomy at the self-computed threshold")
    A("")
    if fn:
        A(f"**False negatives: {fn['n']}.** "
          f"{fn['by_group']['EX']} explicit and {fn['by_group']['IM']} implicit, "
          f"an implicit share of {pct(fn['share_IM'])}. Within-group recall loss "
          f"is {pct(fn['recall_loss_within_group']['EX'])} for EX and "
          f"{pct(fn['recall_loss_within_group']['IM'])} for IM. Of the eight "
          "videos Step 0 established have no transcribable speech, and for which "
          "C2 therefore has nothing to restore, "
          f"{fn['step0_no_meaningful_speech_residuals']['n_still_FN']} remain "
          f"false negatives. {fn['gate_outcome']['fell_back_to_dataset_transcript']} "
          "of the false negatives fell back to the dataset transcript because "
          "the gate rejected their fresh pass.")
    if fp:
        sm = fp["surface_cue_in_seen_transcript"]
        A("")
        A(f"**False positives: {fp['n']}**, all NH by construction. "
          f"{pct(sm['frac'])} of them carry a surface cue in the transcript the "
          f"judge actually read, against {pct(sm['true_negative_baseline_frac'])} "
          "among the true negatives (Fisher exact "
          f"p = {sm['fisher_p']:.3g}). The cue family is the frozen regex set "
          "from the kill-test analysis, used as a text-locating device only and "
          "never as an input to any model or threshold.")
    A("")
    A("## Transcription and gate statistics")
    A("")
    fc = g["full_corpus"]
    A(f"Across all {fc['n']} videos the gate accepted {fc['accepted']} fresh "
      f"transcripts and rejected {fc['rejected']} "
      f"(rate {pct(fc['reject_rate'])}), which fall back to the dataset "
      f"transcript. {fc['hit_30min_cap']} clips reached the 30-minute audio cap "
      f"and {fc['asr_error']} raised an ASR error. Median transcript length goes "
      f"from {fc['median_old_chars']} characters in the dataset to "
      f"{fc['median_gated_chars']} after the fresh pass and the gate, at a "
      f"median normalized edit distance of {fc['median_edit_norm_vs_dataset']} "
      "against the dataset text.")
    A("")
    A(f"Of the {r['coverage']['n_scored_C2']} judged videos, "
      f"{g['transcript_source_in_judge_input']['override_gated_fresh']} were "
      "judged on a gated fresh transcript and "
      f"{g['transcript_source_in_judge_input']['fallback_dataset_transcript']} "
      "on the dataset transcript.")
    A("")
    A("## Determinism")
    A("")
    A(f"{det['n_common']} videos were scored in both this run and the "
      "kill-test C2 run under a byte-identical judge input. "
      f"{det['n_exactly_equal']} of them reproduce their score exactly; the "
      f"largest absolute difference is {det['max_abs_delta']}.")
    if r2b:
        a2 = r2b["auc"]
        m2 = r2b["operating_points"].get("METHOD_self_computed_valley")
        t2 = r2b["threshold_label_free"]
        A("")
        A("## Contrast arm: Qwen3-VL-2B-Instruct")
        A("")
        A("The same full-corpus C2 inputs judged by the smaller model, as a "
          "boundary condition on where the pipeline stops working.")
        A("")
        A(f"AUC hateful vs NH is {pct(a2['C2_full_corpus']['hateful_vs_NH'])} "
          f"(IM vs NH {pct(a2['C2_full_corpus']['IM_vs_NH'])}, EX vs NH "
          f"{pct(a2['C2_full_corpus']['EX_vs_NH'])}). The label-free valley "
          f"lands at {t2.get('value')}"
          + (f" and yields macro-F1 {pct(m2['macro_f1'])} with {m2['fn']} false "
             f"negatives and {m2['fp']} false positives."
             if m2 else ", and the recipe returns no usable operating point."))
    A("")
    A("## Scope")
    A("")
    A("Every figure is on `train_clean`. The ImpliHateVid test split has not "
      "been scored and is reserved. Published supervised numbers on that "
      "dataset are test-split results under full supervision, so nothing here "
      "licenses a comparison against them. Gold EX/IM/NH prefixes were consumed "
      "by the evaluation only: the transcription, the gate, the judge, and the "
      "threshold are all label-free.")
    A("")
    A(f"Full statistics: `{rep_path.split('Hate-follow-up/')[-1]}`. Raw "
      "transcripts, wavs, and per-video scores stay in the gitignored "
      "`results/` tree and are not published.")
    A("")

    with open(dest, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {dest} ({len(L)} lines)")


if __name__ == "__main__":
    main()
