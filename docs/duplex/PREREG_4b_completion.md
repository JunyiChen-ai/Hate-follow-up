# Pre-registration: completing the five-corpus picture at 4B

**Frozen 2026-08-10, before any MHClip-EN, MHClip-ZH or HateClipSeg video was scored by the 4B checkpoint.** Nothing below may be changed once the first new score is written. This document governs `docs/duplex/FOURB_COMPLETION_NOTE.md` and `results/scale_emergence/fourb_completion.json`.

## What is being measured, and what is not

The scale-emergence run at commit d804f51 scored `Qwen/Qwen3-VL-4B-Instruct` on two held-out test splits. On ImpliHateVid the 4B reaches valley macro-F1 0.8568 against the 8B's 0.8823, so it loses. On HateMM it reaches 0.7904 against the 8B's 0.6562, so it wins by 0.134. Ranking moves the other way on both corpora: the 8B's AUC is higher in each case. Two corpora cannot say whether that pattern is a property of the scale or a property of HateMM.

This run completes the picture on the three corpora the 4B has never seen: MHClip-EN, MHClip-ZH and HateClipSeg. The comparison is then five corpora deep on both scales.

**This is a fixed model swap, evaluated everywhere.** One checkpoint, `Qwen/Qwen3-VL-4B-Instruct`, is scored on every corpus, and its five-corpus row is compared with the five-corpus row of the one checkpoint `Qwen/Qwen3-VL-8B-Instruct`. There is no per-corpus model selection anywhere in this design, no mixing of the two arms inside a mean, and no reporting of a best-per-corpus envelope. A reader must be able to substitute either checkpoint for the other as a single deployment decision and read the consequence off one row.

**This is measurement, not a method claim.** No component is being promoted, no rule is being fitted, and nothing here is proposed as part of the framework. The scale-emergence run already disconfirmed the graded angular-rotation mechanism (D3 fired at d804f51), so no mechanism is available to explain a scale effect on the label-free operating point, and none will be offered. The strongest statement this run can license is the mechanism-free empirical one: *at these five corpora, under this frozen pipeline, the label-free operating point prefers the intermediate scale* — or does not.

## Model arm

| arm | checkpoint | text layers | hidden size | status |
|---|---|---:|---:|---|
| 4B | `Qwen/Qwen3-VL-4B-Instruct` | 36 | 2560 | already downloaded and cached; scored on ImpliHateVid and HateMM at d804f51 |
| 8B | `Qwen/Qwen3-VL-8B-Instruct` | 36 | 4096 | already scored on all five corpora; z and hidden states on disk |

The 4B arm runs `src/duplex/extract_duplex_readout.py` **unmodified** under exactly the settings the 8B test runs used: bf16, `device_map=cuda:0`, the frozen `prag` reader, 16 frames from `frames_16` at `max_pixels` 100352, one forward pass per video, raw unclipped `z = logsumexp(W[yes] h) - logsumexp(W[no] h)`, `--transcript-limit 0`, and `--transcript-override-json` pointed at the same `c2_overrides.json` the 8B arm read on that corpus. Same prompt, same frames, same fresh gated transcripts, same output layout, differing only in `--model`. Hidden states are dumped per video, one array of `37 x 2560`, as at d804f51.

No 8B video is rescored. The 8B `z` and hidden states already on disk are read as they stand.

## Corpora

Test splits only. The three new corpora, plus the two already measured at both scales, quoted from committed reports.

| corpus | split | scored by 8B | 4B status | working directory |
|---|---|---:|---|---|
| ImpliHateVid | `test_clean` | 400 | scored at d804f51 | `results/testruns/implihatevid` |
| HateMM | `test_clean` | 215 | scored at d804f51 | `results/testruns/hatemm` |
| MHClip-EN | `test_clean` | 161 | **new** | `results/testruns/mhclip_en` |
| MHClip-ZH | `test_clean` | 149 | **new** | `results/testruns/mhclip_zh` |
| HateClipSeg | `test` (no official split; the 394 annotated videos with decodable media) | 394 | **new** | `results/hateclipseg` |

Coverage asserts of exactly 161, 149 and 394 scored videos, and the same counts of hidden-state arrays, run before any report is written. A shortfall aborts the run rather than reporting a subset. HateClipSeg has no official split; its 394 are the whole annotated corpus that survives the media-availability and decodability attrition documented in `GATED_ANCHOR_HATECLIPSEG_NOTE.md`, and it is reported as such rather than as a held-out split.

MHClip-ZH uses the automatic-language 8B arm, `judge_8b`, as its reference, not the forced-zh contrast arm.

## Frozen readouts

Computed identically for both arms on every corpus, on CPU, from artifacts on disk after scoring. Every convention is inherited from `PREREG_scale_emergence.md` and none is re-chosen here.

**The de-quantized score.** `z` recomputed in fp32 from the stored final state, after the model's own final RMSNorm with `eps = 1e-6` and its own `model.language_model.norm.weight`, against `d = mean(W[yes_ids]) - mean(W[no_ids])` with the frozen id sets `Yes = [7414, 9454, 9693, 9834, 14004, 14080]` and `No = [902, 2152, 2308, 2753, 5664, 8996]`. The stored bf16 `z` is reported alongside and the maximum absolute deviation between the two is reported as a grid-correction check.

**(a) Ranking AUC.** Mann-Whitney AUC of the de-quantized `z` against gold labels. Evaluative; uses labels; is not part of any operating point.

**(b) De-quantized KDE relative trough depth and mode count.** The exact E7/KDE convention of `crossbench_analyze.kde_valley`: Gaussian KDE, Scott bandwidth, 4001-point grid over `[min - 2, max + 2]`, the two highest grid local maxima as modes, the minimum-density grid point strictly between them as the valley, `relative_trough_depth = 1 - density(valley) / min(density(mode_a), density(mode_b))`. Fewer than two modes returns no valley, which is the substantive result for that cell and is reported as a dash rather than as a missing measurement. Label-free.

**(c) Valley macro-F1.** The label-free decision at that arm's own de-quantized valley on that corpus's own test distribution, scored as macro-F1 against gold. Transductive over the unlabeled batch, as every previous run in this program has been.

**(d) Oracle macro-F1.** The macro-F1-maximizing threshold chosen against gold, as a diagnostic ceiling, never part of any rule. This is the macro-F1-max convention of `anchored_operating_point.py` and `scale_emergence_analyze.py`, not the hateful-F1-max convention of the committed `test_c2_*.json` reports; the two differ only on MHClip-EN, where hateful-F1-max gives 0.6888 against macro-F1-max 0.6962. Both conventions are recorded in the JSON.

**Label collapses.** ImpliHateVid `{Hateful: 1, Normal: 0}`; HateMM `{Hate: 1, Non Hate: 0}`; MHClip-EN and MHClip-ZH `{Hateful: 1, Offensive: 1, Normal: 0}`, the standing project mapping. HateClipSeg is reported under **both** shipped collapses, computed by `hateclipseg_prep.video_labels` unmodified: the **offensive union** (positive if any of hateful, insulting, sexual, violence or harm appears at video level) and **hateful strict** (only hateful counts). The label-free threshold is identical across the two collapses by construction, since no fitted quantity reads a label; only the labels the predictions are scored against change.

## Frozen reproduction check

Before any 4B number is interpreted, the analysis recomputes the **8B** cells with the same code and asserts that the stored-bf16 valley macro-F1 reproduces the committed values to within 0.002:

| corpus | committed 8B valley macro-F1 | source |
|---|---:|---|
| ImpliHateVid | 0.8823 | `TEST_RUNS_NOTE.md`, `docs/duplex/reports/test_c2_implihatevid_8b.json` |
| HateMM | 0.6562 | `TEST_RUNS_NOTE.md`, `docs/duplex/reports/test_c2_hatemm_8b.json` |
| MHClip-EN | 0.6962 | `TEST_RUNS_NOTE.md`, `docs/duplex/reports/test_c2_mhclip_en_8b.json` |
| MHClip-ZH | 0.7256 | `TEST_RUNS_NOTE.md`, `docs/duplex/reports/test_c2_mhclip_zh_8b.json` |
| HateClipSeg, offensive union | 0.6522 | `GATED_ANCHOR_HATECLIPSEG_NOTE.md`, `results/hateclipseg/gated_anchor_results.json` |
| HateClipSeg, hateful strict | 0.4479 | same |

All six were verified against the committed reports before this document was frozen and none was misquoted in the run request. A reproduction failure larger than 0.002 aborts the analysis. The de-quantized figures are reported alongside; where de-quantization moves a cell, both numbers appear and the de-quantized one is primary, matching `SCALE_EMERGENCE_NOTE.md`.

## Frozen aggregate

Two means, both defined before any 4B score is seen, both computed the same way for both arms.

- **Four-corpus valley mean**: the unweighted mean of valley macro-F1 over ImpliHateVid, HateMM, MHClip-EN and MHClip-ZH. The committed 8B value under this definition is **0.7401** (`ANCHORED_OPERATING_POINT_NOTE.md`), and the committed four-corpus labeled-oracle mean is 0.8175. This is the mean that is comparable to the TRIAGE reference of **0.808**, which is quoted at one MLLM call per video on our side against TRIAGE's reported 1.73.
- **Five-corpus valley mean**: the same, with HateClipSeg added under the **offensive union** collapse only, which is the primary collapse frozen at `PREREG_gated_anchor_hateclipseg.md` and used throughout the closeout. The strict collapse is reported in every table but never enters a mean. There is no committed five-corpus mean to compare against; both arms' values are computed here for the first time under the same definition.

A cell with no valley contributes no number, and a mean over an arm with any dashed cell is reported as undefined rather than as a mean over the surviving corpora. Unweighted means over corpora of very different sizes are a crude summary and are labelled as such.

## Interpretation boundaries, frozen

- **A mean is not a method.** If the 4B's five-corpus valley mean exceeds the 8B's, the reportable statement is *the label-free operating point prefers the intermediate scale on these five corpora under this pipeline*, with no mechanism attached. D3 already killed the graded angular account at d804f51 and nothing here revives it. The word "emergence" is not used for any result in this run.
- **No per-corpus model mixing.** The tables report the 4B row and the 8B row. No row, mean or headline may combine the better of the two per corpus, and no sentence may recommend choosing scale per dataset.
- **Ranking and thresholding are separate axes and are reported separately.** A 4B win on valley macro-F1 alongside a 4B loss on AUC is the expected shape given d804f51 and is not evidence that the 4B is the better judge; it is evidence that the valley recipe lands better on the 4B's score distribution.
- **A win inside a saturated corpus is suspect.** HateMM's normal class and HateClipSeg's are both documented as carrying hate-adjacent surface features, and the HateClipSeg union collapse is 87.3 percent positive, where a rule that flags almost everything scores well by accident of prevalence. Any 4B advantage concentrated on those two corpora is reported with that caveat attached in the same paragraph, not in a limitations list at the end.
- **The oracle column bounds the whole argument.** Where the oracle is low, neither scale has a threshold problem worth solving, and the note says so.
- **No raw video id** is written to the JSON or to the note.
- **Nothing is selected after seeing results.** The corpora, the collapses, the readouts, the two means, the reproduction tolerance and the interpretation boundaries are all fixed above.

## Budget and stop rule

The 4B scored 615 videos in 126 GPU-seconds at d804f51. The three new corpora are 704 videos on the same 16-frame grid, so the projection is **about 10 GPU-minutes** total, on a single RTX 5090, with the weights already cached and no download. The analysis is CPU only and rescores nothing.

**Stop rule: one hour of wall clock.** If scoring has not completed within an hour of launch, the run stops and reports whatever coverage it reached, and the note reports the shortfall rather than the subset.

## Execution

One GPU job at a time, detached with `setsid nohup`, with a STATUS file, a DONE marker, a pre-flight decode check that the frozen Yes and No id sets come back unchanged from the 4B tokenizer, an input-fingerprint check on each corpus's `c2_overrides.json`, and coverage asserts of 161, 149 and 394 before any report is written.
