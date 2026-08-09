# Pre-registration: scale emergence of the label-free operating point

**Frozen 2026-08-10, before any 4B weight was downloaded and before any new video was scored.** Nothing below may be changed once the first 4B score is written. This document governs `docs/duplex/SCALE_EMERGENCE_NOTE.md` and `results/scale_emergence/results.json`.

## What is being tested

The round-2 record in `idea-stage/IDEA_REPORT.md` (appended at commit db778d6) reports a dissociation between two scales of the same frozen judge on ImpliHateVid `train_clean`, 1,283 videos. A linear probe on hidden states reaches 0.9655 on the 2B against 0.9647 on the 8B, so the hatefulness construct is equally present in both residual streams. Ranking AUC is 0.898 against 0.934, so both order the corpus. But the de-quantized KDE relative trough depth of the raw logit contrast is 0.0315 against 0.4011, and the standard deviation of the cosine between the final state and the Yes-minus-No unembedding direction is 9.1 times narrower on the 2B. The 2B holds the construct and never rotates its residual stream toward the answer direction, so its score distribution is unimodal and no label-free operating point exists on it.

Those measurements were taken with labels in view, on a train split, after the fact. They are hypothesis-generating and are not evidence for anything. This pre-registration is the confirmatory completion: the same readouts, on held-out test splits, with a third scale interposed between the two, with the interpretation boundaries frozen before any number is seen.

This is a **finding**, not a method. There is no success bar to clear and no component that survives or dies. What is frozen is therefore the set of results that would **disconfirm** the reading, so that the finding cannot be rescued by reinterpretation after the fact.

## Model arms

| arm | checkpoint | text layers | hidden size | status |
|---|---|---:|---:|---|
| 2B | `Qwen/Qwen3-VL-2B-Instruct` | 28 | 2048 | already scored; z and hidden states on disk |
| 4B | `Qwen/Qwen3-VL-4B-Instruct` | 36 | 2560 | new; to be downloaded and scored |
| 8B | `Qwen/Qwen3-VL-8B-Instruct` | 36 | 4096 | already scored; z and hidden states on disk |

The 4B arm runs `src/duplex/extract_duplex_readout.py` **unmodified**, bf16, `device_map=cuda:0`, the frozen `prag` reader, 16 frames from `frames_16`, `max_pixels` 100352, one forward pass per video, raw unclipped `z`, `--transcript-limit 0`, and `--transcript-override-json` pointed at the same `c2_overrides.json` the 2B and 8B arms read. Same prompt, same inputs, same readout, same output layout, differing only in `--model`. No 2B or 8B video is rescored: their per-video `z` and their 37-row and 29-row hidden-state arrays are already on disk for both corpora and are read as they stand.

## Corpora

Test splits only, per the owner's standing rule. The prior 2B and 8B train-split numbers stay as background and are never pooled with these.

| corpus | split | ids | scored by 8B and 2B |
|---|---|---:|---:|
| ImpliHateVid | `test_clean` | 401 | 400 (one video has no frames and is skipped by the frozen judge) |
| HateMM | `test_clean` | 215 | 215 |

The 4B arm must reach the same coverage as the 8B arm on each corpus. A coverage assert refuses to write the report otherwise.

## Frozen readouts

Five quantities, computed identically for every model and every corpus. All are computed on CPU from artifacts on disk after scoring.

**The final state.** `hidden_states[-1]` as written by the extractor is the residual stream **before** the model's final RMSNorm. The state that produces the logits is
`h = x * rsqrt(mean(x^2) + eps) * g`, with `g` the model's own `model.language_model.norm.weight` and `eps = 1e-6`. This was verified before freezing: on ImpliHateVid `train_clean` the recipe returns state norms of 226.5 for the 8B and 142.6 for the 2B against the 226.5 and 142.6 in the round-2 table, and `|d|` of 1.6544 and 1.5448 against 1.654 and 1.545, and cosine spread 0.02946 and 0.00323 against 0.0295 and 0.0032. The convention is the round-2 convention.

**The answer direction.** `d = mean(W[yes_ids]) - mean(W[no_ids])`, with `W` the unembedding matrix (`lm_head.weight`, or `model.language_model.embed_tokens.weight` where the checkpoint ties them, which the 2B and the 4B do) and the id sets the frozen `build_binary_token_ids` sets, `Yes = [7414, 9454, 9693, 9834, 14004, 14080]`, `No = [902, 2152, 2308, 2753, 5664, 8996]`. These sets are identical across all three checkpoints, which share a tokenizer; the run asserts this rather than assuming it.

**(a) z distribution.** Standard deviation of the raw `z`; dynamic range, defined as `max(z) - min(z)`; and the **de-quantized KDE relative trough depth**. De-quantization recomputes `z = logsumexp(W[yes] h) - logsumexp(W[no] h)` in fp32 from the stored final state, escaping the bf16 logit grid the stored `z` sits on; the stored `z` is reported alongside and the maximum deviation between the two is reported as a check that de-quantization is a grid correction and nothing more. Trough depth uses the exact E7/KDE convention of `crossbench_analyze.kde_valley`: Gaussian KDE, Scott bandwidth, 4001-point grid over `[min - 2, max + 2]`, the two highest grid local maxima as modes, the minimum-density grid point strictly between them as the valley, and `relative_trough_depth = 1 - density(valley) / min(density(mode_a), density(mode_b))`. Fewer than two modes returns no valley, which is itself the reported diagnostic. The primary trough figure is the de-quantized one; the bf16 one is reported in the same table.

**(b) Angular commitment.** Standard deviation across the corpus of `cos(h, d)`. Mean cosine, `|d|` and mean `|h|` are reported alongside so that a spread difference cannot be confused with a geometry difference.

**(c) Ranking AUC.** Mann-Whitney AUC of the de-quantized `z` against gold labels, hateful versus normal, with the AUC routine already used in `readout_bottleneck_killtest.py`. Evaluative; uses labels; is not part of any operating point.

**(d) Linear-probe AUC.** Ridge regression on the mid-stack hidden state, five-fold cross-validated, AUC over held-out predictions. The frozen layer rule is `round(0.75 * n_layers)`, which reproduces the round-2 choices of layer 21 on the 2B (28 layers) and layer 27 on the 8B (36 layers) and gives layer 27 on the 4B (36 layers). Features are standardized per dimension with the fold's own training-half statistics; the ridge penalty is fixed at `lambda = 1.0` and is never tuned; folds are stratified with `seed = 0`. Evaluative; measures whether the construct is present, not whether it is readable.

**(e) Valley macro-F1 against oracle macro-F1.** The label-free decision at the de-quantized valley of that arm's own test distribution, scored as macro-F1 against gold, and the F1-maximizing threshold chosen against gold as a diagnostic ceiling. No valley means no label-free decision and the cell is a dash, which is the substantive result for that cell rather than a missing measurement.

Every cell of `3 models x 2 corpora x 5 readouts` is computed and reported. Nothing is omitted.

## Frozen disconfirmers

**D1 — the emergence reading dies if the trough is already deep at 4B.** If the 4B de-quantized relative trough depth is at least `0.7 x` the 8B's on **both** corpora, then thresholdability grows smoothly with capability and there is no transition between 2B and 8B to report. The finding is then reported as smooth capability scaling, and the word emergence is removed from the note.

**D2 — the "construct present at all scales" half dies if the 4B probe is weak.** If the 4B probe AUC is more than 0.05 below the 8B's on either corpus, the dissociation is not construct-present-but-unreadable at that scale, and the claim collapses to ordinary capability loss. Reported as such.

**D3 — the angular-rotation mechanism dies if the two orderings disagree.** Rank the three scales by cosine spread and rank them by de-quantized trough depth, per corpus. If the rank orders differ on either corpus, the mechanistic claim that angular commitment is what makes the score distribution bimodal is disconfirmed, and the note reports the trough ordering as an observation with no mechanism attached, even if the trough ordering itself is monotone.

D1, D2 and D3 are independent. Any of them may fire without the others.

## Frozen analysis discipline

- No model, layer, corpus, threshold or readout is selected after seeing results. The probe layer rule, the KDE convention, the direction convention and the id sets are all fixed above.
- All six model-by-corpus cells are reported for every readout, including cells that are dashes.
- Labels enter readouts (c), (d) and (e) only. Readouts (a) and (b) never see a label; they are the label-free half and they are the half the finding is about.
- No raw video id is written to `results/scale_emergence/results.json` or to the note.
- The train-split 2B and 8B numbers from round 2 are quoted only as background, in a separate table, never merged into the test table.

## Optional descriptive arm

The pre-registered optional arm was Qwen3-VL-8B **base**, non-instruct, on ImpliHateVid test only, to locate the geometry's origin in pretraining scale rather than instruction tuning. **This arm cannot be run: no non-instruct Qwen3-VL checkpoint is published.** The Qwen3-VL family on the Hub is Instruct, Thinking, FP8 and GGUF variants only, at 2B, 4B, 8B, 30B-A3B, 32B and 235B. Checked before freezing.

The substitute declared here, before any scoring, is `Qwen/Qwen3-VL-8B-Thinking`: the same pretraining scale under a different post-training recipe. It bears on the same question from one side only — it can show that post-training recipe moves the geometry, but it cannot show that pretraining alone produces it. It is **descriptive**, carries no disconfirmer, and is run on ImpliHateVid test only and only if the confirmatory arm completes inside budget. If it is run, its numbers appear in a separate table, clearly marked, and never inside the three-scale table.

## Budget

The 8B arm scored 400 ImpliHateVid videos in 116 s and 215 HateMM videos in 60 s on this machine. The 4B arm is smaller and reads the same inputs, so the projection is under five GPU-minutes for both corpora, plus roughly 8 GB of download. The optional arm adds roughly two GPU-minutes plus 16 GB of download. The projection is far under the three-GPU-hour stop rule. If measured GPU time exceeds three hours the run stops and reports.

## Execution

One GPU job at a time, detached with `setsid nohup`, with a STATUS file, a DONE marker, a pre-flight decode check on the Yes and No id sets, and a coverage assert before any report is written.
