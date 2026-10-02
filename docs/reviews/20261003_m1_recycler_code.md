# Independent code review: M1 Recycler

Date: 2026-10-03. Reviewer: independent agent `/root/m1_grounder_code_review`, same model as the parent session. **PASS after the reporting correction below; no remaining observation-affecting bug found.**

Reviewed `experiments/20261003_m1_recycler/{README.md,recycler.py,measure.py,analyze.py,selfcheck.py,smoke_report.py,launch/}` under rule 6. The accepted proposal defines a paper-based VAR transfer, including explicit differences from author code; no proposal review was reopened. The reviewer changed only independent test artifacts and this report, ran no GPU job or benchmark evaluation, read no real GT or candidate performance, and computed no hashes.

## Correction and confirmation

The first `analyze.report` implementation retained `r['pai_raw']` from Amplifier, although its arm names were `base/eager/recycle`. This would raise `KeyError` when producing the complete report. The reviewer reported it; the author changed it to `recycle_raw`. The independent synthetic preparation/report subsequently reached `ANALYSIS_DONE` with the corrected raw paired comparison. No reader change was needed.

## Computation and lifecycle

- Sink channels are the stable descending absolute-value top two of the first prefix token's **decoder block-input residual**. Stable sorting gives smaller channel indices precedence on exact ties. The indices are established once per layer during native prefix capture and retained for the global question, forced answer and subsequent window queries.
- `sink_mask` uses FP32 activation values, exact RMS over all hidden channels and the maximum absolute activation on the selected channels. It uses `>=20`, with zero RMS mapped to zero. This matches the frozen declaration rather than silently adopting the author's epsilon/strict-inequality variant.
- Capture appends masks for the native prefix, original global question and original answer. `end_capture` requires every modified layer and exactly the current cache length. Ordinary native visual/speech reads do not append masks. Each intervened query obtains fresh current-token masks at each layer from its current residual states, so previous interventions can affect subsequent-layer detection as intended.
- Only visual-window calls activate redistribution. Layers 0–34 are modified; layer 35 and the vision encoder retain the original attention interface. All suffix query rows participate. Original cached K/V and masks remain fixed, and cache cropping/current-mask reset prevents one window query from becoming context for another.
- GQA repeats each KV head over its contiguous query-head group. Masked softmax is FP32 after the declared original-dtype QK/scaling arithmetic. Selection is per query row/head: visual mass at least `.2`, non-sink visual mass at least half of visual mass, and positive recipient/sink mass.
- For selected rows/heads, sink probabilities are multiplied by `.4`; the removed `.6` mass is added to non-sink visual keys in proportion to their original probability. Other entries remain unchanged. The two supports are disjoint. The implementation preserves the FP32 probability sum and zero causal entries. Text-only sink mass can trigger redistribution, as explicitly declared. No-sink or no-recipient cases leave probabilities unchanged.

The native prefix/global/answer computation is not altered by capture. Both local branches retain their original question strings, but only visual queries use the new attention computation. Native speech, numeric global and answer text are copied unchanged into all output arms. No GT is read by scoring or channel/sink/head selection.

## Independent nonempty-path verification

Artifacts:

- `runs/20261003_m1_recycler/independent_review/check_recycler.py`
- `runs/20261003_m1_recycler/independent_review/check_recycler.json`

The independent CPU test uses actual multimodal Qwen3-VL with 36 small language layers, GQA, DeepStack and real image-processor grids expanding to 4 and 6 image tokens. Local versions: Torch 2.7.1 / Transformers 4.57.6. Both FP32 and BF16 were exercised.

The detector itself was separately tested at width 4096, including selected versus unselected spikes, zero RMS, stable top-two ties, persistence across appended tokens, and the inclusive `phi == 20` boundary using a constructed exact case. A width-64 model cannot naturally reach the threshold because its maximum normalized coordinate is 8. Therefore the actual-model test **explicitly substitutes controlled nonempty sink masks**, marks that fact in its JSON, and does not claim natural sink detection or semantic validity. The substitution leaves the real capture/current-query hooks and attention implementation active. Every detector call was independently checked against the actual block-input residual, before learned normalization.

| Independent check | FP32/BF16 result |
|---|---|
| Separate-head/row probability oracle | Correct GQA and redistribution |
| Text-only sinks, visual sinks, no sinks, no recipients and future query sinks | Declared behavior |
| Probability sum and masked future entries | Conserved within tolerance; causal zeros exact |
| Native prefix/global/answer mask alignment | Exact lengths and concatenation at every modified layer |
| First-token channel selection on actual model states | Matches independent stable ordering |
| Active layer/query scope | Layers 0–34, including earlier suffix rows |
| Future suffix token perturbation | Earlier suffix hidden rows remain exact |
| Nonempty operation reaching outputs | Changes final Yes/No margin and `read_video` frame curves |
| Cache K/V, captured masks, model weights and original inputs | Unchanged after cropped queries |
| Repeated query and native restoration after registry removal | Exact |
| mRoPE/DeepStack | Positions unchanged; actual multimodal/DeepStack path exercised |
| Native global, selected answer and speech availability/values | Preserved |

Controlled-sink margin changes relative to matched dense attention were `+.06730837` in FP32 and `+.03970909` in BF16. These are plumbing witnesses only. With three visual queries and one available speech query (`V=3`, `B=4`), the actual outer-model hook recorded 20 smoke calls: 13 collection calls plus 7 native-restoration calls. Every standalone arm records 7 calls. The complete 96-frame, 4 fps curves matched their window scores.

The author's separate selfcheck was also inspected. The independent test goes beyond an empty-sink identity result: nonzero redistribution changes real model outputs while causality and original caches remain intact. Deployment-version, pretrained-weight activation rates and real GPU memory/runtime still belong to the declared smoke, not this CPU evidence.

## Evaluation and reporting

The complete-manifest preparation gate requires identical keys in all three raw arms and checks. It validates native global/window parity with `base_gridA`, speech/stance preservation, exact window counts/bounds, finite 4 fps curves, equal lengths, channel/mask shapes, all 35 layer visits and call counts. This code was tested with a complete synthetic manifest; removing one arm's video was rejected. No real 333-video coverage claim is made before the run.

Synthetic records passed the actual preparation and report functions after the field correction. Nonfinite decoded scores and changed decoded globals were rejected before GT loading. The report allows canonical r6 outputs to omit stance, checking it when present. Paired uncertainty uses video-level differences. No evaluator formula was duplicated: raw commands invoke `src.eval.evaluate_four_datasets`, and separate arm tags use unchanged r6 flags `--noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`. Commands were intercepted, not executed. Shell syntax checks passed; the launcher waits for all arm exits before reporting.

The new `smoke_report.py` received a static narrow check. It reads native predictions/masks/checks without GT, checks five-video native parity, speech/global preservation, frame shapes, calls and probability conservation, and excludes the stress video from first-two-per-corpus cost extrapolation. It does not turn sink or transferred-mass statistics into a performance claim.

## Cost and interpretation boundary

Call formulas are correct: standalone `3+B`; paired collection `3+B+2V`; smoke adds `3+B`. The method still incurs residual inspection, mask capture, dense per-row/head attention and redistribution despite no additional deployment forward count. The recorded native baseline prefix time includes capture overhead shared by collection arms. Treat that as a baseline containing extra work, not a pure native timing when computing an overhead ratio; otherwise the relative added cost could be understated. This qualification is already recorded in the run config.

The method is the declared paper-based adaptation: transferred fraction `.6`, inclusive exact-RMS detector, layers 0–34, text-only sink activation and current-query sink scanning. It is not an exact reproduction of author code. Controlled-mask activation and increased visual attention establish implementation behavior, not that Qwen's selected channels identify semantically redundant content or correctly localized evidence.

**Disposition: PASS for the declared GPU smoke and complete paired experiment if plumbing passes.** The reporting field correction is resolved; no additional method edit is required by this review.
