# Independent narrow code review: Integrator R4

Date: 2026-10-03. Reviewer: independent agent `/root/m1_grounder_code_review`, same model as the parent session. **PASS; no observation-affecting correction required.**

Reviewed the R4 changes in `integrator.py`, `aligned.py`, `measure_aligned.py`, `analyze_aligned.py`, the two aligned launch scripts, the shared token-region mapper and the final README declaration. This is the requested implementation review of the last family revision, not a new proposal review. The reviewer changed no experiment code, ran no GPU job or real benchmark evaluator, read no real GT or new result files, and calculated no hashes. R3's development motivation was read only in the declaration.

## Exact edge and token mapping

`aligned_edges` begins with the complete native lower triangle. For each window, it opens visual query rows to nonvisual, non-ASR keys globally and to ASR keys assigned to that window. Future image-key access remains closed; nonvisual query rows keep their causal direct edges. Visual tokens must have exactly one window assignment, and visual/speech overlap is rejected.

The visual row set uses actual processor-expanded image-token IDs. `token_regions` independently reproduces image-token expansion and requires exact equality with encoded input IDs before using tokenizer offsets. It checks each frame's expanded image count. Frame timestamps use half-open read windows, with a timestamp at the final endpoint assigned to the last window. ASR word allocation uses the same proportional rounding rule as `window_text`; timestamp prefixes are shared among windows receiving nonempty word allocations. No new word aligner or GT-dependent allocation is introduced.

Independent tiny-tokenizer cases verified:

- Different image grids expanding to 4, 6 and 2 image tokens.
- An ASR segment crossing two 8-second windows, including agreement with the actual `window_text` strings.
- A token deliberately spanning two allocated words: character-overlap mapping shares that complete token between the corresponding windows.
- Shared ASR timestamp tokens, an out-of-extent/unassigned ASR word and an empty-speech placeholder. Unassigned ASR gets no newly opened visual edge.
- A frame exactly at duration 24 seconds maps to the last read window.
- Altered expanded IDs, duplicate/missing image-window membership and a frame outside the declared duration are rejected.

The resulting full matrix was compared entry by entry with an independent row/key/window-membership definition. The policy token remained globally accessible from visual rows; an unassigned ASR token did not gain direct access.

The token-sharing boundary matters for interpretation: an indivisible token overlapping two word spans can be assigned to both windows. This is token-level support derived from approximate word spans, not strict isolation of individual words. More generally, retained causal edges and globally accessible non-ASR text can relay remote information in deeper layers. The README correctly rejects a complete information-isolation or online-causality claim.

## Actual multimodal execution and regression

Independent artifacts:

- `runs/20261003_m1_integrator/independent_review/r4/check_aligned.py`
- `runs/20261003_m1_integrator/independent_review/r4/check_aligned.json`

The CPU fixture connects that tokenizer/mapping to actual randomly initialized multimodal Qwen3-VL with 36 small language layers, GQA, mRoPE and DeepStack. It uses real image-processor tensors. Local versions: Torch 2.7.1 / Transformers 4.57.6; FP32 and BF16 both passed.

| Check | FP32 | BF16 |
|---|---:|---:|
| Actual layer attention mask equals the aligned edge matrix | Pass | Pass |
| First-frame layer-1 cached-value change from a matched ASR token change | .00594418 | .02343750 |
| Same cached-value change from an unmatched ASR token change | 0 | 0 |
| Native first-frame cache change from either later ASR change | 0 | 0 |
| mRoPE positions, cropped-cache versus copied-cache suffix read | Exact | Exact |
| Original inputs, parameters and native restoration | Exact | Exact |
| R1/R2 public encoding paths versus explicit original edge matrices | Exact hidden states and all K/V | Exact hidden states and all K/V |

The layer-1 matched/unmatched comparison verifies the newly selected **direct** path. It does not establish deeper-layer separation; the test explicitly records this limitation. All 36 language layers are visited during the modified prefix, and the hook is inactive for ordinary suffix queries. `encoding_edges` therefore introduces the intended custom graph without changing the existing R1 `all` default or R2 `text` graph.

The actual `read_video` path was also executed, including native, explicit-causal visual, aligned visual and native-restoration passes. Modified prefixes receive visual queries only; speech values are copied from the native branch. Original global margin, selected answer, speech availability and speech values remain exact. Frame reconstruction and all returned input tensors were checked.

With `V=3`, `S=2`, `B=5`, the real outer-forward hook measured 28 smoke calls: 20 paired collection calls plus 8 native-restoration calls. Native standalone records 8 calls; causal/aligned standalone record 11. This agrees with `9+B+2V`, smoke plus `3+B`, and deployment `6+B`.

## Full pipeline and cost

The independent analysis fixture contains **333 synthetic records** with a 215/118 split, not real videos or labels. Strict R4/arm configs, manifest durations, saved masks and all three arms passed `analyze_aligned.prepare`. The shared report completed with the expected per-video sample counts. Wrong mask mode, missing arm coverage, a duration mismatch and a nonfinite decoded curve were rejected before array/GT loading in the respective validation tests.

The full-run gate checks exact 333-key coverage, R4 config and arm-config equality, the fixed model/input/window/grid settings, native read parity, original global/stance/speech, window extents and counts, `ceil(4T)` frame lengths, mask shapes, edge counts, all 36 layers and actual calls. Smoke has its separate fixed five-video coverage gate and cannot invoke evaluation/report stages through its CLI.

The wrapper reuses the existing same-experiment canonical analysis; it does not copy AUC or decoder formulas. Intercepted commands confirmed the canonical raw evaluator and unchanged r6 settings: `--noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`. Base/causal/future have separate tags. R4 inputs and decoded/analysis outputs use distinct `r4_*` paths, and the launcher waits for all successful arm processes before reporting. Both launch scripts passed shell syntax checking.

The standalone timing composition is consistent with the revised scheduling: original native prefix/global, original answer and speech, mapping, then the modified prefix/global/forced answer and visual queries. It excludes unused native visual-query work from modified deployment timing. Paired timing includes the one shared native speech pass and both visual-control passes. Dense-mask construction and the second prefix remain real costs. GPU runtime, full-input memory and deployment-version parity are not inferred from the CPU fixture and remain the declared smoke's responsibility.

**Disposition: PASS for the declared R4 GPU smoke, then the complete paired experiment if plumbing passes.** No semantic alignment, localization gain or promotion claim follows from these implementation checks.
