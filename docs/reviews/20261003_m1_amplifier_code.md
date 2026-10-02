# Independent code review: M1 Amplifier

Date: 2026-10-03. Reviewer: independent agent `/root/m1_grounder_code_review`, same model as the parent session. **PASS after the report-integrity correction below; no remaining observation-affecting bug found.**

Reviewed `experiments/20261003_m1_amplifier/{README.md,amplifier.py,measure.py,analyze.py,selfcheck.py,launch/}` against the accepted proposal and rule 6. This is an implementation review, not another proposal review. The reviewer changed no candidate source, ran no GPU job or benchmark evaluator, and read no real GT or candidate performance. PASS permits the declared deployment smoke and experiment; it does not establish effectiveness.

## Correction verified

The initial report checked decoded coverage/rate/length but omitted decoded finite-score and native-global assertions. Canonical evaluation filters nonfinite frames, so that omission could allow mismatched comparison support to pass reporting. The author added finite decoded curves and unchanged `extra.z_video` checks before GT loading. Stance is checked when present, since the existing r6 output does not serialize it. The author also added explicit window/attention-list length checks to prevent `zip` truncation in preparation.

The reviewer confirmed these changes and executed synthetic failure cases: a nonfinite decoded score and a changed decoded global both raise before any GT access. A normal decoded fixture with no stance field still passes. No method computation changed for this correction.

## Intervention and reference

- `ImageAmplifier` selects exactly language layers 2 through 35 for the 36-layer model. Other language layers, the vision encoder and inactive calls delegate to the saved original SDPA function. The registry is restored on close.
- The dispatcher receives Q/K after Qwen normalization and mRoPE and after cache concatenation. It recomputes only the last query row, with the existing scaling and mask, adds `.5 * abs(logit)` on the actual expanded prefix image-key positions, performs FP32 softmax and casts probabilities to the query dtype before multiplying V. GQA maps each contiguous query-head group to its KV head. The original SDPA output is retained for every earlier query row.
- The visual support is padded with false entries for global/answer/query tokens; those tokens are not accidentally amplified. Cached multi-token suffixes require an explicit offset mask, avoiding upper-left causal-mask misuse. Alpha 0 and .5 use the same last-row computation. Alpha 0 is an arithmetic control, not an assertion of equality with native SDPA.
- Prefix, global question/answer and speech queries run with the intervention inactive. Each visual intervention crops back to the original cache length in `finally`, and per-query layer visits are checked. The prefix/global/answer K/V remain reusable; the intervened suffix's last-token state can change, as declared.
- The language-only reference uses the shared `prefix_messages(..., with_frames=False)`, resets RoPE and creates a new cache after releasing the visual cache. System text, ASR and policy are retained; image entries/timestamps and the frame-bearing introduction are removed. The original native answer is forced, and visual query token sequences are asserted identical. The reference's global margin is diagnostic only. This is the declared language-only reference, not an isolated pixel deletion.

Read-only inspection of the actual `uoa-lab2` Transformers 5.15.1 attention/SDPA source confirmed the dispatched tensor layout, GQA path, mask handling, scaling argument and output layout used by the wrapper. Its SDPA interface disables implicit causality when an explicit mask is supplied. No deployment GPU execution was performed by this reviewer.

## Token contrast

The implementation computes `1.1 * amplified_token_logits - .1 * reference_token_logits` before the original Yes/No token-set logsumexp aggregation. It uses the shared FP32 output-head projection and original answer-token IDs. Vocabulary log-normalizers contribute a common additive constant to all contrasted tokens and therefore cancel from the final Yes-minus-No margin; full-vocabulary projection is unnecessary for this declared arithmetic. APC is absent as explicitly declared.

An independent full-vocabulary numerical check agreed with the answer-token subset margin within `6.56e-7`. A separate counterexample produced a token-first margin of `2.4951005`, versus `1.6306838` when class margins were incorrectly contrasted first. Thus the implementation follows the intended order rather than the generally different scalar-margin shortcut.

## Independent executable verification

Artifacts:

- `runs/20261003_m1_amplifier/independent_review/check_amplifier.py`
- `runs/20261003_m1_amplifier/independent_review/check_amplifier.json`
- Synthetic analysis outputs under the same directory's `synthetic_analysis/`.

The CPU fixture used actual randomly initialized multimodal Qwen3-VL with 36 small language layers, GQA, DeepStack and real image-processor grids expanding to 4 and 6 image tokens. It used synthetic token IDs and the shared message-construction routine. Local versions: Torch 2.7.1 / Transformers 4.57.6. Both FP32 and BF16 passed:

| Independent check | Result |
|---|---|
| GQA oracle using separate per-head computation | Correct head mapping and amplified row |
| Layer scope | Exactly 34 intervened layers, indices 2–35 |
| Query-row scope | First two layers and all earlier query rows unchanged; last row changes from layer 2 |
| mRoPE/DeepStack | Positions unchanged, actual multimodal/DeepStack path exercised |
| Prefix K/V and independent queries | Exact after crop; independent-copy and reused-cache reads agree |
| Registry/native restoration | Registry restored; native global/window reads exact after wrapper removal |
| No-image reference | No image tokens/pixels; system/ASR/policy retained; original questions/answer used |
| Speech/global/input/parameters | Exact preservation |
| Actual reader and call hook | With `V=3`, `B=4`: smoke 26, native/eager standalone 7, PAI standalone 13 |
| 4 fps reconstruction | Complete 96-frame curves agree with the window scores |

The maximum amplified-versus-alpha-zero answer-token differences were `.0435150` in FP32 and `.0466162` in BF16. These demonstrate activation in the fixture, not semantic correctness or benchmark gains. The author's separate text-model selfcheck also passed and was read as additional evidence. Deployment pretrained-model parity, runtime and memory still require the declared GPU smoke.

## Evaluation and cost

The scoring path reads shared frames/ASR/manifest fields and no GT. Preparation requires the complete two-corpus manifest in all three raw arms and checks, exact native global/windows against `base_gridA`, unchanged speech and stance, finite 4 fps curves, equal lengths, matching token shapes, all 34 layer visits and exact call formulas. Saved logits independently reconstruct the eager and PAI visual margins before evaluation.

Synthetic records passed the actual preparation/report functions with correct `base/eager/pai` fields and no template leftovers. Command interception confirmed the canonical raw evaluator and unchanged r6 invocation: `--noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`. Each arm has a separate output tag; the launcher waits for all successful exits before reporting. Shell syntax checks passed. Paired uncertainty resamples video-level differences. The declared performance gate checks both corpora against native, current r6 and matched eager control, and the report keeps `mechanism_supported: false`.

Actual outer-call accounting is consistent: native/eager `3+B`; PAI deployment `6+B+V`; complete collection `6+B+3V`; smoke adds `3+B`. The deployment time includes the native prefix/global/answer, amplified visual reads, shared native speech, language-only prefix/global/forced-answer and all reference queries. It excludes the collection-only native/eager visual controls. The wrapper still computes original SDPA before replacing the final row, and expands GQA K/V for that row; this extra work is real and must remain in measured runtime. Outer call counts alone do not establish equal per-call cost or the README's provisional runtime estimate.

**Disposition: PASS for the declared GPU smoke, followed by the complete paired experiment if plumbing passes.** Attention mass rises by construction, and neither that rise nor successful implementation establishes temporal grounding. The declared raw-order, matched-control and later component/support checks remain necessary for a mechanism claim.
