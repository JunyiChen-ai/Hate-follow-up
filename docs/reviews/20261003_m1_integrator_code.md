# Independent code review: M1 Integrator

Date: 2026-10-03. Reviewer: independent agent `/root/m1_grounder_code_review`, same model as the parent session. **Decision: PASS under RESEARCH_ITERATION_RULES.md rule 6. No observation-affecting bug requiring a fix was found.**

Reviewed the declared R1 in `experiments/20261003_m1_integrator/{README.md,integrator.py,measure.py,analyze.py,selfcheck.py,launch/}` and the accepted proposal review. This is the single independent implementation review. The reviewer did not edit candidate code, launch a GPU job, execute benchmark evaluation, or read corpus GT or candidate performance. PASS permits the declared deployment smoke and experiment; it is not evidence of effectiveness or promotion.

## Effective computation

- `allowed_edges` implements `j <= i OR visual[i]`. Every visual row can access the supplied prefix; every other row retains its causal direct edges. The mask does not accidentally restrict keys to visual positions. Every row retains its diagonal, so there are no completely masked rows.
- The hook replaces the attention mask at every language self-attention module. The context checks an exact ordered visit to all layers and clears the intervention before the global question, forced answer and local queries. The deployment code asserts 36 layers in the saved mapping at preparation time.
- Read-only inspection of the actual `uoa-lab2` Transformers 5.15.1 source confirmed that `Qwen3VLTextAttention.forward` passes the supplied mask to the SDPA interface. The SDPA implementation sets `is_causal = q_length > 1 and attention_mask is None and is_causal`; a nonempty explicit mask therefore disables the second causal restriction. The language model's earlier `create_causal_mask` output is replaced at self-attention, rather than intersected with the new graph.
- Visual membership comes from actual processor-expanded `input_ids == image_token_id`, checked against image-grid token counts and the prefix cache length. Padding is rejected. Vision start/end tokens, timestamps and other scaffolding are not incorrectly included in the visual set. The hook does not alter image processing, mRoPE positions or DeepStack feature injection. In Qwen's inspected source, DeepStack still adds the corresponding visual embeddings at visual positions between the first language layers.
- A fresh cache and reset `rope_deltas` are used for each encoding. All arms keep the original input tensors, selected native answer text and downstream native global margin. Modified global margins are diagnostic fields only. Both local modalities use the original ordinary suffix queries; missing speech is skipped exactly as in the native branch. Each query is cropped back to the shared answer-conditioned cache length.

The declaration correctly limits the causal statement to **direct mask edges**. Later nonvisual representations can change indirectly through visual states. Likewise, the original answer text and numeric global score are fixed, while their hidden states need not be fixed under the new prefix.

## Independent executable witness

Artifact and reproducible script:

- `runs/20261003_m1_integrator/independent_review/check_integrator.py`
- `runs/20261003_m1_integrator/independent_review/check_integrator.json`
- Synthetic analysis outputs under the same directory's `synthetic_analysis/`.

The independent check ran on CPU with Torch 2.7.1 and Transformers 4.57.6. It used an actual randomly initialized **multimodal** Qwen3-VL model, 36 language layers, GQA, two DeepStack injection levels, and real image-processor outputs for different image grids. The grids expanded to 4 and 6 image tokens. This is a small-model implementation witness, not a run of the deployed pretrained model.

| Check | FP32 | BF16 |
|---|---:|---:|
| Native visual cached-value maximum change after changing a later text token | 0 | 0 |
| Future-mask visual cached-value maximum change after the same text change | 0.10917455 | 0.06030273 |
| mRoPE positions native versus modified prefix | Exact | Exact |
| Cache crop versus independent-copy suffix read | Exact | Exact |
| Cached prefix K/V after crop | Unchanged | Unchanged |
| Native restoration, input tensors and model parameters | Exact | Exact |
| Original answer/global and ordinary query sequence across arms | Preserved | Preserved |

The changed visual cached values, with unchanged native visual values, demonstrate that future input reaches stored visual memory rather than merely changing the final hidden output. Hooks were inactive on suffix reads. The test also executed the actual `read_video` orchestration, including all three arms and the restored native pass, and verified frame-level reconstruction and modality availability.

The author's separate `selfcheck/future_visibility.json` also passed its FP32/BF16 text-model checks. Its default implicit-versus-explicit causal drift is recorded rather than treated as parity: maximum hidden differences were approximately `8.34e-7` and `.03125`; the fixed math-kernel comparison passed. Keeping the complete explicit-causal control and requiring improvement against it is appropriate. The independent multimodal check and read-only inspection of deployment 5.15.1 cover different layers of evidence; neither substitutes for the declared five-video GPU smoke on the pretrained deployment model.

## Evaluation, alignment and cost

- No GT enters the reader. It uses the shared frame, ASR, window and prompt helpers. Frame curves use `ceil(duration * 4)` elements and 4 fps frame centres mapped to the existing 8-second windows. The synthetic reader check verified the complete curve, including a visual-query window without a sampled frame.
- `prepare` requires all three raw arms and checks to match the complete two-corpus manifest. It checks original global and window read parity against `base_gridA`, window boundaries, speech availability, finite frame curves, 4 fps rates, lengths, token mapping and measured calls. No partial video selection is performed.
- The raw evaluator command calls `src.eval.evaluate_four_datasets`. The decoder command calls the unchanged r6 entry with `--noleak --transform nscore --key calib --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`. Three arms have separate input/output tags and independent unchanged unlabeled fits. The launcher waits for all three successful exits before reporting.
- Synthetic records exercised `prepare` and `report` without reading benchmark files. The report uses `future_raw - base_raw`, the correct diagnostic fields and the correct arm names; no Factorizer or other candidate computation remained. Evaluator/decoder command construction was intercepted and verified without running either process. The synthetic report reached `ANALYSIS_DONE`.
- Paired uncertainty resamples per-video within-AUC differences, not individual frames. Reported promotion checks require the declared within gain against matched native, current r6 and explicit-causal, with the declared pooled-loss tolerances on both datasets. `mechanism_supported` remains false; raw overall within and global diagnostics alone do not establish the README's later modality/source mechanism claims.
- The actual outer-model hook counts `9 + 3B` forwards for full paired collection and `12 + 4B` for smoke. With four available local branches, the independent reader measured exactly 28 smoke forwards. Native standalone is `3 + B = 7`; future and causal standalone are `5 + B = 9`, including the original native prefix/global read. The modified-arm timing includes mapping, the second prefill, modified global-question computation, forced original answer and all local reads, plus the shared original prefix/global cost. It does not pretend that diagnostic global computation or fresh prefill is free.

Dense mask/kernel memory and pretrained GPU runtime remain deployment measurements. The declared longest-prefix smoke is the appropriate next check; this review makes no speed or VRAM claim from the CPU fixture.

**Final disposition: PASS; no required code correction.** Proceed with the declared GPU smoke, retain native-restoration exactness and the explicit-causal comparison, then use the complete paired canonical evaluation. No performance or mechanism conclusion was drawn in this review.
