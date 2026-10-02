# Independent narrow code review: Integrator R2

Date: 2026-10-03. Reviewer: independent agent `/root/m1_grounder_code_review`, same model as the parent session. **PASS after the selfcheck correction below; no remaining observation-affecting bug found.**

Scope is the R2 change in `experiments/20261003_m1_integrator/{integrator.py,measure.py,analyze.py,selfcheck.py,launch/}` and its README declaration. This supplements the R1 implementation review; it does not reopen the method or proposal review. No candidate implementation was edited by the reviewer. No GPU job or benchmark evaluation was launched, and no real GT or R2 performance was read. R1's development motivation was read from the declared README.

## Required correction and confirmation

The first R2 `selfcheck.py` revision stored argparse options in `a` and later overwrote `a` with a suffix model output. The next dtype iteration and final filename selection would fail at `a.future_keys`. The reviewer reported this; the author renamed the suffix outputs to `suffix_original` and `suffix_changed`. The corrected source and completed `selfcheck/future_text_visibility.json` were checked: both FP32 and BF16 entries are present, with `future_keys: text` and passing visibility/restoration checks. This affected the validation script, not the method reader. No further correction was required.

## R2 computation

The new mask is exactly `j <= i OR (visual[i] AND NOT visual[j])`. All native causal edges remain. Additional direct edges go only from a visual query to a later nonvisual key; future image-key edges are no longer opened. Nonvisual query rows and all visual-key columns retain their causal direct edges. The same explicit additive mask reaches all 36 language layers during the prefix pass, then the existing context clears it before ordinary suffix processing.

Expanded-image membership, original inputs, mRoPE, DeepStack, fresh caches, native global/answer, both ordinary local queries, missing-speech handling, 4 fps reconstruction and call accounting retain their reviewed R1 paths. `added_edges` now counts later nonvisual positions for each visual query and is checked against saved token membership in preparation. The existing SDPA behavior still applies: a supplied explicit mask disables the implicit causal flag, so it does not silently remove the new edges.

The README correctly limits the claim to direct edges. Nonvisual keys include ASR, policy, timestamps and scaffolding; their deeper representations can carry image information. This does not isolate ASR or eliminate indirect cross-frame access.

## Independent actual-model verification

Reused and adapted the R1 CPU multimodal witness, with artifacts under:

- `runs/20261003_m1_integrator/independent_review/r2/check_integrator_r2.py`
- `runs/20261003_m1_integrator/independent_review/r2/check_integrator_r2.json`

The fixture uses actual Qwen3-VL with 36 small language layers, GQA, DeepStack and two real image-processor grids expanding to 4 and 6 image tokens. Local versions are Torch 2.7.1 / Transformers 4.57.6. It tests FP32 and BF16 without CUDA.

| Observation | FP32 | BF16 |
|---|---:|---:|
| Native visual cached-value change from a later text-token change | 0 | 0 |
| R2 visual cached-value change from that text-token change | 0.11494116 | 0.07031250 |
| Actual attention-input mask equals the exact R2 graph | Pass | Pass |
| Native versus R2 mRoPE positions | Exact | Exact |
| Independent-copy versus cropped-cache suffix read | Exact | Exact |
| Original inputs, parameters and native restoration | Exact | Exact |
| Original answer/global, query sequence and speech availability | Preserved | Preserved |
| Default R1 versus explicitly requested `all`: hidden states and all cached K/V | Exact | Exact |

The synthetic prefix adds 52 direct edges under R2, matching the recorded count. Hooks are inactive on suffix reads. Executing the real reader orchestration with four local branches measured 28 smoke outer forwards; native standalone remains 7 and modified standalone 9. General counts remain `12 + 4B` smoke, `9 + 3B` full collection and `5 + B` modified deployment. No GPU runtime or memory conclusion is inferred from this small-model test.

## Round isolation and evaluation

The `all` default retains R1. The reader requires `r1_{main,smoke}` for `all` and `r2_{main,smoke}` for `text`; opposite combinations were independently confirmed to fail before model/input access. Resume checks preserve the mask mode. Config and each check record contain the mode; older R1 records/configs default to `all` in preparation.

Synthetic R2 records passed the complete three-arm preparation and report path, including matching native reads, 36-layer mapping, updated edge counts and actual call counts. The coverage gate still requires every two-corpus manifest key in all raw arms and checks; decoded validation still requires the same keys, native global, finite 4 fps curves and equal lengths. No subset selection was introduced.

CLI interception confirmed that `--run-name r2_main` selects only `r2_main`, `r2_main_decoded` and `r2_main_analysis`, while the default selects R1. The two launch scripts propagate the revision consistently and passed shell syntax checking. Raw evaluation still calls the canonical evaluator, and each arm's decoding uses the same reviewed r6 flags and separate output tag. The synthetic report completed without template/schema errors; no real evaluator process was run.

**Disposition: PASS for the declared R2 GPU smoke and subsequent complete paired run.** Preserve the matched explicit-causal comparison and the README's direct-edge claim boundary; this review establishes implementation integrity, not an effectiveness or fusion claim.
