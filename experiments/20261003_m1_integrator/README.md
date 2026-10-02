# M1 Integrator: future-aware visual prefix memory

Declared2026-10-03, candidate10, before implementation or performance. Independent
of visual contrast R2, which is still running. These candidates are not combined.
Development host sc474397; GPU host selected from live availability after review.
All eventual test comparisons are development-selected. Current method unchanged.

## Hypothesis and source

The current prefix puts timestamped frames before transcript and policy. Standard
causal language attention prevents those image-token representations from reading
later transcript words or later images, although the eventual question can attend
to both. This is a directional restriction on the stored representation, not a
claim that native Qwen has no cross-modal reasoning. Local query restriction and
media isolation have failed in earlier candidates. Here the new hypothesis is
that allowing joint context during visual-memory encoding improves the evidence
subsequently read by both original local branches.

Source: Pei et al., [Rethinking Causal Mask Attention for Vision-Language Inference](https://arxiv.org/pdf/2505.18605),
arXiv2505.18605v1 §3.1 Eq7; [author repository](https://github.com/TerryPei/FutureMask)
identifies the ICLR2026 version. The source proposes future-aware masks, including
visual queries accessing subsequent visual/text input while text-query rows keep
their causal mask. We transfer its full mask to the existing hateful-video prefix.
No first invention of future-aware attention or general video use is claimed.
The source's lighter pooling variant is NOT used. No guarantee that its results
on other models/tasks transfer to Qwen or this task. Independent rule4 review must
check existing hateful-video detection/localization applications of this source.
Independent proposal review PASS: `docs/reviews/20261003_m1_integrator_proposal.md`.

Distinct from Grounder: opens encoding edges, does not restrict window-query
attention. Distinct from Factorizer: opens future visual/text paths instead of
removing cross-window/cross-modal history. Distinct from VCD: one altered prefix,
no corrupted images, contrastive margins, branch averaging or additional model.

## Exact R1 computation and constants

Reuse original Qwen3-VL-8B BF16,20 stored frames, repaired ASR, all token strings,
ordering, positions, processor outputs, policy,8s windows and4fps outputs.
All36 language layers and all heads receive the same mask in prefix prefill only.
Let i,j be prefix token positions, V the actual image-token positions obtained
from model.config.image_token_id in the processor-expanded input. Allow an edge
iff j<=i OR i in V. Thus each visual row sees the whole supplied prefix, and
every other row retains its original lower triangle. No padding in single-video
prefill; assert mask coverage, expanded image-token counts and input lengths.
This preserves direct text-row edges, not strict causal independence of all
prefix text states: deeper text states may receive later input indirectly through
visual states. All that information belongs to the already supplied input.
Do not modify vision encoder attention, position ids, embeddings, Q/K/V weights,
or any suffix-query mask. Apply at language self-attention inputs without cache
stitching; re-encode the complete prefix under this mask. Input includes no answer
tokens at this stage. Once encoded, all suffix/global/local computations use the
normal causal implementation and the resulting fresh prefix cache.

Compute native global z_v and its Yes/No answer with ordinary prefill first.
Then re-encode the future-aware prefix, compute its global question margin only
as diagnostic, append the ORIGINAL native answer, and run the original visual
and available-speech window queries. The downstream global remains native z_v.
Do not blend native and modified local reads. This isolates changes to local
reading from a changed native verdict; raw-window means can still change the
downstream video key and must be analyzed rather than assumed fixed.

No new scalar constants, learned parameters, per-corpus routing or random choice.
Native branch outputs are recorded for evaluation but not used by deployment.
Three complete paired arms: native; explicit-causal mask through the same hook;
future mask (main). The explicit-causal arm is a numerical implementation control,
not a candidate selected by performance. Original r6 CLI and independent unlabeled
fits per arm unchanged. No annotations enter any reader, fitting or routing path.

## Cost and validation

Native standalone3+B outer forwards (B=all visual+available speech branches).
Main standalone5+B: native prefix+global question, then modified prefix+global
question+forced native answer+B local questions. The native prefix can be released
before the modified one; no simultaneous full caches. Paired collection costs
3*(3+B), smoke restoration another3+B. Reuse existing frames/ASR/weights, no new
encoder or preprocessing. Dense prefix-mask work may slow prefill and grow O(P²)
mask memory; no claim of zero cost. Estimate main12–20GPUmin full333 on5090;
paired three-arm collection30–45min. Replace with five-video actual smoke cost.

Code review checks actual new edges, all-layer activation, suffix restoration,
exact native/global/answer, causal-mask numerical control, deepstack/RoPE,
timestamps,4fps, calls and no GT dependency. Small actual Qwen FP32/BF16 witness;
five-video deployment plumbing uses first2manifest videos per corpus plus
HMMhate_video_114. Full run then canonical raw and r6 evaluation. Same promotion
gate as iteration entry: within>=.01 both, no pooled loss>.005 or within loss>.01.
Apply this against matched native, current r6 and the explicit-causal control.
The previous Factorizer witness already showed BF16/backend drift between
implicit and explicit causal SDPA; record this drift rather than asserting false
bitwise equivalence. Native and restored-native outputs must remain exact.

If main qualifies, ablate future-text access with source visual-to-visual mask
(allow j<=i OR i,j both visual); also source visual-to-text mask to separate the
two added edge types. Those are mechanism controls, not permission to select a
different best arm silently. Native/explicit-causal remove all new edges.
The source's text type includes ALL nonvisual input tokens (ASR, policy,
timestamps and scaffold); the v2v control removes all future-text edges, not
only speech access. These comparisons can isolate future-input-text access,
not ASR-specific fusion. A later ASR-specific claim needs its own source control.
Report all three final metrics, raw within and modality order, global-diagnostic shifts,
no-frame/single-window effects, native-verdict strata, paired uncertainty and
positive/negative examples. A gain confined to video shifts or decoder artifacts
does not establish improved local evidence. Every novelty component needs the
two-corpus .01 removal criterion. Only after these checks can promotion be claimed.

## Implementation and local witness

Implemented2026-10-03, independent rule6 review PASS. `integrator.py` changes
only prefix language self-attention masks; `measure.py` records native/causal/
future arms and an actual outer-model forward-hook count. No cross-experiment
import; runner scaffolding adapted from the archived Factorizer with its temporal
group mapping removed. `analyze.py` invokes the canonical evaluator and original
r6 entry, without duplicating either metric or decoder code.

CPU check `selfcheck.py` passes actual tiny Qwen language-model FP32/BF16:
changing later text leaves native earlier visual states exact, but changes future
visual states AND their later-layer cached values; suffix hooks are inactive,
native restoration and weights remain exact. Implicit/explicit causal hidden
drift8.34e-7/.03125 disappears under a common math backend. Source:
`runs/20261003_m1_integrator/selfcheck/future_visibility.json`. These are synthetic
implementation checks, not corpus results or full multimodal validation.

Independent actual multimodal witness (36 language layers, heterogeneous image
grids, DeepStack) passes FP32/BF16, including mRoPE parity, native restoration,
ordinary suffix masks and real forward counts. Report:
`docs/reviews/20261003_m1_integrator_code.md`; artifacts:
`runs/20261003_m1_integrator/independent_review/check_integrator.{py,json}`.
No real GT/performance was read. Also inspected lab-server's actual deployment
Qwen attention and SDPA source: a nonempty explicit mask disables the implicit
is_causal flag. GPU numerical parity and full-input cost remain to be checked.

GPU smoke target sc448960 (lab-server), live idle250MiB/0%, after git sync and
complete preflight. Repository HF_HOME points to existing Qwen weights; use
HateVLM torch2.11/transformers5.15.1, not the old HateVideo CUDA build. Commands:
`bash experiments/20261003_m1_integrator/launch/run_lab.sh smoke` then, if passed,
`.../run_lab.sh main`. Outputs `runs/20261003_m1_integrator/r1_{smoke,main}/`.
Pre-existing unrelated home STRAY entries are not task outputs and are left alone.

## GPU smoke passed; proceeding to full333

Host sc448960, artifacts returned to local
`runs/20261003_m1_integrator/r1_smoke/plumbing_summary.json`. All5 original
globals and292 native branches exactly match base_gridA; forced-answer/global
preservation, native restoration, unchanged inputs, expanded visual masks,
actual forward counts and finite4fps curves pass. Largest5829-token prefix
fits17.9192GiB. Explicit-causal branch drift reaches.72398, reinforcing the
matched dense-mask control. Future-read changes are nonzero on all5 inputs;
this confirms activation, not correctness. Smoke wall57.8s includes restoration.

First2/corpus extrapolation: native209.6/286.5s, future292.1/341.0s, paired three
arms560.8/859.9s for HMM/HCS; standalone total10.55min, paired23.68min. Small
sample estimates exclude the stress video (future8.72s, paired23.91s); actual
full cost will replace them. No GT or performance was read in smoke, no constants
changed. Full333 three-arm run follows on the same idle host after sync/preflight.

## R1 complete: retained for one revision, not promoted

Host sc448960, complete333 paired readings returned locally. All333 native
globals and13939 native branches exactly match base_gridA. Canonical results:
`runs/20261003_m1_integrator/r1_main_decoded/<arm>/metrics.json`.

| Corpus | arm | ROC | PR | within | raw within |
|---|---|---:|---:|---:|---:|
| HateMM | base | .897119 | .694235 | .750782 | .680011 |
| HateMM | causal | .896821 | .694265 | .749962 | .683615 |
| HateMM | future | .891985 | .681604 | .766819 | .678186 |
| HateClipSeg | base | .716825 | .671072 | .637349 | .610130 |
| HateClipSeg | causal | .716990 | .671300 | .639037 | .610385 |
| HateClipSeg | future | .709646 | .670246 | .636054 | .599374 |

Within delta+.016037 (84videos, paired95%CI[-.007256,.043421]) / -.001295
(99,[-.017212,.016490]). HMM PR-.012630 and HCS ROC-.007179 fail promotion.
Raw within falls.001825/.010756. Visual-only ordering falls.018899/.005350;
speech-only+.003721/.003458, speech eligible82/97. Actual standalone future
448.89/346.57s, native357.50/293.00s (total1.223x); paired wall32.58min,
peak17.9195GiB. Sources `r1_main_analysis/{summary,cost,branch_diagnostics}.json`.

Post-scoring error analysis read complete raw/decoded predictions, test GT arrays,
and those reports. HMM largest gains are hate_video_279 (+.64) and329 (+.60),
both unchanged raw within;329 has only one read window, so its+.007143
contribution to the corpus mean cannot represent new M1 timing evidence.
These two cases contribute.014762 of the.016037 mean gain. No semantic claim
about their harmful content follows. No new transcript/frame inspection here.
Original explicit-causal control differs little in aggregate, so the large
future effect is not explained solely by the attention-mask backend change.

## R2 declaration: future text keys only

Rule9 permits a revision after HMM within exceeds+.01. Before any R2 performance
is inspected, change ONE design choice: future visual query rows can directly
access future NONVISUAL keys only, not future image keys:
allow(i,j) = j<=i OR (visual(i) AND NOT visual(j)). This is the source v2t
edge type already listed in the initial control plan; R2 explicitly promotes it
to a revised main hypothesis instead of silently selecting a control result.
R1 did not isolate future visual and text edges, and its visual ordering fell.
Hypothesis: retain future textual interpretation while reducing direct access to
other future images. This is tentative, not a causal conclusion from R1 metrics.
Text includes ASR/policy/timestamps/scaffold, and text states themselves can
carry other images; it does NOT isolate speech or remove all cross-frame flow.

All36layers/heads, original inputs, native global/forced answer, both original
window branches, ordinary suffix attention, r6, constants and gates unchanged.
No partial-depth scan, weighted blend or label-conditioned routing. Three paired
arms base/causal/future, with future meaning v2t for R2. Save mask mode in config
and every check; retain R1 files. Counts and expected costs equal R1 (~33min
paired, ~13.3min standalone full333). Five-video plumbing first, then full333.
No v2v control GPU run unless a qualifying main warrants mechanism controls.
CPU/independent code review checks exact removed/retained edges, actual prefix
activation, native restoration, token mapping and canonical report selection.

R2 implementation and independent review PASS:
`docs/reviews/20261003_m1_integrator_r2_code.md`. Actual small multimodal Qwen
FP32/BF16 witness confirms exact v2t edges and changed visual KV from later text,
unchanged native restoration/weights, R1 default regression and R2 report paths.
The CPU selfcheck variable-shadowing failure was fixed before any GPU job.
Artifacts `runs/20261003_m1_integrator/selfcheck/future_text_visibility.json`
and `independent_review/r2/`. Launch `bash .../launch/run_lab.sh smoke r2`,
then `main r2`; analysis `bash .../launch/run_analysis.sh r2`.

R2 GPU smoke PASS on sc448960: returned
`runs/20261003_m1_integrator/r2_smoke/plumbing_summary.json`. Five native globals
and292branches exact; restoration, forced answer/global, v2t mapping, actual
calls and finite4fps pass. Maximum prefix5829, peak17.9192GiB, smoke wall57.8s.
No GT/performance read. Full333 R2 started on sc4489602026-10-03 at05:54NZ
after fresh machine preflight, outputs `runs/20261003_m1_integrator/r2_main/`.

## R2 complete: visual-only gains warrant a branch-specific revision

Complete333 returned from sc448960; native333 globals/13939branches exact.
Canonical `runs/20261003_m1_integrator/r2_main_decoded/<arm>/metrics.json`:

| Corpus | arm | ROC | PR | within | raw within |
|---|---|---:|---:|---:|---:|
| HateMM | base | .897119 | .694235 | .750782 | .680011 |
| HateMM | causal | .896821 | .694265 | .749962 | .683615 |
| HateMM | future | .892960 | .686597 | .774351 | .684553 |
| HateClipSeg | base | .716825 | .671072 | .637349 | .610130 |
| HateClipSeg | causal | .716990 | .671300 | .639037 | .610385 |
| HateClipSeg | future | .708410 | .664196 | .640054 | .604598 |

Within+.023569 (84,95%CI[.000564,.049579]) / +.002705 (99,[-.012687,.018964]);
HMM PR-.007638, HCS ROC-.008415/PR-.006877 fail promotion. Visual-only raw
ordering+.005225/+.017995 (both intervals include0); speech+.003225/+.000915
(82/97 eligible). Max-branch raw+.004541/-.005532. Large HMM279/329 final gains
still sum1.24/84=.014762 with no raw ordering change; no new timing in329.
These reports, complete raw/decoded predictions and test GT arrays were read
post-scoring, not used in scoring/fit. No new media/ASR inspection. Sources:
`r2_main_analysis/{summary,branch_diagnostics,cost}.json`.
Standalone future449.22/346.48s vs native356.98/292.43s, paired32.58min,
peak17.9195GiB. These are measured on sc448960.

## R3 declaration: visual integration with original speech branch

Before any R3 result, declare one branch-specific revision. R2 changes the shared
prefix seen by BOTH branches, although the mechanism targets visual tokens.
Visual raw ordering improves on HCS but combined raw ordering falls; the modified
speech branch offers little ordering gain. This motivates testing whether native
speech avoids collateral changes. This is a hypothesis, not an established cause.
Use exactly R2 future-text visual reads and original native speech reads, with
native numeric global and native answer. Same Qwen model, no independent-model
ensemble, no score blending/calibration, no new constants. Both corpora use the
same rule. Matched control uses R2 explicit-causal visual plus native speech.
Base unchanged. Windows combine by the existing max rule; unchanged r6 follows.

First run this complete333 branch ablation from exact saved R2 reads, no GPU.
It is explicitly a revised main candidate, not a silently selected control. Calls
for a future deployment: native prefix/global/answer3 + modified prefix/global/
forced answer3 + V visual + S speech =6+B (one more than R2). Existing saved
outputs allow exact scoring but do not measure this deployment's runtime.
Measured base+future collection time is only an upper bound, since it includes
unused branches; do not present it as R3 runtime. If the revised main qualifies,
implement/verify standalone branch scheduling and measure cost before promotion.
Same performance gates and mechanistic controls apply; test metrics remain
post-scoring development-selected evidence. No constant scan or branch choice
based on video labels; the branch rule is fixed before this run.

R3 independent code review PASS: `docs/reviews/20261003_m1_integrator_r3_code.md`.
Synthetic complete333 checks confirm exact branch substitution, missing-speech
handling, baseline reconstruction and canonical evaluator/r6 calls. Review found
and fixed missing R2 config/mapping validation, shared8s/ceil4T extent validation,
and cached-vs-manifest duration validation before running. Real GT/performance was
not used by the reviewer. Detached CPU experiment started on sc474397; outputs
`runs/20261003_m1_integrator/r3_main/` and `r3_main_decoded/`.
