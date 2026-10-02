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
