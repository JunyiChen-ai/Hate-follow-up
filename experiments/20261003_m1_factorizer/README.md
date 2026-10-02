# M1 Factorizer: separate temporal evidence during prefix encoding

Declared2026-10-03; sixth proposed candidate of the M1 autonomous iteration.
Proposal/code PASS; five-video GPU checks passed; full three-arm run started
on sc474399 at02:34NZ. Attributor is archived and Eraser remains running.
Development host uoa-lab1/sc474397; compute host selected by live availability.

## Hypothesis and precise change

Grounder and Selector restricted attention only after the full video prefix was
encoded. They did not prevent a window's cached media representation from being
conditioned on earlier media from other windows. Those failed results do not
establish that this mixing causes localization errors. The hypothesis here is
that preserving distinct temporal evidence during encoding, then allowing the
known window question to access context, reduces attribution of a video's topic
to irrelevant windows. This is an empirical hypothesis, not decontamination by
assertion or an upper-bound claim about the frozen model.

Compute the native original global margin and selected Yes/No exactly as current.
For local reading, rebuild the same prefix with a temporal block-causal mask in
every language layer. Input tokens, positions, frames, transcript and questions
are unchanged. Media tokens may read earlier media in their own window and
earlier non-media scaffolding; scaffolding may read only earlier scaffolding.
Thus scaffolding cannot relay another window's media into a media block. All
window-query layers subsequently have ordinary causal access to the entire
factorized prefix. Recompute the global question/selected-answer extension on
this prefix, forcing the original native answer, and use the ordinary isolated
visual/speech window queries. Keep original native z_video downstream.

This moves cross-window integration to the question/answer and window-query
stages. Both modalities in a temporal block may interact subject to the original
causal order; there is no new bidirectional attention. Global context remains
available at local query time. Local decisions are NOT independent of other
windows or of the selected verdict, and the method does not guarantee correction
of a wrong global judgement. It only removes cross-window media mixing from the
cached prefix states used by those queries.

## Exact partition and constants

- Frozen Qwen3-VL-8B native BF16; same20-frame input, repaired Whisper loader,
  policy, questions,8-second windows,4fps and seed0. No prompt changes.
- Use `src/window_token_regions.py` for exact expanded-token membership. Every
  media token receives one group: its earliest containing window. This explicit
  tie rule applies to shared ASR timestamp/boundary tokens and prevents them
  bridging blocks. Unassigned media tokens form one separate context group N.
  This includes out-of-duration ASR; no input is silently removed/reassigned to
  an in-duration window. Record assignment/shared/unassigned counts.
- Non-media scaffolding has group -1. For prefix positions q,k, permit attention
  iff k<=q and either (both are scaffolding) or (q is media and k is scaffolding
  or has q's group). No other prefix media path is allowed. Vision encoder is
  unchanged; its per-image attention boundaries must be verified for Qwen3-VL.
- Prefix masks only, all language layers, no head/depth/threshold sweep. Keep
  native visual/speech missing-branch behavior, raw max and unchanged r6 algorithm
  with an independent unsupervised fit for each arm. No labels/ensemble/smoothing.

## Source and distinction requiring independent review

Source family: independently encoded context blocks with joint query access,
e.g. [Block-Attention for Efficient Prefilling, ICLR2025]
(https://openreview.net/pdf?id=7zNYY1E2fq). That work uses block fine-tuning and
position re-encoding for RAG acceleration. We borrow its attention factorization,
not its trained checkpoint, position scheme or speed claims. Proposed use is
frozen-model temporal-evidence separation, and adds a prefill rather than claiming
free caching. [APE](https://infini-ai-lab.github.io/APE-Page/) is another relevant
parallel-encoding antecedent; its temperature/scaling corrections are not used.
Independent rule4 review must search whether the source mechanism is already
used in hateful video detection/localization, and distinguish it from ordinary
clip-feature extraction. Novelty would be an effective task transfer only.

Independent review PASS: `docs/reviews/20261003_m1_factorizer_proposal.md`.
[SafeWatch/PEPE](https://proceedings.iclr.cc/paper_files/paper/2025/file/beac6bfb7eac3d651307c16ac747df01-Paper-Conference.pdf)
already isolates policy blocks for video safety evaluation; this
is a close application of block attention, with a different isolation target.
CLARA, MultiHateLoc and Few-Shot Hate Videos already use independent clip feature
encoding. Therefore neither block attention in video moderation generally nor
local-then-global video processing is a contribution claim. Only the effective
frozen-MLLM temporal-media KV transfer is being tested. Scaffold isolation is an
implementation constraint ensuring block independence, not a separate novelty.

Local evidence read: archived Grounder/Selector README and reports already
logged in the iteration; current `src/mllm_judge.py` prefix/query construction;
`src/window_token_regions.py` membership. No current Attributor/Eraser performance
or new GT read. Attributor's numerical zero-value diagnostic is not localization
evidence. This candidate's block construction is not a revision of query-depth
or head thresholds from either archived attention experiment.

## Experiment, falsification and cost

1. One independent proposal review, then implementation and one code review.
   CPU small-model checks: native explicit-mask parity; same-position outside-
   group embedding intervention leaves a target group's prefix K/V unchanged
   in all layers; no scaffold bridge, no future access, no shared-token bridge.
   Check original input/model restoration. This checks the implemented graph,
   not whether its semantic interpretation is correct.
2. First two manifest videos per corpus plus the largest observed native prefix
   (HateMM/hate_video_114,5829 tokens): GPU smoke for native full baseline
   parity, token/position alignment, missing branches, memory and timing. No GT.
   The extra case is selected only from `prefix_tokens` metadata in current
   `runs/20260926_glr/base_gridA/predictions.jsonl`, not scores or labels, because
   dense masks may choose a kernel whose memory grows quadratically with P.
3. Full333 native/explicit-causal/Factorizer read; canonical evaluator raw and r6, all3
   metrics, paired-video bootstrap2000 seed0. No partial performance screening.
   Primary localization target: within+.01 both with no pooled drop beyond.005
   or within drop beyond.01 against matched base/current and explicit-causal.
   The explicit-causal arm uses the same dense-mask plumbing as Factorizer but
   retains all ordinary causal prefix edges; it keeps native z_video/answer too.
   Any final metric+.01
   allows up to3 declared revisions; otherwise archive under rule9.
4. If positive: wrong temporal partition with original group sizes (permute
   the token-to-group assignment vector over media-token positions separately
   within modality, with numpy default_rng(0); do not permute group names), all
   media in one group with scaffold isolation retained, and native-prefix late
   query restriction as a reference. Specifically use Grounder late: native
   prefix; final ceil(L/4)=9 query layers see own-window media of both modalities,
   scaffold and causal query tokens, but not other media/global Q/A; earlier
   query layers have ordinary access. The allowed graphs and amount of masking
   differ, so this comparison alone cannot identify a causal effect of stage.
   These controls test alignment and generic context reduction/scaffold effects.
   Include common-shift diagnostic preserving native raw max order. Every
   claimed novel component must pass14g on both corpora; otherwise narrow/remove.
   Record the number/fraction of changed allowed prefix edges: an unchanged
   graph (e.g. a single available group) is not an effective wrong-partition
   intervention. Token permutation can also disrupt within-frame structure;
   therefore its performance alone cannot uniquely prove temporal semantics.
5. Report raw and decoded changes, correct/wrong native verdict, sparse positives,
   branch coverage, gain/loss cases. Matched representation interventions can
   test dependence on out-of-window media, but cannot alone prove GT correctness.

Reuse existing frames/ASR/weights. Standalone base3+B outer forwards, Factorizer
5+B: native prefix+global query, additional factorized prefix+global question+
forced answer, then B available modality windows. Two-arm measurement6+2B;
the full explicit-causal plumbing control adds3+B, for actual9+3B per video.
No backwards; one extra full prefill per new video, not one per window. Prefix
mask is P-by-P shared across heads; peak matrix size at observed P~5800 is about
68MB in BF16, in addition to model/cache and kernel workspace. No speed promise:
masked SDPA may select a slower kernel. Five-video GPU smoke completed on
sc474399: first-two-per-corpus extrapolation gives8.1/12.0 minutes for the
complete three-arm measurement. This small sample is not representative;
budget20-40 GPUmin for both corpora and report actual cost after completion.
All output paths under `runs/20261003_m1_factorizer/`. Development-selected;
current method, paper and Overleaf unchanged until supported promotion.

CPU self-check `selfcheck.py` found implicit/default versus explicit causal
SDPA drift (FP32 hidden8.34e-7, KV1.12e-6; BF16 hidden/KV.03125 on the tiny model).
Holding the math backend fixed gives exact mask parity in both precisions.
All-layer unaffected-group K/V, scaffold independence, native restoration and
ordinary query access pass exactly. These are synthetic numerical checks, not
performance evidence. Source `runs/20261003_m1_factorizer/selfcheck/invariance.json`.
To avoid attributing kernel effects to factorization, the explicit-causal arm is
included in the complete primary experiment before any GPU performance read;
GPU smoke records its drift rather than requiring unjustified bitwise equality.
The original native baseline/restoration must still reproduce exactly.

## GPU plumbing check and full launch, 2026-10-03

Source: `runs/20261003_m1_factorizer/r1_smoke/plumbing_summary.json`, returned
from sc474399 before full launch. Five native global margins and292 native
branch reads exactly match current base_gridA; native restoration, branch
availability,4fps length, finite scores and token-group alignment all pass.
Largest5829-token prefix fits17.919GiB. Explicit-causal window drift reaches
.724, supporting the already declared matched-mask control, not a performance
claim. No GT or performance was inspected. The initial load-only failure is
preserved in `r1_smoke_load_failure`; repository HF_HOME now links existing
model weights, with no change to scoring code. Full command:
`bash experiments/20261003_m1_factorizer/launch/run_lab2.sh main`, detached on
sc474399, outputs `runs/20261003_m1_factorizer/r1_main/`.
Attributor completed negatively after this proposal was fixed and is archived;
Eraser remains running. Neither outcome changed the Factorizer declaration.
