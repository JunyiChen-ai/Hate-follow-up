> Archived: no final main metric improved by .01; selective restriction harmed within-video ordering on both corpora.

# M1 Selector: window-selective attention heads

Date: 2026-10-02. Status: full paired experiment completed; archived under rule 9.
Independent review: `docs/reviews/20261002_m1_selector_proposal.md`.
Close general antecedents include Gaze Heads and IGAR; this is a target-task
transfer and online label-free routing test, not the invention of attention steering.
Host planned: uoa-lab2 / sc474399 (recheck availability before launch).
Scope: next candidate in `experiments/20261002_m1_iteration/README.md`.
Frozen Qwen3-VL-8B; same inputs, questions, global verdict, isolated branches,
window grid and r6 algorithm. No labels in selection, fitting or scoring.

## Evidence motivating this candidate

Read `runs/20261002_m1_grounder/r1_full_analysis/{summary,window_rank_diagnostics,read_diagnostics}.json`
and all raw/decoded arm `metrics.json`, plus the 4-fps test labels only in the
post-scoring analysis. Grounder's last-quarter restriction had no main-metric
gain of .01. Correct versus wrong local support changed logits very little,
despite almost disjoint support. Restricting every query layer substantially
hurt both corpora. These observations do NOT prove where a model stores context.
They motivate testing selective intervention at the heads actually retrieving
local evidence, while keeping other heads and the global context accessible.
No layer-depth scan or new prompt variant is proposed.

## Fixed mechanism

For each window branch, at every language attention layer:

1. Compute the current layer's pre-restriction scaled query/key affinities from the last query token
   to cached prefix media tokens (frames and transcript). No second model call.
2. A head is selected when its mean attention per token on this window's media
   exceeds its mean attention per token on other-window media. Compare these
   means using log-sum-exp affinities minus log token counts; the attention
   denominator cancels. The comparison threshold is equality, not a fitted value.
   Later-layer affinities already reflect earlier-layer routing; they are not
   counterfactual affinities from an entirely unmodified base forward.
3. In selected heads, block direct access to other-window media for that branch's
   query tokens. Keep its own causal query tokens, policy/scaffolding, and global
   question/answer context. Unselected heads retain the original full context.
4. Use the same SDPA attention computation and ordinary final Yes/No log odds.
   No score subtraction, label-based calibration, ensemble or extra encoder.

If local or other-window media is empty, do not select any head. Apply the same
rule to both visual and speech questions; both local modalities are included
in temporal support. The existing prompt still specifies which modality to judge.
Prefix encoding/global query/answer extension are untouched. Contextualized K/V
and earlier query states remain: this is selective direct access, not complete
information isolation or a solution to all wrong-verdict propagation.
Routing conditions on the whole observed question: its last position chooses
the head mask applied to all its positions. Attention edges to future query
values remain blocked, but the routing choice itself uses the complete query.
Do not claim strict autoregressive independence of earlier query positions.
The entire question is known input for one Yes/No read, not a generated sequence.

Implementation should obtain Q/K after native position encoding/cache update,
compute only one query row for selection, and perform one native SDPA call for
the full query. It must not materialize full attention probabilities merely to
choose heads. The ordinary causal mask must remain active in every head/arm.
Do not change weights, precision, input encoding, prompts or downstream fitting.

Constants: seed 0; 20 existing frames; repaired Whisper ASR; existing 8-second
grid; original policy and shared `yesno_question`; all text layers/all heads are
eligible under the same content-dependent rule. No threshold/depth/head scan.

## Controls and decision sequence (declared before performance)

First, two videos per corpus for input/mask/cache/numerical/cost checks only.
Then complete both corpora for paired `base` and `select` arms. Base uses the
same attention wrapper and per-head mask shape without removing any key.
Compare its logits to the original reader, and report any numerical drift.

If the full candidate has no .01 main-metric gain anywhere, archive per rule 9.
If it shows a qualifying gain, continue the following prespecified controls:

- `all_heads`: restrict other-window media in all heads, retaining global Q/A.
  This differs from Grounder's all-local arm, which also removed global Q/A.
- `permuted_heads`: replay the select arm's head-selection trajectory, rotated
  by half the head count at each layer. Same number of restricted heads per
  layer/window; tests the identity of selected heads.
- `shifted_support`: replay the select arm's selected heads, but use the
  half-rotated, token-count-matched media mask from Grounder. Retain the original
  query, its copied speech, and global context. It is a wrong-prefix-support
  control, not a full replacement of all local information.
- CPU `shift_only`: reconstruct base plus each video's select-minus-base mean
  max-logit shift. Preserve raw ordering, then refit the same r6 algorithm.

All control predictions remain separate; no arm averaging. The replay controls
are experimental interventions using the select trajectory, not deployment.
For any mechanism claim, select must beat base within by >=.01 on both corpora,
with no drop beyond .005 pooled/.01 within, and beat each claimed-component
control by >=.01 on a common main metric on both corpora (rule 14g). Report raw
ordering and shift-only comparison, video-paired bootstrap, positive/far-negative
rank changes, frame availability, selection frequency by layer and failures.
If routing identity has no effect, do not call head selection the explanation.
Evidence remains development-selected; all failed arms and later revisions kept.
Record the selected-versus-rotated head-set overlap and the fractions of
layer/window decisions selecting every head or no head. A highly overlapping
permutation is not a strong head-identity intervention. Added on the independent
reviewer's recommendation before implementation or outcome inspection.

## Cost and reuse

Reuse the same existing frames, ASR, model and prefix cache per video. A deployed
select arm has exactly the current 3 + number-of-observed-modality-window-branches
model forwards. Added work is one QK row and a head-specific mask per text layer;
no repeated per-window MLLM calls. Grounder measured about 9.2 GPU minutes for
one complete 333-video arm on this 5090. Estimate 10–16 GPU minutes for select,
subject to actual mask-kernel overhead; the initial paired run about 20–30 minutes.
Measure before full launch. If overhead exceeds 25%, report and first remove
avoidable full-matrix selection work; do not hide the residual cost.

One complete corpus always runs on one machine. All results return locally before
STATUS updates. Use the canonical evaluator and current r6 CLI; do not copy metric
logic. No paper changes until an improved, explained mechanism is supported.

## Implementation and plumbing checks

Independent code review: `docs/reviews/20261002_m1_selector_code.md`.
Shared token mapping was promoted without a functional change from the archived
Grounder implementation to `src/window_token_regions.py`; AST comparison of both
functions matched exactly. Existing lab2 HateVLM environment reused unchanged.

GPU smoke (`runs/20261002_m1_selector/r1_smoke`, sc474399): all five arms on four
videos. Every crop/copy comparison and every window's wrapper-base versus ordinary
SDPA margin is exactly equal. Select branch time is 1.151 times ordinary SDPA;
including prefix computation, estimated standalone time is 1.101 times ordinary.
These four videos are plumbing/cost checks, not an outcome-selected pilot. No
GT or performance metric was inspected. Independent CPU checks also validate
GQA grouping, log-mean-affinity equivalence, exact trajectory replay/rotation,
branch-order/cache invariance and edge-causal masking under fixed routing.

Run `launch/run_lab2.sh main` detached for the two full arms; immediately rsync
`runs/20261002_m1_selector/r1_main/` back, then run `launch/run_analysis.sh` locally.
It first checks full paired/current-reader alignment, evaluates both raw arms and
the unchanged r6 runs, then reports video-paired bootstrap, head selection and
cost. Initial report always leaves mechanism_supported=false until the declared
controls have actually run and passed. Analysis GT never enters GPU reading or fit.

## Full results and decision (2026-10-02)

Host sc474399 / uoa-lab2, all 333 videos, paired reader wall time 1054.2 seconds.
Outputs immediately returned to local `runs/20261002_m1_selector/r1_main/`.
Analysis completed locally at `runs/20261002_m1_selector/r1_main_analysis/`.
All original global and window margins are reproduced exactly for every video.

Official final sources: `runs/20261002_m1_selector/r1_main_decoded/{base,select}/metrics.json`.
Order: pooled ROC / pooled PR / within macro ROC, development-selected.

| Corpus | Base | Selector |
|---|---|---|
| HateMM | .897119 / .694235 / .750782 | .897708 / .691099 / .709021 |
| HateClipSeg | .716825 / .671072 / .637349 | .713008 / .667350 / .622332 |

Paired within differences: HateMM −.041761 (84 mixed videos, 95% bootstrap
[−.075893, −.010154]); HCS −.015018 (99, [−.043671, +.014567]).
Raw within also falls: .680011 → .653309 and .610130 → .580374;
its paired intervals exclude zero on both corpora. Head selection was active:
layerwise selected-head fractions range .133–.580 / .133–.604, all-head
selection .038 / .036 and no-head selection .127 / .121. This is not a no-op.
Attention affinity did not identify a restriction that improved localization.
These observations do not establish that attention heads lack useful information.

Standalone prefix-plus-branch wall estimates: base 319.10 / 258.96 seconds,
select 336.15 / 274.32 seconds (HateMM / HCS); mean deployed forwards
36.52 / 60.05 per video, exactly the same for both arms. Additional per-window
model calls: zero. Timing excludes trajectory serialization and paired verification.
Sources: analysis `cost.json`, `head_selection.json`, `alignment.json` and `summary.json`.

No main metric gains .01, so archive immediately under rule 9. The full
all-head/permuted-head/shifted-support and shift-only controls were not run;
only the four-video plumbing smoke exercised the three GPU control arms.
There is no positive mechanism claim. Current method and paper remain unchanged.

Test-read log: inspected raw and decoded official metrics, analysis summary,
paired per-video outcomes, routing frequencies and costs after predictions were
fixed. Analysis used `data/gt_4fps/{HateMM,HateClipSeg}.npz`; GT never entered
reader/routing/unsupervised r6 fitting. Together with Grounder, the failure
motivates testing evidence attribution while retaining joint context, rather
than another head/depth threshold for removing context.
