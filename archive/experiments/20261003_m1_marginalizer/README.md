**归档原因：规则4来源机制已用于hateful video detection（MARS、RAMF）；未实现、未跑GPU，不能用概率平均的差异主张新机制。**

# M1 Marginalizer: enumerating the appended verdict

Date 2026-10-03. Status: proposal, before implementation/performance.
Fourth independent candidate in the ongoing M1 iteration. Attributor is still
running; its performance has not been read. This candidate tests a different,
cheaper intervention and may run independently on another free laboratory GPU.
Planned host lab-server / sc448960, live availability required.

## Motivation and distinction from earlier failures

The current reader commits to one generated video Yes/No answer and conditions
every local read on that answer. Both media and the model's interpretation of
those media are therefore in its context. Earlier append/no-append analysis
(`experiments/20261002_verdict_analysis/README.md`) found heterogeneous effects,
not stable gain; the scalar latent-offset model in
`archive/experiments/20261002_revisable_prior/README.md` lost video information.
Neither explicitly evaluated the same local query under both possible appended
answers. Grounder/Selector restricted access to media and hurt localization;
this candidate preserves every media token. It also differs from GLR
(`experiments/20260926_glr/README.md`), which scores likelihood of spoken words
under speaker descriptions and already failed; that proposal is not repeated.
No new GT inspected for this proposal. Earlier result documents are development
analysis, not proof that the new hypothesis will work.

## Fixed computation

Same frozen Qwen3-VL-8B, native BF16, current20frames/repaired ASR/policy,
8-second windows and existing visual/speech queries. Encode full prefix once;
read original global margin z_v. Keep this exact global score downstream.
Construct two independent continuations of the same global question, ending
in the canonical assistant answer Yes or No. Each continuation receives every
local visual/speech query in isolation, as in current code. These are two
interventions on generated dialogue, not extra labels or altered media.

For each existing modality/window obtain margins z_yes and z_no, then
p_bar = (sigmoid(z_yes) + sigmoid(z_no))/2;
z_marg = logit(p_bar).
Compute stably with log-sigmoid/log-sum-exp (no clipping thresholds).
This is uniform marginalization over an untrusted binary dialogue answer,
not a calibrated posterior over the true video label. The uniform distribution
is fixed, not fit to GT and not replaced by the confident global margin.
Global media/context and visual/speech query wording remain available.
The resulting local score is independent of which single answer was selected;
it need not be correct and does not remove errors in global media understanding
or the unchanged final video-level score.

Raw window score max over observed modalities; missing speech omitted exactly
as current. Feed z_marg plus unchanged z_v to unchanged r6 algorithm, separate
unsupervised corpus fit per arm. No ensemble of independent models, no temporal
postprocessor added, no prompt wording scan, no dataset-specific constants.

Hypothesis: averaging opposite verdict conditions reduces local rankings that
are driven by a mistaken or overgeneralized appended answer while retaining
content-sensitive evidence shared across those conditions. It may instead
weaken useful context or simply behave like deleting the answer. Both outcomes
are tested, not explained away. Invariance to answer selection is algebraic;
semantic correctness and localization gain require experiment.

## Sources and novelty question

Marginalizing untrusted conditioning variables and counterfactual invariance
are general ideas, not inventions here. Background:
https://arxiv.org/abs/1703.06856 (counterfactual fairness uses an explicit causal
model, which this reader does NOT provide), and
https://arxiv.org/abs/2203.11171 (self-consistency marginalizes sampled reasoning
paths; this proposal exhaustively enumerates a single binary answer and uses
local conditional probabilities, without generating reasoning paths).
Rule4 independent review must actually search hateful-video detection/localization
for dual-verdict conditioning/marginalization, including self-consistency prior
art. If this source mechanism has already been applied there, STOP. No claims
of causal fairness or novel Bayesian algebra.

## Predeclared experiment and controls

1. Independent proposal and code reviews, once each per repository rules.
2. First two manifest videos per corpus: plumbing only, no GT or pilot AUC.
   Native chosen-answer windows/global margin must match current reader; verify
   two independent caches, exact token seams and no cross-window contamination.
   Extreme-logit analytic probability checks; finite scores, correct 4fps bounds.
3. Complete333-video run, both main corpora on one host. Store Yes/No and a
   no-verdict branch (same prefix, no global question/answer turn). Primary arms
   are native chosen answer and uniform marginalization. No performance read
   until coverage complete. Canonical evaluator only, raw and unchanged r6,
   all three metrics and paired within bootstrap2000, seed0.
4. If no final metric gains .01 anywhere, archive. If any gain, run diagnostic
   r6 arms from stored reads: Yes-only, No-only, no-verdict, mean logits instead
   of mean probabilities, half-video-rotated opposite-answer windows, and base
   shifted per modality by the candidate-minus-base within-video mean.
   Shift control preserves each original modality's order; report raw max order
   too, since modality-wise offsets can change the max. Add a common max-mean
   shift to both modalities as a control preserving raw max window ordering.
5. Promotion: within +.01 on both corpora versus paired/current, no main metric
   drop beyond .005 pooled/.01 within. Claimed novel components must pass14g.
   If a single fixed answer or removing verdict performs equivalently, no claim
   that enumeration caused the improvement; simplify or reject the candidate.
6. Mechanism analysis after scoring: per-window answer sensitivity z_yes-z_no,
   rank changes, correct/wrong global Yes/No, positive-frame coverage, and
   deterministic gain/loss cases with original transcripts/frames. Verify that
   any gain survives shift-only controls and depends on matching both reads to
   the same time interval. Distinguish invariance by construction from improved
   evidence localization. No labels enter scores, fit, routing or constants.

## Inputs and cost

Reuse all existing inputs and shared prefix. Original standalone3+B forwards;
new4+2B: prefix, global question, two answer extensions, twice B local queries.
Full evaluation also adds B no-verdict queries, counted separately. Approximately
2x branch computation and <=2x deployed latency (shared encoding); initial
estimate10-20 GPUminutes for paired333 including no-verdict on5090, subject to
smoke timing. Peak memory close to native reader by processing one verdict
continuation at a time, no backward. Same single model, not independent-model
ensemble. All runs save host/config/log/PID and readable code provenance, no
hashes. Results returned locally before STATUS updates. Development-selected;
paper/Overleaf/currentmethod unchanged until all gates pass.


## Proposal decision

Independent review STOP: `docs/reviews/20261003_m1_marginalizer_proposal.md`.
Primary sources https://arxiv.org/html/2601.15115v1 and
https://arxiv.org/html/2512.02743v1 already use opposite hate/non-hate
assumptions on the same media and combine their results. Exact local probability
averaging was not found, but this aggregation change is insufficient under
rule4. No implementation, GPU experiment, or performance/GT inspection.
