> Archived 2026-10-02: no qualifying gain; shared offset absorbs discriminative video signal and severely degrades pooled performance. Current r6 and paper unchanged. Historical run configs retain original source paths; code now resides here.

# Revisable global prior with a shared-context random effect

Declared 2026-10-02, before candidate implementation/results. Host: uoa-lab1 / sc474397.
User authorized implementing and testing a mechanism in which global judgement is
a soft prior and local evidence can overturn an incorrect verdict. Scope is this
specific redesign, not an open-ended search or a paper edit.

## Problem and prior failures

The global Yes/No dialogue turn conditions every local query. Paired analysis in
`experiments/20261002_verdict_analysis/README.md` found both useful separation and
large regressions; current r6 has no stable average within gain from that turn.
Both conditions already see the full frames and transcript.

HVL (`experiments/20260911_hvl/README.md` §8) added generated hypotheses and a
second verdict: sequential reads collapsed to continuation; revision restated
the first answer. Two-level round 1 (`experiments/20260926_twolevel/README.md`
§10.2) used a global latent variable but overcounted correlated local reads.
Hard within-video centering also lost information. ICC tempering and a mixture
of video summaries did not preserve pooled PR. These are not new proposals.

## Candidate: model the common context bias, rather than count it repeatedly

Use the **nostance** Reader: full original multimodal context, independent visual
and speech queries, no appended global Q/A. Retain the separately measured global
log odds as a noisy observation. No generated critique, extra reader or ensemble.

For each video introduce a single shared random effect b, representing a common
offset in global and window readouts induced by their shared context. A latent
binary video regime V has prior pi. Conditional on V=0, both temporal chains are
off. Conditional on V=1, use the same explicit-duration visual/speech chains and
duration integration as r6; all-zero paths remain possible, as in r6 (V therefore
denotes the active regime, not literally the OR of all states).

After corpus normal-score transforms of global and each modality's local reads:

- global read x ~ Normal(a_V + b, sigma_global^2);
- local read y_wm ~ Normal(mu_m,H_wm + b, sigma_m^2);
- b = tau * u, with fixed symmetric Gauss-Hermite nodes/weights approximating u ~ N(0,1).
- H_wm is the OR of the one/two 4-second cells covered by that observation window.

Fit pi, a, mu, observation variances, tau and chain start probabilities jointly
from unlabelled reads by EM. Integrate b and V; never substitute the selected
Yes/No answer. Local observations can overturn a weak/incorrect global read.
Because the global observation informs b, it can change conditional local
inference, not merely multiply all frame probabilities by one constant.
The shared effect models cross-window dependence; it is not a fitted post-hoc
temperature or a subtraction of each video's empirical mean.

Implementation: exact forward-backward and duration averaging conditional on
each (V,u), then posterior averaging. Gaussian means and tau use weighted least
squares in the M-step; variances and mixture/start probabilities use expected
sufficient statistics. Fit all test videos of a corpus with no GT access.

Output: posterior video-regime log odds plus centered rank of conditional
P(h_t=1 | V=1, all reads), preserving the established video-plus-local-rank form.
Save unconditional frame probabilities separately for diagnostics. No thresholds
or added output calibration.

## Inputs, cost, and limits

First test uses the matched old Reader family:
`runs/20260910_spvl/mllm/q3vl-8b/{full,nostance}` (333 videos).
Baselines: current r6 on both, already available under
`runs/20261002_verdict_analysis/main_{full,nostance}`.
DeHate nostance/current caches can be used for external follow-up if main evidence
warrants it. These are not the latest main-table Reader caches.

New MLLM calls: zero for the cached pilot. Deployment replaces the existing
stance read process with nostance; same 20 frames, transcript and two local
branches. New cost is CPU quadrature (initial estimate 10–30 minutes for both
main corpora, measured before any larger follow-up). Latest main-family nostance
re-extraction is only warranted if cached pilot passes; estimate its GPU cost
from existing reader timings first. No GT fits, thresholds or routing.

## Constants and first-round arms

- seed 0; 4 fps evaluator; same 4-second cells and 8-second source windows;
- same minimum segment shape 4 (= two read windows), six duration grid points
  from two windows to video length, prior uniform in length;
- 7 Gaussian quadrature nodes initially; check 15 nodes on frozen fitted params
  regardless of metric outcome; if substantial discrepancy occurs, treat as
  numerical insufficiency and refit at 15 (then 31 if still insufficient), not a
  performance-driven selection. Numerical criterion: median absolute change in
  video posterior > .01 or median conditional frame posterior change > .01;
- EM max 100 iterations, relative log-likelihood convergence 1e-6; variance floor
  1e-6 and probability floor 1e-9 are numerical safeguards;
- initialization: global/local class means at 10th/90th percentiles, pi=.5,
  chain start probabilities .5, tau=.5 in normal-score units;
- no parameter sweep; constants and procedure identical for both datasets.
- Ordered global and local means initialize the semantic orientation. If an
  active emission mean ends below its background mean, do not silently relabel
  or clip it: record a semantic identification failure and report the fit as such.
  A negative fitted tau may be reflected to positive, since the fixed u prior is
  symmetric and this is exactly the same marginal model.

Rule-4 independent proposal review: PASS, `docs/reviews/20261002_revisable_prior_proposal.md`.
The shared offset dependence is the change; soft global variables already existed
in r1. Conditional modality OR must precede u integration; EM includes joint
cross-moments and counts the global observation once per video.

Rule-6 independent code review: PASS, `docs/reviews/20261002_revisable_prior_code.md`.
Independent path enumeration and scalar/batch parity are in
`runs/20261002_revisable_prior/checks/checks.json` (errors below 1.2e-15).
The missing-speech behavior is inherited from current r6: absent observations
have no emission likelihood, but do not force the speech latent chain to zero.
This affects 14 HateMM / 3 HCS videos and is held the same across candidate arms.

## Execution

Environment reused: HateVideo, Python 3.11.8, NumPy 1.26.4, SciPy 1.17.1; seeded
CPU linear-system witness passed on sc474397. No environment rebuilt, no GPU
allocated. Repository rules override the skill's hash-ledger recipe.
Machine check found remote checkouts at different versions and unrelated files
in other projects/home directories; no remote execution or synchronization is
used for this local CPU task, and those files are untouched.

Launch: `bash archive/experiments/20261002_revisable_prior/launch/run_cpu.sh`, detached via
setsid/nohup, logs and PID under `runs/20261002_revisable_prior/`.
Three arms are sequential, each containing both complete corpora. A separate
reference rerun checks that promoting chain kernels to `src/temporal.py` did not
change the existing r6 outputs (`launch/run_reference.sh`).
Analysis after runs: HateVideo Python `archive/experiments/20261002_revisable_prior/analyze.py`.

Numerical note (before interpreting any candidate metrics): the first HateMM
7-node fit converged but 15-node evaluation changed median conditional frame
probability by .0109 and some videos much more. Thus 7 nodes are insufficient.
Continue the same fit at 15, 31, 63, then 127 nodes if needed, initialized from
the previous fit; no selection by metrics. In addition to the declared median
criterion, require the 95th-percentile video/posterior change <= .05 so a stable
median cannot hide a substantial inaccurate subgroup. If a fit reaches its
iteration limit without convergence, continue to at most 300 iterations at that
resolution. These are numerical repairs, not new mechanism rounds. If 127 nodes
are still insufficient, report a numerical limitation rather than an idea result.

Three candidate arms: full random effect; tau fixed zero (old independence
assumption); omit global likelihood (local evidence alone). The controls share
the candidate's data transforms and fitting protocol. Reference r6 is unchanged.

Numerical repair 2 (before interpreting metrics): GH15 remained inaccurate
(HateMM posterior p95 changes .239/.147). GH31 continuation was stopped and its
partial outputs retained. Replace fixed GH with bounded adaptive trapezoid
integration of the **same continuous Gaussian random-effect model**. Conditional
on any temporal path, u has variance s² = 1/(1 + tau² sum_i 1/var_i), independent
of the path. Its mean lies between bounds obtained by using the smallest/largest
class mean for each observation. Integrate the union of these mean bounds ±8s
with spacing at most s; use log(step)+log N(u;0,1), half endpoint weights, **no
renormalization**. Check frozen parameters at half spacing. Warm starts: full
from GH15, no_global from GH7; both complete corpora, max 300 iterations. The
independent arm has no integration and is reused. This is solely numerical
repair, with no metric-driven parameter selection. `launch/run_adaptive.sh`.
Sparse Numba forward/backward (cache disabled) accelerates the same recursion;
NumPy batch and original scalar implementations remain as parity references.

Numerical repair 3, declared before reading performance: continuous ordinary EM
was still improving slowly after 89 iterations on HateMM (LL increment .516;
4 seconds/iteration). Use parameter-expanded EM for the same marginal model.
The E-step adds per-video E[u] and E[u²], counting each video once. Maximize an
expanded Normal(center,spread) prior, with center=mean E[u] and
spread=mean E[u²]−center²; after the existing joint regression/residual-variance
M-step, reduce to N(0,1) by adding signed_tau*center to all active means and
setting tau=abs(signed_tau)*sqrt(spread). This leaves observation variances and
pi/start probabilities unchanged. Apply only with continuous integration; keep
the same LL convergence and monotonicity checks. Full/no_global use this solver
and the original GH15/GH7 warm starts; no metric selection. The partial ordinary
EM run is retained as numerical development evidence. `--px` selects this solver.

Independent numerical follow-up: PASS (same code-review record, no new general
review). Sparse-vs-NumPy error <=2.85e-14; adaptive continuous marginals against
96 explicit-path analytic integrations <=1.20e-9; joint sufficient-stat error
<=4.23e-7. Six real-video frozen fits include a 250-cell sequence. Sixteen
expanded/reduced-prior tests agree to 3.55e-15; full/no_global 12-step PX-EM
sequences are monotone and the independent arm is unchanged. Reproducible checks
and outputs are under `runs/20261002_revisable_prior/review/`.

Execution-only change: after numerical checks, the no-global ablation was launched
on a second local CPU core while full-model fitting continued. The sequential
parent dispatcher was stopped without stopping its full-model child. Each arm
still processes both complete corpora on sc474397; no corpus splitting. This
reduces elapsed development time and changes no fitted quantities or constants.
No-global launch: `launch/run_no_global_px.sh`. Final selection is based only on
the completed fits' convergence, ordered means and integration diagnostics.

## Decision and mechanism checks

Report pooled ROC, pooled PR, within (84 HateMM / 99 HCS); use paired video
bootstrap for within differences. Relative to matched Append+r6, require no drop beyond .005 pooled/
.01 within and at least .01 improvement in the same metric on both datasets for
promotion. This is the repo's gate, not a guarantee of success. Novelty claims
additionally require the declared component ablation gate in rule 14g.

User's mechanism claim is checked on original global errors and correct global
predictions separately (GT only in diagnostics), plus injected contradictory
global observations with local reads fixed. Brute-force small-state enumeration
checks exact marginals and observation-time mapping. Independence arm must
recover the same model with tau=0. Fit code must never read GT.

All result-driven decisions are development-selected. Test-read log for design:
read prior analysis report and subgroup/case findings, prior two-level and HVL
READMEs; common read shifts motivated a shared latent effect. No new GT inspected
for this candidate before declaration. Unsuccessful redesign will be reported
and archived; current method/paper will not change on weak evidence.

## Completed full-model finding: reject this candidate

The full continuous fit converged with ordered means on both corpora; the largest
posterior change on halving quadrature spacing was below 1.7e-9. Numerical
implementation is therefore not the explanation for the observed regression.
No principal metric gains .01 against matched Append+r6, so rule 9 calls for
archiving this mechanism. This bounded user task does not authorize an unrelated
open-ended candidate search. No paper, Overleaf or current-method replacement.

Mechanism intervention (`r1_full_px/diagnostics/global_interventions.json`):
holding all local reads and fitted parameters fixed and replacing the global
observation with its 10th/90th normal-score quantiles changes conditional local
posteriors in all 333 videos. The median per-video maximum change is .124 on
HateMM and .072 on HCS; the independent arm changes exactly zero. Thus the global
observation does enter local inference. Nevertheless, median local-rank Spearman
is .994/.992, so much of this response is probability change rather than useful
reordering. Prior-to-posterior regime signs change in 29/215 and 13/118 videos;
these are **not counts of corrected original Yes/No errors**, because V denotes
an active temporal regime and the model prior is not the raw MLLM verdict.

Failure diagnostic (`analysis/failure_diagnostics.json`): with one presence label
per video and the canonical evaluator, raw global observation video ROC is
.933/.806; revised regime odds are only .637/.625. The inferred common offset
alone has ROC .872/.814 and Spearman .852/.698 with the global observation.
The model allocates 80%/68% of global conditional emission variance to the
shared effect (visual 95%/87%, speech 80%/64%). This supports a specific failure:
the nuisance term absorbs discriminative video-level signal as well as possible
context bias. An all-off path with a high common offset can explain sustained
high reads, so free revision can destroy a correct global ranking. This is an
empirical failure of this decomposition, not proof that every revisable prior
must fail or that appending a verdict is necessary.

Post-fit GT-read log: read `data/gt_4fps/{HateMM,HateClipSeg}.npz` test entries,
official final predictions/metrics, and global interventions. The only resulting
decision is rejection/archiving; no label or diagnostic changed fits, thresholds,
output weights or global/local routing. We did not rescue the model by restoring
the old video score while advertising its discarded revised score.

### Final metrics and controls

All values below are pooled frame ROC / pooled frame PR / within-video macro ROC.
Within uses 84 HateMM and 99 HateClipSeg mixed videos; total coverage 215/118.
Development-selected, matched older Reader family, same official 4-fps evaluator.

| Method | HateMM | HateClipSeg |
|---|---|---|
| Append + current r6 | 0.894242 / 0.691086 / 0.763920 | 0.715843 / 0.671566 / 0.637100 |
| No append + current r6 | 0.896761 / 0.696719 / 0.765482 | 0.705523 / 0.658747 / 0.631225 |
| Shared-effect revisable prior | 0.569308 / 0.360292 / 0.713359 | 0.513904 / 0.489523 / 0.632432 |
| Same model, no shared effect | 0.843232 / 0.588602 / 0.760175 | 0.721597 / 0.658269 / 0.631490 |
| Same model, no global observation | 0.571902 / 0.361639 / 0.704287 | 0.514209 / 0.490220 / 0.633764 |

Authoritative metric sources (each table method, in order):
- `runs/20261002_verdict_analysis/main_full/metrics.json`;
- `runs/20261002_verdict_analysis/main_nostance/metrics.json`;
- `runs/20261002_revisable_prior/r1_full_px/metrics.json`;
- `runs/20261002_revisable_prior/r1_independent/metrics.json`;
- `runs/20261002_revisable_prior/r1_no_global_px/metrics.json`.

The shared-effect component itself damages HateMM within by .0468 against its
independence control, and sharply damages both pooled metrics on both corpora.
Adding the global observation to the shared-effect model changes within by
+.0091/−.0013; it does not pass the declared component gate. Its weak gain cannot
justify the shared-effect failure. Independence alone also falls short of current
r6; it is a control, not an alternative proposal selected after seeing outcomes.

Against Append+r6, within difference is −.0506 (95% paired video bootstrap
[−.1026,−.0026]) on HateMM and −.0047 [−.0322,+.0217] on HCS. Source:
`runs/20261002_revisable_prior/analysis/summary.json`, 10,000 paired resamples.

Error groups and examples (exploratory, selected after seeing per-video scores):
- Original wrong-No mixed videos: mean within difference +.0765 on HateMM
  (n=3; only 1 improves, 2 worsen), −.0571 on HCS (n=5). Neither supports reliable
  correction; both bootstrap intervals include zero.
- `hate_video_279` recovers .125→.765, but no-global also gives .765 and no-append
  r6 already gave .765. This recovery is not evidence for the new global mechanism.
- Correct-Yes `hate_video_25` collapses .946→.002; `hate_video_138` .996→.128.
  Correct original video judgement does not protect local ordering under this model.
- HCS wrong-No `yt_8sIY4W1hBQM` drops .895→.393.
These are numerical cases, not new claims about uninspected semantic content.
Per-video results: `runs/20261002_revisable_prior/analysis/per_video.csv`.

### Cost, reproducibility and disposition

Zero additional MLLM calls, zero GPU time; all old reads are reused. The final
PX full fit (including final prediction, precision check and evaluation) took
339 seconds on both corpora; no-global took 464 seconds. These are **warm-started** timings, not from-scratch deployment estimates:
full depends on GH7→GH15 fits; no-global depends on its GH7 fit. Several earlier
numerical runs were discarded or stopped. Complete logged timings and completion
status are in `runs/20261002_revisable_prior/analysis/costs.json`; those development
costs should not be hidden behind the final solver time. A new video still needs
the normal MLLM reading process; using cached reads here does not remove that cost.

Final reproduction order after obtaining the declared runs: `select_numerics.py`,
`analyze.py`, `diagnose.py --tag r1_full_px`, `diagnose.py --tag r1_independent`,
`explain_failure.py`. `numerical_selection.json` records resolution selection with
no metric reads. Frozen-code review and proposal review are in `docs/reviews/`.
The shared scalar kernel extraction reproduces all 333 original curves exactly.

Archived as a negative result under rule 9: no full-model gain >=.01, pronounced
pooled regression, no supported shared-effect contribution. Current r6 and paper
remain unchanged. This experiment establishes that this particular latent-offset
redesign is unsuitable; it does not establish that a useful revision mechanism
cannot be built. Any future design needs additional evidence that distinguishes
context bias from sustained genuine content, rather than treating shared strength
itself as nuisance. No further candidate, GPU extraction, or paper edit is launched.
