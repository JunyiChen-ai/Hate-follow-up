**归档原因：完整333无任一最终主指标提升.01；HateMM PR明显下降，原始联合排序两语料下降。**

# M1 Reinforcer: query-specific visual residual steering

Candidate13, declared2026-10-03 while Recycler R1 runs and Integrator R4 is in
code review. No Recycler/R4 performance had been inspected at declaration.
Proposal review PASS; implementation and independent code checks precede GPU. Independent candidate, not combined with
Integrator/Recycler. Same frozen Qwen3-VL-8B, original r6 and canonical4fps.
All eventual results/development decisions are development-selected.

## Source and hypothesis

VISTA, *The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language
Models via Visual Information Steering*, arXiv2502.03628v2 / ICML2025:
https://arxiv.org/html/2502.03628v2, sections2.3/2.4/3.1.
VSV takes the difference between layerwise last-token residuals for contexts with
and without images, adds the resulting direction during inference, and preserves
residual norm. SLA combines final logits with averaged preceding-layer logits.
Paper defaults are lambda.17 for LLaVA, gamma.3 and five preceding layers. We
transfer this paper-defined mechanism, not claim source novelty. Its analysis
of visual-information loss during long generation does not itself establish a
failure in our one-token binary reading. A separate experiment is required.

Hypothesis: a visual query's image-dependent residual direction can reinforce
its own visual evidence before the final decision; intermediate semantic readout
may recover useful evidence missed by the final Yes/No projection. This differs
from PAI's uniform image-attention boost and final-logit reference subtraction;
no attention mask/weight edit. It differs from DoLA's contrast against a selected
early layer; SLA uses a fixed preceding-layer mean with positive weight.

Official code inspected:
https://github.com/LzVv123456/VISTA and its `llm_layers.py`, `steering_vector.py`,
`llava/model/language_model/llava_llama.py`, `pope_eval.py`, `run_pope.sh`.
It is NOT identical to paper equations: it wraps MLP outputs rather than the
post-block residual; unit-normalizes directions and adds cosine-dependent gain;
POPE uses lambda.01; one-pair direction goes through PCA; layer range25..30 is
inclusive (six layers), and intermediate head projections omit the final norm.
This candidate explicitly chooses PAPER residual equations/defaults instead of
mixing those variants. The Qwen3/local-query scope below is our adaptation.
Proposal review should flag any incorrectly described source and actually search
for prior use of VISTA/VSV in hateful-video detection/localization.

## Exact R1 transfer and fixed adaptations

Retain original20frames, full ASR, policy, prompts,8s windows,4fps grid, native
whole-video numeric read and its original forced Yes/No answer. Speech queries
remain exactly native. Only visual-window queries receive the intervention.

For each visual window question, save the LAST query token's output residual
from every language decoder block during its ordinary native read. Independently
construct `Judge.prefix_messages(..., with_frames=False)` with original full ASR
and policy, its declared no-frame wording, no images/frame timestamps; encode it,
ask original global question, force the ORIGINAL native answer, then ask exactly
the same visual-window query tokens. Save its last-token block-output residuals.
The per-layer direction is native minus no-image residual (FP32 subtraction).
It is query-specific and image-dependent, but also includes effects of removing
image tokens/timestamps and changing their surrounding introduction. It is not
a pure semantic image-causal effect. Reference global score is diagnostic only.

Replay that visual question against the original cached visual prefix/global QA.
At EVERY decoder block output, for every suffix-query token residual h, form
u=h+.17*d[layer], then replace h by u*||h||2/||u||2. Compute update and norms in
FP32, cast back to original BF16. If ||u||=0, leave h unchanged; if h=0 and u is
nonzero the preserved norm makes the output zero. All36layers, all suffix rows,
no intervention on cached prefix/global QA, no extra generated words. Suffix cache
is cropped after each query, and captured source directions always come from
unsteered passes. Save/restore each context's Qwen mRoPE state when switching
from the no-image cache back to the native visual cache. Unlike a full VISTA prefill replay, this freezes prefix/global
representations to retain original branches and reusable computation.

SLA reads last-token residuals AFTER each block's steering. For final layer35,
use the ordinary final RMSNorm+LM head. For block outputs30..34, apply that SAME
final RMSNorm+LM head separately and average token logits; combine .7 final+.3
mean. Aggregation happens before the existing Yes/No token-set logsumexp; compute
only the existing12 answer-token columns because that produces the exact binary
margin of full-vocabulary logits. These norm/indices choices explicitly define
the Qwen transfer; do not call it a bitwise copy of the author's implementation.
No APC, thresholds, learned parameters, temperature/constant scan or GT routing.

Two complete main arms: native and full VSV+SLA. Hooks during native capture are
read-only; verify exact historical native reads. Keep native intermediate and
steered final/intermediate answer-token logits so the declared component removals
require no new GPU calls. Do NOT evaluate/select component variants until the
main result warrants them. Full is the declared main, not the best of four arms.

## Cost and falsification

V visual windows, S available speech, B=V+S. Native3+B. Full deployment and paired
native/full collection both need6+B+2V: native prefix/global/answer and all native
branches, no-image prefix/global/native-answer plus V reference questions, and
V steered visual replays. Smoke restoration adds3+B. Original positive reads are
necessary to derive directions, not free cached data on a new video. Reuse media,
ASR/weights; no new preprocessing or dataset. Initial complete333 estimate20–35
GPUmin on5090, peak expected<32GiB with prefills executed sequentially, both unmodified KV caches retained simultaneously on GPU until
reference reads finish (then delete reference; no native rebuild or CPU offload), and36x4096 FP32 directions/window on CPU; replace with actual five-video smoke.
Source efficiency claims for long generation do not cover our many local queries.

Required independent code check: source norm algebra, all36layers, post-block
hook placement vs Qwen DeepStack, exact prefix/global/speech restoration, per-query
direction/cache isolation, no-image condition and identical question tokens,
SLA indices and normalized projections, token-first margin equivalence, actual
forward counts,4fps/complete333 alignment and canonical evaluator/r6 plumbing.
Small actual multimodal Qwen FP32/BF16 plus real first2/corpus and HMM114 smoke;
no GT/performance in smoke. Full follows only independent proposal/code + smoke.

Same gates: final within+.01 both vs native/current; no pooled loss>.005 or
within loss>.01. On qualifying main, evaluate VSV-only and SLA-only cached
controls; a claimed component must lose>=.01 on a common main metric in both
corpora when removed. Compare query-matched vs within-video shuffled directions
with its precise fixed-seed0 mapping declared before replay to test whether the
query-specific direction matters. That is a mechanism control, not alternative
candidate selection. Report raw/modality order, final three metrics, unchanged-
order/single-window contributions, uncertainty, cost and actual gain/loss cases.
Any observed gain must be distinguished from a score-level shift. If no main
metric gains>=.01, archive without tuning; otherwise rule9 permits revision.

## State

Independent proposal review PASS: `docs/reviews/20261003_m1_reinforcer_proposal.md`.
Implementation underway; no GPU run, labels or performance read for this proposal. Constants fixed from the cited paper before our run, with no local scan; the
source authors tuned their settings on100MSCOCO validation images. Nearby
FBHM/LSV uses supervised steering for static hateful memes (arXiv2605.31349v2),
so no claim of first activation steering for multimodal hate is intended.

Own numerical checks completed before GPU: `selfcheck.py` runs actual36-layer
small multimodal Qwen including two DeepStack injections, FP32 and BF16 on CPU.
Native read is exact, direction is nonzero, steering changes final answer logits,
original KV/mRoPE recover exactly, norm-preservation/zero-norm algebra passes,
and12-column token-first margin matches full-vocabulary construction within
1.2e-7. This small hidden64 witness does not replace the forthcoming real8B
smoke. Artifacts `runs/20261003_m1_reinforcer/selfcheck/numerics.json`.
Independent review is separate and pending; no GT or real performance read.

Independent code review PASS (no production fix):
`docs/reviews/20261003_m1_reinforcer_code.md`; artifacts
`runs/20261003_m1_reinforcer/independent_review/check_reinforcer.json`.
Actual36-layer multimodal smallQwen FP32/BF16, all-row norm oracle, unsteered
sources, SLA and native preservation, dualKV/RoPE/crop and actual outer-call
counts pass. Complete-reader test changes ONLY the width assertion4096→64 in
an independent test copy; production still asserts real8B width. Synthetic333
prepare/report and invalid config/coverage/alignment/global/nonfinite rejection
pass; canonical evaluator and unchanged r6 commands verified. No real GT read.

GPU target sc448960 (lab-server): existing reviewed HateVLM, media and weights.
Fresh preflight `runs/20261002_m1_iteration/preflight/machines_reinforcer_code.txt`;
source sync must complete before launch. Other-project home STRAY entries are
outside task scope; project target clean. Five-video smoke first via
`bash experiments/20261003_m1_reinforcer/launch/run_lab.sh smoke`, then local
`python experiments/20261003_m1_reinforcer/analyze.py --smoke --stage prepare`.
Only after validation, `.../run_lab.sh main` complete333 and
`bash .../launch/run_analysis.sh`. Results returned before STATUS update.

Five-video8B GPU smoke PASS on sc448960; returned and locally validated before
full launch. Source `runs/20261003_m1_reinforcer/r1_smoke/plumbing_summary.json`:
all5global/292branch native scores exact, forced answer/speech/native inputs
restored, nonzero directions change window scores. Peak18.0980GiB, smoke41.9s
including extra native restoration. First2/corpus estimated full cost339.50s
HateMM and543.29s HateClipSeg (14.71min total; deployment=paired collection),
versus capture-inclusive base208.49/286.19s. HMM1145829-token check passed,
excluded from extrapolation. No GT/performance inspected in smoke. Full333 next,
unchanged declared constants; fresh preflight `machines_reinforcer_main.txt`.

Full333 initially stopped after103 complete paired records on a strict diagnostic
assertion, before any GT/performance was read. The no-image query at position104
had94query/700cache tokens; BF16 single-row finalNorm re-evaluation differed from
native full-query normalization (maxhidden.03125, answerlogit.0028019).
Logs preserved `runs/20261003_m1_reinforcer/failure_103{,_numeric}.log`.
Fix only the diagnostic: read-only hooks check exact actual last-block→finalNorm
input and actual finalNorm output→Judge return. Independent single-row re-evaluation
is recorded, not required bitwise-equal across kernel shapes. Scores still use
original native final output; SLA unchanged. Independent narrow fix check PASS,
all9 synthetic token/input/RoPE arrays exact before/after; source
`independent_review/norm_capture_scores_unchanged.json`. Original5 smoke + failed
manifestposition104 will run afresh via `launch/check_normalization.sh`, no GT.
Only after six-video8B probe passes may the original103 paired records be resumed.

Six-video8B probe PASS on sc448960, returned locally:
`runs/20261003_m1_reinforcer/normalization_probe/checks.json`. Failed case is
HateMM `hate_video_215`; actual normalization boundaries and historical native
reads pass exactly despite repeatable single-row diagnostic drift. Five original
smoke cases still pass. No GT read. Resume the complete paired collection from103;
this is a diagnostic-only correction, no prediction formula or stored score change.


## R1 complete and archived, 2026-10-03

Run host sc448960. Full333 returned locally before evaluation;333global and13939
native branch values exactly match base_gridA. Canonical final source
`runs/20261003_m1_reinforcer/r1_main_decoded/{base,full}/metrics.json`, raw source
`r1_main/{base,full}/metrics.json`; reports `r1_main_analysis/`. Development-selected.

| Corpus | Arm | ROC | PR | within |
|---|---|---:|---:|---:|
| HateMM | native | .897119 | .694235 | .750782 |
| HateMM | full | .891214 | .661409 | .750129 |
| HateClipSeg | native | .716825 | .671072 | .637349 |
| HateClipSeg | full | .722658 | .670072 | .636515 |

Final within deltas−.000653/−.000834 (n84/99; paired95% CI
[−.016251,.014118]/[−.014649,.012634]); HMM PR−.032826. No final main gain>=.01,
so rule9 archive with no SLA/VSV component selection or parameter tuning.
Post-scoring reads: raw/decoded predictions, canonical metrics, checks and the
original two `data/gt_4fps` arrays, for reporting only. Visual raw ordering changes
−.033063/+.004810; native speech exactly unchanged; raw max ordering changes
−.034783/−.017511. Thus an image-dependent internal update is verified, but a useful
localization mechanism is not. No new media/ASR semantics inspected for this run.

Actual completed-video standalone sums (`r1_main_analysis/cost.json`):
full640.66/548.81s, total19.82min; capture-inclusive native358.24/293.70s, ratio1.825x.
Native capture includes intermediate projections, so this is not overhead relative
to a bare uninstrumented native run. Full mean forwards74.57/123.92 versus native
36.52/60.05; exact full6+B+2V, native3+B. Peak18.106GiB. Initial run stopped after103
records, then the scoring-invariant finalNorm diagnostic repair resumed230records;
reported908.4s resume wall is NOT whole333 time. Initial completed portion approx285s
plus resumed908.4s is about19.9min paired collection, excluding failed probes and
repair validation. All retained records use the identical scoring computation;
there was no fresh single uninterrupted final-code333 run, and no promotion claim.
Current method, paper and Overleaf unchanged. This is cumulative archive13.
