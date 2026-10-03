# M1 autonomous iteration, 2026-10-02

User instruction: 进入自主迭代，修改第一个模块，要能够做出为什么涨点的机制。
Scope: modify frozen-MLLM reading, hold r6 inference/evaluator fixed. Continue
according to RESEARCH_ITERATION_RULES, with genuine mechanistic controls; no
paper polishing substitutes for empirical improvement. No new dataset or ensemble.
All development decisions are development-selected. Test GT is used only after
scoring for evaluation/error analysis, never fitting/routing/threshold selection.

Completed candidates:
1. `archive/experiments/20261002_m1_grounder/README.md`: late restriction had no
   qualifying gain; correct/wrong support almost indistinguishable.
2. `archive/experiments/20261002_m1_selector/README.md`: selective-head restriction
   changed routing but reduced final within by .0418 / .0150; no qualifying gain.
   Full baseline reproduction exact; controls beyond smoke not run per funnel.

3. `archive/experiments/20261003_m1_marginalizer/README.md`: proposal STOP;
   MARS/RAMF already use opposite hate/non-hate assumptions on the same media.
   No implementation or GPU run.

4. `archive/experiments/20261002_m1_attributor/README.md`: complete uniform FP32
   run, native parity/numerics passed; final within drops.1296/.0695 and every
   main metric drops. No control performance evaluated after failed primary gate.

5. `archive/experiments/20261003_m1_eraser/README.md`: actual media deletion and
   full re-encoding, native parity passed; all final metrics fell, within−.1312/
   −.1311. Raw ordering also worse. Archived with no extra controls.

6. `archive/experiments/20261003_m1_factorizer/README.md`: full native/explicit-
   causal/factor experiment complete; no qualifying gain. HCS finalwithin−.0700
   and raw−.0654; paired native exact. No further controls.

7. `archive/experiments/20261003_m1_contraster/README.md`: full333 paired run,
   native exact. Final within+.0001/+.0002; all13939 branches selected layer2.
   No qualifying gain; no additional controls or layer-pool revisions.

9. `archive/experiments/20261003_m1_allocator/README.md`: proposal STOP; published
   SHAP use in target video detection task; no implementation/GPU run.

8. `archive/experiments/20261003_m1_visual_contrast/README.md`: eighth candidate,
  clean/corrupt visual-window contrast with native speech/global retained.
  Proposal/code PASS; five-video GPU plumbing passes (292 exact native branches).
  Full333 R1 missed promotion because of HMM PR; cached matching controls
  supported localization only on HCS. R2 local-image corruption had no qualifying
  main gain and degraded visual raw ordering. Archived after that revision; no donor
  control or further schedule/contrast tuning. R1 best numbers remain recorded.

Running candidates:
- `archive/experiments/20261003_m1_integrator/README.md`: tenth candidate, future-aware
  visual prefix memory from FutureMask; R1 full333 complete on sc448960, native
  exact. HMM within improves but pooled losses prevent promotion; both visual
  raw orderings fall. HMM gains are concentrated in two unchanged-raw-order cases.
  Numbers and sources remain in the candidate README and STATUS, not a third table.
  R2 future text keys only full333 complete, HMM within gain but pooled fails;
  HCS visual ordering improves without combined improvement. R3 declares
  new visual with native speech; cached complete-branch test complete: HMM pooled restored, HCS fails.
  R4 last modification complete: same-window future ASR with native speech still fails HCS pooled. Family archived as twelfth; max3 modifications exhausted.
  Independent of VCD, not combined.
- `archive/experiments/20261003_m1_recycler/README.md`: candidate12, paper-defined VAR
  with explicit Qwen3 sink-channel adaptation; independent proposal/code PASS,
  real-input GPU smoke PASS, full333 complete; no qualifying gain, eleventh archive.

11. `archive/experiments/20261003_m1_amplifier/README.md`: candidate11, tenth
  archive. Complete333 PAI transfer had no qualifying main gain; HMM PR drops.
  Native reads exact, matched eager near baseline, no component controls after
  failed main gate.

Cumulative archives17 (fifteen performance/mechanism failures, two proposal novelty STOPs).

Initial candidate rationale:
Reason: all previous score-subtraction, hypothetical counterfactual questions,
generated hypotheses and latent common-offset decompositions failed. A previously
suggested layer-dependent attention intervention has not actually been run.
Read-only prior sources: STATUS; 20260928 headroom and its round-2 review;
20260927 error analysis; CVA archive; 20261002 verdict analysis and revisable-prior
archive. Distinguish failed scalar-cache proxies from an untested internal
computation. Each candidate gets a separate declaration, review and result record.

Acceptance: same metrics/gates on both main corpora; baseline-matched reading and
unchanged downstream inference; claimed mechanism must pass component ablation
and an intervention capable of falsifying its explanation. Each new reading cost
and all failures are recorded. Current paper remains unchanged during search.

Maintenance2026-10-03: independent Allocator proposal review encountered a
content-derived tie-breaking seed in the archived August
`project_shared_shapley_rank_transport.py`. Replaced it with fixed seed0 under
CLAUDE prohibition; script not executed, historical results not reinterpreted.

Unselected possibilities inspected while full jobs run, not additional candidates:
- Soft expected-answer embeddings (SoftThinking source previously noted) would
  differ mainly near uncertain native verdicts. Read only native z_video from
  `runs/20260926_glr/base_gridA/predictions.jsonl`, no GT: smaller Yes/No probability
  exceeds.05 in18/215 HMM and14/118 HCS, and its median is about2e-6 in both.
  This is an input-distribution observation, not a performance screening result;
  no temperature, routing threshold, method or experiment was selected.
- PAI, [arXiv2407.21771](https://arxiv.org/abs/2407.21771), is another known
  input-attention/contrastive-decoding source. Only abstract/source discovery
  was inspected at this point; no claim of target-task novelty or implementation.
  At that time the running work was visual-contrast R2 and Integrator R1;
  visual contrast has since been archived. PAI has now been fully source-read
  and separately declared as candidate11; SoftThinking remains unselected.
- VAR was inspected during Integrator R2 / Amplifier R1 collection and is now
  separately declared as candidate12: `archive/experiments/20261003_m1_recycler/README.md`.
  Independent proposal review PASS; paper/code differences and Qwen3 channel
  adaptation declared before implementation, no full-run performance read.

Candidate13 proposal: `archive/experiments/20261003_m1_reinforcer/README.md`, VISTA
paper-defined residual steering + preceding-layer logits, Qwen local-query scope.
Author-code discrepancies are declared explicitly. Independent proposal review
PASS; implementation, independent code review and real-input checks PASS;
full333 completed after a scoring-invariant diagnostic repair; no qualifying
main gain, archived as13. No additional component variants selected. VTI/PTI were only inspected
as adjacent steering sources, not declared as candidates or run; no external data
has been added. Pairwise Ranking Prompting was located but not yet method-read.

Candidate14 proposal: `archive/experiments/20261003_m1_projector/README.md`, paper-defined
ACG transfer, orthogonal attention-output correction with same-pass masked
reference. Explicit per-head Qwen3 adaptation and matched last-row eager control;
proposal/code/real-input review PASS; full333 complete, no qualifying main gain;
archived as14. No further controls/tuning. This is distinct from residual steering, sink transfer
and image-logit boosts. PRP was not selected after finding the prior PWC failure.

Candidate15 proposal: `archive/experiments/20261003_m1_highlighter/README.md`, VGA visual-
semantic value guidance restricted to local frame support, preserving native
context. Paper and official Qwen2.5 code read; source discrepancies and Qwen3
adaptations declared. Independent proposal/code review and real5video checks PASS;
full333 complete and canonically evaluated; no qualifying gain, archived15. FV-Action inspected but not selected because existing
M1 already uses binary window reads. VideoTree/VAP only discovered, no new data.


Unselected input-timing direction inspected2026-10-03 after Projector completion:
read `data/asr_whisper_large_v3/PROVENANCE.md`, `src/window_token_regions.py`,
`src/video_inputs.py` and unlabeled loaded ASR segment durations on the333manifest.
HMM has2283nonempty segments,414>8s/217>30s, median2.04s/max478.92s;
HCS1555segments,313>8s/216>30s, median2.44s/max289.98s. These are cache boundary
spans, not evidence that speech lasts throughout the span or that timestamps are
correct. Existing window text uses proportional word assignment. This motivates
checking acoustic timing uncertainty, but no method has been declared or selected,
no word alignment extracted, and no GT read for this timing inspection. Ordinary
forced alignment alone would be an input change, not a rule4 novel mechanism.
Primary source inspected: *Whisper Has an Internal Word Aligner*, arXiv2509.09987v1,
methodsII-A–C; it proposes character teacher forcing and unsupervised head filtering.
Word-confusion-network SLU arXiv2401.02921 and uncertainty-DTW arXiv2211.00005 only
source-discovered; no claim of applicability, source reproduction or target novelty.

Follow-up timing diagnostic (post-scoring, not method selection): reused canonical
per-video speech within from `runs/20261003_m1_projector/r1_main_analysis/branch_diagnostics.json`
(which reads original test GT), joined to loaded ASR spans and manifest durations.
Artifact `runs/20261002_m1_iteration/timing_diagnostic/summary.json`. On82/97eligible
speech videos, rank correlation of fraction(nonempty spans>8s) with speech within
is−.009/−.324. Maximum span itself is+.199/−.130, video duration+.211/−.039.
These exploratory correlations are neither alignment-error measurements nor causal
proof, and do not support a common two-corpus failure explanation by themselves.
Lowest speech cases were listed automatically, but audio has not been listened to
and their transcript accuracy/semantics have not been assessed. No new constants,
scoring code, frame inputs or experimental candidate changed from this diagnostic.

Inspected two listed cases using original ASR JSON, base_gridA window scores and
original GT arrays: HMM hate_video_349 and HCS bit_8I3rasu4mSiz. Both have positive
GT over nearly the entire video, with a short negative tail; their raw speech
ordering gives the tail a higher score than many positives. Thus the lowest
speech-within cases do not isolate word-timing failure. Source ASR clips were
read as text only; audio not listened to, no semantic or transcription-accuracy
claim. This weakens treating coarse timestamps as the established next mechanism;
no timing candidate selected or implemented.

Candidate16 proposal: `archive/experiments/20261003_m1_stabilizer/README.md`, PAS-derived
head-dependent native temporal RoPE phase during image prefix encoding. Paper/code
read; multi-image Qwen3 pairing/axis adaptations and source theorem limitations
explicit. Independent proposal/code review PASS; full333 Slurm run and canonical
evaluation complete. Final within gains only+.000607/+.001170, no final metric+.01;
archived16. Each arm used its own global and both window branches, not a fixed verdict.

Adjacent unselected sources inspected during Highlighter collection: VideoTree
arXiv2405.19209v1 sections3.1–3.3 (visual clustering, relevance-guided hierarchical
keyframe captioning); Temporal Tree of Thought arXiv2608.27871 sections3.1–3.2
and appendix algorithms (contiguous feature-based hierarchy, iterative retrieval/
expansion with answer confidence); official MTLA repository README
https://github.com/TalRemez/MTLA (localized attention confidence and rollout voting).
No proposal, implementation, new frame extraction, external data or score screening
was made from these reads. Their full source recipes do not automatically satisfy
our no-ensemble/no-postprocessing and new-video-cost constraints; any adaptation
needs a separately declared complete mechanism and independent review.


Candidate17: `archive/experiments/20261003_m1_preserver/README.md`. MAD-RAG-derived
preservation of reference visual question attention outputs in the normal full
context read. Paper/official Qwen code and discrepancies inspected; two-cache/full
suffix adaptation declared, cost and controls fixed before performance. Independent
proposal/code review PASS; real5video smoke reproduces native and alpha0 exactly.
Full333 completed on sc448960 Slurm job55 and canonically evaluated. No final
metric+.01; raw visual/max ordering worsened in both. Archived17; no alpha tuning.

2026-10-03 resumed autonomous work on user instruction, explicitly no Overleaf.
Lab nvidia-smi failure was traced to /dev/nvidiactl EPERM from user.slice
50-slurm-gpu-only.conf, not established hardware failure. qian_pilot Slurm has
local partitions for all four lab hosts. Stabilizer will use its existing reviewed
HateVLM environment in a local-sc474399 GPU allocation, not alter system policy.


Additional primary method reads while candidate16 full run and candidate17 code
review proceed: VideoTree2405.19209v1 sections3.1–3.3 and implementation details
(adaptive breadth, relevance-dependent tree depth, caption-based reasoning);
Temporal Search2507.02946v1 algorithms1–2/section3 (interval proposals, confidence
plus self-evaluation, global keyframe descriptions, sequential/best-first search);
ReMem2607.24794v1 source discovered/opened, not yet fully method-audited. These are
unselected future directions, not candidate18. No new input extraction, method
constants, data or model was selected. Their confidence-calibration and overhead
claims on QA do not establish validity for binary hate-window localization.

ReMem full method3.1–3.3 subsequently read: entity/query feature fusion, two-step
semantic/temporal graph diffusion and event-budget routing. Its projection-weight
provenance and some equation/text consistency require author-code inspection before
any faithful adaptation. No candidate selected. VTimeCoT ICCV2025 primary PDF
method3.1–3.3 also read (visual progress bar, VideoCLIP-XL retrieval/highlights,
iterative tool reasoning/cuts;3step maximum). This is a full tool/retrieval mechanism,
not merely timestamp overlays, and has not been proposed or implemented here.
Read-only source under third_party/vtimecot_source_read/. No videos were modified
and no new feature models/data downloaded.


Unlabeled input-coverage diagnostic2026-10-03: actual native frame timestamps
from src.video_inputs.frame_paths plus all_test manifest/fixed8s windows, no GT.
`runs/20261002_m1_iteration/frame_support/summary.json`:1030/3768HateMM windows
and1233/3591HCS windows contain no sampled native frame;114/215 and118/118videos
have at least one such window. This is sparse visual coverage, not evidence that
those windows contain missed hateful visual content. No new sampler/candidate,
frame extraction or scoring change is selected from this count alone.


Unselected source follow-up while Preserver job55 runs: EcoFrame2608.03918v1
sections2,4.1–4.4,5.1 and appendixA.3/B/C read, with the official repository
https://github.com/AK-DREAM/EcoFrame (README-only as checked2026-10-03).
It combines whole-vocabulary answer entropy, progressive budgets4/8/16/32,
pre-RoPE query/image attention from layers19–21, attention-times-distance candidate
expansion, and CLIP-relevance-times-distance reselection. Defaults m4,
attention exponent.5/relevance exponent1; length factor clamps sqrt(duration/120)
to[1,4]. Source uses different entropy thresholds by benchmark; any adaptation
here must fix one shared setting and declare it first. Binary normalized Yes/No
entropy is not the source's full-vocabulary entropy, and a confident answer is
not a correctness guarantee. No candidate, encoder/cache extraction, new scoring
or threshold scan selected from this reading. VAP2605.01662v1 introduction and
method start only: it requires a video-diffusion interpolator, not free evidence
from the existing MLLM. No VAP implementation/weights requested. New searches
also surfaced VideoRoPE2502.05173 and WRWS2609.37345; no complete method read or
adaptation made for either. These observations do not change current M1.

Candidate18 proposal: `experiments/20261003_m1_explorer/README.md`, bounded
uncertainty/support-aware acquisition of actual local video frames, with a
pre-RoPE attention-times-distance proposal and cached native context. Self-designed
adaptation, not full EcoFrame reproduction; source/data/cost/control differences
explicit. Independent proposal/code review PASS; real5video smoke and full333 R1 complete.
HCS improves in all three main metrics, but HMM misses the dual-corpus goal and its
raw ordering falls. R2 entropy-only cache replay passed independent review but
still misses the HateMM goal. R3, the second revision, always acquires the first
two eligible local frames and retains the .3 entropy gate for two more. Its narrow
code review passed before the fixed5 GPU smoke; full333 scoring remains pending.
Numbers and GT/case-read log are in the candidate README/STATUS.
Input audit read old frame provenance/prep script without GT: old JPEG timestamps
are nominal seek times and may hide a .5s retry. New proposal distinguishes nominal
support from verified new-frame PTS, preserves the native baseline and records
the limits of legacy source-index exclusion. All original inputs remain unchanged.
