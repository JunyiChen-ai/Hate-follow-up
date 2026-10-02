# M1 Projector: orthogonal attention-output guidance

Candidate14, declared2026-10-03 while Integrator R4 runs and Reinforcer R1 is in
independent code review. Neither candidate's full performance has been read.
Independent proposal; not combined with any earlier intervention. Frozen
Qwen3-VL-8B, native20frames/fullASR/policy/prompts/8s/4fps and unchanged r6.
Development-selected. No implementation or GPU run before proposal review.

## Source and testable hypothesis

Source: Jo, Bae and Kim, *Attention-Space Contrastive Guidance for Efficient
Hallucination Mitigation in LVLMs*, CVPR Findings2026, arXiv2601.13707v2.
Primary sources:
https://arxiv.org/html/2601.13707v2 (Algorithm1, equations4–8, sections4.1/5.3),
https://openaccess.thecvf.com/content/CVPR2026F/html/Jo_Attention-Space_Contrastive_Guidance_for_Efficient_Hallucination_Mitigation_in_LVLMs_CVPRF_2026_paper.html

ACG contrasts ordinary attention output with a same-pass approximation obtained
by masking image keys. It removes the correction's component along the masked
output before adding guidance. Source uses the current last text token and all
layers by default; Qwen-VL gamma1.4 is transferred unchanged here. No Qwen3 result
or official implementation has been verified; this is a paper-defined transfer.
The source explicitly acknowledges visual information already inside text values
and redistribution under masking. Orthogonality is geometric, not proof that
language bias is removed or the retained direction contains only image meaning.

Hypothesis: ordinary visual reads may dilute useful image-dependent evidence
with a dominant text-aligned attention component. A per-head perpendicular update
could change their ordering without scalar logit subtraction or separate-reference
prefills. Whether it helps binary window discrimination must be tested here.

Closest earlier attempts: Grounder/Selector alter which keys/heads are read;
Amplifier scales image attention logits; Recycler redistributes sink attention;
Reinforcer steers post-block residuals using separately encoded image/no-image
contexts. Projector instead changes the vector-valued attention output before
its output projection, using a query-dependent same-pass masked reference. VCD
contrasts final logits. Those failures do not establish this different mechanism.
PWC pairwise window ranking has already failed (DIRECTIONS A2); PRP rediscovery is
not selected as a new candidate. ASCD was source-discovered but not selected.

## Exact R1 Qwen3 adaptation (fixed before scoring)

Only VISUAL window questions change. Native shared prefix, native global numeric
score, forced native answer and every native speech question remain identical.
No recomputed global score or new question wording. Crop suffix KV after each
branch. All36 language self-attention layers (0..35), all32 attention heads,
LAST query row only; every preceding query row retains native SDPA computation.
Visual key mask is actual expanded image-token IDs across all20images, including
noncontiguous blocks. No temporal selection, sink/head selection or threshold.

For the last row, use native normalized/rotated Q,K,V with GQA repetition and
same causal/additive mask. Compute scaled QK in the current BF16 model dtype and
softmax in FP32, cast probabilities back to model dtype before AV. O is ordinary
attention output; U uses identical logits with visual-key positions set to -inf.
All non-image positions, including scaffolding/frame timestamps/ASR/global QA
and current-question tokens, remain available to U. U is a surrogate, not a
true no-image model execution. At each HEAD separately, in FP32:

    delta = O - U
    unit = U / (||U||2 + 1e-8)
    perpendicular = delta - sum(delta * unit) * unit
    guided = O + 1.4 * perpendicular

Cast guided to original dtype, substitute the last row before concatenating
heads and applying native o_proj. Per-head geometry, epsilon1e-8, GQA and
noncontiguous20frame mask are explicit adaptations; paper does not specify our
Qwen3 implementation details. If U=0, unit=0. If no image keys exist, correction
is zero. Final decision remains original finalNorm/head and12-token Yes/No
logsumexp. No SLA, image-noise branch, additional encoder or model ensemble.

Three main arms: native; matched last-row eager control (same reconstruction,
no guidance); Projector. All share unchanged global/answer/speech. Audit native
historical equality; compare against BOTH native and eager, not only one control.
Store per-layer norm/projection diagnostics and final answer-token logits, plus
all windows and native inputs, without feature/content hashes or GT access.

## Cost, checks and falsification

V visual windows, B=V+available speech. Deployed3+B outer model forwards exactly
as native, with two last-row attention reductions in each affected layer; paired
native/eager/guided collection3+B+2V. Smoke restoration adds3+B. Native prefix
cache reused, no second context/cache or new media preprocessing. Initial paired
333 estimate15–25GPUmin on5090, deployment10–15min; actual five-video smoke must
replace estimates and measure peak memory (<32GiB required), not hide inner
attention cost behind unchanged outer-call count.

Before GPU: independent proposal review with actual target-task novelty search;
independent code review and actual small multimodal Qwen FP32/BF16, all36layers,
GQA/causal correctness, image-mask validation, vector algebra, native restore,
prefix/global/answer/speech/KV invariance and actual calls. First2per corpus plus
long-prefix HMM114 smoke with no GT. Complete333 uses canonical evaluators and
unchanged r6; all config/constants identical across corpora. No local scan.

Same acceptance: final within+.01 both vs native/current and matched eager,
no pooled loss>.005 or within loss>.01. No final main gain>=.01 =>archive.
A qualifying result triggers removal of orthogonal projection at the SAME1.4
scale, plus a per-head norm-matched random-direction intervention (fixedseed0,
exact sequence rule declared before run). Component claim needs removal loss
>=.01 on a common main metric both corpora. A projection diagnostic alone is
not mechanism evidence. Report raw visual/speech/max ordering, final3metrics,
unchanged-order/single-window contributions, gain/loss cases, uncertainty and
new-video cost. Actual local evidence use needs content-sensitive controls;
do not infer localization semantics from output-vector orthogonality alone.

Proposal source clarification before implementation: the source guidance scales
were selected by scans on CHAIR; ours borrows Qwen-VL1.4 without a local scan,
not a claim of parameter-free source design. The source newer-model section
covers LLaVA-NeXT7B/13B, not Qwen3. Epsilon and BF16 make orthogonality approximate,
and per-head orthogonality need not survive o_proj or imply semantic isolation.

Independent proposal review PASS:
`docs/reviews/20261003_m1_projector_proposal.md`. Primary target-task searches
found no verified prior ACG use; this is a scoped search conclusion, not proof
of absolute novelty. No four-category STOP. Implementation begins after PASS.

Own selfcheck implemented and passed (before GPU): exact per-head formula,
zero/gamma0 cases, masked GQA last-row computation, noncontiguous image positions,
actual36-layer smallmultimodal Qwen with DeepStack FP32/BF16, and native KV/mRoPE
restoration. Source `selfcheck.py`, artifact
`runs/20261003_m1_projector/selfcheck/numerics.json`. BF16 eager differs from
native (maximum answer-logit drift.01171 in this synthetic model), confirming
that the matched eager arm is necessary. Geometric projection residuals are
small numerical values, not evidence of semantic debiasing. Independent review
is separate and pending. No real GT or candidate performance inspected.

Independent code review PASS, no production fix:
`docs/reviews/20261003_m1_projector_code.md`; artifacts
`runs/20261003_m1_projector/independent_review/check_projector.json`. Actual small
multimodal Qwen36 layers/DeepStack FP32/BF16,20 synthetic images/100 noncontiguous
expanded image keys, exact head/GQA oracle, last-row-only intervention, original
other rows and causal edges, registry/KV/mRoPE/global/answer/speech/weights/native
recovery, actual B4/V3 deployment7/paired13/smoke20 calls all pass. Synthetic333
prepare/report and matched-eager gate checks pass, no real GT/performance read.

GPU smoke target sc474399 after Integrator R4 completion; source git sync and
fresh preflight `runs/20261002_m1_iteration/preflight/machines_projector_smoke.txt`
required. Existing reviewed HateVLM/model/media reused. Run
`bash experiments/20261003_m1_projector/launch/run_lab.sh smoke`, return files,
then `python experiments/20261003_m1_projector/analyze.py --smoke --stage prepare`.
Other-project home STRAY entries excluded from task; source must be clean.
