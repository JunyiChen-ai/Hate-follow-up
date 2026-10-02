Archived2026-10-03: proposal STOP under rule4; prior SHAP use in hateful-video detection. No implementation/GPU run.

# Allocator: joint-window coalition attribution

Declared 2026-10-03 before implementation or performance. Ninth candidate,
prepared while visual contrast r1_main runs; that candidate's results have not
been read. Proposal review pending. This candidate does not amend or retune the
archived Eraser. It transfers Kernel SHAP to a different estimand and jointly
groups the two modalities of each window. Development host sc474397.

## Hypothesis and boundary

Eraser measured the effect of removing one modality-window from an otherwise
complete video and lost within-video accuracy on both corpora. It did not test
how a window contributes in the presence of different subsets of other windows.
For an OR-like classifier, two redundant positive events can each have zero
leave-one-out effect. Shapley allocation instead shares their contribution.
This is a mathematical distinction, **not evidence that redundancy caused the
observed Eraser failure**. Context removal can also change meaning, and a model's
decision attribution need not match annotation spans.

Each player is one 8-second window's sampled frames AND transcript words,
retained/removed together. A coalition is physically reconstructed and freshly
encoded. Its value is the frozen MLLM's global Yes/No class margin. The output
local score is its estimated Shapley contribution, not the original window
answer plus an arbitrary correction. Joint players allow a contribution to
depend on the combined visual/speech evidence. They do not separately identify
cross-modal synergy and cannot support that stronger claim.

Sources: [Kernel SHAP, NeurIPS 2017](https://arxiv.org/pdf/1705.07874),
[Play Fair: Frame Attributions in Video Models, ACCV 2020 author repository](https://github.com/willprice/play-fair),
[Explaining by Removing, JMLR](https://arxiv.org/abs/2011.14878).
The repository also tried Shapley across three precomputed modality-evidence
views in August (`archive/experiments/idea_discovery-2026-08/idea_discovery/
run_shared_coalition_game.py` and `run_interaction_coalition_game.py`): eight
subsets of an external evidence model, not temporal raw-media subset re-encoding.
Allocator must be evaluated as the latter transfer, not first project use of SHAP.
No new Shapley algorithm claim. Rule4 review must search actual hateful-video
detection/localization use, and assess the distinct method versus archived
leave-one-out attribution. SHAP use for text hate or static memes alone is not a
first-ever hate-attribution claim; only video-localization transfer is proposed.

## Frozen computation and predeclared constants

- Qwen3-VL-8B BF16, native20 frames/ASR/policy/system/global question,8-second
  windows,4fps, native original global margin retained downstream. No labels.
- Partition in-range frames by timestamp (last endpoint inclusive).
  Partition each original ASR segment's words by the same rounded fractional
  cuts used by `window_text`. Keep original segment timestamps and join retained
  words in original order. Words or frames that belong to no legal window stay
  as fixed background in EVERY coalition; never assign invented timestamps.
  Retain unchanged original segment text verbatim when
  all its words remain. Drop a segment only if none remain. Preserve chronological
  order; never permute the video itself. Full coalition must reproduce the exact
  native input, processor tensors and margin.
- A player is active if it owns any frame or ASR word. There are M active players;
  entirely empty windows are dummy players with contribution0. Empty coalition
  contains only that fixed background plus the same policy/question; when there
  is no unmatched media, it contains no frames/transcript. Prefix prose reports
  actual remaining frame count; retained media keep their original timestamps.
  Thus the attribution also includes changes to media-count/availability prose.
- Compute each unique coalition value with a fresh prefix and global-question
  read,2 outer forwards. No forced original verdict in these value evaluations.
  The original full-video margin, not the empty or sampled margin, remains z_v.
- M<=6: enumerate all2^M coalitions and exact finite Shapley-kernel regression.
  M=0/1: handle directly. M>6: evaluate all M singletons and M complements, plus
  32 random complementary pairs of interior coalitions, and empty/full.
  Interior size s in {2,...,M-2} sampled with probability proportional to
  1/[s(M-s)], then choose s distinct players uniformly. numpy.default_rng(0)
  reset per video; no content-derived seed. Cache duplicate subsets by a tuple
  of player indices, with no digest; retain repeated rows' statistical weight.
- Kernel weight per subset: w(S)=(M-1)/[C(M,s)*s*(M-s)]. Singleton/complement rows
  get exact weight1/M. Each of the64 sampled interior rows gets
  W_int/64, where W_int=(M-1)*sum_{s=2}^{M-2}1/[s(M-s)]. This is importance
  sampling of the interior weighted objective; complementary draws are paired,
  not64 independent samples. No extra kernel factor after this sampling.
- Minimize weighted squared residuals of v(S)-v(empty)-sum_{i in S}phi_i,
  subject to sum(phi)=v(full)-v(empty). Use float64 constrained least squares
  (eliminate one coordinate), no ridge/L1/feature selection. The singleton rows
  guarantee rank on the constraint space. Log condition number, weighted fit
  error, efficiency residual, exact mask coverage and deduplication counts.
  Finite sampled estimates are not exact Shapley values; label them as estimates.
- Write one honest `z_joint` channel; use unchanged r6 dynamic-modality support,
  with the same independently label-free fitted r6 algorithm and constants.
  Raw4fps curve repeats phi by window. Do not duplicate one score into both
  modality channels. Native dual read and native joint query are paired controls;
  the joint control distinguishes attribution from merely changing two branches
  to one. Original global retained in all arms.

## Cost and staging

Reuse current frames, ASR, weights. No new encoder or training. No experiment
cross-import; reusable raw subset construction belongs in shared src if needed.
Let U be unique coalitions including empty/full. Deployed Allocator costs2U
outer forwards per video; paired measurement adds native answer/window reads
(1+B) and N native joint questions, reusing the full coalition global prefix.
For M>6, U<=2M+66; for M<=6, U=2^M. Every new subset needs real prefill cost.
No cached-prefix reuse across changed media, no claim that this is free offline
work. Input audit gives at most33,030 unique subset reads before deduplication
(18,398 HateMM /14,632 HCS); mean active players16.98/29.00, maximum125/42.
Most sampled subsets are shorter than full videos. Based on Eraser cost,
estimate100-200GPUmin for333, to be replaced using
five-video GPU plumbing. This is a higher-cost candidate, justified only if
coalition-sensitive allocation improves localization measurably.

Proposal once, implementation, independent code review once. CPU: exact small
additive/OR/interaction games, dummy/efficiency/symmetry, exhaustive Shapley vs
kernel fit, sampling weights and reconstruction conservation. Five-video GPU:
first2 manifest per corpus plus HateMM/hate_video_114, no GT; native global and
every dual branch exact, full/empty reconstruction, fresh restoration, complete
physical partition, finite values,4fps, cost. No hidden sampling-budget tuning.
Full333 arms base/joint/allocate, canonical raw and r6 three metrics, within
n84/99, paired bootstrap2000 seed0. Target final within+.01 on BOTH corpora versus
current/paired base and no pooled loss>.005 or within loss>.01; require meaningful
advantage over joint control for an attribution contribution. Rule9: any final
main gain>=.01 allows up to3 declared revisions; no gain archives the candidate.

Only if primary gain qualifies: compare to joint leave-one-out and singleton
scores (already evaluated subsets), shuffled allocation across windows seed0,
and native joint query. Measure raw ordering, native verdict strata, modality
coverage, costs and semantic examples. Coalition fit error/efficiency are
implementation diagnostics, not proof of localization mechanism. To claim
redundancy handling, show cases with multiple independently sufficient subsets
and weak leave-one-out effects, along with controls and counterexamples. Each
claimed novel component must satisfy rule14g on both corpora. If no such
mechanism evidence appears, do not invent a redundancy explanation.

## Read log

Read STATUS, archived Eraser/Factorizer/Attributor/Contraster README and current
VCD declaration; error-analysis20260927 and SDL/NGA historical README. Older
claims that supervised probes prove an information ceiling are not adopted.
Eraser loss motivates testing a different influence estimand, not retuning its
constants. A no-GT input audit of manifest/ASR found24 of333 videos with original words
not covered by legal window_text intervals (ASR exceeds manifest duration). This
requires the explicitly fixed-background rule above for exact full reconstruction;
it does not trim or repair the baseline ASR. No new GT or in-progress
visual-contrast metrics read for this design.
All later test-based comparisons are development-selected. Current default and
paper unchanged while the experiment is pending.

Input audit source: `runs/20261003_m1_allocator/input_audit.json`, host sc474397,
no GT read. Fixed background: HateMM18 videos/705 words; HCS6 videos/7 words;
no unmatched sampled frames. This is an input-duration issue, not inferred from
performance. Full/background reconstruction will preserve these words exactly.

Decision source: `docs/reviews/20261003_m1_allocator_proposal.md`. Prior-source
implementation limits are explicitly retained there; not a negative performance
result, and not evidence that the proposed temporal game was already evaluated.
