# Prior falsification map

Audit date: 2026-08-07. Scope: every falsified or retired mechanism in the EMNLP2 predecessor
project (April 2026) plus the two mechanisms this project has already killed. Purpose: decide
which deaths were caused by the compromised substrate and which indict the design itself, so
that a second mechanism is not a rerun of something already falsified.

All quoted figures are copied from the repo files cited beside them. No hateful transcript
content is reproduced anywhere in this document; where source files quote material, it is
paraphrased neutrally.

## The two substrates

Every EMNLP2 falsification ran on a substrate that is now known to be compromised in three
independent ways, and the current project has measured all three.

**Model.** Qwen3-VL-2B only. `docs/duplex/reports/c2_fullcorpus_2b.json`, section
`operating_points.diagnostic_zero`, shows the 2B at its own decision boundary produces 647 true
positives, 417 false positives and 2 false negatives: recall 0.9969 against precision 0.6081.
The 2B does not commit. The same file's `threshold_label_free` gives a relative trough depth of
0.0066 with 271 of 1000 bootstrap resamples finding no valley at all, against 0.6004 and 0 of
1000 for the 8B in `c2_fullcorpus_8b.json`. The 2B score distribution is not bimodal in any
usable sense.

**Readout.** Clipped renormalized `P(Yes)`. `docs/duplex/KILLTEST_POSTMORTEM.md`, section "The
censoring artifact behind the 8B P1 pass", records that 622 of 1155 videos had a divergence of
exactly zero because both readers hit the `EPS = 1e-4` guard, with censoring rates of 75.9% for
NH, 34.7% for EX and 19.7% for IM. The same section records the 2B failure in the opposite
direction: 99.9% of `s_lit` logits fall on a 0.25 grid and `s_lit` takes 21 distinct values.
`docs/duplex/READOUT_POSTMORTEM.md`, section "Where the phenomenon went", records that under the
raw unclipped logit the same stratum is a range from −3.0 to −21.75 with an interquartile range
of [−17.69, −9.75].

**Input.** 300-character truncated transcripts with unrepaired ASR failures.
`docs/duplex/reports/channel_restoration_verdict.json`, `P1_flip_asymmetry`, records that
restoring the speech channel flipped 57 of 110 dismissed hateful videos (0.5182) against 18 of
552 dismissed NH (0.0326), asymmetry 0.4856, Fisher exact `p = 1.30e-35`. Channel starvation is
causally established, not conjectured.

**Current substrate.** Qwen3-VL-8B, raw unclipped `z`, fresh full-length Whisper large-v3
transcripts, label-free KDE-valley threshold. `docs/duplex/reports/c2_fullcorpus_8b.json`:
AUC 0.9505 (bootstrap 95% [0.9387, 0.9608]), macro-F1 0.8796, accuracy 0.8800, one MLLM call per
video. The oracle gap in the same file is −4.3e-05: the label-free threshold is at the
label-peeked optimum. Error budget: 105 false positives, all NH, of which 83 (0.7905) carry a
surface cue against 0.5331 among true negatives, Fisher `p = 3.02e-07`, with 75 carrying an
identity marker; and 49 false negatives, 43 of them IM, of which 7 are the 8 videos Step 0
established have no transcribable speech.

## Summary table

Death cause key: **S** substrate-bound, **D** design-bound, **M** mixed, **U** unresolved.

| # | Design | Verdict and where recorded | Cause | Revival |
|---|---|---|---|---|
| 1 | PP-v1 Observe-then-Judge: video observer writes ≤40-token (group, stance) sentence, text-only judge scores it | MISS. EN oracle 0.7391 vs baseline 0.7764, ZH 0.7785 vs 0.8121, rank almost unchanged (AUC-like 0.7250→0.7170 EN). `docs/experiments/prompt_paradigm_runs.md` §v1; `STATE_ARCHIVE.md:497`; root cause in `docs/proposals/prompt_paradigm_v2.md` §0 | **M**, D dominant | Partial: only as an added channel, never as a replacement |
| 2 | PP-v2 Factored Verdict: `score = P_T × P_S`, target-presence times derogatory-stance, both video-grounded | MISS. EN oracle 0.7267, ZH 0.7785, both below baseline. `STATE_ARCHIVE.md:498`; ruling recorded in `docs/proposals/prompt_paradigm_v3.md` §0 (P_S alone on ZH = 0.8121 = ZH baseline oracle exactly) | **M**: operator D, "no separable T/S signal" S | Yes, without the product |
| 3 | PP-v3 Polarity-Calibrated Probes: `σ(0.5·(logit p_E − logit p_C))` over "violates" and "fully consistent" framings | MISS. EN 0.77640 = baseline 0.7764, ZH 0.81208 = 0.8121, both ties not beats; AP1 self-binding fired (prob-avg = logit-fused on EN); Ablation A leak on ZH. `docs/experiments/prompt_paradigm_runs.md` §v3 | **M**, S dominant | Yes, but only with a paraphrase placebo arm |
| 4 | PP-v4 Modality-Split: frames-only call and text-only call, rank-space noisy-OR against train reference | MISS. EN fused 0.7640 vs baseline oracle 0.7764, ZH 0.7987 vs 0.8121; Clause 3 leak (text-only oracle 0.7640 = fused); Clause 5 AP1 violation (prob_avg = rank-nor); **Clause 6 PASS** (corr 0.295/0.493, rescue 54–77%). `docs/experiments/prompt_paradigm_runs.md` §v4 | **D** | No |
| 5 | PP-v5 Per-Rule Disjunction Readout: one constrained decode of K binary chars, score = max per-rule probability | MISS. EN oracle 0.7205, ZH 0.7651; per-rule train variance 8.87e-5 / 9.53e-5 against a pre-registered 0.01 bar; rule-1 monopoly identical on two disjoint rule sets. `STATE_ARCHIVE.md:502` | **M**, kill statistic S, fingerprint D | Low-priority partial |
| 6 | PP-v6 Coarse Axes Prompt: replace the 9/8-rule block with a 2-axis target × hostility statement, one call | **No verdict recorded.** Pipeline still running at director-commanded shutdown, waves 1–3 done, wave 4 as jobs 8211/8212. `STATE_ARCHIVE.md:503`, `:659–662`. `report_v6.json` is not present in this repo | **U** | No, as specified |
| 7 | MS-v1 low-mode-crispness selector: route Otsu vs prior-quantile by `R = σ_lo/(μ_hi−μ_lo)` at τ=0.08 | Withdrawn before implementation. The only shrinkage λ that beats EN ACC (0.7702) regresses mF1 0.6532→0.6513, and λ had no frozen value. `docs/proposals/meta_selector_v2.md` §0 | **S** | Moot, superseded |
| 8 | MS-v2 bucket-Pareto impossibility and rank-stable bucket shift | Self-withdrawn. No bucket-aligned threshold beats both metrics on either dataset; the winning shift is up on EN and down on ZH, "the shift direction is not unsupervised". `docs/proposals/meta_selector_v2.md` §§2, 4–5 | **S** | Claim is already dead on 8B |
| 9 | MS-v3 Unique-Value Otsu: Otsu on `unique(U)`, one confidence state = one unit of evidence | MISS. EN 0.7702/0.6513 (ACC beat, mF1 −0.0019), ZH 0.7450/0.5574 catastrophic. `docs/proposals/meta_selector_v3.md` §3 | **S** | Moot: premise was 2B quantization |
| 10 | MS-v4 GHT sweep: Barron 2020 generalized histogram thresholding, 3575 configs per dataset | 0 strict-both wins on either dataset; EN ceiling is the one-atom-up point, ZH ceiling is exactly the published baseline. `docs/experiments/meta_selector_v4_literature_notes.md` | **S** | Moot |
| 11 | MAD affine-quantile selector `q = 0.60 + 7.83·MAD(pool)` | **RETRACTED as scientifically invalid.** Strict-both came entirely from sub-atom floating-point drift; sub-FP/label concordance EN 61.2%, ZH 52.0% (chance). `docs/experiments/meta_selector_runs.md` "RETRACTION" and job 8019 | **D** | Never |
| 12 | Scanfold selector: pick the threshold method label-free by cluster-quality criterion | **0 of 6 criteria pass.** Three datasets, three different labeled-best methods; on ZH the labeled winner scores silhouette 0.55 against Otsu's 0.84 and KDE-valley 0.04 against 0.93 yet beats it by 5.4pp. `docs/experiments/selector_scanfold_notes.md` | **M**, S dominant | Moot; the discipline lesson survives |
| 13 | Cross-config prompt fusion: 6 prompt configs × 8 fusion operators × subsets ≤4 | H1 passes on exactly 1 cell of 408 (withdef+minimal, logit_avg; EN 0.7702/0.6723, ZH 0.8188/0.7914) at a margin of 1 and 2 videos; **H2 fails, 0 of 10 label-free methods reach it**. `docs/experiments/crossconfig_fusion_notes.md` | **M**, D dominant | No, AP1 by construction |
| 14 | Boundary rescue: second structured-review call on the top-2 predicted-hateful videos nearest the threshold | Claimed strict-beat on 3 datasets, but the margin is **exactly one flipped video per dataset**, and "pure score-rank (no rescue)" alone strict-beats EN at +0.0124/+0.0112. k=10/10 lost on EN (−0.050) and ZH (−0.020). `docs/experiments/boundary_rescue.md` §§Summary, Ablations | **D** | No |
| 15 | Triplet-judge / multi-expert band rescue: entropy-band routing plus 3-MLLM majority | 0 of 336 triplet cells reach 4/4; the eventual 4/4 comes from a single external judge (gemma-27B) selected best-of-8 on test, plus a dataset-tailored IH prompt. `docs/triplet_judge_round_2026_04_20.md`; `docs/label_free_universal_rule_attempts_2026_04_18.md` §"WINNING RECIPE" | **D** | No, AP1 and AP3-adjacent |
| 16 | Label-free flip gates: SCG, G1/G7/G9/G10, DPP, Bayes fusion, OUT, asymmetric flips, quote-count | All fail to transfer. SCG 4/4 on gemma-27B only, 0/4 on Qwen3-VL-8B and llava. Quote-count gate AUROC ∈ [0.31, 0.56] over 32 cells, majority below 0.5. `docs/label_free_universal_rule_attempts_2026_04_18.md` §§3, 10, 11; `docs/triplet_judge_round_2026_04_20.md` §Side tests | **D** | No |
| 17 | MADP Multi-Aspect Definition Probe: `hateful = target_real_group ∧ hostile_framing ∧ ¬protected_context` | 0/4 on Qwen3-VL-8B (EN 0.7640, ZH 0.8121, HM 0.8186, IH 0.8080); on gemma-27B ZH drops 122→113 because "MADP's conservatism blocks too many good flips"; `target_real_group` catches 1 of 4 bad flips and loses ≥2 good ones. `docs/label_free_universal_rule_attempts_2026_04_18.md` §6 and §"Analysis of gemma's 4 ZH bad flips" | **M**: conjunction D, elicitation S | Yes, without the conjunction |
| 18 | EAA v1/v2/v3: evidence-anchored abstention, minimal abstention, devil's-advocate | v1's `evidence` field induced confirmation bias, Qwen3-VL-8B HateMM 0.8419→0.7907; v2 abstention used <5% of the time; v3 produced 46% abstention on gemma-27B. `docs/label_free_universal_rule_attempts_2026_04_18.md` §§7–9 | **D** | No, but the side effect is a binding constraint |
| 19 | TokenSAR gate: relevance-weighted full-span token uncertainty as the flip gate | **Falsified.** AUROC 0.50–0.66 across judges and datasets; ZH/qwen-32B is 0.505, literal chance. Best gated configuration 1/4. `docs/label_free_universal_rule_attempts_2026_04_18.md` §"Attempt: TokenSAR" | **D** | No |
| 20 | Duplex reading (current project): instructional literal-vs-pragmatic divergence `D` | **KILL** on P5a. The thoroughness-matched effort placebo reproduces `D` to +0.0051 (2B) and +0.0027 (8B) against a required 0.05 gap. IM concentrates in the high-lit/high-prag cell at 0.610/0.795, not the predicted 0.117/0.042. `docs/duplex/KILLTEST_POSTMORTEM.md` | **D** | No |
| 21 | Duplex readout (current project): label-free linear direction at the judgment token to recover a suppressed percept | **KILL** on P1 and P2. Probe 0.8520 against raw `z` 0.8751 (Δ −0.0231) where the bar was `z + 0.10`; Δ(IM) − Δ(EX) = −0.0102 against +0.03. `docs/duplex/READOUT_POSTMORTEM.md` | **D**, narrow S caveat | Low-priority partial |
| 22 | MARS reproduction (external, arXiv 2601.15115): 4 calls, describe → assume-hateful → assume-benign → synthesize | Underperforms a single joint call on the same backbone: 0.6584/0.7450/0.6977 against our 0.7640/0.8121/0.8047. The assume-hateful stage inflates positives, 66 on EN against naive's 11. `docs/experiments/naive_baseline_and_mars_reproduction.md` §§Final results, Observations | **M**: bias D, 47–75% parse failure S | Not ours to revive; it is a no-fly zone |

## Row-by-row justification

### 1. PP-v1 Observe-then-Judge — mixed, design dominant

The design put a natural-language bottleneck between a video-grounded observer and a text-only
judge. `docs/proposals/prompt_paradigm_v2.md` §0 names two mechanisms for the failure and the
first one is the load-bearing evidence: observer descriptions repeatedly used abstract stance
words ("mocking", "critical", "satirical") that occur in both benign satire and identity-targeted
derogation, so the judge returned roughly 0.3 where the answer was 0.02 and roughly 0.6 where the
answer was 0.9. That is an information-theoretic loss, not a capability loss. A larger observer
writes a better sentence but the ≤40-token budget and the abstractness of English stance
vocabulary do not depend on model scale, and the judge still never sees the pixels or the audio.
This project's own kill-test reaches the same structure from the other side: `KILLTEST_POSTMORTEM`
§"What survives" names the description bottleneck as the *structural* way to remove evidence from
the input, which is a statement that it does remove evidence.

The substrate-bound part is narrower than it looks but real. The second named mechanism,
calibration drift on a text-only judge with a base "Yes" rate of 0.24 on EN and 0.43 on ZH, was
measured on clipped renormalized probability with Otsu and GMM cuts. Under raw `z` with a KDE
valley the threshold moves with the distribution, so a uniform calibration shift is far less
costly. Against this, the recorded oracle drop of 0.037 on EN with an AUC-like drop of only 0.008
is a genuine localized ranking loss near the decision boundary, not a pure calibration effect,
and it is not explained away by the readout.

**Binding lesson.** A second call must not receive a lossy summary as its only view of the video.
If a description is produced, it must be shown to carry information the raw input does not, and it
must be added to rather than substituted for the evidence.

**Revival.** Partial. Only in the inverted form where the description is an extra channel
alongside the restored transcript and frames, and only with the v1 counterfactual run as an
explicit arm.

### 2. PP-v2 Factored Verdict — mixed, and the phenomenon is alive on the new substrate

Two claims died together and they deserve separate verdicts. The **operator** claim is
design-bound: a product of two sub-probabilities zeroes out positives that are tall on one axis
and moderate on the other, which is arithmetic and transfers to any substrate.
`docs/proposals/prompt_paradigm_v3.md` §0 states this directly, and the worked example in §2 of
that document shows v2 scoring 0.45 where the evidence-only channel says 0.9.

The **phenomenon** claim is substrate-bound and I now believe it was wrong. The director's ruling
recorded in v3 §0 concluded "factor the question into more atomic sub-questions is a dead
mechanism class for this task at 2B", with the supporting observation that the 2B baseline is
already near its per-call oracle ceiling. That ceiling was a property of a model that says Yes to
1064 of 1283 videos at its own boundary (`c2_fullcorpus_2b.json`, `diagnostic_zero`). More
decisively, the current substrate's own error anatomy shows the target/stance axis has unfinished
work: `c2_fullcorpus_8b.json` `error_anatomy_at_method_threshold.false_positives` records that 83
of 105 false positives carry a surface cue against 0.5331 among true negatives (Fisher
`p = 3.02e-07`), with 75 carrying an identity marker, 24 violence and 15 explicit hate discussion.
`READOUT_POSTMORTEM` §"What survives" names this in plain language: "That is topic-versus-stance
confusion, not a readout defect either." The half of the error budget v2 was designed to attack is
still there and is now measured.

**Binding lesson.** Never combine independently-elicited axes with a hard AND. Any factored design
must beat both the best single axis and the joint single call at the same operating point.

**Revival.** Yes, with the aggregation replaced. Minimal decisive retest below in section (iii).

### 3. PP-v3 Polarity-Calibrated Probes — mixed, substrate dominant

The recorded kill is weak on inspection. Both oracles tied the baseline to four decimal places
rather than falling below it, and the diagnostic that carried the interpretation was that "Call 2
compliance probe is near-saturated (EN pos_mean 0.895 / neg_mean 0.776) and adds noise, not
orthogonal signal" (`docs/experiments/prompt_paradigm_runs.md` §v3). A class separation of 0.12 in
clipped renormalized probability space at 2B, where `KILLTEST_POSTMORTEM` records 99.9% of logits
on a 0.25 grid and 21 distinct values, is exactly what a censored instrument produces from a
signal that may be perfectly healthy in raw logit space. The AP1 self-binding clause (prob-space
average matching logit-space fusion on EN) fired at n=161, where the two orderings coinciding is
unremarkable. And the entire comparison chain for v3 was later found corrupt: `STATE_ARCHIVE.md`
lines 500 and 517–540 record that v3's Call 1 deleted a 45-character sentence from the user
message, shifting 78% of EN and 73% of ZH scores with a maximum single-video drift of 0.2151, and
that v3's own Ablation A integrity check was silently violated.

Against revival there is a real design-bound objection that does not depend on the substrate. Two
calls over identical input differing only in the polarity of the question is structurally close to
prompt ensembling, and this project has since shown that a prompt-difference statistic can be
entirely reproduced by a matched placebo (`KILLTEST_POSTMORTEM` P5a: +0.0051 and +0.0027) and that
a pure paraphrase shifts the median score by one 0.25 logit quantum in every group.

**Binding lesson.** Any two-prompt differencing scheme must ship a length-matched, effort-matched
paraphrase placebo as a pre-registered arm, and the effect must exceed the wording-noise floor.

**Revival.** Yes, conditionally. See section (iii).

### 4. PP-v4 Modality-Split — design-bound, the cleanest in the record

This is the row that most deserves to stay dead, because the ablation that would have excused it
passed. Clause 6 confirmed the motivating phenomenon empirically: rank correlation between the
visual and text specialists was 0.295 on EN and 0.493 on ZH, well under the 0.7 bar, and
cross-rescue rates ran 54–77% against a 30% bar. Variable modality carriage is real in this data.
The mechanism still lost, and the reason recorded in `docs/experiments/prompt_paradigm_runs.md`
§v4 is structural: "The MLLM's own cross-modal attention is a stronger fusion operator than
post-hoc rank-noisy-OR." A larger model has better cross-modal attention, so 8B strengthens the
thing that beat the design. Clause 3 leaked (text-only oracle 0.7640 equals fused 0.7640) and
clause 5 fired (prob-average matches noisy-OR on both), meaning neither the split nor the specific
aggregator was load-bearing.

This project reached the same conclusion independently on a different mechanism.
`KILLTEST_POSTMORTEM` §"What survives" records that in the middle tercile of `s_prag`, `s_prag`
alone separates hateful from normal at 0.665 (2B) and 0.771 (8B) while the two-reader difference
`D` reaches only 0.534 and 0.651, and states the generalization: "One well-specified reading is
worth more than the difference between two readings."

**Binding lesson.** Do not deliberately blind a call and then recombine post hoc. Any second call
must add information the first call could not have had, not withhold information the first call
did have.

**Revival.** No.

### 5. PP-v5 Per-Rule Disjunction Readout — mixed

The kill statistic and the diagnostic fingerprint point in different directions. The fingerprint
is strong and design-bound: rule 1 held ten times the mass of rules 2..K on EN and three times on
ZH, and `STATE_ARCHIVE.md:502` notes the monopoly was identical across two completely different
constitutions, which cannot be a fact about hate content. That is decode-position geometry.

The number that fired the kill is contaminated. Per-rule variance of 8.87e-5 and 9.53e-5 was
computed on `P('1')/(P('1')+P('0'))` at each decode position, that is, the same renormalized
two-way probability the readout postmortem retired, on a model whose logits sit on a 0.25 grid.
Variance of order 1e-4 in that space is roughly what quantization alone predicts. A retest in raw
per-position logit space is cheap and would separate the two explanations.

The reason not to prioritize it is elsewhere. `c2_fullcorpus_8b.json` `oracle_gap` is −4.3e-05:
the label-free threshold on the current score already sits at the label-peeked optimum. There is
no headroom on the threshold side of the same forward pass, and v5 is a readout manipulation of
the same forward pass.

**Binding lesson.** A readout-side mechanism must state, before running, how much headroom exists
between the current label-free operating point and the oracle on the same scores. On the current
substrate that headroom is zero.

**Revival.** Low-priority partial.

### 6. PP-v6 Coarse Axes Prompt — unresolved, and the pre-registration is void

There is no Gate-2 verdict anywhere in this repo. `STATE_ARCHIVE.md:503` records the pipeline as
"running at director-commanded shutdown" with waves 1–3 complete and jobs 8211/8212 still in
flight; lines 659–662 confirm those jobs were not cancelled and that `report_v6.json` was expected
to appear post-shutdown. It is not in this repo, and `results/` here contains only the current
project's four directories.

Independently of the missing numbers, the pre-registration cannot be honoured. Its clause 4
required v6 to strict-beat "v3 p_evidence" at EN 0.7702 / ZH 0.8188, and `STATE_ARCHIVE.md:500`
establishes that this target was never a valid comparison because it was scored on a different
prompt. Its motivating phenomenon, the v5 per-rule variance, is the contaminated statistic
discussed above. A design whose anchor and whose bar are both void is not a pending result; it is
a retired design.

**Binding lesson.** A prior-art bar must be verified byte-identical to the artefact it claims to
reproduce before it is used as a gate.

**Revival.** No, as specified. Rewriting the policy block is in any case a prompt-engineering
lever with no hateful-video-specific mechanism left standing behind it.

### 7–10. The meta-selector family — substrate-bound, and already revived without a retest

MS-v1, MS-v2, MS-v3 and MS-v4 are one mechanism class: choose a label-free threshold on the 2B
`binary_nodef` score. `docs/experiments/meta_selector_runs.md` records roughly 120 pilots, and the
cumulative result at line 170 is "0 strict-both unified hits across 22 qualitatively-different
feature families, including 623808 individual rule-variants in job 8113 alone". The GHT session
(`docs/experiments/meta_selector_v4_literature_notes.md`) added 3575 configurations per dataset of
the most general published histogram-thresholding framework and reported zero.

The reason is stated correctly in that same document and it is a statement about the substrate,
not about thresholding: "the frontier is a property of the quantized score surface, not of the
threshold criterion". `docs/proposals/meta_selector_v3.md` §1 quantifies it: about 40 atoms, with
6 atoms absorbing over 60% of the EN pool and 5 absorbing over 75% of the ZH pool.

That surface no longer exists. On the current substrate the same label-free recipe is not merely
viable, it is optimal: `c2_fullcorpus_8b.json` gives a KDE valley at −4.854 producing macro-F1
0.8796 against a label-peeked oracle of 0.87956, gap −4.3e-05, with a relative trough depth of
0.6004 and zero bootstrap resamples failing to find a valley. The 2B counterpart in
`c2_fullcorpus_2b.json` gives an oracle gap of 0.0172, trough depth 0.0066 and 271 of 1000
resamples with no valley. The impossibility result was an artefact of a degenerate score
distribution, and the current project has already demonstrated the revival by construction.

**Binding lesson (survives).** Never target a gain that lives between adjacent score atoms or
inside floating-point noise. This is now doctrine, not preference, and it is enforced in the
current project's own analysis by reporting bf16 tie rates (`READOUT_POSTMORTEM` records 0.74% of
IM-versus-NH pairs tied at bf16 resolution).

**Revival.** Moot. Do not import the impossibility claim into this project.

### 11. MAD affine-quantile rule — design-bound, never revive

`docs/experiments/meta_selector_runs.md` "RETRACTION" is unambiguous: at whole-atom granularity
the rule scores EN 0.7702/0.6513 and ZH 0.7919/0.7577, both failing, and the reported strict-beat
came entirely from sub-atom floating-point drift, with within-cluster sub-FP/label concordance of
61.2% on EN and 52.0% on ZH. Fifty-two percent is a coin.

This is the purest placebo in the record and its lesson is fully substrate-independent. The
current project applies the same discipline to itself: `READOUT_POSTMORTEM` volunteers that when
the permutation placebo is given the same freedom over layers that the sweep enjoyed, its 95th
percentile rises to 0.9049 above the observed peak of 0.8938, so "no layer-shopped version of this
result would survive its own placebo".

**Binding lesson.** Every selected hyperparameter, layer, cell or judge must be given to its own
placebo with the same degrees of freedom.

### 12. Scanfold selector probe — mixed, substrate dominant

Zero of six pre-registered non-self-referential criteria recover the labeled-best threshold method
on all three datasets. The interpretation recorded in
`docs/experiments/selector_scanfold_notes.md` is explicitly a calibration claim, not a geometry
claim: on ZH the labeled winner sits almost on top of the negative mode with silhouette 0.55
against Otsu's 0.84 and KDE-valley depth 0.04 against 0.93, and wins anyway by 5.4pp, "because the
2B model crushes Chinese hateful content's Yes-probabilities toward zero. This is a calibration
issue, not a separation issue."

Model under-confidence crushing the positive class toward the negative mode is precisely the 2B
pathology this project has measured and left behind. On the 8B C2 distribution the two modes sit
at −17.74 and +12.66 with a trough at 40% of mode density, so geometry and labels have no reason
to disagree, and the −4.3e-05 oracle gap shows they do not.

**Binding lesson (survives).** A per-dataset choice of threshold method, prompt or definition is
label-peeking unless the choice rule is itself label-free and frozen in advance. The honest
label-free baseline recorded in that document is 5.37pp ACC and 18.3pp macro-F1 worse on ZH than
the number the project had been quoting. The current project's single frozen KDE-valley recipe
complies; nothing should be added that reintroduces a per-dataset switch.

### 13. Cross-config prompt fusion — mixed, design dominant

One cell of 408 per dataset produced an oracle strict-beat on both datasets, at a margin of one
video on EN and two on ZH, with no replication on any other pair, triple or quadruple and under
only one of eight fusion operators. `docs/experiments/crossconfig_fusion_notes.md` reports 816
atom-level evaluations. A single hit at that multiplicity, at that margin, with no replication, is
a selection artefact, and that reasoning transfers to any substrate. The document is honest that
H2 failed outright: no label-free method reaches the cell, the closest being renyi at exactly
baseline ACC and 0.0069 below baseline macro-F1.

Separately, fusing six prompt framings of the same query is multi-prompt pooling and is forbidden
outright by anti-pattern 1 and the two-call cap.

**Revival.** No.

### 14. Boundary rescue — design-bound

`docs/experiments/boundary_rescue.md` presents a strict-beat on three datasets. Read the numbers
and the claim dissolves. The mechanism flips exactly one video per dataset, three videos across
525 test videos, and every flip is in the same direction. `k_above = 2` was not derived; the
ablation ladder in the same document shows k=10/10 losing on EN (−0.050 ACC) and ZH (−0.020), and
the k=0/2 configuration is the fourth configuration tried. The rescue prompt uses a different hate
definition per dataset, which the document concedes is "the only per-dataset variability". Most
damaging, the document's own diagnostic reports that pure score-rank with no MLLM call at all
strict-beats EN at +0.0124/+0.0112, better than the full mechanism, so the second call is not
load-bearing where the gain is largest. The ZH comparison target was also unstable: this document
pins ZH baseline at 0.8121 while `docs/triplet_judge_round_2026_04_20.md` lists the 2b stage-1 ZH
protocol baseline as 0.7919.

**Binding lesson.** A mechanism whose effect size is one to three samples, selected after a search
over its own hyperparameter and prompt, is a selection artefact regardless of substrate. Effect
sizes must be reported against bootstrap intervals; the current project's own C2 AUC interval is
[0.9387, 0.9608], which is the scale a real gain must clear.

### 15–16, 18–19. The flip-gate family — design-bound, and jointly a real negative result

Across `docs/label_free_universal_rule_attempts_2026_04_18.md` and
`docs/triplet_judge_round_2026_04_20.md`, the project established something worth carrying
forward. The oracle diagnostic shows every tested judge had enough information to pass the target
on all four datasets, and states the conclusion directly: "The bottleneck is NOT judge capability
— it is our inability to label-free-ly distinguish which of a given judge's flips are correct from
which are wrong." Every signal tried failed: stage-1 posterior (good and bad flips overlap),
verdict-token logprob (`p_chosen ≈ 1.0` constant on Qwen3-VL-8B), gemma's Yes/No log-ratio
(inverted on ZH, good median 18.75 against bad median 21.75), MADP sub-fields, rationale length,
quote count (AUROC 0.31–0.56 over 32 cells, majority below chance), and TokenSAR (AUROC 0.50–0.66,
0.505 on ZH/qwen-32B).

This project independently confirmed the same negative from the inside. `READOUT_POSTMORTEM`
§"What survives" records that a label-free linear direction estimated from high-purity anchors
"recovers the verbalized signal and stops there": 0.9548 against 0.9518 on EX versus NH, 0.9077
against 0.9150 on IM versus NH, 0.8520 against 0.8751 on the dismissed stratum. Neither the text
the model writes nor the activations it computes at the judgment token separate its correct
judgments from its incorrect ones any better than the score it already reports.

**Binding lesson.** Self-assessed confidence is not an available signal in this project, from any
readout, at any level. Do not propose a second mechanism whose gate is the model's own certainty.

The hand-crafted variants (SCG, G1–G10, quote-count) add a second lesson. Their failure to
transfer is documented: SCG reaches 4/4 on gemma-27B alone and 0/4 on Qwen3-VL-8B and llava, and
the quote-count rule that reaches 4/4 on gemma was found by grid search over `(thr_to1, thr_to0) ∈
{0..3}²` on test data and reaches 1/4 on both other judges. The document labels it "ILLEGAL" and
records it "to document the sin and the non-transfer". Non-transfer across judges is the fingerprint
of engineering on one model's generation style.

### 17. MADP — mixed, and the closest prior art to a topic-vs-stance mechanism

MADP is the most direct precedent for anything in the target/stance family and it must be
confronted head-on by any new proposal. It elicits three booleans in one call
(`target_real_group`, `hostile_framing`, `protected_context`) and combines them as
`target ∧ hostile ∧ ¬protected`.

Two things make its death partly substrate-bound. It was run on 300-character truncated
transcripts, and its axes were read as generated Yes/No text rather than as raw logit differences,
so each axis was quantized to a binary with no graded score. But the crucial fact is that MADP was
*not* a 2B result: it scored 0/4 on Qwen3-VL-8B, the current model, at EN 0.7640, ZH 0.8121,
HM 0.8186, IH 0.8080. The model is not the excuse here.

What is design-bound is the operator, and the recorded failure mode is identical to PP-v2's.
On gemma-27B, ZH drops from 122 to 113 correct because "MADP's conservatism blocks too many good
flips", and `target_real_group` catches one of four harmful flips while losing at least two good
ones. Two independent falsifications of a hard conjunction over independently-elicited sub-answers,
four months apart, on different models, is enough to treat the operator as dead.

**Binding lesson.** The target and stance axes may be elicited, but they may not be conjoined as
booleans and they may not be multiplied. Any combination must be graded, must be monotone in each
axis, and must be shown to beat the joint single call.

### 20. Duplex reading — design-bound

The kill fired on a placebo clause, which is the strongest kind of kill available. Telling the
model to read harder moved the divergence statistic as much as telling it to read for conveyed
meaning: 0.5047 against 0.4996 on the 2B and 0.6069 against 0.6042 on the 8B, where the
pre-registration required the placebo to sit 0.05 below. Both models. The fingerprint confirms it:
IM videos land in the high-literal, high-pragmatic cell at 0.610 and 0.795 rather than the
predicted low-literal, high-pragmatic cell at 0.117 and 0.042, and the literal reader gives IM a
median `s_lit` of 0.269 on the 8B against 1.1e-07 for NH. The instructed reader reads the implicit
hate it was told to ignore.

`KILLTEST_POSTMORTEM` states the general form: "A successor that needs the model not to see
something must enforce that structurally, through a bottleneck that removes the evidence from the
input, rather than by asking the model to ignore it."

**Binding lesson.** Instructions cannot suspend comprehension. Any design requiring the model to
withhold or restrict its own reading must do so by removing evidence from the input, and must
carry an effort-matched placebo.

### 21. Duplex readout — design-bound, with a narrow substrate caveat

The probe recovered the verbalized ranking and stopped: 0.8520 against raw `z` 0.8751 on the
dismissed stratum where the bar was `z + 0.10`, and Δ(IM) − Δ(EX) = −0.0102 against a required
+0.03. Three of the six clauses passed, so the direction is real, stable (mean cosine 0.9661 over
100 resamples) and not an anchor artefact (empirical `p = 0.01`); it is simply redundant. The
postmortem bounds its own claim honestly: one token position, one prompt, one linear family, one
model.

The narrow substrate caveat is worth stating because it is easy to miss. The probe ran on C0
input, before channel restoration. Its dismissed stratum of 78 IM and 552 NH was partly an
artefact of starved input, and channel restoration subsequently flipped 57 of the 110 dismissed
hateful videos. So the probe's headline contrast was computed on a stratum that no longer exists
in the same form. Against this, the full-sample contrasts (EX versus NH 0.9548 against 0.9518, IM
versus NH 0.9077 against 0.9150) show the same redundancy and are not stratum-defined, and
restoring the channel feeds both the internals and the mouth the same new evidence, so redundancy
is the expected outcome. The caveat licenses curiosity, not a retest.

### 22. MARS reproduction — external, and a warning about the family

MARS is not our mechanism but it is the most complete published instance of the observe-then-judge
family, and reproducing it produced a directly relevant negative. Four calls per video (objective
description, assume-hateful evidence gathering, assume-benign evidence gathering, meta-synthesis)
scored 0.6584 / 0.7450 / 0.6977 on EN / ZH / HateMM against a single-call 0.7640 / 0.8121 / 0.8047
on the same backbone. The document identifies the mechanism: the assume-hateful stage biases the
positive class, producing 66 positive predictions on EN against the naive baseline's 11.

The 47–75% Stage-1 JSON parse failure rate is a 2B format-following artefact and is correctly
labelled as such in the document, so the absolute numbers are not the point. The point is that a
structured adversarial decomposition, faithfully implemented, lost to one prompt, and that its
loss came from a stage whose job was to argue one side.

**Binding lesson.** A decomposition that asks the model to argue for a side inherits that side's
bias into the final verdict. Symmetric decompositions do not cancel this; MARS is symmetric and
still positive-biased.

## Closing section

### (i) The substrate-independent hard lessons, ranked

Ranked by how much of the design space each one closes and by the strength of the recorded
evidence.

**1. Post-hoc recombination of deliberately narrowed calls loses to the model's own fusion inside
one call.** PP-v4 is the decisive case because its motivating phenomenon passed its own test
(correlation 0.295/0.493, cross-rescue 54–77%) and the mechanism still lost, isolating the failure
to the recombination structure. `KILLTEST_POSTMORTEM` reaches the same conclusion on a different
mechanism: `D` adds nothing over `s_prag` in the ambiguous tercile (0.534/0.651 against
0.665/0.771). This closes every design of the form "blind two calls, then combine".

**2. Hard conjunctions and products over independently-elicited sub-answers destroy more true
positives than false ones.** PP-v2 (`P_T × P_S`, EN oracle 0.7267) and MADP
(`target ∧ hostile ∧ ¬protected`, 0/4 on Qwen3-VL-8B; gemma ZH 122→113) are two independent
falsifications, four months and several model scales apart, with the same recorded failure mode.

**3. Any statistic built on the difference between two prompts must clear a matched placebo.**
The effort placebo reproduced the duplex divergence to +0.0051 and +0.0027 on two models; the
layer-shopped permutation placebo in the readout probe reaches a 95th percentile of 0.9049 above
the observed 0.8938. A paraphrase alone shifts the 2B median by one 0.25 logit quantum in every
group.

**4. Self-assessed confidence is not an available gating signal, at any level of the stack.**
TokenSAR AUROC 0.50–0.66 with 0.505 on one cell; verdict logprob `p_chosen ≈ 1.0` constant; quote
count AUROC 0.31–0.56 with the majority below chance; gemma's ZH log-ratio inverted. And from
inside the model, the linear probe recovers the verbalized ranking and never exceeds it (0.8520
against 0.8751). Text-level and activation-level attempts failed independently.

**5. Free-text bottlenecks between calls lose the discriminating detail.** PP-v1's observer
descriptions collapsed benign satire and identity-targeted derogation onto the same abstract stance
vocabulary, costing 0.037 and 0.034 of oracle accuracy while leaving rank almost untouched.

**6. Gains of one to three samples, or gains obtained after selecting `k`, a threshold, a judge, a
fusion cell or a layer against test metrics, are selection artefacts.** Boundary rescue (one flip
per dataset after four configurations); cross-config fusion (1 cell of 408, margin 1–2 videos);
the MAD rule (sub-atom drift, 52.0% ZH concordance); the quote-count rule (grid search over 16
threshold pairs, 1/4 on the other two judges).

**7. Adding an output field to a judge prompt is not free; it moves the verdict distribution.**
EAA v1's `evidence` field dropped Qwen3-VL-8B on HateMM from 0.8419 to 0.7907; RTG target-first
scored 1/4 and target-after-verdict 2/4; CONS exclusion reminders scored 1/4. Eliciting a target
or a stance changes the answer even when nothing else changes.

**8. A per-dataset choice of method, prompt or definition is label-peeking unless the choice rule
is itself label-free and pre-registered.** Zero of six criteria recover the baseline's own
(EN→Otsu, ZH→GMM) pairing; the honest label-free baseline is 5.37pp ACC and 18.3pp macro-F1 worse
on ZH.

**9. Never target a gain that lives between adjacent score atoms or in floating-point noise.**
Whole-atom discipline is doctrine after the MAD retraction and the six phantom cells caught by
atom quantization in the cross-config probe.

**10. State every method against the best available reading of the baseline it claims to beat, and
verify byte-identity of any reproduced baseline.** The v3 prompt-deletion bug shifted 78% of EN
and 73% of ZH scores while passing an ablation that was supposed to catch exactly that;
`READOUT_POSTMORTEM` generalizes it: "A hypothesis about a hidden channel should be stated against
the best available reading of the visible one."

### (ii) The topic-vs-stance / observe-then-judge neighbourhood, and what a new proposal must show

Five prior designs sit inside this family, and a new proposal has to be distinguishable from all
five, not just from the one it most resembles.

| Prior design | What it occupied | How it died |
|---|---|---|
| PP-v1 | Observe (video) → judge (text-only) cascade | Text bottleneck; abstract stance words merge satire and derogation |
| PP-v2 | Target-presence × stance-valence, both video-grounded, product | AND-gate compression of one-tall positives |
| MADP | `target_real_group ∧ hostile_framing ∧ ¬protected_context` in one call | Conjunction too conservative; blocks more good than bad |
| MARS (published) | Describe → pro-hate evidence → anti-hate evidence → synthesize | Positive-class bias from the assume-hateful stage |
| MATCH-HVD (published) | Hate-evidence agent + non-hate-evidence agent + judge | Occupied, not falsified by us; see (iv) |

A new proposal in this family must show all seven of the following, and it should state them as
pre-registered arms rather than as post-hoc defences.

**a. That it is not a text bottleneck.** Whatever the first role emits must either be added to the
raw input rather than substituted for it, or be shown to carry information the raw input does not.
The v1 counterfactual (second call sees only the summary) must be run as an arm and must lose.

**b. That the combination is graded and monotone, never a conjunction or a product.** It must beat
the best single axis on its own and it must beat the joint single call, at the same operating
point, on raw `z`. This is what PP-v2 and MADP both failed.

**c. That it beats the current one-call number under the same frozen recipe.** The bar is
`c2_fullcorpus_8b.json`: macro-F1 0.8796, AUC 0.9505 with bootstrap 95% [0.9387, 0.9608], at the
self-computed KDE valley of −4.854, one call per video. A gain inside that interval is not a gain.

**d. That the gain cannot come from the threshold.** The oracle gap on the current scores is
−4.3e-05. There is nothing left on the threshold side. Any proposal that improves the score
distribution must say so explicitly and must not claim credit for threshold effects.

**e. That it attacks the located error budget and states which half.** The 105 false positives are
all NH, 83 of them carrying a surface cue against 0.5331 in true negatives (`p = 3.02e-07`), 75
carrying an identity marker: this is topic-versus-stance confusion and it is the half a stance
mechanism should move. The 49 false negatives are 87.8% IM and include 7 of the 8 videos Step 0
proved have no transcribable speech: no stance mechanism can reach those, and a proposal must
pre-commit to not costing them. A proposal that predicts a generic macro-F1 rise without naming
which of these two piles moves is not falsifiable.

**f. That it survives an effort-matched and verbosity-matched placebo second call.** The placebo
must be the same length, the same output schema and the same nominal thoroughness, differing only
in the named role. PP-v3 never had one; the duplex reading kill-test did, and that is why it died
cleanly in a day.

**g. That eliciting the axis did not simply move the verdict.** Given the EAA and RTG results, the
proposal must report the marginal shift its extra elicitation induces on the judge's own score
distribution, separately from the claimed gain, so that a distribution shift is not mistaken for
new information.

### (iii) The most substrate-bound deaths, ranked by cheapness and promise of a retest

**Rank 0, already revived, no retest needed: the meta-selector family (rows 7–10, 12).** The
impossibility claim was a property of a score surface with about 40 atoms. The current substrate's
own numbers refute it: KDE-valley macro-F1 0.8796 against oracle 0.87956 (gap −4.3e-05), trough
depth 0.6004, 0 of 1000 bootstrap resamples without a valley, against the 2B's 0.0172 gap, 0.0066
trough depth and 271 of 1000 failures. Cost: zero. Action: do not import the impossibility claim
into any new pre-registration, and do not re-run any of the 120 pilots.

**Rank 1, cheapest live retest: PP-v3 polarity probes (row 3).** The kill rested on the compliance
probe reading pos_mean 0.895 against neg_mean 0.776 in clipped probability space at 2B, which is
the signature of a censored instrument rather than of an absent signal. Minimal decisive retest:
score `train_clean` once with the deflected framing under raw `z` on the restored C2 input, and
ask whether `z_deflected` adds AUC over `z_prag` beyond a length-matched paraphrase placebo of the
same framing. Cost by the PREREG estimate (662 videos in roughly 7 minutes on the 8B) is about
15 minutes per arm, three arms. Caution: this is a two-call design that lives one step from
prompt ensembling. The placebo arm is what makes it legal, and without it the retest should not be
run at all.

**Rank 2, highest promise: the target/stance axis, retested without the conjunction (rows 2 and
17).** Unlike everything else on this list, its motivating phenomenon has been independently
re-measured on the new substrate and is present at `p = 3.02e-07`. Minimal decisive retest: elicit
a stance-axis `z` and a topic-axis `z` on the same restored input and ask two questions. First,
within the high-topic stratum, does stance-`z` separate the 105 NH false positives from the true
positives that share their surface markers? Second, does any graded monotone combination beat the
single joint `z` at its own KDE valley? A negative on the first question kills the family outright
and is worth knowing; a positive on the first with a negative on the second reproduces the PP-v4
lesson and also kills it. Neither outcome is ambiguous. Note the two-call cap is binding, so topic
and stance cannot both be extra calls on top of the existing judge.

**Rank 3, cheap but low prior: PP-v5 per-rule readout (row 5).** Re-reading the K decode positions
in raw logit space instead of clipped renormalized probability is nearly free and would settle
whether the 1e-4 variance was quantization. The prior is poor: the rule-1 monopoly replicated
identically across two disjoint rule lists, and the oracle gap leaves no headroom on the readout
side of the same forward pass.

**Rank 4, do not retest as designed: PP-v1 (row 1) and the duplex readout (row 21).** v1 is worth
revisiting only in the inverted form described in (ii)(a). The readout probe's C0 stratum caveat
is real but its full-sample redundancy result is not stratum-dependent, and a rerun would most
likely reproduce it.

**Not retestable at all:** rows 11, 13, 14, 15, 16, 18, 19, 20, 22. Placebo exploits, ensembles,
selection artefacts and placebo-killed mechanisms do not become true on a better substrate.

### (iv) What the published baselines already occupy: the novelty no-fly zones

Read from `docs/baseline_briefs/`. These are published methods reproduced in the predecessor
project, so any proposal that lands inside one of them is a reproduction, not a contribution.

**Observe-then-judge with adversarial dual-stance is occupied twice over.** MARS
(`naive_baseline_and_mars_reproduction.md` §Part B, arXiv 2601.15115, "Training-Free and
Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning") runs objective
description, assume-hateful evidence, assume-non-hateful evidence, then meta-synthesis.
MATCH-HVD (`match_hvd.md` §Scope) runs a hate-evidence agent (stage 2a), a non-hate-evidence agent
(stage 2b, the same runner with `ifhate` flipped) and a judgement stage (2c) that adjudicates the
two evidence sets and emits a label-free yes/no verdict. Between them, "two opposing-stance
evidence calls plus a synthesizing judge" is fully published and training-free.

**Target identification by attribute probing, then classify, is occupied by Pro-Cap.**
`procap.md` §"8 VQA probes" lists the upstream probes verbatim: race, gender, animal presence,
person presence, nationality, which animal, disability, religion. That is the target axis of a
target-versus-stance factorization, elicited by probing a frozen VLM and consumed through a text
bottleneck. `procap_v3.md` upgrades the captioner to a 16-frame video model and adds the
supervised RoBERTa head. Pro-Cap v3 scores 0.8700/0.8700 on ImpliHateVid
(`docs/results_2026_04_16_v2.md`), so this is also a strong baseline, not just an occupied idea.

**Re-asking conditioned on a disclosed prior verdict is occupied by LoReHM.** `lorehm.md`
§"Upstream pipeline" describes the RSA branch: when the retrieval-derived label disagrees with the
basic prediction, the model is re-asked with a prompt that discloses the disagreeing label. The
predecessor project's boundary rescue was structurally the same idea and died as a selection
artefact.

**Label-free harmful-content detection via confidence readout plus agent self-improvement is
occupied by ALARM.** `alarm.md` §Pipeline describes a Label stage that reads output logits for a
confidence score, an embedding stage, a retrieval stage, a pairwise Experience stage that compares
two items side by side, and a reference stage that uses retrieved pairs as in-context examples.
The framing, "label-free harmful meme detection via LMM agent self-improvement" (KDD 2026),
overlaps this project's own framing directly. It also collides with anti-pattern 3 (retrieval over
a task-relevant pool) and anti-pattern 1 (multi-stage agent), so it is doubly out of bounds.

**Retrieval-augmented and modular-adapter approaches are occupied by LoReHM and Mod-HATE.**
`mod_hate.md` describes LoRAHub-style black-box composition over hate-speech, meme-caption and
hate-explanation modules with a K-shot labeled support set. Both require task-relevant external
supervision and are forbidden here regardless of novelty.

**Temporal localization of the hate-bearing segment is partly occupied.**
`match_hvd_2b5_jina_clip.md` describes MATCH-HVD stage 2b.5, which aligns 32 frames to transcript
segments with Jina-CLIP-v2 and picks the best-matching frame per evidence sentence. A proposal
that localizes where in the video the offending signal sits should expect this as prior art.

**What is left unoccupied.** After removing the five zones above, the space that remains for a
second mechanism in the target/stance neighbourhood is roughly: operating on the *input* rather
than on the judgment, in a way that changes what evidence reaches a single frozen judge, and
reading the change in that judge's raw `z` rather than in a second judgment. That is the shape of
the one mechanism this project has confirmed rather than killed, channel restoration, which moved
AUC from 0.9334 to 0.9505 and flipped 57 of 110 dismissed hateful videos while moving 18 of 552
NH. A stance mechanism built in that shape would be a counterfactual over the input, not a second
opinion over the output, and it would not be any of the five published designs above.
