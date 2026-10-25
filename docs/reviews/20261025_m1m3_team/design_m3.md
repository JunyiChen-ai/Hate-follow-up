# M3 designer: proposals for the video level (2026-10-25)

Written by the M3 designer of the M1 / M3 novelty team. No repo file was edited, no GPU job was run, and no hash or
checksum was computed. All scripts and outputs are in `scratchpad/m1m3_novelty/work_m3/` (scripts `*.py`, outputs
`out/*.txt`). Metrics come only from `src/eval/evaluate.py` (`pooled`, `within_video_macro`, imported in
`work_m3/lib.py`). Every number is development-selected: it was computed on test, and test GT was read for analysis
(rule 10). The read log is in §5.

**Reference** (the current method, raw key, as in the charter): HateMM .8955 / .6883 / .7508, HateClipSeg (HCS)
.7137 / .6666 / .6373, DeHate .6994 / .1568 / .6431 (pooled ROC / pooled PR / within). Sources:
`runs/20260926_twolevel/final_rawkey/metrics.json` and `runs/20260927_dehate_external/r6_rawkey/metrics.json`.
`work_m3/p0_reference.py` reproduces all nine numbers exactly from the predictions files.

## Summary

- **Ranked proposals:**
  1. **Act-span verdicts.** Ask the reader the whole-video verdict on growing beginnings and shrinking endings of the
     video. The act's start and end are where those answers change.
  2. **Counter-verdict reading.** Read every window also under the opposite of the model's own answer. The video key
     then rests on the evidence that survives that opposite answer.
  3. **Segment-pooled video node** (the "one model" idea). Make the video node the top level of the M2 segment model,
     with window evidence counted per segment. **I ran its CPU pre-check and it fails.**
- **None of the three is likely to pass rule 14g on both corpora.** The reasons are in §1 and in each proposal's
  estimate. Proposal 1 is the only one whose design can carry extent information, which is what any HCS gain needs.
  Its Stage-0 check costs about 2 GPU minutes.
- **Two findings matter more for the lead than the proposals:**
  - **§1.2.** Against the raw-key reference, the existing product composition (the method's interval output) gains
    pooled PR on both corpora: HateMM +.025, HCS +.011. Its HateMM ROC is −.0051, just past the .005 floor. It is not
    novel, and it needs P(V) as a probability.
  - **§1.3.** The window-mean term of the key does its work as an extent term, not as evidence for presence. This
    supports the paper's M3 sentence ("orders videos by how much of them the act covers") with an ablation.

## 1. Facts that constrain every M3 proposal

### 1.1 Presence is decided by the verdict; extent is what is missing

`work_m3/out/p0_reference.txt`, `p0_saturation.txt`. "Presence" means the video has at least one GT-hateful frame.
"Share" means the GT-hateful frame fraction, used among hateful videos only.

| | HateMM | HCS |
|---|---|---|
| verdict Yes / videos; GT-hateful | 146 / 215; 85 | 108 / 118; 103 |
| non-hateful videos with verdict Yes | 64 | 10 |
| video AUC for presence: z_video / key K / window mean | .935 / .927 / .890 | .794 / .803 / .802 |
| ρ(share): K / z_video / window mean | .372 / .190 / .416 | .450 / .321 / .533 |
| videos with P(V \| K) > .95 (calibrated key) | 57 % | 86 % |

Consequences:

- **HCS.** Presence is saturated: 103 of 118 videos are hateful, and 86 % of videos have P(V | K) > .95. An element
  that changes only presence cannot move HCS pooled. An HCS gain must come from extent (share) ordering or from the
  within-video order.
- **HateMM.** The verdict already ranks presence best: z_video .935 against K .927. A principled presence posterior
  reaches .941 (§1.3), yet loses pooled. A HateMM gain must therefore come from one of three places:
  - share ordering among its 85 hateful videos. The headroom between "perfect presence" and "perfect presence plus
    true share order" is +.053 / +.199 (`experiments/20260928_headroom/README.md` A1 vs A2);
  - the edge negatives of its long-hate videos (`experiments/20261014_within_diagnosis/README.md`);
  - interleaving frames of different videos (§1.2).

### 1.2 Finding: the product composition against the raw-key reference (not a proposal; not novel)

`final_full` is the method's own interval output: frame score = log P(V = 1 | K) + log P(h_t = 1 | V = 1), where
P(V | K) comes from the label-free two-Gaussian mixture on K. The 2026-10-25 M3 round compared it with the calibrated
key and recorded a HateMM ROC loss of .007. Against the raw-key reference (`work_m3/out/p0_reference.txt`):

| | HateMM | HCS | DeHate |
|---|---|---|---|
| final_full − current, pooled ROC / PR | −.0051 / **+.0250** | **+.0254 / +.0113** | −.0047 / −.0009 |

- **Rule 14g.** Going back to the additive key costs pooled PR .025 and .011, so this component passes 14g on PR on
  both main corpora.
- **G1.** It misses G1 by .0001 on HateMM ROC.
- **Novelty.** It is the two-part / hurdle form (Cragg 1971; STPN; Cai & Sun 2009), so it has no novelty.
- **Calibration.** It needs P(V) as a probability. That means the two-Gaussian mixture the user removed from the
  additive key on 2026-10-07. Whether that is acceptable inside a product is the user's decision.
- **Why it moves pooled.** P(h | V = 1) interleaves frames of different videos better than "key first, then rank".
  The cost comes from the 64 HateMM false-Yes videos, which get P(V) ≈ 1.

### 1.3 Finding: the window-mean term is an extent term, not presence evidence

From proposal 3's CPU run (`work_m3/out/pG_segment_re.txt`):
- When window evidence is counted per segment, the model's own video log-odds becomes a copy of the verdict: Spearman
  with z_video .991 (HateMM) and .957 (HCS).
- On HateMM it ranks presence slightly better than K (.941 against .927). It loses the share ordering (ρ .179 against
  .372) and loses pooled.
- So the window mean helps the key through extent, not through presence. That is the empirical basis for the story
  sentence "orders videos by how much of them the act covers". The story designer can cite it.

## 2. Proposals (ranked)

### Proposal 1 — Act-span verdicts (rank 1)

**1. Name and mechanism.** The reader answers only the question it answers well: the unchanged whole-video verdict.
It answers it on growing beginnings [0, c_j] and shrinking endings [c_j, D] of the video (J = 8 cuts, c_j = jD/8).
- The act has started by c_j if the beginning up to c_j already violates.
- The act is still running after c_j if the ending from c_j violates.
- The act's span is where both hold.

It replaces the **window-mean term of the key** ("key arm"). The span can also be passed to the within-video order
("joint arm"). Formulas, declared, with no fitted constant except the C47–49 scale rule:
- a_j = running maximum of the beginning reads up to cut j, with a_J = max(z_video, all beginnings);
- b_j = running maximum of the ending reads from cut j to the end, with b_0 = max(z_video, all endings);
- part k = [c_{k−1}, c_k] lies in the span with probability π_k = [σ(a_k)/σ(a_J)] · [σ(b_{k−1})/σ(b_0)];
- extent E = mean_k π_k;
- key arm: K_B = z_video + s · logit E, where s = sd(window mean) / sd(logit E) over the corpus (the fixed rule of
  candidates 47–49);
- joint arm: frame score = K + centred rank(logit p_t + log π_{k(t)}), with the r6 key and r6 M2 posterior p_t
  unchanged.

**2. Named failure of the current form.**
- **The verdict says whether, not how much.** ρ(share) of z_video is .190 / .321.
- **The window mean inherits window errors.** It is a classify-and-count estimate (.416 / .533), and every window's
  topic error goes into it.
- **The two cheaper fixes already failed:**
  - asking for the share directly collapses onto the verdict. In candidate 47, 142 of 146 verdict-Yes HateMM videos
    answered "more than three quarters" with p > .99, and ρ with z_video was .965 / .904;
  - reading pieces alone loses context. Candidate 49's quarters gave ρ(share) .278 / .419.
- **What is missing on HateMM is the act's start and end.** 37 of 84 within-eligible HateMM videos are at least 75 %
  hateful, and 94 % of their negatives lie in the first or last 10 % (`experiments/20261014_within_diagnosis/README.md`).
- **Truncation keeps context.** A beginning keeps everything before its cut; an ending keeps everything after it.

**3. Closest prior work.**
- **lit_m3 has no entry for this operation.**
  - T3AL and FreeZAD (A1, A2) use the video level only to select the class.
  - ASOT (C11) obtains the share as a transport marginal of per-frame costs. Here the extent is read from verdicts on
    truncated inputs.
- **FV-Action, "Your VLM Already Knows When" (arXiv 2608.08315; I read pp. 1–5).**
  - It asks a coarse-to-fine scan of per-clip questions ("Does this clip show {query}? Answer Yes or No."), takes the
    peak, and emits a fixed-width window.
  - Those are per-clip reads, the same family as r6's window reads. It does not use verdicts on truncated videos.
- **Iterative visual prompting (Wake et al., arXiv 2408.17422).** Search snippet only, not opened. It narrows a window
  around the start and the end with frame tiles.
- **Deletion and insertion curves (RISE, Petsiuk et al., BMVC 2018).** Recalled, not opened. They remove or insert
  input progressively and track a classifier score, but they evaluate saliency maps; they do not estimate extent.
- **Hateful video.** Not found (two web searches on 2026-10-25 plus lit_m3). This is a search boundary, and the rule-4
  reviewer must search again.

**4. Closest past attempts and closed families.**
- **Candidate 49** (quarters read alone, archived item 39): P1 .278 / .419.
  - Its first quarter equals this design's first J = 4 beginning, and its last quarter equals the last J = 4 ending.
    That is why J = 8 is declared: it keeps more context per read and gives a finer resolution.
  - The other reads differ: the beginnings and endings are cumulative, and the reads are combined as onset and offset
    (order statistics), not counted.
- **Eraser** (leave one window out) and **CVA** (exclude one interval verbally): removal of one window. Here input is
  truncated from the ends.
- **HVL state chain**: sequential persistence ("continue" in 77–82 % of windows). Here persistence is what makes a
  beginning stay Yes after the act starts.
- **Closed-family check:**
  - not M3-a (no new question; the inputs change);
  - not M3-c (the context is cumulative, not segment-alone);
  - not M1-e (it is not the verdict's dependence on one window).

**5. Rules and cost.**
- **Rule 3.** One frozen MLLM and one question; no ensemble; no per-corpus constant. s is a fixed corpus-statistic
  rule.
- **Rule 4.** No target-task precedent found. The expected objection is "only an input layout". The answer: the
  contribution is the onset / offset estimator, not the prompt.
- **Rule 5.** The truncated inputs are new reads. Their cost is reported below, and no input novelty is claimed.
- **Rule 13.** One J and one formula for every corpus.
- **Rule 14g (key arm).** K_B against the r6 key must gain ≥ .01 on one pooled metric on both corpora. z_video alone
  is also reported.
- **Rule 14g (joint arm).** Removing log π must cost ≥ .01 within on both corpora.
- **Where a HateMM gain could come from:** share order among its hateful videos, and the edge negatives. The bounds
  (`out/pB2_bound.txt`, `pB4_within_bound.txt`) use an oracle that says Yes exactly when the truncated piece holds a
  GT-hateful frame:
  - extent ρ(share): J = 8 gives .757 / .730, against the window mean .416 / .533;
  - joint-arm within: J = 8 gives +.092 (HateMM) / +.067 (HCS).
- **Cost:**
  - 7 beginnings and 7 endings per video, about 7 full-prefix prefill equivalents;
  - at candidate 49's measured rate (≈ .4 s per full-prefix equivalent on a 5090; 316 s for 333 videos with 2.37
    equivalents each), that is ≈ +2.8 s per video;
  - HateMM + HCS ≈ 16 min, DeHate ≈ 63 min;
  - the current reading costs ≈ .6–1.6 s per video, so per-video GPU time roughly triples. CLAUDE.md asks for this to
    be justified, and the justification is that full-context video questions collapse onto the verdict and
    context-free pieces lose context;
  - J = 4 costs 3 equivalents (≈ +1.2 s per video);
  - the code exists: `src/prefix_questions.py` `subset_inputs` / `read_subset_verdict` with t1 = 0 or t2 = D.

**6. Pre-check.**
- **CPU, run now.**
  - **J = 4 edges with real reads** (`out/pB3_edges.txt`): candidate 49's first and last quarters are both Yes in 65 %
    (HateMM) and 78 % (HCS) of verdict-Yes videos. Among hateful videos that is 78 % / 79 %, against an oracle 66 % /
    73 %. So at J = 4 most videos tie at extent 1.
  - **Stand-in: the declared formulas on candidate 49's context-free quarters** (`out/pB5_standin.txt`,
    `pB5b_standin_rankform.txt`):

    | | ρ(E, share), window mean in brackets | key arm pooled, vs current | joint arm within, vs current |
    |---|---|---|---|
    | HateMM | .305 [.416] | −.017 / −.043 | +.0065 |
    | HCS | .374 [.533] | −.027 / −.037 | +.0019 |
    | DeHate | — | — | +.0051 |

    The stand-in reads see only their quarter, unlike the proposal's beginnings and endings. So these numbers say
    "not supported by the nearest cached reads", not "refuted".
- **GPU Stage 0** (GT-free; 20 videos per main corpus; ≈ 2 min including model load). Kill if PB1 fails on both main
  corpora.
  - **P0**: the full-span read reproduces z_video within 0.05. Cache and plain-call reads differ within the bf16
    tolerance used for candidate 49 (|dz| < 1).
  - **PB1**: among verdict-Yes videos, the first-eighth beginning and the last-eighth ending are both Yes in ≤ 60 %.
  - **PB2**: adjacent beginnings drop by more than 2 logits in ≤ 20 % of pairs.
- **GPU Stage 1** (all 333 videos, plus DeHate for reporting).
  - GT-free: PB1, and PB3: ρ(logit E, z_video) < .90.
  - **P1**: ρ(E, share | hateful) > .416 / .533 on both corpora, and the partial ρ given the window mean > 0 on both.
  - Gates:
    - G1: no metric below the current method beyond the noise floor;
    - G2 (14g): as in item 5;
    - selection order: key arm, then joint arm. If only the joint arm passes, label it "selected between two declared
      arms".

**7. Story paragraph.**

> Ordering videos by how much of them the act covers needs the act's extent. A frozen reader answers "whether"
> reliably and "how much" badly. Asked directly for the share, it answers "more than three quarters" for nearly every
> video it calls violating. Pieces of the video read alone lose the context that makes a moment part of the act. The
> naive estimate counts the windows read as violating, and so inherits every window's topic error. We ask the reader
> only the question it answers well, on growing beginnings and shrinking endings of the video. The act has started by
> time t if the video up to t already violates. It is still running after t if the video from t on still violates.
> Each truncated verdict keeps all the context on one side of its cut, and the act's span is where the two answers
> overlap (*act-span verdicts*). Following the deletion and insertion curves of saliency evaluation (Petsiuk et al.,
> 2018), we use progressive truncation, but as an estimator of the act's extent rather than a test of an explanation.

**8. Honest estimate.**
- **Novelty:** moderate. The estimator appears new for this task as far as searched; its ingredients (truncation,
  verdict, order statistics) are known.
- **Chance of passing 14g on both corpora:** key arm about 10 %, joint arm about 15–20 %.
- **Main risks:**
  - beginnings and endings saturate at Yes: the J = 4 edge reads are already Yes for most videos;
  - endings lack the early context. That is the context that made candidate 49 lose on HCS.
  - per-video GPU cost roughly triples.

### Proposal 2 — Counter-verdict reading (rank 2)

**1. Name and mechanism.** Every window branch is also read after the forced opposite of the model's own one-word
answer: "No" when z_video > 0, else "Yes". The key's window term becomes the mean read under that opposite answer, so
the windows can revise the verdict instead of echoing it.

It replaces the **window-mean term of the key**, which is now computed under the model's own answer. M1's window reads
and M2 stay as they are. Arms:
- additive raw key K_A = z_video + mean_w ℓ_opp(w);
- product: P(V | K_A) · P(h | V = 1). This needs P(V) as a probability (see §1.2).

**2. Named failure of the current form.**
- **The verdict is in every window's context, so a wrong Yes inflates its own windows.** The windows of false-Yes
  normal videos rise by 3.16 / 2.90 logits (`experiments/20261002_verdict_analysis/README.md` §4).
- **HateMM has many false Yes.** 64 of its 146 verdict-Yes videos are non-hateful.
- **The cascade passes information one way.** The verdict goes into the windows, and they come back only as their mean
  (McDonald et al. 2007: "Valuable information is passed in both directions").
- **The proxy shows the windows' dependence on the verdict marks wrong verdicts.** This is CPU, `out/pA_stance_contrast.txt`.
  The no-stance read stands in for half of the counter-verdict.
  - Among verdict-Yes videos, a smaller lift (stance read minus no-stance read) means more likely GT-hateful: AUC
    .661 (HateMM), .731 (HCS), .585 (DeHate). K's AUC there is .896 / .758 / .676.

**3. Closest prior work.**
- **McDonald et al., ACL 2007** (lit_m3 C1, read §1–§2): the document label and the sentence labels are decoded
  jointly.
- **ConCoRD and Maieutic prompting** (EMNLP 2022, lit_m3 E6, abstracts only): the model's beliefs are revised by
  consistency between them.
- **MARS (ICASSP 2026) and RAMF** are target-task precedents. They read the same media under a hateful and a benign
  assumption and combine the reasoning into the video label (verified in
  `docs/reviews/20261003_m1_marginalizer_proposal.md`).
- **Exact difference:** here the opposite assumption is applied to the local reads, and only the evidence that
  survives it enters the video key. The local reads used for the time level stay one-sided, and no reasoning text is
  generated.
- **Hateful video:** the dual-assumption mechanism has been used for detection.

**4. Closest past attempts.**
- **Candidate 4, Marginalizer:** average each window over both forced answers. It was stopped by rule 4 on MARS /
  RAMF, and the review says a new candidate "needs a different substantive operation on evidence beyond enumerating
  and aggregating opposite verdict conditions".
- **08-10 answer contrast:** the judge's hidden states under forced Yes vs No gave a leading direction with Spearman
  .983 against z.
- **verdict_analysis:** stance vs no stance.
- **HVL revision:** the verdict re-asked after a summary; it restated itself.
- **Closed families:** this is outside M3-a and M3-e because it adds new reads. It sits next to the Marginalizer STOP.

**5. Rules and cost.**
- **Rule 4: STOP is likely.** The video key would come from reads under the opposite assumption, which is close to
  MARS's assumption-based video decision.
- **Rule 14g on HCS is blocked structurally.** HCS presence is saturated (§1.1).
  - In product form the three keys give HCS .7380–.7382 / .6780–.6797 (`out/pA3_product.txt`).
  - In additive form, the verdict-free window term loses HCS share ordering: ρ(share) falls from .444 to .404, and
    pooled moves −.011 / −.010 (`out/pA2_keys.txt`, old-ASR pair).
- **Where a HateMM gain could come from:** presence among the 146 verdict-Yes videos. The headroom from perfect
  presence is +.031 / +.047 (`experiments/20260928_headroom/README.md` A1).
- **Proxy pooled numbers** (old-ASR pair; linear extrapolation to the opposite answer, so indicative only):

  | | HateMM | HCS | DeHate |
  |---|---|---|---|
  | product with the extrapolated key, vs current additive | −.002 / +.023 | +.025 / +.012 | +.000 / +.007 |
  | the counter-verdict element alone (product with it vs without it) | +.006 / −.001 | .000 / +.002 | — |

- **Cost:**
  - one more branch per window and modality on the cached prefix; the opposite answer is a turn of about 5 tokens, and
    no new prefix prefill is needed;
  - about +40–60 % of per-video read time;
  - HateMM + HCS ≈ +2–5 min, DeHate ≈ +10–15 min.

**6. Pre-check.**
- **CPU:** done above. It cannot be completed on CPU, because no forced-opposite read is cached.
- **GPU** (one extra window pass):
  - GT-free: the median within-video sd of ℓ_own − ℓ_opp is > .5 logit, and ρ(mean ℓ_opp, z_video) < .9;
  - GT: AUC(GT hateful | verdict Yes) of K_A is greater than that of K (.904 / .743 with the fixed-ASR reads,
    `out/pF_video_sets.txt`) on both corpora; ρ(share) is not more than .05 below the window mean;
  - gates G1 and G2 as for proposal 1.
- **Run it only if** the lead wants the information (one cheap pass) and accepts the rule-4 risk.

**7. Story paragraph.**

> The verdict enters every window's read, so the windows of a video the reader wrongly calls violating are read as
> confirming it: such windows rise by about three logits. A cascade cannot revise its verdict, because the evidence
> that should test it was produced under it. We read each window a second time under the opposite answer and let the
> video level rest on the evidence that survives (*counter-verdict reading*). Following fine-to-coarse structured
> models (McDonald et al., 2007), information then passes in both directions between the video and its moments.

**8. Honest estimate.**
- **Novelty:** weak to moderate, with a likely rule-4 STOP.
- **Chance of passing 14g on both corpora:** at most 5 %. HCS is blocked by presence saturation.
- **Main risk:** like DVD's target read, it may follow the hate definition. That loses on HCS protocol GT and gains on
  DeHate. The proxy already shows this pattern.

### Proposal 3 — Segment-pooled video node (rank 3; CPU pre-check done: not supported)

**1. Mechanism.** The video node becomes the top level of the segment model itself. Window reads share their
segment's level (a random effect per segment), so a segment, not a window, is the unit of evidence. The model's own
P(V = 1 | verdict, reads) then replaces the key and its separate calibration.

**2. Named failure.**
- Counting every window as independent evidence makes the joint video node overconfident: the EM evidence is 5–14×
  the true slope, and the Bayes node used as the intercept gave .8230 / .5576 (`experiments/20260926_twolevel/README.md`
  §9, §10.2).
- Reads of one segment are correlated: .50 / .59 between adjacent windows after removing the video mean.

**3. Prior work.**
- Segmental HMMs with random segment levels (Russell 1993; Holmes & Russell 1999; Kim & Smyth, JMLR 2006). Recalled;
  Kim & Smyth is cited in the revisable-prior record.
- The mixed HMM (Altman, JASA 2007; lit_m3 C7) has a random effect per sequence.
- Not used for hateful video.

**4. Closest attempt.**
- **The revisable prior** (archived): one shared offset per video, in the verdict and in every window. The offset
  absorbed the presence signal, and pooled fell to .569 / .360 and .514 / .490.
- **Difference:** here the effects are per segment and on windows only. A random level can explain a high stretch
  only as long as one segment lasts, and the verdict observes V directly.

**5–6. Pre-check (run).** `work_m3/pG_segment_re.py`; output `out/pG_segment_re.txt`; criteria
`pG_declared_criteria.txt`, written before the results.
- **Model:** a simplified one-chain semi-Markov model on the 8 s windows, with the max read normal-scored. The hate
  minimum is 2 windows and the mean length is 10 windows. EM is label-free.
- **Brute-force check:** the semi-Markov forward-backward matches brute-force enumeration to 1.8e-15
  (`pG_selftest.py`).

| | HateMM | HCS |
|---|---|---|
| fitted levels: s² (window) / τ² gap / τ² hate | .104 / .925 / .426 | .233 / .426 / .354 |
| C1: video AUC of the model's log-odds vs K | .941 vs .927 (pass) | .783 vs .803 (**fail**) |
| ρ(video log-odds, z_video) | .991 | .957 |
| C3: video node × r6 M2 posterior (product), vs current | −.013 / −.017 (**fail**) | +.023 / +.020 |
| video log-odds + r6 rank (additive), vs current | −.006 / −.022 | −.030 / −.029 |
| C4: same product without segment pooling | .8614 / .6279 (pooling +.021 / +.043) | .7355 / .6664 (+.001 / +.020) |

C2 passes formally: each τ² < (μ_h − μ_g)². The window noise s² becomes small, however, so the segments become short.

**Verdict: not supported** (C1 fails on HCS, C3 fails on HateMM).
- Counted correctly, the windows add almost nothing to presence beyond the verdict.
- The node loses the share information that the window mean carries. This is finding §1.3.
- Its HCS product gain is the §1.2 product effect, not the node.

**7. Story.** As in item 2 above. It would only work with an extent read (proposal 1) as an added observation of the
node.

**8. Estimate.** Do not pursue as M3 novelty. The one useful output is the §1.3 ablation fact.

## 3. Dropped ideas (with the number that drops each)

1. **Per-video random effect in the segment model** (lit_m3 top 1). This is the archived 2026-10-02 revisable prior:
   regime video ROC .637 / .625 against .933 / .806, and pooled .569 / .360 and .514 / .490. Its segment-level variant
   is proposal 3, which also failed.
2. **Shared null from pseudo-benign videos, share = 1 − π̂0** (lit_m3 item 3; C2FPL; conformal p-values).
   - The scout's Storey π0 with a verdict-No null gave .205 / .398.
   - Conformal p-values against a reference set are the same estimator family.
   - The pseudo-benign set is chosen by the verdict, which is circular.
3. **Share as a corpus-solved transport variable** (ASOT; unbalanced OT from 2026-08).
   - Without the temporal term, the solution is soft classify-and-count with one corpus threshold, shrunk toward a
     constant.
   - Over temperatures and corpus mass, ρ(share) is ≤ .418 on HateMM (window mean .416) and ≤ .564 on HCS (.533)
     (`out/pE_ot_share.txt`).
   - It is a re-composition of the same reads (M3-e).
4. **Video sets gating local models** (Kukleva 2019; BP-AR-HMM) (`out/pF_video_sets.txt`).
   - Label-free sets of 2 or 3 follow the key: ρ .75–.91.
   - AUC(hateful | verdict Yes) is .64–.84 against K's .90 (HateMM) and .66–.69 against .74 (HCS).
   - ρ(share) is .18–.40 against .416 / .533.
   - The 09-28 reading-condition groups were inert too.
5. **Threshold-ladder share question.**
   - Candidate 47's η is the mean cumulative logit of this same ladder, read from one distribution: ρ(η, z_video)
     .973 / .958, P1 .228 / .303.
   - The model's belief is "more than three quarters" for 142 of 146 verdict-Yes HateMM videos, so separate Yes/No
     branches would read the same belief.
   - It is M3-a in spirit. If the lead wants it closed formally: about 11 GPU minutes, with the GT-free criteria
     already written (history §3.2 P1).
6. **Verdict support size** (leave-one-window-out mass, attribution spread; history §3.2 P2) (`out/pC_redundancy.txt`).
   - Support fractions give ρ(share) .20–.24 on HateMM. On HCS they give .07 from the Eraser and .23 from attribution.
   - The partial ρ given the window mean is ≤ +.15.
   - It costs about 10× native.
7. **Verdict on windows deleted in order of their reads** (deletion curve) (`out/pD_deletion_bound.txt`).
   - Even with an oracle verdict on the remainder, three deletion levels give ρ .419 / .503 (window mean .416 / .533).
   - Window-level resolution needs one re-encode per window, which is Eraser-scale cost.
8. **Verdict on the complement of M2's inferred act** (ERASER-style comprehensiveness). With an oracle complement
   verdict, ρ(share) is .246 / .331, below the window mean (`out/pH_complement_bound.txt`).
9. **Group testing on random pools of windows.** Not run.
   - Random pools saturate faster than contiguous ones (P(pool hits) ≈ 1 − (3/4)^k).
   - Scattered windows keep less context than candidate 49's quarters, which already lost (.278 / .419).
10. **Verdict-free (no-stance) window mean as the key's window term.** Proxy: HCS −.011 / −.010 pooled
    (`out/pA2_keys.txt`). It is proposal 2's additive arm without the counter-verdict, and it fails on HCS.
11. **Averaging both forced verdicts** (Marginalizer). Stopped under rule 4 (MARS / RAMF); see proposal 2.

## 4. Recommendation to the lead

1. **Run proposal 1's Stage 0** (about 2 GPU minutes, GT-free).
   - If PB1 fails on both main corpora (beginnings and endings saturate), the current reader has no remaining M3 route
     that is not an M1 input change.
   - The in-window-frame reads already raise HCS ρ(share) to .615 (history §1.6). That route is an input change, and
     adoption is the user's call.
2. **Proposal 2** is cheap information (one window pass), but it is likely to be stopped under rule 4 and cannot pass
   14g on HCS. I would not spend a rule-4 review on it unless the user accepts the MARS overlap.
3. **For the user's decision**, independent of novelty:
   - the product composition (§1.2) is the only M3 change in the record that gains ≥ .01 pooled PR on both corpora
     against the raw-key method;
   - it needs a calibrated P(V), and its HateMM ROC is .0001 past the noise floor.
4. **For the story designer:** §1.3 is an ablation fact that supports describing M3's window term as an extent term.

## 5. Test-read log (rule 10) and files

- **GT read (test split only), analysis only, never in any fit:** `data/gt_4fps/{HateMM,HateClipSeg,DeHate}.npz`.
- **Columns derived from GT:** `runs/20261002_verdict_analysis/analysis/per_window.csv` (`gt_fraction`). I also read
  `per_video.csv`, using its `global_z` column.
- **Runs read:**
  - `runs/20260926_twolevel/{final_rawkey,final_m2,final_full}/predictions.jsonl` and `final_rawkey/params.json`;
  - `runs/20260927_dehate_external/{reads_gridA,r6_rawkey}/predictions.jsonl`;
  - `runs/20260926_twolevel/final_dehate/{final_m2,final_full}/predictions.jsonl`;
  - `runs/20260926_glr/base_gridA/predictions.jsonl`;
  - `runs/20260910_spvl/mllm/q3vl-8b/{full,nostance}/predictions.jsonl`;
  - `runs/20261002_verdict_analysis/main_full/predictions.jsonl`;
  - `runs/20260928_infer/dehate/reads_nostance/predictions.jsonl`;
  - `runs/20261025_m3_coverage/{keys,seg_main,seg_dehate}/*.jsonl`;
  - `runs/20261003_m1_eraser/r1_main/{checks.jsonl,base/predictions.jsonl}`;
  - `runs/20261002_m1_attributor/r1_main_fp32_mem/tokens/`.
- **What was found:** §1–§3.
- **What it changed:**
  - proposal 3 was dropped after its declared check;
  - J = 8 (not 4) was declared for proposal 1, after seeing the J = 4 edge saturation in candidate 49's reads and the
    oracle bounds by J;
  - proposal 2 was moved below proposal 1.
- **External paper read:** arXiv 2608.08315 pp. 1–5 (PDF fetched into the session's tool-results folder, read as
  data). Search snippets for arXiv 2408.17422 were not opened.
- **Scripts** (`work_m3/`):
  - `lib.py`, `p0_reference.py`;
  - `pA_stance_contrast.py`, `pA2_keys.py`, `pA3_product.py`, `pA4_window_level.py`;
  - `pB_span.py`, `pB2_bound.py`, `pB3_edges.py`, `pB4_within_bound.py`, `pB5_standin.py`;
  - `pC_redundancy.py`, `pD_deletion_bound.py`, `pE_ot_share.py`, `pF_video_sets.py`;
  - `pG_segment_re.py`, `pG_selftest.py`, `pG_declared_criteria.txt`;
  - `pH_complement_bound.py`.
  - Outputs are in `work_m3/out/`.
