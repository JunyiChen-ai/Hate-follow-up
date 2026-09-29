# Three redesigns of the inference after the reads (declared before any run, 2026-09-28)

User direction (2026-09-28): the method works, but the inference after the reads (M2–M4) is plain. Make it less plain
along its own mechanisms. Gains are not required; no metric may drop, and each redesigned part must still show in the
ablation. All three redesigns run on the cached reads `runs/20260926_glr/base_gridA` (Qwen3-VL-8B, fixed ASR), CPU
only, no new MLLM calls. The reference is the current method `r6_bma` (`runs/20260926_twolevel/final_m2`).

Everything not named below stays as in `r6_bma`: normal-score reads, EM without labels, no stance-leak term, minimum
segment of two reading windows (k = 4), mean lengths averaged per video on [two windows, video length] with a prior
uniform in length (G = 6), calibrated video key, centred within-video rank. Code: new flags of
`experiments/20260926_twolevel/twolevel_r2.py` (as the DVD round did); launch scripts and checks live here.

Closest earlier attempts were checked in `research-wiki/DIRECTIONS.md` before writing this; each section names them.

## 1. Evidence whose reliability depends on what the reader saw (M2)

**Mechanism.** A window read is a measurement whose noise depends on the input the reader had for that window. The
visual branch of a window that holds none of the 20 prefix frames sees no picture of its own; the speech branch of a
window with a few words has little to judge. Today every read of a modality shares one pair of emission
distributions. Here the emission distributions are fitted per reading condition, still without labels.

**Conditions (label-free, from the inputs of the reading run).**
- Visual: the window holds at least one of the 20 uniform frames (`f1`) or none (`f0`). Frame times from the file
  names in `data/frames_k20`. Share of `f1`: HateMM 72.7 %, HateClipSeg 65.7 % (counted before this declaration).
- Speech: the number of words the speech branch was given (the window's transcript slice, `src.video_inputs.window_text`
  on the fixed ASR, exactly as the reading run built it), split at the corpus median (`w0` at or below, `w1` above).
  Medians: HateMM 18, HateClipSeg 15 words. Windows without speech have no speech read, as now.
- The conditions are written once by `window_conditions.py` to `runs/20260928_infer/conditions/<reads>.json`.

**Model change.** Each (modality, condition) pair is an emitter with its own non-hate mean, hate mean and variance,
estimated by EM. The chains, the normal scores (per modality over the corpus) and everything else are unchanged.
With one condition per modality this is exactly `r6_bma` (plumbing check).

**Closest earlier attempts.** K7 changed the reads (a frame per window) and failed its gate; the silent-window rule
forces the speech chain off where there is no speech read. Nobody has fitted the evidence per condition.

**Arms** (`runs/20260928_infer/`): `c1_cond` (both conditions), `c1_vis` (visual condition only), `c1_sp` (speech
condition only), `c1_shuf` (control: the condition labels permuted within the corpus, seed 0, same number of
parameters), `c1_full` (interval output).

## 2. Three read levels: no topic, topic without attack, attack (M3)

**Mechanism.** The window read mostly tracks whether the window talks about the target group; attack adds a smaller
step on top (`experiments/20260912_pwc/README.md` §7c: mentioning the group is worth +8.1 / +7.0 log-odds, being a
true hate window +4.3). So the reads have three levels, and attack occurs inside stretches of topic. The time level
gets three phases per modality chain: 0 = off topic, 1 = topic without attack, 2 = attack. Hate = phase 2. Phase
changes are nested: 0 ↔ 1 and 1 ↔ 2 only; when phase 1 ends it goes to 0 or 2 with equal probability. Each phase
lasts a negative-binomial number of cells (k = 4, as now). Mean lengths: phases 0 and 1 share one mean (non-attack),
phase 2 has its own; both averaged per video on the same grid and prior as `r6_bma`. The emission means per emitter
are ordered at initialisation (10th, 50th, 90th percentile) and fitted by EM; the start distribution over the three
phases is fitted by EM.

**Pre-check (reads the gold; analysis only; `topic_runs_check.py`).** The mechanism needs topic stretches to be longer
than hate stretches. On the 183 videos that have window-level topic reads (`runs/20260912_tad/e0/predictions_topic.jsonl`,
`t` = "does this window mention the target group", older ASR loader, same 8 s grid):
- runs of gold-hate windows (hate share ≥ .5) against runs of high-topic windows (`t` above the corpus median);
- for each gold-hate run, the length of the high-topic run that contains its first window, divided by its own length;
- the mean act read (`a`) in three groups: no topic and no hate, topic and no hate, hate.
Proceed only if, on both corpora, the median of that ratio is ≥ 1.5 and the median high-topic run is longer than the
median hate run. Otherwise this direction is closed here without a model run.

**Closest earlier attempts.** §15 of the twolevel README added a "slip" phase with the same read distribution as hate,
allowed in any video; it failed because short slips and short hate are inseparable. Here the middle phase has its own,
lower, read level and is nested under topic, which is a different identification. TAD subtracted the topic read and
lost; here nothing is subtracted, the topic level is a state.

**Arms.** `l3_m2`, `l3_full`, `l3_free` (ablation: direct 0 ↔ 2 changes allowed, i.e. no nesting; phases change to
either other phase with equal probability). Reported: fitted level means per emitter, share of cells in each phase.

## 3. One segment structure for both modalities (M3)

**Mechanism.** Today each modality has its own chain and the fusion is OR; the earlier shared-chain arm forced the two
modalities into one state and lost .043 within on HateClipSeg. In between: one segmentation per video whose segments
carry a joint label (visual on or off, speech on or off), so the boundaries are shared while each modality keeps its
own state. At a boundary the label changes to any of the other three with equal probability. Segment lengths:
negative-binomial with k = 4; the label 00 uses the gap mean, the other three the hate mean; both averaged per video
as in `r6_bma`. Emissions: the visual read observes the visual bit, the speech read the speech bit, through the same
pair / single window rule as now. Hate = visual or speech bit on.

**Closest earlier attempts.** `final_sharedchain` (one state for both reads) and `final_m2` (independent chains) are
the two ends; both exist and are the ablations.

**Arms.** `j_m2`, `j_full`.

## 4. Decision rule (all three, declared before running)

Noise floor as in rule 7: pooled .005, within .01. Reference `final_m2` (= `r6_bma`).

- **No drop:** on HateMM and HateClipSeg none of the six numbers of the arm's `_m2` variant is below `final_m2` by more
  than the floor. An arm that drops is closed.
- **Mechanism visible (needed for any novelty claim, rule 14g):** the arm's within is above `final_m2` by ≥ .01 on
  both corpora, with a paired bootstrap interval over videos (`experiments/20260927_dvd/analyze_dvd.py`, pooled and
  within). For §1 the shuffle control `c1_shuf` must not reach the same gain. An arm that does not drop but does not
  reach .01 is kept only as a design detail, not a contribution, and is reported as such.
- Only an arm that passes both parts goes to the eight-MLLM check (`runs/20260910_spvl/mllm/<m>/full`, against
  `runs/20260926_twolevel/robust/<m>_r6`; for §1 the conditions are rebuilt with the older ASR loader those reads
  used) and to DeHate (external, not a gate). Rule: within not below `<m>_r6` by more than .01 on ≥ 7 of 8 models per
  corpus.
- No rescue tuning. Grid and prior are those of `r6_bma`.

## 5. Plumbing checks (before any number is read)

1. `twolevel_r2.py --selftest` extended: the three-phase chain and the joint-label chain against brute-force
   enumeration over paths (log-likelihood and per-cell P(hate), ≤ 1e-8).
2. `c1_plumb` (the new code with no conditions and kind `two`) reproduces `final_m2` to 4 decimals on all six numbers.
3. The scoring code never opens a GT file; the pre-check and the analysis do, and are not part of any method.
4. EM log-likelihood monotone (asserted, as before).

## 6. Cost

No model calls. CPU on uoa-lab1. `r6_bma` takes about 40 s per corpus; the three-phase chain has 36 augmented states
against 8 and the joint chain 64, so a few minutes per corpus.

## 7. How to run

```
bash experiments/20260928_infer/launch/run_all.sh
```

## 8. Test-read log (rule 10)

- 2026-09-28, before writing the model code: `topic_runs_check.py` reads `data/gt_4fps/{HateMM,HateClipSeg}.npz` (hate
  share per 8 s window) on the 183 videos with topic reads, for the §2 pre-check only. Results in §9.
- Counts used in §1 (frame share, word medians) are label-free.

## 9. Results (2026-09-28, uoa-lab1, CPU; development-selected)

### 9.1 Pre-check of §2: topic stretches are not longer than hate stretches. Direction closed by its declared rule.

`runs/20260928_infer/precheck_topic/table.txt` (183 videos with topic reads; hate = gold share ≥ .5 per 8 s window;
topic = `t` above the corpus median):

| | HateMM (84 videos) | HateClipSeg (99 videos) |
|---|---|---|
| hate runs, windows q25 / 50 / 75 | 2 / 4 / 9.5 (n 119) | 1.2 / 3 / 5 (n 318) |
| high-topic runs, windows q25 / 50 / 75 | 1 / 2 / 6 (n 177) | 1 / 2 / 6 (n 258) |
| topic run holding the hate run ÷ hate run length, q25 / 50 / 75 | 0 / .67 / 1.5 | 0 / 1.0 / 5.5 |
| hate runs whose first window is not in a high-topic run | 33.6 % | 42.5 % |
| mean act read: no topic and no hate / topic without hate / hate | +0.2 / +12.1 / +11.2 | −2.8 / +2.0 / +4.3 |

- High-topic runs are shorter than hate runs, and a third to a half of hate runs begin outside any high-topic run. So
  the topic read is not a slow stretch that contains hate; it flickers at the window scale.
- On HateMM there is no middle level: windows that mention the group without hate read as high as hate windows. On
  HateClipSeg a middle level exists (+2.0 against +4.3) but is not nested in time.
- The declared criterion (median ratio ≥ 1.5 and median topic run longer than median hate run) fails on both corpora.
  **Direction 2 is closed here.** Deviation from §2, recorded: the three-phase arms (`l3_*`) were still run, after the
  pre-check and only for information, because they cost minutes. Their numbers cannot pass the gate by the
  declaration, whatever they are.

### 9.2 Plumbing

- Self-test (all four chain kinds against brute-force enumeration, the model averaging, the corpus-shared pairs):
  max difference 4.3e-14.
- `c1_plumb` (the new code, no conditions, kind `two`) reproduces `final_m2` exactly: .8971 / .6942 / .7508 and
  .7168 / .6711 / .6373, same EM log-likelihoods (−8621.3714, −8107.3928).

### 9.3 Runs

Sources: `runs/20260928_infer/<arm>/metrics.json`, `runs/20260928_infer/analysis/table.txt` (paired bootstrap over
videos, 4000 draws, seed 0), `runs/20260928_infer/<arm>/run.log` (fitted parameters). `run_all.sh` stopped at `j_m2`
on a bug (the emission array was sized for two previous-level ids; fixed to `n_plv × S`, which does not touch the
two-phase arms); `run_rest.sh` ran the remaining arms and the analysis.

Pooled ROC / pooled PR / within; differences to `final_m2` with 95 % intervals (within):

| arm | HateMM | HateClipSeg | within vs `final_m2`, HateMM; HateClipSeg |
|---|---|---|---|
| `final_m2` (r6_bma) | .8971 / .6942 / .7508 | .7168 / .6711 / .6373 | — |
| §1 `c1_cond` (visual by own frame, speech by word count) | .8971 / .6942 / .7511 | .7169 / .6711 / .6350 | +.0003 [−.003, +.004]; −.0024 [−.014, +.009] |
| §1 `c1_vis` | .8971 / .6942 / .7504 | .7168 / .6711 / .6376 | −.0004; +.0003 |
| §1 `c1_sp` | .8971 / .6943 / .7522 | .7169 / .6711 / .6351 | +.0014; −.0022 |
| §1 `c1_shuf` (control, labels permuted) | .8971 / .6940 / .7497 | .7168 / .6713 / .6385 | −.0011; +.0012 |
| §1 `c1_full` (intervals) | .8910 / .7132 / .7511; F1@.3/.5/.7 .321 / .292 / .232 | .7392 / .6796 / .6350; F1 .239 / .134 / .065 | `final_full`: .325 / .295 / .235 and .246 / .145 / .077 |
| §3 `j_m2` (one segmentation, joint label) | .8969 / .6943 / .7588 | .7163 / .6708 / .6298 | +.0080 [−.016, +.036]; −.0075 [−.018, +.003] |
| §3 `j_full` | .8888 / .7052 / .7588; F1 .330 / .300 / .245 | .7349 / .6767 / .6298; F1 .224 / .133 / .075 | |
| `final_sharedchain` (one state for both reads) | .8969 / .6943 / .7434 | .7148 / .6702 / .5949 | −.0074; −.0425 [−.073, −.013] |
| §2 `l3_m2` (three levels, nested; pre-check failed) | .8963 / .6929 / .6879 | .7169 / .6716 / .6367 | **−.0629 [−.106, −.024]**; −.0006 |
| §2 `l3_free` (three levels, free changes) | .8965 / .6936 / .7217 | .7168 / .6715 / .6391 | −.0291 [−.069, +.007]; +.0017 |
| `final_nocoupling` (reference ablation) | .8961 / .6881 / .6367 | .7144 / .6689 / .5801 | −.1141; −.0572 |

Pooled differences of every arm except `l3_*` are within ±.0005; `l3_m2` pooled ROC −.0008 / +.0001.

**§1, condition-dependent evidence: no drop, no visible mechanism. Closed for a novelty claim; not adopted.**
- All six numbers are within the noise floor of `final_m2`, and the permuted-label control moves the numbers by the
  same amount. So the conditions carry nothing the ranking uses.
- The fitted emitters differ a little: on HateClipSeg the speech read of a window with few words separates hate from
  non-hate less (levels −.93 / +.32 against −.52 / +.86 for many words; slope 2.4 against 2.9); on HateMM the visual
  read of a window without its own frame has a lower non-hate mean (−.55 against −.85). These shifts change the
  posterior of a few windows but not the order inside videos.
- Not adopted: it adds parameters and a preprocessing step (word counts, frame times) for no measured effect.

**§2, three read levels: fails. Closed (by the pre-check, and confirmed by the run).**
- HateMM within −.063 (interval excludes 0); HateClipSeg unchanged. Free changes (`l3_free`) lose less (−.029) but
  still lose.
- What EM does with the middle phase: it puts its mean near 0 on the normal-score scale (HateMM visual levels
  −1.17 / +0.01 / +1.16), which triples the evidence slope (3.8 → 9.9), and it assigns 43–49 % of the cells of
  violating videos to the middle phase. Only the top phase counts as hate, so much of the hate stretch is called
  "topic without attack". This is the round-1 failure again (EM makes the evidence too strong and the persistence
  stops mattering), reached through a third level instead of a geometric chain.

**§3, one segmentation with a joint label: no drop, no visible mechanism. Closed for a novelty claim; not adopted.**
- HateMM within +.008, HateClipSeg −.0075, both intervals include 0; pooled unchanged.
- It sits between the two existing ends: independent chains (`final_m2`) and one shared state (`final_sharedchain`,
  −.043 on HateClipSeg). Sharing only the boundaries costs HateClipSeg a little, where the two modalities' reads are
  the most independent, and gains a little on HateMM.
- Fitted joint-label shares under V = 1 (HateMM): 00 .35, 01 .16, 10 .17, 11 .32; HateClipSeg: .28 / .20 / .21 / .30.
  A third of the hateful time is carried by one modality alone, which is why forcing one state loses.

### 9.4 Conclusion

None of the three redesigns of the inference passes the mechanism rule, and one (three levels) drops. The current
method `r6_bma` stays. Together with `experiments/20260928_headroom` this says the same thing from the model side:
on the current reads, changing how the reads are combined moves within by less than the noise floor, unless the
change makes the evidence stronger than the reads warrant, in which case it drops. A more elaborate inference on
these reads is possible but does not show in the ablation, so it cannot be claimed.

Code: `twolevel_r2.py` now describes a chain by its phases (`KINDS`), carries the previous cell's level id in the
augmented state, and fits emitters per (modality, condition). With the defaults it is the same model and reproduces
`final_m2` exactly (§9.2).

## 10. DeHate (external, not a gate; declared 2026-09-29 before running)

User request (2026-09-29): run the three redesigns on DeHate too, on a machine other than uoa-lab1. Same flags as §9,
reads `runs/20260927_dehate_external/reads_gridA` (1151 videos, `experiments/20260927_dehate_external`),
reference `runs/20260926_twolevel/final_dehate/final_m2` (r6_bma on DeHate: .7011 / .1582 / .6431). Arms: `c1_cond`,
`c1_shuf` (control), `j_m2`, `l3_m2`, and `c1_plumb` (must reproduce `final_m2`). Conditions rebuilt for the DeHate
reads with the fixed ASR loader (the DeHate reads used it). Launch `launch/run_dehate.sh`; output
`runs/20260928_infer/dehate/`, analysis `runs/20260928_infer/dehate/analysis/table.txt` (paired bootstrap over
videos). Nothing is decided on DeHate; the numbers are reported next to the main-corpus ones.

### 10.1 Results (2026-09-29, uoa-lab3 = sc474398, CPU; copied back to `runs/20260928_infer/dehate/`)

Sources: `runs/20260928_infer/dehate/<arm>/metrics.json`, `dehate/analysis/table.txt` (paired bootstrap over 1151
videos), `dehate/<arm>/run.log`. Conditions on DeHate: 84 % of windows hold a frame, word median 20.

| arm | DeHate pooled ROC / PR / within | within vs `final_m2` |
|---|---|---|
| `final_m2` (r6_bma) | .7011 / .1582 / .6431 | — |
| `c1_plumb` | .7011 / .1582 / .6431 | 0 (exact) |
| §1 `c1_cond` | .7012 / .1582 / .6444 | +.0013 [−.004, +.009] |
| §1 `c1_shuf` (control) | .7012 / .1582 / .6439 | +.0009 [−.000, +.002] |
| §3 `j_m2` | .7009 / .1579 / .6488 | +.0058 [−.007, +.019] |
| §2 `l3_m2` | .7010 / .1583 / .6268 | −.0163 [−.045, +.013] |

Pooled differences are all within ±.0003. The picture is the one of §9 on the two main corpora:
- condition-dependent evidence: nothing, and the permuted control moves the same amount;
- joint segmentation: a small within gain (+.006) with an interval that includes 0, pooled unchanged;
- three levels: a loss (−.016), same fitted shape as before (middle level near 0, slope 9.1, 45 % of the cells in the
  middle phase).

Across the three corpora, the joint segmentation is +.008 / −.0075 / +.006 and never outside its interval; the
condition-dependent evidence is within ±.003 everywhere; the three-level chain loses on HateMM and DeHate. The
conclusion of §9.4 stands.

## 11. Two M1 ablations on DeHate: joint branch, and branches that see each other (declared 2026-09-29 before running)

User request (2026-09-29): the two reading-module ablations discussed for the main corpora, run on DeHate. Both were
measured on HateMM / HateClipSeg in the SPVL era only (`experiments/20260910_spvl/README.md` §9; `research-wiki/DIRECTIONS.md`
A2 "分支互相可见"): a joint branch instead of dual costs HCS within .028 (HateMM +.002); branches that can see each
other cost .034 / .009. Neither has a DeHate number, and neither was re-measured under the fixed ASR loader or the
current time level.

**Reads (GPU, uoa-lab2, `experiments/20260922_til/til_measure.py` with two new flags; everything else as
`runs/20260927_dehate_external/reads_gridA`: grid A, 20 frames, fixed ASR loader, stance turn, evidence wording).**
- `reads_joint`: `--branches joint`. One branch per window that shows the window's transcript and asks whether the
  window is one of the segments where the violating content occurs (the SPVL `joint` + `evidence` wording). The model
  still sees the whole prefix (all frames, full transcript, its own verdict). Read key `z_joint`; the time level then
  has one chain.
- `reads_seq`: `--branches dual --isolation sequential`. The same two branches per window as now, but run in order
  (window 1 visual, window 1 speech, window 2 visual, ...) on one cache that keeps every branch's tokens, so branch i
  sees the question text and assistant header of every earlier branch (no answers are generated, as in the SPVL
  causal-mask arm). The isolated reads deep-copy the prefix cache per branch instead.
- Cost: the isolated DeHate reads took 1.3 s / video (1341 videos, 28 min); the joint arm has half the branches, the
  sequential arm the same number without the cache copies. About 1 h of GPU in all.

**Time level (CPU, same machine):** `r6_bma` flags on each read set, arms `m1_joint`, `m1_seq`; reference `final_m2`
(the isolated dual reads). Analysis `analyze_dvd.py --datasets DeHate`, paired bootstrap over videos, in
`runs/20260928_infer/dehate/analysis_m1/table.txt`. Also reported: the window-level agreement between the read sets
(Spearman of `z` per video) from the predictions, label-free.

Nothing is decided on DeHate (external). The question is only whether the two main-corpus findings (a single joint
answer loses what the picture carries; branches that see each other drift) show up on a third corpus.

### 11.1 Results (2026-09-29, uoa-lab2 = sc474399; copied back to `runs/20260928_infer/dehate/`)

Reads: `reads_joint` 1341 videos, 0 errors, 0.8 s / video (17 min); `reads_seq` 1341 videos, 0 errors, 1.2 s / video
(27 min). The whole-video verdict is identical to the isolated reads in both (verdict Spearman 1.000: same prefix,
same in-place first branch), so every difference below comes from the window reads.

Time level `r6_bma` on each read set (`analysis_m1/table.txt`; pooled ROC / pooled PR / within; paired bootstrap over
videos, 4000 draws, arm minus `final_m2`):

| arm | DeHate | vs `final_m2` (95 % interval) |
|---|---|---|
| `final_m2` (isolated dual, current) | .7011 / .1582 / .6431 | — |
| `m1_joint` (one joint branch per window) | .7031 / .1565 / .6700 | +.002 [−.010, +.013] / −.002 [−.018, +.010] / **+.027 [−.001, +.056]** |
| `m1_seq` (dual, branches see earlier branches) | .6859 / .1554 / .6123 | **−.015 [−.025, −.006]** / −.003 [−.013, +.009] / **−.031 [−.060, −.001]** |

Read agreement with the isolated dual reads (label-free, `analysis_m1/reads_agreement.txt`; per-video Spearman of the
window score `z`): joint median .80 (q25 .57, q75 .92), 12.2 % of windows change sign; sequential median .73 (q25 .50,
q75 .87), 20.7 % change sign. EM evidence slope: joint chain 3.08; isolated visual / speech 3.65 / 2.85.

**Correction to the §11 preamble.** The main-corpus number quoted there for branches that see each other (.034 / .009)
compared the SPVL causal-mask arm with the SPVL-r2 base; the causal-mask arm was measured on round-1 SPVL (joint
branch, rules question, no stance turn) and against its own base it is HateMM −.014, HateClipSeg +.011
(`experiments/20260910_spvl/README.md` §9, row "causal mask" vs row "SPVL"). The two corpora point in opposite
directions there. DeHate is the first measurement of visibility under the current reads (dual + evidence + stance).

**Reading.**
- Joint branch vs dual, within, three corpora: HateMM +.002, HateClipSeg −.028, DeHate +.027 (interval lower bound
  −.001). The dual branch is not a robust gain: only HateClipSeg supports it, and it was already not a claimable
  component under rule 14g (HateMM < .01). The joint branch also halves the branch count. Whether to keep dual is a
  user decision; nothing is changed here.
- Branches that see each other: DeHate −.015 pooled ROC and −.031 within, both intervals excluding 0, with the
  largest read drift of the two arms (21 % of windows change sign, and the share of non-hateful videos with
  P(V) > .5 rises .565 → .591). Visibility helped on no corpus and hurts on DeHate. Isolation stays.

## 12. Every leave-one-out ablation of `r6_bma` on DeHate (declared 2026-09-29 before running)

User request (2026-09-29): the §11 run covered only two reading-module ablations; the request was the whole
leave-one-out table that was given for the main corpora (each component removed alone, everything else as in
`r6_bma`). Main-corpus sources: `experiments/20260926_twolevel/README.md` §20.1 (time level, video and within terms),
§14.4 (reading components under `r3`), `experiments/20260910_spvl/README.md` §8–§9 (fixed windows, isolation).

**Reads (GPU, uoa-lab2, `til_measure.py` with three new flags; everything else as `reads_gridA`).** Each arm removes
one part of the reading module:

| read set | flag | removed |
|---|---|---|
| `reads_noframes` | `--frames 0` | the 20 frames in the prefix; no visual branch (a video without speech gets one joint branch per window, as `spvl.py`) |
| `reads_noctx` | `--no-transcript-context` | the full transcript in the prefix (window speech branches still show their window's text) |
| `reads_nostance` | `--stance none` | the verdict turn: the window branches follow the prefix directly; the verdict is read on a copy |
| `reads_asrwin` | `--windows asr` | the fixed 8 s grid: one window per ASR segment, gaps unscored, a video without transcript has no windows |

The main-corpus reading ablations (§14.4) were measured on the SPVL read family of 2026-09-10 (old ASR loader);
these DeHate arms are on the current read script with the fixed loader, so they are the same removals, not the same
runs. Cost: about 25 min per read set (the no-frames set less), 1.5 h of GPU in all.

**Time level (CPU, same machine), all on the isolated dual reads `reads_gridA` unless stated, flags as
`experiments/20260926_twolevel/launch/run_final.sh`:**

| arm | removed or replaced | main-corpus row |
|---|---|---|
| `abl_nocoupling` | time coupling (independent cells) | §20.1 (DeHate already known: +.020 [−.017, +.057]; re-run here so every arm sits in one table) |
| `abl_k1` | minimum length (k = 1) | §20.1 |
| `abl_sharedchain` | per-modality chains + OR (one shared chain) | §20.1 |
| `abl_rawscale` | normal scores (raw reads into EM) | §20.1 |
| `abl_rawkey` | calibrated key (raw K) | §20.1 |
| `abl_nokey` | video term | §20.1 |
| `abl_norank` | within term | §20.1 |
| `abl_fixed80` | per-video mean length (BMA) replaced by the fixed 80 s of `r3_m2` | §19 (r6 vs r3) |
| `abl_r2nl`, `abl_linstd` | EM evidence vs reads / corpus std, both under the fixed 80 s chain (`r2_noleak` vs `diag_lin_or_k4` flags) | §10.9 / §11 |
| `abl_noframes`, `abl_noctx`, `abl_nostance` | `r6_bma` on the three read sets above | §14.4 |
| `m1_joint`, `m1_seq` | §11 (already run) | §14.4, SPVL §9 |
| `spvl_fixed` vs `spvl_asrwin` | fixed 8 s windows vs ASR segments, under the SPVL composition (`compose.py --intercept zv_plus_mean --residual rank`), because ASR windows are longer than two cells and cannot enter the time level (`20260926_twolevel` §12.2) | SPVL §8 |

Analysis: `analyze_dvd.py` paired bootstrap over videos (4000 draws), base `final_m2` for the main table
(`analysis_all/`), base `abl_r2nl` for the evidence pair (`analysis_linstd/`), base `spvl_fixed` for the window pair
(`analysis_asrwin/`). For the two SPVL-composed arms the video-level diagnostics (share of videos with P(V) > .5,
video AUC) use the raw verdict, since the composition has no calibrated key; the three main metrics are unaffected.
Read agreement of the three new fixed-grid read sets with `reads_gridA`: `analysis_all/reads_agreement.txt`.

Launch: `launch/run_dehate_all.sh` (resumable: read sets with a `DONE videos` line are skipped). Nothing is
decided on DeHate (external); the table says which main-corpus effects show up on a third corpus.

### 12.1 Results (2026-09-29, uoa-lab2 = sc474399; copied back to `runs/20260928_infer/dehate/`)

Reads: four sets, 1341 videos each, 0 errors (`reads_noframes` 8 min, `reads_noctx` 22 min, `reads_nostance` 24 min,
`reads_asrwin` 19 min). Time level and analysis: CPU, same machine, no failures. Tables: `analysis_all/table.txt`
(base `final_m2`), `analysis_linstd/table.txt` (base `abl_r2nl`), `analysis_asrwin/table.txt` (base `spvl_fixed`),
`analysis_all/reads_agreement.txt`.

DeHate, arm minus base, within unless stated, 95 % paired-bootstrap interval; main-corpus columns are the numbers
quoted in the request (HateMM / HateClipSeg; `20260926_twolevel` §20.1 for the time level, §14.4 for the reading
components under `r3`, SPVL §8–§9 for windows and isolation):

| removed or replaced | HateMM | HCS | DeHate | interval |
|---|---|---|---|---|
| video term (pooled ROC) | −.325 | −.147 | **−.150** | [−.200, −.098] |
| within term | −.251 | −.137 | **−.143** | [−.182, −.104] |
| transcript context | −.039 | +.001 | **−.049**; pooled ROC −.078 | [−.079, −.021]; ROC [−.124, −.030] |
| fixed 8 s windows (ASR segments instead; SPVL composition) | −.089 | −.052 | **−.065**; pooled ROC −.018 | [−.106, −.026] |
| branch isolation (sequential, §11) | −.014 | +.011 | **−.031**; pooled ROC −.015 | [−.060, −.001] |
| time coupling | −.114 | −.057 | −.021 | [−.057, +.016] |
| minimum length (k = 1) | −.030 | −.009 | −.016 | [−.041, +.010] |
| EM evidence (reads / corpus std instead; fixed 80 s chain) | +.010 | +.024 | −.014 (EM is better by .014) | [−.040, +.012] |
| per-modality chains + OR (shared chain) | −.007 | −.043 | −.013; pooled ROC −.0006 | [−.035, +.007]; ROC excludes 0 |
| stance turn | −.014 | −.011 | −.010 | [−.028, +.009] |
| normal scores (raw reads into EM) | **+.011** | **+.011** | −.010 | [−.032, +.012] |
| frames | −.060 | −.109 | −.003; pooled ROC −.017 | [−.035, +.029]; ROC [−.061, +.021] |
| calibrated key (pooled ROC / PR) | −.002 / −.006 | −.003 / −.005 | −.002 / −.001 | ROC [−.003, −.001] |
| per-video mean length (fixed 80 s instead) | −.002 | −.002 | **+.011** (fixed 80 s is better) | [−.009, +.032] |
| dual branches (joint branch instead, §11) | +.009 | −.074 | **+.027** (joint is better) | [−.001, +.056] |

Absolute numbers of the new arms (pooled ROC / PR / within): `abl_nocoupling` .7010 / .1581 / .6226; `abl_k1`
.7011 / .1575 / .6274; `abl_sharedchain` .7005 / .1577 / .6300; `abl_rawscale` .7012 / .1585 / .6334; `abl_rawkey`
.6994 / .1568 / .6431; `abl_nokey` .5508 / .0881 / .6431; `abl_norank` .6982 / .1591 / .5000; `abl_fixed80` .7009 /
.1578 / .6539; `abl_r2nl` .6995 / .1569 / .6468; `abl_linstd` .6996 / .1570 / .6328; `abl_noframes` .6846 / .1507 /
.6398; `abl_noctx` .6237 / .1422 / .5938; `abl_nostance` .6999 / .1619 / .6330; `spvl_fixed` .6993 / .1570 / .6406;
`spvl_asrwin` .6817 / .1427 / .5753.

Read agreement with `reads_gridA` (per-video Spearman of window z; verdict Spearman; windows changing sign): no
frames .72 / .81 / 12.6 %; no transcript .74 / .55 / 14.9 %; no stance .90 / 1.00 / 8.4 %. Without the transcript the
verdict itself changes (Spearman .55) and the share of non-hateful videos judged violating falls from .565 to .309,
with hateful ones from .833 to .474: the verdict without transcript misses most hateful videos. Without frames the
time level reads (`z_joint`, `z_speech`): the joint chain, present only on videos without speech, gets a negative EM
slope (−0.93) and carries nothing; the speech chain is as before (slope 2.81 vs 2.85).

**Reading.**
- On DeHate the removals whose interval excludes 0: the video term, the within term, the transcript context, the
  fixed 8 s windows, and branch isolation. The first four hold on all three corpora (transcript context: HateMM and
  DeHate, not HCS). Isolation holds on HateMM and DeHate, not HCS.
- The time-level components (coupling −.021, minimum length −.016, EM evidence −.014, OR −.013, stance −.010) all
  point the same way as on the main corpora but with intervals that include 0: DeHate within has a wider interval
  (± .02 to .04) than the main corpora, and the effects are smaller than on HateMM.
- Two components go the other way on DeHate: normal scores help here (−.010 when removed; +.011 / +.011 on the main
  corpora), and the fixed 80 s mean length beats the per-video integration (+.011). Neither passes the noise floor
  on any corpus; both stay as decided (§20.1, §19).
- Frames carry almost nothing on DeHate (within −.003), against −.060 / −.109 on the main corpora: DeHate hate is in
  the speech (the SPVL composition on speech-only reads is within .003 of the full reads). The joint branch is better
  than dual here (+.027) for the same reason (§11.1).
- Nothing is decided on DeHate. For the main-corpus claims: the components that pass rule 14g there (coupling, video
  term, within term, frames) keep their sign on DeHate except frames, which is flat here.
