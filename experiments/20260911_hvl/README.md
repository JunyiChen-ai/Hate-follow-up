# HVL — hypothesis–verification localization (2026-09-11, in progress)

Status: proposal + E0 pilot. Development-selected numbers only (rule 10). Runs on uoa-lab3 / uoa-lab2 / lab-server,
conda `HateVLM`, Qwen3-VL-8B-Instruct for development; cross-model check in E5.

## 1. Why: the five mechanism gaps of SPVL-r2 and what the cross-model study showed

SPVL-r2 (`experiments/20260910_spvl/`, current method) gains come from how the inputs are organised (fixed 8 s
windows, one shared prefix, rank composition); the model itself does the same computation as plain prompting.
That is why the pipeline lifts all seven MLLMs alike (spvl README §11) and why it is thin. The decision step
of SPVL-r2 is missing, at each window, the following information (mechanism problems: what is absent,
predicted failure, counterfactual):

| gap | what the window decision lacks | predicted failure | evidence |
|---|---|---|---|
| G1 no temporal reasoning | the decisions for windows i−1, i+1 | continuation and boundary windows scored low; within capped by single-window accuracy | within stuck at .70 / .60 on all 7 models; letting windows see each other (causal mask) did not help (−.014 HateMM) |
| G2 stance is one bit | who is targeted, in what form, where the model saw it | implicit-target windows missed; heated but untargeted windows in non-hateful videos over-scored | stance conditioning only +.008 / +.012; transcript context useless on 3 of 7 models |
| G3 frameless windows | a frame inside the window (20 uniform frames, ≈ 22 windows) | silent-hate windows guessed from neighbours | silent-hate subset .69–.72 / < .5; visual branch wins mostly in windows that have speech |
| G4 crude extent | a calibrated "how much of the video" | short-hate long videos pushed down, whole-hate short videos pushed up in the pooled ranking | z_video + mean(w) with unmatched scales; alternatives within noise |
| G5 unfiltered context | which sentences matter for this window | short single-topic videos: transcript is noise | HCS +.010 without transcript; context inert on 3 of 7 models |

## 2. Method (four steps, one frozen MLLM, prefix KV cache; `hvl.py`)

Paradigm claim (to be earned by the ablations of §6): localization as a **hypothesis–verification loop** —
the model first states a structured hypothesis about the whole video (who is targeted, in what form, where),
verifies it window by window as a **temporal state sequence**, and the verification revises the verdict.
Verdict and localization condition each other; in every earlier label-free method (per-segment classifiers,
anomaly scoring, our SPVL) they are one-directional.

| step | input | output | gaps |
|---|---|---|---|
| S1 verdict + hypothesis | shared prefix (rules + 20 timestamped frames + timestamped transcript) + whole-video question | z_video (Yes/No log-odds); then a greedy fixed-format hypothesis `TARGET / FORM / EVIDENCE(start-end: note)`, appended to the context as the model's own turn | G2, G5 |
| S2 temporal verification | prefix + hypothesis + per-window questions asked **in sequence**: "relative to the hypothesis, is this window *start / continue / stop / none*?"; window i follows the model's own answer for i−1 (the chain is one growing cache, no isolation); one chain for the visual branch, one for the speech branch | per window and branch: restricted log-softmax over the four state words; presence log-odds = logsumexp(start, continue) − logsumexp(stop, none); window score = max over branches | G1, modality module |
| S3 revision | prefix + hypothesis + verification summary (how many windows were judged present, which, the strongest window's transcript) + the whole-video question again | z_rev | G4, loop closure |
| S4 composition | z_rev as the video intercept; centred rank of window scores as the within-video residual (`compose.py`, same as SPVL) | frame scores on the 4 fps grid | — |

Optional parts (each an ablation arm): `--fill-frames` adds the window-centre frame (`data/frames_w8`) for every
window that has no uniform frame inside it (G3; ≈ +11 frames, +1k prefix tokens); `--context cited` rebuilds
the prefix with only the transcript sentences overlapping the cited evidence ± 1 segment (G5; +1 prefix forward).

Sources: VERA (verbalised, verifiable questions for video anomaly), Holmes-VAU / HiProbe-VAD (multi-level
reasoning over segments), sequence-labelling formulations of temporal grounding (BIO-style states), MIL-style
weak supervision (video label constrains segment scores; here the constraint is the model's own hypothesis, no
label); internal: 2026-08 masked isolation, SPVL-r2 (§9–§10 of its README). None of these runs a self-generated
hypothesis through a temporal-state verification with feedback to the verdict on hateful video.

## 3. Inputs and cost

Same caches as SPVL (Whisper transcripts `data/asr_whisper_large_v3`, frames `data/frames_k20`, optional
`data/frames_w8`). Per video: one prefix forward (≈ 2.8k tokens), one verdict read, one greedy generation of
≤ 220 tokens, N window steps per branch (sequential: cache extends, no copies; ≈ 60 tokens each), one revision
read. Expected 8B time ≈ 3–4 s / video on a 5090 (SPVL-r2: 1.5 s); no new preprocessing.

## 4. Constants (declared before the runs; both corpora identical)

K = 20 uniform frames (pixel cap 100352 / 65536), S = 8 s windows, rules / reader / system / whole-video question
verbatim from the 2026-08 judge (`src/mllm_judge.py`), hypothesis question `HYP_QUESTION` (hvl.py), greedy
decoding, ≤ 220 generated tokens, states `start / continue / stop / none` with token sets from the tokenizer
(first token of each casing / spacing variant; sets must be disjoint), presence log-odds as in §2, FILL = −12 for
windows with no branch, seed 0, verify gate |Δz| < 3 nats between the cached verdict and a plain call.
Composition tags: `izv_rev_rrank` (method), `ispvl_rrank` (first verdict), `izv_rev_plus_mean_rrank`,
`izv_plus_mean_rrank` (SPVL-r2 intercept), `izv_rev_rnone` (verdict only).

## 5. Plan and gates (rules 4, 6, 8, 9, 14)

E0 (this file §7): is the structured hypothesis reliable? E1 proposal review. E2 code review. E3 pilot on the
within-defined subset (HateMM 84 + HCS 99; selection reads GT, rule 10). E4 full run + ablations; promotion gate
vs SPVL-r2 (HateMM .8919 / .6831 / .6976, HCS .7119 / .6664 / .6001): no metric down beyond noise (pooled .005 /
within .01), ≥ 1 metric up ≥ .01 on both corpora. Paradigm claim needs (rule 14g): removing the loop (S3 off) and
replacing the temporal states by independent Yes/No each cost ≥ .01 within on both corpora. E5 cross-model
check (Qwen3-VL-4B, InternVL3.5-8B). Expected: within HateMM ≥ .71, HCS ≥ .61.

## 6. Ablation arms (pre-declared)

full (hyp + sequential four-state + revision + dual); − revision; independent Yes/No instead of states; − hypothesis
(one-bit stance, as SPVL); joint branch; + fill-frames; context cited; intercept variants from compose.

## 7. E0 — hypothesis reliability (test read, rule 10)

Run: `runs/20260911_hvl/e0_hypothesis/` (within subset, S1 only). Analysis `analyze_e0.py` → `metrics_cited.json`:
cited evidence spans painted 1 / 0 and scored by the shared evaluator; TARGET / FORM distributions; relation of
"none" hypotheses to z_video; coverage and precision of cited frames against GT (reported only).
Decision rule (declared before reading): cited-span within ≥ .60 (HateMM) / .55 (HCS) and hypotheses that vary
with content (not a renaming of z) → P1 stands; otherwise the hypothesis is demoted to context filtering (G5)
and P2 (temporal states) is tested on its own.

Results (`runs/20260911_hvl/e0_hypothesis/`, 183 videos, 3.2 s/video incl. generation; `metrics_cited.json`):

| | HateMM (84) | HateClipSeg (99) |
|---|---|---|
| hypotheses with evidence spans / "none" | 79 / 3 (all three z<0) | 82 / 7 (5 of them z<0) |
| items per hypothesis | 5.0 | 4.9 |
| TARGET distribution | black people 43, jews 10+3, none 8, african americans 4, white people 3, lgbtq+ 1 … | none 25, jewish people 8+6, white people 4, black people 3, men 2, muslims 2, women 2 … |
| cited-span within (macro ROC) | **.571** | **.506** |
| cited-span pooled ROC | .588 | .494 |
| coverage of GT-positive frames / precision of cited frames | .61 / .66 | .41 / .53 |

Reading (decision rule above): (a) passes — the hypothesis varies with content (specific targets, rule numbers
spread over 1–9) and "none" is not simply z<0 (HateMM: 3 of 3; HCS: 5 of 7, plus 25 "none" targets with
evidence); (b) **fails** — the model's own cited spans localize barely above chance (.571 / .506, below the
declared .60 / .55; per-window-alone gives .628 / .576). The model can say who and how, but not reliably where.
Consequence: the cited spans are not used as scores anywhere; the hypothesis is kept only as conditioning
for the verification step, and whether that conditioning helps is decided by the E3 pilot (hyp vs no-hyp
arms, plus the permuted-hypothesis control suggested by the proposal review). P1 as a paradigm claim is on
hold until the pilot says the verification uses the hypothesis.

Proposal review (rule 4, 2026-09-11): 放行. Closest works: VideoHV-Agent (2603.04977, think-then-verify for
long-video QA, retrieval of windows, no state chain, no verdict revision), GtS/VAGU (2507.21507, coarse-to-fine
anomaly, no self-hypothesis), AnomalyRuler (2407.10299, induced rules from normal frames), MARS (2601.15115,
video-level multi-stage reasoning for hateful video, no localization), LELA (2602.09637, per-frame independent).
Reviewer suggestions adopted as control arms: sequential chain with binary Yes/No (separates order from the
state vocabulary); shuffled window order (does the chain carry time?); hypothesis from another video (is the
hypothesis used?); record state-transition counts; report Spearman(z_rev, z_video) and Spearman(z_rev, #present).
Reviewer risk noted: the speech chain skipped no-speech windows, so its "previous window" differed from the
visual chain's; fixed after the first pilot arm (sequential mode now visits every window in both chains).

