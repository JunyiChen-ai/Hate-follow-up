# NGA — negative-only group adaptation (label-free)

Status: 2026-09-12 proposal + run. Development-selected numbers only (rule 10). Successor to
`experiments/20260912_sdl/` (archived after three rounds); the constraint it drops is the one that
falsified SDL.

## 1. What the predecessors established

Four things are now measured rather than assumed, and each one closes a door:

| finding | source |
|---|---|
| the per-window judgement is substantially topic detection: mentioning the target group is worth +7.2 / +6.9 log-odds, being a GT-positive window +6.3 / +4.7 (count-weighted) | `experiments/20260912_pwc/` §7c |
| no read-out over the frozen scores recovers ordering — relative, absolute and joint-context all land in the same .55–.69 accuracy band | `runs/20260912_pwc/e0b/` |
| **with real labels**, a classifier over every cached per-window feature, including the branch hidden state, does not beat the unfitted read (.758 / .621) | `experiments/20260912_tad/` §5d |
| adapting the model with a "one window positive per hateful video" bag constraint flattens the curve: positive-window rate 45.8 % → 12.3 %, within-video std 4.46 → 1.73, within .5092 | `experiments/20260912_sdl/` §5e |

The last row is the immediate motivation. Hateful videos have a median positive fraction of .24 (HateMM)
and .47 (HCS), so "one window up, all others down" is a false statement about the data, and a model asked
to satisfy it complies by destroying the ordering.

## 2. Method

Keep exactly the part of the pseudo label that is true at window level and drop the rest.

> If the model's own whole-video verdict says a video is **not** hateful, then **every** window of that
> video is not hateful. If it says the video **is** hateful, nothing follows about any particular window.

```
per video v (test corpus, no labels anywhere):
  y_v = 1[ z_video(v) > 0 ]          pseudo label from the FROZEN model, fixed before training
  y_v = 1  ->  no loss term at all
  y_v = 0  ->  mean_i relu( s_i + m )        push every window below -m, hinge on the raw log-odds
```

Everything else is SPVL-r2 and is unchanged: prefix, windows, dual branches, read-out, composition,
evaluator. Trained parameters are a rank-8 LoRA on the language layers' attention projections; the prefix
is encoded with the adapter disabled and no gradient, and the stance turn is removed (SDL round 1: it
leaks the bag label into every branch). Inference matches training — adapter off for prefix and verdict,
on for the window branches.

### Why this is the mechanism the diagnosis calls for

Non-hateful videos contain windows that refer to protected groups. The topic/act cross-tabulation says the
frozen model scores such windows high regardless of the label — that is the confound. Pushing them down on
**genuine negatives** is the only label-free statement available that separates "mentions the group" from
"attacks the group", and it is a statement about window content, not about the video. If the adapted model
transfers that to the context windows of hateful videos, the within-video ordering improves without any
constraint ever being placed on a hateful video.

Falsifiable prediction, checked with `diagnose_topic_confound.py`: after adaptation the *mention* main
effect in the cross-tabulation shrinks while the *GT* main effect is retained or grows. If both shrink,
the adaptation is indiscriminate and the direction is dead.

## 3. Inputs and cost

No new cache, no new preprocessing. Only pseudo-negative videos produce a training step: 69 of 215 HateMM
and 10 of 118 HCS videos (from the frozen verdict), i.e. 79 of 333 — training is about a quarter of SDL's.
Inference is unchanged from SPVL-r2 plus the adapter.

**Declared risk, before the run:** only 10 HCS videos are pseudo-negative. The HCS side of the adaptation
is supervised by 10 videos and may not move at all; if the effect is HateMM-only, that is expected and must
not be reported as a corpus-specific method (rule 13 — the method is identical, the supervision available
differs).

## 4. Constants (declared before the run)

Inherited from SPVL-r2 unchanged. New: LoRA rank 8 / alpha 16 on `q,k,v,o_proj` of the language layers;
AdamW lr 1e-4, weight decay 0, linear warmup 10 %; 3 epochs; gradient accumulation 4; hinge margin m = 2
log-odds; `--grad-windows 4` sampled windows per step (unbiased estimate of the mean over the video's
windows); pseudo label `1[z_video > 0]` from `runs/20260910_spvl/mllm/q3vl-8b/full/predictions.jsonl`,
fixed; stance turn removed; adapter scope = window branches; seed 0.

## 5. Plan and gates

**Gate (declared before the run).** Within must rise by ≥ .01 over the frozen baseline on the same code
path (.6968 / .6020) on **both** corpora, with no pooled metric falling more than the .005 noise floor.
Reported with both the adapted intercept and `--intercept frozen`, so a within-video reordering is
separated from a pooled shift.

Controls, run only if the gate passes:

- **shuffled pseudo-labels** — permute which videos count as negative; the gain must not survive;
- **identity distillation** — same number of updates, target = each window's own frozen score; separates
  "this objective moved it" from "a rank-8 perturbation of this size moves it";
- **frozen baseline on this code path** (`--train 0`, exact by construction: the LoRA B matrix is zero-init);
- **oracle** — real video-level labels instead of the pseudo labels (rule-10 diagnostic, never a method arm).

## 5b. Results

All numbers through the shared evaluator, 333 videos, both corpora. Frozen reference on the same code
path: HateMM .8920 / .6825 / **.6968**, HCS .7132 / .6675 / **.6020**
(`runs/20260910_spvl/mllm/q3vl-8b/full/`).

| round | anchor | HateMM ROC / PR / within | HCS ROC / PR / within |
|---|---|---|---|
| r1 negative-only, no anchor | 0 | .8913 / .6797 / .6787 | .6960 / .6554 / .5875 |
| r2 + anchor on pseudo-positive videos | 0.1 | .8928 / .6838 / **.6812** | .7092 / .6641 / **.5996** |
| r3 (declared scan, rule 7) | 0.03 | running | running |

Mechanism diagnostics against the frozen curve:

| | Spearman(adapted, frozen) | within-video std | windows judged positive |
|---|---|---|---|
| frozen | — | 4.46 / 5.40 | 45.8 % / 50.7 % |
| r1 (anchor 0) | .829 / .787 | **1.33 / 1.45** | **2.8 % / 0.0 %** |
| r2 (anchor 0.1) | .950 / .938 | 4.40 / 5.50 | 48.2 % / 50.8 % |

Reading. Round 1 confirmed the failure mode the anchor was designed for: with no term on pseudo-positive
videos the objective is satisfiable by lowering every window everywhere, and the resulting compression
costs within even though the ordering itself is largely preserved. Round 2's anchor removes the
compression completely — spread, positive rate and ordering all return to the frozen values — but by the
same token it pins the model, and the small movement that remains is net slightly negative
(−.016 HateMM, −.002 HCS, the latter inside the noise floor).

The two rounds bracket the trade-off: too little anchoring and the model pays for the negative constraint
with a global shift; too much and there is nothing left to learn. Round 3 is the single declared
intermediate value.

**Supervision budget, which bounds all of this:** only 79 of 333 videos are pseudo-negative (69 HateMM,
**10 HCS**). The HCS side of the objective is supervised by ten videos, which is the declared risk in §3
and is consistent with HCS moving least in every round.

## 6. Test-read log (rule 10)

- 2026-09-12: the §1 findings are the PWC, TAD and SDL test reads already logged in those READMEs.
