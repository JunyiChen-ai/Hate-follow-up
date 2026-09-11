# SDL — self-distilled localization: a label-free MIL objective from the model's own verdict

Status: 2026-09-12 proposal. No run yet. Development-selected numbers only (rule 10).

Predecessors, all archived as negative results today: `experiments/20260912_pwc/` (read-out change),
`experiments/20260912_tad/` (auxiliary read, subtraction and decision re-specification). Together they
establish the premise of this one.

## 1. What the last two rounds established

Every attempt to improve the within-video ordering **downstream of a frozen read** has failed, and the
failures are quantified rather than assumed:

| what was changed | result |
|---|---|
| ask relatively instead of absolutely (pairwise A/B/C, swap-symmetrised, sparse graph) | accuracy .654 / .570 vs .664 / .565 for `sign(z_i − z_j)`; every read-out form lands in the same .55–.69 band (`runs/20260912_pwc/e0b/`) |
| subtract an auxiliary topic read | within .6159 / .5793 vs .6926 / .6007; monotone decreasing in beta; adding it also hurts (`runs/20260912_tad/e0/`) |
| re-specify the decision as a five-way speech act | argmax is `attacks` on 79.6 % / 56.0 % of windows and the three carve-outs take 1.9 % / 9.2 %; within .6945 / .6076 (`runs/20260912_tad/e0_acts/`) |
| fit a classifier **on the test labels** over the reads, window shape, ImageBind audio and neighbour context | .742 / .626 (logreg), .614–.793 / .548–.563 (GBT) vs **.758 / .621** for the unfitted read |
| fit the same classifier on the branch **hidden state** (64 / 128 / 256 dims) | .747 / .608, .746 / .594, .698 / .552 |

The last two rows are the decisive ones: **with real labels, nothing built on top of the frozen model's
per-window output beats the raw output.** The information is not being lost by the read-out, the
composition or the question form. It is not there.

What is missing is not information about the window — it is *pressure to discriminate inside a video*.
The model was instruction-tuned to answer questions about a clip as a whole. Nothing in its training, and
nothing in any prompt, ever asked it to say which part of a hateful video is the hateful part. That is
exactly the quantity `within-video macro ROC` measures, and it is exactly what the topic/act diagnosis
shows is absent: the per-window score separates *videos* (topic) far better than it separates *windows of
one video* (act).

## 2. Method

Supply that pressure without any human label, using the one judgement this model is reliably good at: its
own whole-video verdict (pooled ROC .88 / .67 with no localization at all; SPVL README §11 Table A).

```
per video v (test corpus, no labels anywhere):
  y_v   = 1[ z_video(v) > 0 ]                       pseudo video label, from the frozen model
  s_i   = window Yes/No log-odds of window i        differentiable, same branch as SPVL-r2
  loss  = - y_v · log sigmoid( max_i s_i )
          - (1 - y_v) · mean_i log sigmoid( -s_i )
          + lambda · mean_i sigmoid(s_i)            sparsity, standard in MIL localization
```

Trained parameters: LoRA (rank 8) on the attention projections of the language layers only. The vision
tower, the processor, the prompts, the windows, the composition and the evaluator are untouched. The
shared prefix is computed **without gradient** and its KV cache is treated as a constant for the step, so
what is adapted is the decision path that reads a window question against a fixed video representation;
one 2.8k-token no-grad forward plus N short branch forwards per step.

Inference after adaptation is byte-identical to SPVL-r2 except for the adapter weights: same prefix, same
verdict, same stance turn, same dual branches, same `z_video + mean_i(a_i)` intercept, same centred-rank
residual.

### Why this is the mechanism the diagnosis calls for

`max_i s_i` is the only part of the loss that touches a pseudo-positive video, and it forces exactly one
window up while the sparsity term pushes the rest down — a *within-video* contrast that no prompt can
express. On pseudo-negative videos every window is pushed down, which is what supplies the negative
evidence for the topic dimension: windows that merely mention a protected group appear in both classes,
so the objective cannot be satisfied by tracking topic. That is the falsifiable content: after adaptation
the topic main effect in the §1 cross-tabulation should shrink while the GT main effect grows.

It is also the label-free counterpart of what the weakly supervised baseline does with human video labels
(MultiHateLoc, MIL over video-level annotations), which gives the paper a direct claim: the frozen MLLM's
verdict is a **better** weak supervisor than the dataset's own video labels — MultiHateLoc with real video
labels reaches .7535 / .4880 / .6070 and .5062 / .4925 / .5128, the frozen verdict alone already reaches
pooled ROC .88 / .67.

### Relation to test-time adaptation

Training runs transductively on the test corpus with no labels, which is the same setting as T3AL (the
label-free comparator in the results table, which adapts per video at test time). Two variants are run:
corpus-level (one adapter for all test videos) and per-video (T3AL-style, adapter reset per video).

## 3. Inputs and cost

No new cache, no new preprocessing, no new model weights beyond the adapter. Per training step: one
no-grad prefix forward (~2.8k tokens) + N branch forwards with gradient (N ≈ 22, ~60 tokens each).
Estimated 10–20 s per video-step on one RTX 5090; 333 videos × 3 epochs ≈ 3–6 h. Inference cost is
unchanged from SPVL-r2 (2 forwards per video) plus loading a rank-8 adapter.

## 4. Constants (declared before any run; identical for both corpora, rule 13)

Inherited unchanged from SPVL-r2: K = 20 frames, pixel cap 100352 / 65536, S = 8 s, all prompt material,
Yes/No token sets, greedy read-out, dual branches with max, stance = verdict, intercept and residual.

New, declared here:

| constant | value |
|---|---|
| adapter | LoRA rank 8, alpha 16, dropout 0, on `q_proj, k_proj, v_proj, o_proj` of the language layers |
| optimiser | AdamW, lr 1e-4, weight decay 0, linear warmup 10 % |
| epochs | 3 |
| batch | one video per step, gradient accumulation 4 |
| sparsity lambda | 0.05 |
| pseudo label | `1[z_video > 0]` from the **frozen** model, computed once before training and never updated |
| confidence filter | none in the main arm; `|z_video| < 2` dropped is a declared ablation |
| seed | 0 |

## 5. Plan and gates

### E0 — does the adapter move the within-video ordering at all? (kill test)

Adapt on the within subsets only (HateMM 84 / HCS 99 videos), evaluate on the same videos. This is a
sanity run, not a result: if the objective cannot move within even where it trains, the direction stops.
Declared stop rule: within must rise by ≥ .02 over the frozen baseline (.6926 / .6007 on the same code
path) on at least one corpus without falling on the other.

Diagnostics recorded: pseudo-label accuracy against the video-level GT (a rule-10 read, reported only);
the fraction of windows with `s_i > 0` before and after; the §1 topic/act cross-tabulation before and
after; the training loss curve.

### E1 — full corpus, both variants

333 videos, corpus-level and per-video adapters, three metrics through the shared evaluator. Controls,
each a separate run:

- **shuffled pseudo-labels** (video labels permuted within corpus): if the gain survives, it is not the
  verdict's information;
- **mean-pool instead of max-pool** on pseudo-positive videos: removes the within-video contrast while
  keeping everything else — the direct test of the mechanism claim;
- **frozen baseline** re-run on the same code path;
- **oracle**: real video-level labels instead of the pseudo labels (rule-10 diagnostic only, never a
  method arm) — bounds how much of the gap is pseudo-label noise.

### E2 — gates

Rule 8 comparison gate vs T3AL and promotion gate vs SPVL-r2 (.8919 / .6831 / .6976 and
.7119 / .6664 / .6001), noise floor pooled .005 / within .01. Rule 14g ablation on every claimed
component. Cross-model repeat on at least one further MLLM if E1 passes.

## 5b. Round 1 — the stance turn leaks the bag label (falsified, `runs/20260912_sdl/e1_max/`)

333 videos, 3 epochs, grad-windows 4, 999 steps, 48 min training + 11 min inference on lab-server.

| composition | HateMM ROC / PR / within | HCS ROC / PR / within |
|---|---|---|
| SDL round 1 | .8719 / .6065 / **.5026** | .6760 / .6122 / **.5091** |
| same run, frozen intercept (`--intercept frozen`) | .8908 / .6763 / .5026 | .7114 / .6632 / .5091 |
| frozen SPVL-r2 reference | .8920 / .6825 / .6968 | .7132 / .6675 / .6020 |

Within collapsed to chance. It is not a numerical collapse of the scores: the within-video standard
deviation of the window curve **rose** from 4.46 to 6.90 (HateMM) and 5.40 to 6.72 (HCS), and only 1.9 % /
0 % of videos ended with a constant curve. What collapsed is the *correlation with the labels*
(Spearman against the frozen curve fell to .495 / .442).

Diagnosis, measured: the adapter learned to shift whole videos by their pseudo label rather than to
separate windows inside a video — mean adapted window score is **−17.67 on pseudo-negative HateMM videos
versus +1.09 on pseudo-positive ones** (HCS: −15.69 vs −3.40). The cause is structural. SPVL-r2 conditions
every window branch on a stance turn that contains the model's own whole-video Yes/No — and that Yes/No
**is** the MIL bag label. A branch that can read it satisfies the objective without looking at the window.

Fix carried into round 2: `--stance none`, so the branches see the prefix only. The stance turn is worth
+.008 / +.012 within in the frozen method, below the noise floor on HateMM.

## 5c. Round 2 — the BCE form of the objective has no gradient here (falsified before completion)

With the stance turn removed the objective is sound, but the measured per-video losses in the first 40
videos were 0.0335, 0.0000, 0.0010: essentially zero. Two reasons, both structural rather than accidental:

1. **The frozen model already satisfies the bag constraint.** Positive videos have a window at s ≈ +17,
   negative videos sit at s ≈ −14. There is nothing for `-log sigmoid(max_i s_i)` or
   `-mean log sigmoid(-s_i)` to correct at bag level.
2. **The read-out is saturated.** Window log-odds run to |s| ≈ 15, where `sigmoid` and `logsigmoid` have
   vanishing derivatives, so even the violated terms contribute almost no gradient.

The run was stopped rather than spending an hour to confirm a no-op.

## 5d. Round 3 — non-saturating hinge on the raw log-odds

Same objective, expressed so that it bites where the information is:

```
positive video:  relu(m - s_top)                       + lam * sum_{i != top} relu(s_i + m)
negative video:  mean_i relu(s_i + m)
m = 2 (log-odds), lam = 0.05, estimated on the sampled windows as in round 2
```

Constant gradient while a constraint is violated, exactly zero once satisfied. The terms that actually
carry information are the ones the BCE form could not express: the windows of a **negative** video that
the frozen model scores positive (45.8 % / 50.7 % of all windows are judged positive), and the non-top
windows of a positive video. Those are also where the topic/act confound lives — non-hateful videos
contain windows that mention a protected group, and pushing them down is the only label-free signal
available for "mentioning is not attacking".

Declared risk: in videos with a large hateful extent (HCS median positive fraction .47) pushing every
non-top window down is wrong. If round 3 shows that pattern, round 4 replaces max with top-K.

## 6. Risks, stated before the run

- The pseudo label is the frozen verdict, so on videos the verdict gets wrong the objective trains the
  wrong thing. Video-level verdict ROC is .88 / .67; on HCS in particular a third of the supervision may
  be wrong. Measured, not assumed: pseudo-label accuracy is reported in E0.
- MIL with max-pooling is known to latch onto one window per video and collapse the rest; the sparsity
  term and the fraction-positive diagnostic are there to catch it.
- Adapting on the test corpus is transductive. It is the same setting as the T3AL comparator and uses no
  labels, but it must be labelled as such everywhere and cannot be presented as inductive generalisation.
- The prefix is treated as a constant, so the adapter cannot change how the video is represented, only how
  a window question is read against it. If the ceiling is in the representation rather than the decision,
  this will fail; that is the falsifiable part.

## 7. Test-read log (rule 10)

- 2026-09-12: the diagnostics in §1 are the PWC and TAD test reads already logged in those READMEs.
