# EventVAD: what the release is missing, and what was put in its place

EventVAD, ACM Multimedia 2025, <https://github.com/YihuaJerry/EventVAD> at
`25cacd88a82af389776d2b397239f39961ac2d27` (2025-07-09), paper
arXiv:2504.13092.

The method is two training-free stages. Stage 1 turns a video into events:
CLIP ViT-B/16 and RAFT optical flow per frame, a dynamic spatiotemporal graph
over frames, graph attention propagation, then statistical boundary detection.
Stage 2 hands each event's 16 frames to VideoLLaMA2.1-7B-16F and reads an
anomaly score out of the answer.

This note records four things: the gaps in the release, the reconstruction of
each, the evidence behind every inferred choice, and the choices that remain
inferences. `smoke_cpu_eventvad.py` re-checks every claim below that can be
checked mechanically, so a clone at a different commit or a silently amended
upstream is caught rather than assumed. Porting patches -- the changes made to
run the method on this study's corpora rather than to make it exist -- are in
PATCHES.md under "EventVAD".

---

## 1. The gaps

Four, of which the first is fatal.

### G1. `graph_propagation` is imported and defined nowhere

`src/event_seg/uniseg_processor.py` line 7:

```python
from graph_operations import graph_propagation  # 确保正确导入
```

and line 58 calls it. `src/event_seg/graph_operations.py` is a **byte-identical
duplicate** of `src/event_seg/video_processing.py` (`diff` is empty) and
defines exactly one function, `process_video`. No file in the release defines
`graph_propagation`. The import therefore cannot resolve, `main.py` raises
`ImportError` before it decodes a frame, and **the released event-segmentation
pipeline has never been run in the state it was published in.**

That matters beyond the missing function. It means the released
`src/event_seg/config.py` is not a tested preset: no value in it was ever
exercised end to end. Section 3 below takes the paper's published
hyperparameters as authoritative wherever the two disagree, and that is why.

Graph attention propagation is not a detail of the method. The paper's own
Table 5 makes it the single largest component:

> | RAFT | GANP | Thinking | AUC (%) | Δ (%) |
> | --- | --- | --- | --- | --- |
> | ✗ | ✗ | ✗ | 73.93 | -- |
> | ✗ | ✓ | ✗ | 77.81 | +3.88 |
> | ✓ | ✗ | ✗ | 75.35 | +1.42 |
> | ✓ | ✓ | ✓ | 82.03 | +8.10 |

so the missing function is worth more than either of the other two components
the ablation isolates.

### G2. The scoring prompt is the literal string `"prompt"`

`src/score/event_score.py` line 23:

```python
abnormal_prompt = "prompt"
```

Nothing else in the release records what was actually asked. A second, smaller
sign that this file was never run against the paper's own outputs: its parser
is

```python
score = float(output.strip())
```

which cannot read the answer the paper's Figure 2 shows the model producing --
prose ending "Therefore, the final score is 0.8."

### G3. The RAFT checkpoint path is a placeholder

`src/event_seg/feature_extractor.py` line 22: `model='/path/raft-things.pth'`.

### G4. `src/evaluate.py` does not compile

Line 44 reads `for line in f:s`. `python -m py_compile` fails with
`IndentationError: unexpected indent`. This study replaces the evaluator
anyway, so nothing depends on it; it is recorded because it is a fourth
independent indication of the state the release was published in.

---

## 2. Reconstructing G1: graph attention propagation

The paper specifies the module in Eq. (5) through Eq. (8), section 3.2. Quoted
in full:

> To improve frame-by-frame representation while preserving temporal
> consistency, we propose a training-free attention mechanism based on
> orthogonal feature projection in spacetime. This mechanism amplifies segment
> contrast through graph-guided message passing, enhancing event boundary
> distinguishability. Our approach builds upon graph attention networks with
> orthogonal constraints.
>
> Given the propagated node features F⁽⁰⁾ = [f₁, ..., fₙ]ᵀ ∈ ℝⁿˣᵈ from Section
> 3.1, we project the features into orthogonal subspaces to prevent dimension
> collapse as Eq (5),
>
> &nbsp;&nbsp;&nbsp;&nbsp;Q = QR(𝒩(0,1)ᵈˣᵏ), K = QR(𝒩(0,1)ᵈˣᵏ), V = QR(𝒩(0,1)ᵈˣᵈ)
>
> where QR(·) denotes orthonormal columns via QR decomposition, d = 640 is the
> fused feature dimension, and k = 64 is the projected dimension. These fixed
> orthogonal matrices maximize feature retention.
>
> For node fᵢ the attention weight for neighbor fⱼ in the dynamic graph is
> calculated as:
>
> &nbsp;&nbsp;&nbsp;&nbsp;Attenᵢⱼ = Softmax((fᵢQ)(fⱼK)ᵀ / √dₐ)(fⱼV)  (6)
>
> The indicator function E₍ᵢ,ⱼ₎ corresponds to temporal connectivity in Section
> 3.1, ensuring attention respects event-induced topology. This constraint
> prevents attention dispersion to irrelevant frames. Features update
> iteratively via:
>
> &nbsp;&nbsp;&nbsp;&nbsp;fᵢ⁽ᵗ⁺¹⁾ = fᵢ⁽ᵗ⁾ + Σ_{j ∈ 𝒩ᵢ} Attenᵢⱼ · E₍ᵢ,ⱼ₎  (7)
>
> After each iteration, features are centered as:
>
> &nbsp;&nbsp;&nbsp;&nbsp;fᵢ⁽ᵗ⁺¹⁾ ← fᵢ⁽ᵗ⁺¹⁾ − (1/|F|) Σ_{k ∈ F} f_k⁽ᵗ⁺¹⁾  (8)

and section 4.1 fixes the iteration count: "the graph attention propagation is
only a single iteration".

`graph.py:graph_propagation` implements

```
scoreᵢⱼ = (fᵢQ)·(fⱼK) / √dₐ           for j ∈ 𝒩ᵢ
aᵢⱼ     = softmax over 𝒩ᵢ of scoreᵢⱼ
fᵢ     ← fᵢ + Σ_{j ∈ 𝒩ᵢ} aᵢⱼ · E₍ᵢ,ⱼ₎ · (fⱼV)
fᵢ     ← fᵢ − mean_k f_k
```

for one iteration. The dimensions close: `fᵢQ ∈ ℝ⁶⁴`, so the score is a
scalar; `fⱼV ∈ ℝ⁶⁴⁰`, so the message is the same width as the residual it is
added to.

**Four choices are inferences, not transcription.**

**G1-a. `dₐ = k = 64`.** The paper writes `√dₐ` and never defines `dₐ`.
Everything entering the dot product has been projected to k = 64, and scaled
dot-product attention divides by the square root of the key dimension, so
`dₐ = k`. The alternative reading, `dₐ = d = 640`, would divide the logits by
3.16× more and flatten the softmax toward uniform. `cfg.ortho_dim` sets both.

**G1-b. The softmax is over 𝒩ᵢ, one scalar per neighbour.** Eq. (6) as printed
folds `(fⱼV)` inside the `Attenᵢⱼ` symbol, which would make `Attenᵢⱼ` a
640-vector and the softmax's axis undefined. The reading above -- softmax the
scalar logits over the neighbour set, then use them to weight the value
vectors -- is the only one under which Eq. (7)'s sum is dimensionally
consistent and under which the phrase "the attention weight for neighbor fⱼ"
denotes a weight.

**G1-c. `E₍ᵢ,ⱼ₎` is the edge weight of Eq. (4), not a 0/1 indicator.** The
paper calls it "the indicator function", which suggests membership, but the
same symbol `E` is the temporal-decay adjacency matrix defined in Eq. (4), and
the attention is *already* restricted to `j ∈ 𝒩ᵢ` in Eq. (7), which makes a
membership indicator a no-op. Reading it as the edge weight is the reading
under which the term does work: it is what lets the time decay reach the
propagation rather than only the graph's topology. `--gat-edge-term indicator`
selects the other reading, and it is a one-line change, so the choice is
testable rather than baked in.

**G1-d. Seed 42 for Q, K, V.** The paper fixes none. Upstream seeds its one
other fixed random projection -- the 2 → 128 flow projection in
`feature_extractor._init_random_ortho` -- with `np.random.seed(42)`, so 42 is
reused. `cfg.ortho_seed` exposes it. Because the matrices are orthonormal and
fixed, the seed rotates the subspace rather than changing its dimension; it is
not a tuned quantity.

---

## 3. Reconstructing G2: the hierarchical prompt

Figure 2's "Event-Centric Anomaly Scoring" panel is the only place in the
paper where the prompt appears. Transcribed exactly:

> **Input**
>
> \#Question: Are there any obvious or potential anomalies in the video?
>
> \#Instruction: Let's think step by step to judge the anomaly of the video.
> Finally, output the anomaly score based on the thinking process.
>
> **Output**
>
> \#Answer: A man is seen with a gun, pointing it at the employees. He takes
> money from the register and … Therefore, the final score is 0.8.

Section 3.4 says what the two input lines are doing:

> To address this, we construct event semantic units as spatiotemporal
> primitives, creating an optimized feature representation framework for
> visual-language models. For event unit analysis, we propose a
> semantic-driven hierarchical prompting framework. This framework directs
> multimodal large language models to produce structured outputs: first
> generating video content descriptions and then deriving anomaly scores. When
> processing a video, the multimodal large language models initially generate
> descriptive text by identifying surface and latent semantic features. It
> subsequently outputs an anomaly score based on this description, enabling
> cross-modal evaluation against predefined criteria.
>
> By implementing a two-stage reasoning framework, we establish a
> self-correction mechanism. This architecture allows systematic score
> derivation through video content analysis, ensuring contextual relevance and
> reducing scoring inaccuracies.

Section 4.2 rules out anything more elaborate:

> Moreover, compared to LAVAD, EventVAD's prompt setup is very
> straightforward. We achieve multi-stage reasoning in MLLM through
> hierarchical prompting to enhance its scene-understanding capability.

and section 4.3 identifies which line Table 5's third column ablates:

> (3) the implementation of deliberative reasoning in MLLM outputs before
> anomaly scoring

> In addition, the quantitative experiments show the structured MLLM output
> can be specified, so that it can summarize the video content before
> outputting the abnormal score, and this process of letting the model think
> can better help him understand these small fragments and give a more
> reasonable score.

`prompt.py` therefore builds

```
#Question: Are there any obvious or potential anomalies in the video?
#Instruction: Let's think step by step to judge the anomaly of the video. Finally, output the anomaly score based on the thinking process.
```

as arm `paper`, the default and the number to quote. `#Answer:` is the model's
turn, so it is not part of the input; `mm_infer` appends VideoLLaMA2's own chat
template and generation prompt after this text.

Two named alternatives exist, both second conditions on the same test split:

* `no_thinking` drops the `#Instruction` line, the Table 5 "Thinking = ✗" row.
* `bounded` appends one sentence fixing the score range. See G2-c.

**Three choices are inferences.**

**G2-a. "Hierarchical prompting" is one call, not two.** Section 3.4's "two
stage reasoning framework" describes the two roles inside the single answer --
describe, then score -- and Figure 2's `#Answer` shows both in one continuous
output ending "Therefore, the final score is 0.8". Upstream's `event_score.py`
makes exactly one `mm_infer` call per segment, with `do_sample=False`, and the
port keeps that: one call per event, greedy, no sampling and no aggregation.

**G2-b. The parser must read prose.** Upstream's `float(output.strip())` is
kept as the first thing tried, so an output it could have read is read
identically. The sentence patterns -- "the final score is X", "score: X" --
and a trailing-number fallback follow. Every event records which rule fired,
and `frame_eval.json` reports the histogram, so a run that mostly falls through
to the fallback is visible rather than silent.

**G2-c. The score range is `[0, 1]`, and out-of-range answers are rescaled.**
The paper says scores are evaluated "against predefined criteria" and never
states the criteria; Figure 2's single example is `0.8`. Frame-level ROC-AUC
is computed **pooled across videos** (`frame_eval_common.evaluate`), so a
model that answers `8` on one event and `0.8` on another would corrupt the
ranking, not merely rescale it. `normalise_score` divides by the smallest
containing decade (10 or 100) and clamps beyond that, and every rescale is
counted in `range_rules`. If that counter is large on the real run, the
`bounded` arm -- which states the range in the prompt -- is the honest
condition to report beside it, and the difference between them is a
measurement rather than a guess.

---

## 4. Reconstructing G3: the RAFT checkpoint

`raft-things.pth` is the entry named by the placeholder path. It comes from
the `models.zip` that `princeton-vl/RAFT`'s own `download_models.sh` fetches:

    https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip

installed at `/home/jehc223/data/checkpoints/raft/`.

| file | sha256 |
| --- | --- |
| `models.zip` | `4be6101b271f58ec49866da5cf609fd17e86e9cae2483f70630ef4a295dc66bd` |
| **`raft-things.pth`** | **`fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`** |
| `raft-chairs.pth` | `c6c75465cf995d137f89ca2e2d08594ed390befbb8859f7f65c48bcc8feb0fd7` |
| `raft-kitti.pth` | `b9d170362415e1a27bd8402ee966a3ddf0d60df9b2df2c0b4949f5ced490a9e6` |
| `raft-sintel.pth` | `90630d2e7d488a0d3ccb5e8194524850c4c05c732ea4ff99799822c7fa5c5cbf` |
| `raft-small.pth` | `c7d41b9cc88442bb8aa911dbb33086dac55a226394b142937ff22d5578717332` |

It loads into `RAFT(Namespace(small=False, mixed_precision=False,
alternate_corr=False, dropout=0.0))` -- upstream's exact argument set -- with
`strict=True` after stripping the `module.` prefix, 5 257 536 parameters.

---

## 5. Where the code and the paper disagree, and which wins

The released config was never executed (G1), so the paper is the authority.
Both readings are reachable; `--preset upstream` selects the config literals.

| quantity | paper | `config.py` | default here |
| --- | --- | --- | --- |
| α, semantic-motion fusion | **0.75** | `clip_weight = 0.8` | 0.75 |
| γ, time decay | **0.6** | `time_decay = 0.05` | 0.6 |
| CLIP L2-normalised in the node feature | **yes**, Eq. (1) → Eq. (3) | only inside the similarity | yes |
| moving average | centred, Eq. (11) | trailing, `mode='valid'` | **trailing** (see G7) |
| Savitzky-Golay width | 60 at 30 fps | `ema_window = 2.0` s | 2.0 s |
| MAD multiplier | 3 | 3.0 | 3.0 |
| GAT iterations | 1 | 1 | 1 |
| projection dim k | 64 | 64 | 64 |
| fused dim d | 640 | 640 | 640 |
| decode FPS | 30 | native | min(native, 30) |

α = 0.75 and γ = 0.6 are not arbitrary paper values: Table 4 is a 5 × 6 grid
over exactly those two, and section 4.3 closes with

> To maintain a good semantic correlation between frames while having a
> certain time continuity constraint, we choose α = 0.75 and γ = 0.6 for
> segmentation.

Neither 0.8 nor 0.05 appears anywhere in that grid.

### G7. The one place the paper is wrong and the code is right

Eq. (11) defines the moving average as **centred**:

> μᵢ = (1/w) Σ_{k=i−⌊w/2⌋}^{i+⌊w/2⌋} s̃ᵢ

and Eq. (12) divides the smoothed signal by it at the same index. The released
code instead computes `np.convolve(s_smoothed, ones(w)/w, mode='valid')` and
pairs `s_smoothed[w−1+j]` with `ema[j]`, which is a window **trailing** the
sample.

Measured, on the 400-frame synthetic in `segment_events.selftest`, with regime
changes planted at frames 150 and 270:

* the raw divergence sᵢ peaks at exactly 150 and 270 -- the signal is there;
* a Savitzky-Golay filter of width 60 spreads that one-frame peak into a bump
  roughly 60 wide and drops its height from 4.25 to 0.96 against a baseline
  of 0.60;
* the **centred** 60-wide average then covers the bump it is being compared
  against. The ratio tops out at **1.194** against a threshold of **1.281**,
  and **nothing is detected**;
* the **trailing** window compares the bump against the quiet stretch
  preceding it, reaches **1.602** against a threshold of **1.577**, and fires.

A centred normaliser cannot detect a change whose width is its own window.
Eq. (11) as printed is not what produced the paper's numbers, so `ma_mode`
defaults to `upstream`. `centered` is kept so the claim stays checkable, and
`trailing_aligned` keeps the trailing comparison while fixing the index --
upstream reports boundaries at `j + w//2` where the ratio at `j` actually
describes frame `j + w`, leaving them about `w/2` frames, a second at 30 fps,
early.

---

## 6. Choices that are inferences, in one list

Everything below is a decision this port made that the paper and the code
together do not determine. Each is a named flag.

| id | choice | default | flag |
| --- | --- | --- | --- |
| G1-a | `dₐ = k = 64` in the attention scale | 64 | `ortho_dim` |
| G1-b | softmax over neighbours of scalar logits | -- | -- |
| G1-c | `E₍ᵢ,ⱼ₎` is the Eq. (4) edge weight | `weight` | `--gat-edge-term` |
| G1-d | seed 42 for Q, K, V | 42 | `ortho_seed` |
| G2-a | one MLLM call per event, greedy | -- | -- |
| G2-b | prose-tolerant score parser | -- | -- |
| G2-c | score range `[0, 1]`, decade rescale | `paper` arm | `--arm bounded` |
| G7 | trailing moving average | `upstream` | `--ma-mode` |
| P1 | α, γ, CLIP normalisation from the paper | `paper` | `--preset upstream` |
| P2 | decode at `min(native fps, 30)` | 30 | `--max-fps` |
| P3 | γ re-read per second across decode rates | `per_second` | `--gamma-mode` |
| P4 | events carried as boundaries, not re-encoded video | -- | -- |
| P5 | unparsed event scores fill with 0.0 and are counted | 0.0 | -- |

P1 through P5 are porting decisions and are argued in PATCHES.md; they are
listed here so the whole set of judgement calls sits in one place.

---

## 7. What this costs to run, and one thing to flag

**Deployment cost.** EventVAD makes **one MLLM call per event**, not per
video. On this study's corpora that is however many events the segmenter cuts,
which the run reports as `events_per_video_median` and `events_per_video_max`.
That is fine for a reproduced baseline and it is worth stating plainly that it
would not satisfy this project's own two-calls-per-video cap (CLAUDE.md,
anti-pattern 1) if EventVAD were being proposed as a method rather than
measured as a comparison. It is a baseline, so the cap does not apply; the
number is recorded so the comparison is read with its cost visible.

**Where the time goes.** Stage 1 is bound by RAFT, which runs once per adjacent
frame pair. Counted exactly, by probing every video in the three gold cohorts
at the capped decode rate:

| corpus | videos | frames | most common decoded size |
| --- | --- | --- | --- |
| hatemm | 214 | 820 923 | 854x480 (108 videos) |
| mhclip_en | 158 | 159 204 | 404x720 (136 videos) |
| mhclip_zh | 153 | 136 007 | 404x720 (59), 1280x720 (44) |
| **total** | **525** | **1 116 134** | |

Stage 2 is bound by VideoLLaMA2 generation, one call per event, so its cost is
`Σ events` and is not known until stage 1 has run.

**Neither stage has been timed on a GPU.** The GPU was held by another job for
the whole of this port's construction, so every number here was measured on
CPU, where the extractor runs at 1.0 to 3.5 frames per second depending on
resolution -- a floor, not an estimate. Both stages print a running
frames-per-second and an ETA from the first video onward and both are
resumable, and `run_all_eventvad.sh` orders the corpora shortest-first
(mhclip_zh, mhclip_en, hatemm) so the throughput is known from real data after
about a tenth of the total work. Read it there before committing to HateMM,
which is three quarters of the frames.
