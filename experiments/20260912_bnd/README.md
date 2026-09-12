# BND — asking about onset instead of presence (label-free)

Status: **2026-09-12 archived as a negative result.** Development-selected numbers only (rule 10).

## 1. Why this was worth one run

Every ceiling measured on 2026-09-12 asks the model *"is this window violating"* and ranks the answers.
The diagnosed confound — the per-window judgement is substantially topic detection — is **near-constant
inside a video**, because a hateful video talks about the same group throughout. `experiments/20260912_tad/`
tried to difference that constant at the **score** level (subtract an auxiliary topic read) and failed,
because the topic dimension is itself positively predictive of the label.

Asking about **change** differences it at the **read** level instead: at the onset of hate inside a
topically homogeneous video the topic does not change, the act does. It is also the quantity a temporal
localization task actually needs. `experiments/20260911_hvl/` tried `start / continue / stop / none` only
as a *sequential chain with self-feedback*, which died of persistence (77–82 % of answers were "continue");
independent boundary reads had never been tried.

## 2. Method

Two extra reads per window, on the same prefix, stance turn and cache path as SPVL-r2:

```
BEGIN: "Considering the video up to this point, does the content that violates the above rules BEGIN in
        THIS window? Answer Yes only if such content is absent from the earlier part of the video and
        present from this window onward; answer No if it was already present before this window, or if
        there is no such content here."
END:   the same for the last window of a violating stretch.
```

Three parameter-free, label-free curves are built from them and compared against the act read:
`begin` alone; `cumulative = cumsum(sigmoid(begin) − sigmoid(end))`, a presence state integrated from the
two reads; and the equal-weight rank sum of the act read and `cumulative`.

## 3. Result (`runs/20260912_bnd/e0/`, within subsets, 183 videos)

Window-level within-video AUC:

| curve | HateMM (75 videos) | HCS (96 videos) |
|---|---|---|
| act read (baseline) | **.7581** | **.6212** |
| begin | .6466 (−.112) | .5605 (−.061) |
| cumulative | .6146 (−.144) | .5265 (−.095) |
| act + cumulative, rank sum | .6956 (−.063) | .5949 (−.026) |

Diagnostics:

- the begin read is **not** a renamed act read: median Spearman .644 / .731, well below the .9 degeneracy
  threshold. It is a separate measurement — just a worse one.
- the model does have some sense of onset: the argmax of the begin read is the first GT-positive window
  on 22.7 % / 15.6 % of videos, against a chance rate of roughly 1/N ≈ 4–5 %. Above chance, far from
  usable.

**Archived.** Asking about change does not difference out the topic confound; it produces a weaker signal
about a harder question.

## 4. What this closes

With BND, every form of the question has been tried on the frozen model at 8-second granularity:

| form of the question | result |
|---|---|
| is this window violating (absolute) | .758 / .621 — the ceiling |
| which of these two windows is more violating (relative) | same band (`experiments/20260912_pwc/`) |
| is this window violating, minus does it mention a protected group | worse (`experiments/20260912_tad/` r1) |
| what does this window do towards the group (five-way speech act) | collapses to binary (TAD r2) |
| does the violating content begin / end here (change) | worse (this experiment) |

Together with the four GT-supervised ceilings (feature combination, post-decision hidden state,
pre-decision prefix representation) and six rounds of label-free adaptation, the boundary of the frozen
prompting paradigm is measured rather than assumed.
