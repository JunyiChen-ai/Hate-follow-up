# M1 / M3 novelty team — charter (2026-10-25)

## User request (verbatim, 2026-10-25)
"我觉得模块的 novelty，第二个模块在我的 perspective 看来是达到了，但是我觉得第一个模块和第三个模块还没有达到 …
虽然他们有用，但是没有达到我对 novelty 的一个 criteria … 我希望在 improve 剩余两个模块，或者说至少其中一个模块的这种
novelty 和 fanciness … 或许可以看看其他的顶会工作有没有什么思路是我们可以借鉴的，或者改进的 … 现在第一个模块和第三个模块
非常不好讲故事 … core novelty 其实全部都在第二个模块，所以说我希望还有至少一个模块它有 novelty 可以 claim。
你可以 Set up 一个 Research team，从不同角度，根据我们过往已经尝试过的大量实验，再看一下如果要做出 Novelty，还可以往哪些
方面去做。主要 Focus 在第一个模块和第三个模块。"

## Goal of this team
Deliver a ranked set of concrete redesign directions for M1 (window read) and/or M3 (video key + fusion), each:
1. a mechanism-level redesign of the existing module, not an add-on module;
2. with a claimable novelty: closest prior work named (top venue, verified), and the difference stated;
3. not a re-run of a closed family in this repo (closest past attempt named, result, what is different);
4. with a pre-check that can falsify it cheaply (CPU on existing caches if possible, else GPU minutes), and cost;
5. with a story line: which named failure of the naive form it answers (per paradigm_framing.md §3 pattern).

## Current method r6_bma (authoritative description: experiments/20260926_twolevel/README.md §19; framing research-wiki/paradigm_framing.md)
- M1 window read: frozen Qwen3-VL-8B; shared cached prefix = rules + 20 frames + full timestamped Whisper transcript;
  model gives the video verdict z_video (Yes/No log-odds), then a stance turn; then isolated per-8 s window branches,
  a visual and a speech Yes/No question per window, each branch forked from the same prefix.
- M2 (novelty accepted by the user): per-modality normal-score transform, per-video two-class EM Gaussians,
  explicit-duration negative-binomial chains (min 2 windows, BMA over lengths), OR across modalities.
- M3 video key + fusion: K = z_video + mean window z (raw M1 window reads), constant within the video;
  frame score = K + within-video centred rank of the M2 posterior. NO calibration: the user removed the two-Gaussian
  key calibration on 2026-10-07 (code flag `--key raw` in experiments/20260926_twolevel/twolevel_r2.py; STATUS still
  says calib, which is stale). Reference numbers of the current method (pooled ROC / pooled PR / within):
  HateMM .8955 / .6883 / .7508; HCS .7137 / .6666 / .6373 (runs/20260926_twolevel/final_rawkey/metrics.json);
  DeHate .699 / .157 / .643 (runs/20260927_dehate_external/r6_rawkey/metrics.json). Compare against these only;
  never add a "calibrated" column.

## User's novelty criteria (from memory files; must be respected)
- Redesign each working component along its own mechanism, less naive; not "add another thing on top"; not chosen by
  which error type is largest. Conditions: metrics do not drop; ablation shows the component still matters.
- Method paper only. No "analysis paper", no "measurement design" framing for M1. M1 needs a mechanism.
- RESEARCH_ITERATION_RULES.md: rule 3 no ensemble / no post-processing / no per-corpus constants; rule 4 STOP conditions;
  rule 5 a new input is not novelty; rule 13 one uniform method; rule 14g each claimed component's removal drops
  >= .01 on at least one main metric on both HateMM and HateClipSeg.
- Cost (CLAUDE.md): avoid many repeated calls per window; report added calls per video and GPU time.
- HCS keeps the protocol GT (any non-normal dimension counts as positive); user ruling 2026-10-25.
- Main corpora: HateMM, HateClipSeg; DeHate is also a main corpus since 2026-10-08 (report it).

## Facts already established (do not re-derive; cite)
- M3 round 2026-10-25 (experiments/20261025_m3_coverage/README.md): candidates 47-49 archived, 39 CPU composition
  variants failed, DVD target condition re-decoded. With M1 and M2 fixed, M3 composition room is about +.003-.005.
  New video-level questions on the full prefix collapse to the verdict (rho .965/.904); context-free segment reads
  lose coverage information (rho with GT share .278/.419 vs window mean .416/.533; partial rho given window mean <= 0).
  Changing the video level toward the "hate" definition loses on HCS protocol GT, gains on DeHate.
- M1 candidates 41-46 archived (grounded probe, pairwise, Omni audio, speaker, generation-side, topic removal):
  reader-side changes did not move the window order. Only input changes gained (re-show in-window frames; aligned
  word slice), and under rule 5 those are inputs, not novelty.
- Module novelty evaluation 2026-10-25: scratchpad novelty_eval/{rubric,M1_report,M2_report,M3_report}.md.

## Team
Round 1 (parallel): history auditor; literature scout for M1; literature scout for M3.
Round 2 (parallel): M1 designer; M3 designer; cross-module / story designer.
Round 3: adversarial reviewer on all proposals (novelty search, closed-family check, rule check).
Lead: synthesis, cheap pre-checks on existing caches where possible, report to the user.

## Output locations
scratchpad/m1m3_novelty/{history.md, lit_m1.md, lit_m3.md, design_m1.md, design_m3.md, design_story.md, review.md, lead_log.md}
No repo edits by team members. No GPU jobs without the lead's approval.
