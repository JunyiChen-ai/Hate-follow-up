# Code review (rule 6): candidate 41 grounded acceptance, 2026-10-10

Reviewer: one independent agent (same model as the main session), on commit 1be1bc0, using the installed
transformers 5.15.1 Qwen3-VL and cache code, CPU checks with the real Qwen3-VL-8B processor, and the finished
fixed-five smoke (`runs/20261010_m1_grounded_adjacent/smoke`, 76 probed windows).

## Findings

1. BLOCKER. The citation match `|x − t| ≤ .05` fails in floating point for frames whose time sits on the .1 f
   rounding boundary: 109 of 6658 native frames (108 of 5096 covered windows, 2.1 %) are labelled with a value
   whose distance from the frame time is .05000000000000071 (e.g. 18.55 → `[t=18.6s]`). The model quotes the label
   verbatim, so a correct citation of such a frame would have been scored as not grounded and the declared
   decision inverted for that window. Fixed before the 333 run: tolerance + 1e-9 in `grounded.parse`,
   `analyze.prepare` and `selfcheck.py`. The README's ".05 s" stays.
2. MINOR. "share of cited times inside the window" was computed as the grounded count. Fixed: counts replies with a
   number inside [start, end).
3. MINOR. "partial" required no loss beyond noise in both corpora; the README says in the other corpus. Fixed to
   the README's rule.
4. Note. The number regex would match a reply echoing window bounds; all 76 smoke replies are a label or "none".

## Answers

- Adjacent read identical to `placement_controls.suffix_margin` (same string, ids, grid, mask handling,
  positions); the user-turn boundary found by the assistant-header ids is valid (the turn ends in
  `<|im_end|>\n` regardless of what follows); smoke replay max |Δz| 0 over 76 windows; base equals r6.
- Probe positions: get_rope_index over user + probe ids sliced equals a single pass over the whole
  conversation; after an image block text positions continue from the max over axes + 1 on all axes, so the
  greedy steps' positions are what the model would assign. rope_deltas untouched by explicit-position forwards
  and restored; `stance_cache.margin` resets it itself. Cache cropped to prefix + user turn, then to the prefix.
- Decision rule and parsing as declared (except finding 1). No GT in the reading or decision path. Window dicts
  carry only the r6 modalities. Call-count assertions consistent and held on the smoke. Binding checks test what
  they claim; the 18-frame video and empty windows handled; controls built as declared without labels.

Verdict before the fixes: FAIL (finding 1). After the fixes the CPU fixture check passes again; the fixed five are
re-run inside the main job (`SCOPE=both`).
