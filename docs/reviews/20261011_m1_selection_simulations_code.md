# Independent code review: offline selection simulations (2026-10-11)

Scope (read-only, same model as the main session): the three offline scripts whose negative results drive the
decision to abandon label-free per-window selection of the extra visual read —
`experiments/20261007_m1_streamingtom/selection_analysis.py` (pre-read rules),
`experiments/20261010_m1_grounded_adjacent/error_analysis.py` + `error_analysis_mixed.py` (candidate 41 error
analysis), `experiments/20261007_m1_streamingtom/combination_analysis.py` (averages, agreement, neighbour rules).
The GPU reading code behind the stored reads had its own reviews (2026-10-09 placement controls, 2026-10-10
candidate 41). Anchors already established: each script's full / accept-all rule reproduces the stored arm's decoded
metrics exactly and the two oracles agree, so the shared path (records, hybrid construction, sole evaluator, fixed
r6 decoder) was not in question; the review targeted per-rule logic, controls, oracles, tables, GT matching.

## Verdict

No blocker. None of the four negative conclusions (pre-read rules; model citation; agreement between reads;
neighbour agreement) is produced by a logic error. Two controls were implemented defectively; both were re-run
correctly by the reviewer (scratch, same evaluator and decoder flags) and the conclusions did not move.

## Findings

1. `selection_analysis.py` random control (minor, pre-read rules only). `random_q50_{seed}` re-created the
   generator per video, so every video with the same window count received the same window positions (one distinct
   subset per length; e.g. the 12 HateClipSeg videos with 27 windows all took the same 13 positions). Re-run with
   six independent per-video seeds, within gain vs r6 (HateMM | HateClipSeg): `adjacent_native` halves +.0150 +.0166
   +.0039 +.0113 +.0071 +.0180 | +.0087 +.0067 +.0097 +.0012 +.0081 +.0091 (stored two seeds +.0096 / +.0140 |
   +.0121 / +.0054); `local_clean` halves +.0132 +.0207 +.0018 +.0136 +.0107 +.0187 | +.0134 +.0229 +.0100 +.0178
   +.0123 +.0136. Against this band `adjacent_native unc_q50` (+.0167 | +.0104) is inside on HateMM and .0007 above
   the best of six on HateClipSeg; `local_clean unc_q50` (+.0208 | +.0132) is .0001 above the HateMM maximum and
   inside on HateClipSeg. No rule is separated from independent random halves on both corpora. Fixed in the script
   (generator seeded per video, six seeds) and re-run after the review.
2. `combination_analysis.py` neighbour rule (minor, neighbour conclusion). `neighbour_a` had no rate-matched random
   control, its acceptance rate was not reported, and covered windows with no covered neighbour were accepted
   unconditionally (968 of 5096 covered windows, 19 %); overall acceptance 78.0 % / 75.3 %. Reviewer's matched
   control (two seeds): +.0184 / +.0236 (HateMM), +.0119 / +.0106 (HateClipSeg) against the rule's +.0264 / +.0092:
   .0028 above the better HateMM seed, below both HateClipSeg seeds. Fixed: acceptance rate and six-seed matched
   control added; the isolated-window behaviour is kept as run and stated in the code.
3. `error_analysis_mixed.py` (minor, transcription). The mixed-video count came from the GT file (HateClipSeg 100),
   but the run has 118 of the 119 GT videos and the absent one is mixed, so all per-video comparisons and the
   evaluator use 99; the README said "84 / 100". Numbers unaffected. Count and README corrected.
4. Window label at the tail (all three scripts; no effect). GT `y4` is one frame shorter than ceil(duration×4) for
   325 of 333 videos; 11 tail windows shorter than .25 s start beyond the GT and get label 0. None has an
   `adjacent_native` read, and the evaluator truncates to the common length, so oracles and tables are unaffected.
5. `error_analysis.py` probe class order (minor, no effect in the data). A reply containing both "none" and a cited
   shown time would be classed `none` while the decision used `grounded`; 0 such replies. `z == v` counted as down:
   0 windows.

## Checked and found correct

Rule definitions of script 1 (|v| thresholds on the native visual margin; per-video halves by stable argsort;
speech disagreement; empty / covered; oracle on the visual direction) with selection fractions recomputed
independently (unc_q50 .3556 / .3255 = summary); script 2 rules and tables recomputed from raw records (e.g.
HateMM "up shown" n = 331, positive share .5952; "down none" fused-changed 623, help rate .9133), random control
rate-matched with one generator per seed, oracle definitions as documented; script 3 agreement logic (same nonzero
sign of read − v for all candidates; missing read keeps v; `a` present implies `l` and `p` present, 0 exceptions over
7454 windows, so controls are exactly rate matched; `default_rng([seed, vi])` independent per video; agree_al
acceptance .817 = summary); hybrid construction identical to the stored arms' prediction builders and to
`analyze.py`'s curve check; decoder reads the same fields; GT ids: all 215 + 118 run ids in the GT, evaluator
intersects ids.

## Design limitation stated by the reviewer

With independent random halves the within metric of a 50 % application of the same read spans .002 to .021 on
HateMM across six draws. "Inside the random band" therefore means "not distinguishable with this design"; a rule
whose true advantage over random is below roughly .01 to .015 within cannot be detected on these corpora. The
oracles show that headroom exists; the negative results say the tested rules are not detectably better than
random, not that no rule can be.
