# M1 visual contrastive reading

Declared2026-10-03, eighth candidate. Proposal and independent code review PASS; implementation complete,
five-video GPU plumbing passed; no performance yet. Contraster completed without qualifying gain after this
declaration. Development host sc474397; GPU target chosen after live inspection.

## Mechanism and source

The current visual window branch sees full speech and the global verdict. Its
answer can therefore remain high even when the visual evidence is weak. Instead
of deleting context or attributing the global decision, compare the SAME visual
window question with clean and deliberately degraded images, retaining the
entire transcript, token sequence, timestamps and native selected verdict.
The hypothesis is that amplifying the part of each local visual response
sensitive to image evidence improves raw within-video ordering. It is not a
guarantee of truthful evidence: corruption can change real visual semantics or
induce unrelated distribution shift. Whole-prefix corruption is not a temporal
local intervention; any claim of temporally specific correction needs controls.

Source: [Visual Contrastive Decoding, CVPR2024](https://arxiv.org/pdf/2311.16922),
with [author noise implementation](https://raw.githubusercontent.com/DAMO-NLP-SG/VCD/master/vcd_utils/vcd_add_noise.py).
We transfer input-contrastive reading to local binary visual decisions. Unlike
Eraser, this uses window queries with a retained clean read, not differences in
the global answer after deleting a window. Unlike Contraster, it changes visual
inputs and recomputes the full prefix; it does not compare model depths.
Neither input contrast nor VCD is a new algorithm claim. Independent rule4 review
must search prior hateful-video detection/localization use, not only matching
the method name. Do not claim source hallucination results transfer to hate.
Review PASS: `docs/reviews/20261003_m1_visual_contrast_proposal.md`.
[MMSafeAware, ACL2025](https://aclanthology.org/2025.acl-long.832/) already uses
clean/noisy VCD for static image-prompt safety assessment, including a category
containing hate speech. Neither first harmful-content use nor first binary
safety use is available as a claim. Only effective video-localization transfer
is being tested. Inspectable Qwen processor source confirms C,temporal,patch,
patch flattening with temporal copies produced by expand; noise shares that axis.

## Exact computation and constants

- Native Qwen3-VL-8B, BF16,20frames, repaired ASR, policy, global/visual/speech
  prompts,8-second windows,4fps and native answer unchanged. Record the full
  native arm. Native global margin z_v is the only global score used downstream.
- Make a fresh copy of processor-normalized image pixels, preserving every
  image grid, patch count and image/text token. Corrupt pixels in FP32 using
  the author's1000-step sigmoid schedule:
  beta_j=1e-5+(.005-1e-5)*sigmoid(linspace(-6,6,1000)_j),
  abar=product_{j=0}^{500}(1-beta_j), x'=sqrt(abar)*x+sqrt(1-abar)*epsilon.
  Fixed zero-based noise_step500, seed0 reset per video, CPU torch.Generator;
  independent normal noise per RGB spatial pixel and per frame. Qwen's two
  repeated temporal copies of each still image share the same noise; verify
  processor patch ordering and temporal copies before use. No clipping or
  re-normalization, no random content-derived seed, no augmentation averaging.
- Fresh corrupted-prefix encoding, with rope state reset; recompute the global
  question/answer extension on that prefix but force the ORIGINAL native answer.
  Do not use corrupted global margin downstream. Ask only the original visual
  window branches on this prefix; all original speech branch margins stay fixed.
- Let a_i,b_i be native Yes/No class log-odds for the clean/corrupt visual query.
  The new visual margin is2*a_i-b_i. This is coefficient1 contrast applied to
  binary CLASS probabilities after aggregating the native label token sets,
  not logit subtraction between individual spelling variants. Consequently
  rare spellings cannot dominate solely through token-wise ratios. This class
  adaptation is explicitly different from the source's token decoding.
- No source APC truncation: both binary alternatives remain finite to support
  ranking. That is a declared adaptation, not an exact VCD reproduction. Raw
  visual/speech maximum and r6 algorithm unchanged; fit each arm without labels.
  Native visual queries are still made for windows without sampled frames;
  their count and behavior must be reported, with no invented local image data.

## Evaluation and falsification

One proposal review then implementation and one code review. CPU checks must
verify pixel-only intervention, unchanged text/grid, shared temporal noise,
determinism, contrast-coefficient0 and abar=1 identity, class contrast algebra and source-noise
schedule. Five-video GPU plumbing: first2manifest videos per corpus plus
HateMM/hate_video_114(largest native prefix from metadata); no GT/performance.
Check all native global/window reads exactly reproduce base_gridA, fresh native
restoration, no mutated inputs, numeric finiteness,4fps and cost.

Full333 paired native/visual-contrast read, canonical raw and r6 metrics, three
metrics on both corpora, paired-video bootstrap2000 seed0. Target within+.01
both, no pooled loss>.005 or within loss>.01 vs paired/current. Any final main
gain>=.01 permits up to3 declared revisions; no qualifying gain means archive.

Only after a qualifying gain: remove contrast(native), permute corrupted visual
margins across windows within each video(default_rng0; record effective changes),
common-shift control preserving native raw-max order, and a clean-only visual
rescaling2*a_i control(remove image-dependent subtraction). The latter distinguishes
image-dependent correction from merely changing visual/speech relative scale.
Also reconstruct2*a_i-mean_j(b_j) to separate video-level corrupted-image shift
from window-matched subtraction; no extra model calls. This is a diagnostic,
not a guarantee that the corrupted view contains only language bias.
If needed, compare matched-noise interventions restricted to own versus other
windows before claiming temporal specificity; this costly control is not part
of the main run. Full-prefix sensitivity alone cannot establish local grounding.
Every claimed component must pass14g on both corpora. Report raw ranking and
decoded changes, image coverage, branch dominance, native-verdict strata and
gain/loss cases. All evidence development-selected.

## Inputs, cost and read log

Reuse current frames/ASR/weights; no new encoder. One additional full-prefix
encoding and B_visual extra local queries per new video. Native3+B_visual+B_speech
outer forwards; deployed paired/contrast6+2*B_visual+B_speech. No backwards,
no per-window prefix reconstruction, no free-offline-processing claim. Estimate
15-30GPUmin for full333 on5090, replace using smoke. Prefixes can be processed
sequentially to avoid retaining two KV caches. Output
`runs/20261003_m1_visual_contrast/`; no inputs written back to data caches.

Sources already read: STATUS; archived Eraser/Factorizer reports; current reader
and Qwen input layout; source VCD. Eraser and Factorizer raw/final losses motivate
retaining full semantic context, not a claim that noise contrast will work.
No pending Contraster performance or new GT was read for this declaration.
Current method and paper remain unchanged pending evidence.

Implementation and CPU checks completed before performance. Source
`runs/20261003_m1_visual_contrast/selfcheck/noise.json`: noise_step500 gives
abar=.74479872; double-precision schedule reference differs by1.94e-7. Explicit
abar=1 preserves inputs exactly, seed reset reproduces corruption, temporal
copies stay equal and distinct spatial patches receive distinct noise. Class
probability contrast equals2a-b; coefficient0 or identical reads return native.
Reader checks every original input tensor for mutation, keeps corrupted global
margin diagnostic-only, and smoke re-encodes the original prefix after noisy
reading to verify all native globals/branches. Main6+B+V forwards; smoke adds
3+B for that restoration. Independent code review PASS: `docs/reviews/20261003_m1_visual_contrast_code.md`;
independent processor/model/report checks in
`runs/20261003_m1_visual_contrast/independent_review/check_visual_contrast.json`.
No GPU measurement yet.


## GPU plumbing, before performance

2026-10-03, host sc474399 (uoa-lab2), torch2.11.0+cu128/transformers5.15.1.
Five declared videos completed in32.8s including native restoration. Returned
artifacts: `runs/20261003_m1_visual_contrast/r1_smoke/plumbing_summary.json`.
All5 native globals and292 branch margins exactly match base_gridA; all native
restoration reads exact, speech unchanged, pixel identity/determinism/input
immutability, contrast algebra, forward counts and4fps pass. Maximum5829-token
prefix uses17.8235GiB combined peak. Contrast standalone measured1.79/1.16s on
first2 HateMM,3.02/3.74s on first2 HCS; longest stress video10.46s.
First2-per-corpus estimate for full215/118:5.29/6.64min,11.93min total; this is
only a small-sample estimate, stress case not substituted for a corpus average.
Proceed to fixed r1_main full333 paired run, then canonical raw/r6 evaluation.
No GT or candidate performance read in smoke. No method/constants changed.

## R1 complete; retained under rule9, not promoted

Host sc474399, full333 paired read912.9s (15.22min); returned locally before
evaluation. All333 global and13,939 native branch margins exactly reproduce
base_gridA. Canonical final source
`runs/20261003_m1_visual_contrast/r1_main_decoded/{base,contrast}/metrics.json`;
raw source `r1_main/{base,contrast}/metrics.json`; report `r1_main_analysis/`.

| Corpus | Arm | ROC | PR | within | raw within |
|---|---|---:|---:|---:|---:|
| HateMM | native | .897119 | .694235 | .750782 | .680011 |
| HateMM | contrast | .892394 | .680132 | .770392 | .663754 |
| HateClipSeg | native | .716825 | .671072 | .637349 | .610130 |
| HateClipSeg | contrast | .724669 | .672273 | .648014 | .621923 |

Within gain+.019610 (n84, paired95%CI[-.000306,.044377]) /+.010665
(n99,[-.004593,.028197]). Both meet point-estimate gain threshold, but HateMM
PR−.014103 violates the no-loss gate. Raw within−.016257/+.011793; not evidence
of universally better raw window ordering. Two largest HateMM final gains
(.64 and .60 on hate_video_279/329) both have unchanged raw within and account
for most aggregate gain. Need determine what new signal and decoder interaction
produce this. Do not claim robust error correction: wrong-No strata have n3/5.

Cost (standalone): native303.49/246.33s, contrast507.19/404.41s; mean forwards
36.52/60.05 vs57.05/93.48. Combined peak17.824/17.547GiB. No new preprocessing.
All evidence development-selected. Default/paper unchanged. Rule9 authorizes
up to3 declared revisions, but primary mechanistic controls run first.

### Cached mechanism controls: execution definition before their results

The earlier declaration authorized these controls only after a qualifying gain.
They now run on the saved clean a_i/noisy b_i/speech c_i with no new MLLM calls.
All retain native global z_v, original windows/availability, and independently
fit unchanged r6. The exact common-shift and RNG details are fixed here before
execution, not selected from performance:

1. `scale`: visual2a_i, original speech. With corpus normal-score transform this
   positive visual rescaling should preserve the r6 temporal input exactly;
   any within drift beyond roundoff is an implementation concern.
2. `video_shift`: visual2a_i-mean_j(b_j), original speech. Removes window matching
   from the noisy read while preserving each video's clean visual ordering.
3. `shuffle`: visual2a_i-b_perm(i), original speech. Reset default_rng(0) per
   video, permute all visual windows; record changed indices and changed values.
   This breaks matching, not video-level noisy mean or marginal distribution.
4. `common_shift`: let d_v=mean_i(max(2a_i-b_i,c_i))-mean_i(max(a_i,c_i)), with
   absent speech omitted. Add d_v to BOTH native branches. Native raw-max order
   stays unchanged and its video mean equals the contrast mean; downstream
   global key is consequently matched, but no new local ordering is introduced.

Report each arm's three canonical raw/final metrics, paired within differences
versus native and contrast, individual visual/speech raw ordering, largest
gain/loss examples and original-verdict strata. A shuffled/constant control
matching the candidate would defeat the window-grounding explanation. No
mechanism claim until this is tested. This is a control phase, not a new
performance-selected method revision.

R1 case inspection before controls: `r1_main/{base,contrast}/predictions.jsonl`,
`r1_main/checks.jsonl`, `r1_main_decoded/{base,contrast}/predictions.jsonl` and
current repaired ASR were read for hate_video_279/329. The former has2 windows
and transcript "Outro Music"; original global−11.704 remains unchanged. The
latter has ONLY1 read window (7.105s), so both raw curves are constant and within
is.5; r6 splits into2 temporal cells whose ordering flips, producing+.60 final
within. This is a decoder interaction, not new within-window timing evidence.
The former's rawmax ordering also stays unchanged. Report such cases separately
when assessing a grounding claim; do not exclude them from main metrics.
