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
