# M1 Stabilizer: head-dependent temporal RoPE phases during visual encoding

Input clarification before any real-video run:20 is the original sampling budget;
the unchanged native cache has20frames for332videos and18 for HCS
`bit_AxrVklzh9Cyf`, as permitted in `data/frames_k20/PROVENANCE.md`. Use all existing
native frames, do not synthesize/re-extract missing frames; record actual counts.
The reader/validator count assertions now use that actual list. This was found
in Highlighter's noGT full prepare; no performance was read to make the correction.

Candidate16, declared2026-10-03 while Highlighter full run is being launched,
before its performance is read. Independent source mechanism, not combined with
any prior intervention. Frozen Qwen3-VL-8B, original20image frame input, ASR/policy/
questions,8s windows,4fps and unchanged r6/canonical evaluator. Development-selected.
No implementation before independent rule4 proposal review.

## Hypothesis and verified sources

PAS, *A Training-Free Stabilizer for Temporal Encoding in Video LLMs*,
Sun et al., arXiv2511.10979v1:
https://arxiv.org/html/2511.10979v1 ; https://github.com/Bowen-Sun-0728/PAS .
Read sections3–4 and official `README.md`/`inference_eval.py`; read-only copies
in `third_party/pas_source_read/`. Source changes the temporal rotary phase of
visual-token queries by head group during prefix encoding. It does not add image
attention mass, restrict local keys or modify the final scoring rule.

Hypothesis: diversifying temporal phases across heads can reduce an incidental
positional preference in how different frames are integrated, improving video
and window evidence reading. Our native input is a sequence of timestamped
independent images, not the source's temporally merged video tokens. Its temporal
RoPE coordinate follows native multimodal position construction, not actual seconds.
The source's sampling/Nyquist interpretation therefore does not directly transfer.
This experiment tests a positional encoding intervention, not claims a proven
physical-time correction or new visual information.

The public demo explicitly says its scale differs from the paper and shows a
0/10 invocation although the parser default is0/.5. It rebuilds temporal inverse
frequencies from a shortened t-dimensional range, uses a contiguous dimension
mask and adjacent even/odd pairing. Qwen3 uses its native inv_freq, interleaved
axis layout and split-half rotary pairing. We use the paper's post-RoPE phase
operator with the ACTUAL Qwen3 frequencies/pairing, explicitly not demo-bitwise
reproduction. Native implementation inspected in transformers4.57.6; verify the
same mapping on the target5.15.1 before any full run.
The installed sc448960 transformers5.15.1 source was inspected remotely on
2026-10-03: it uses the same interleaved axis assignment and split-half pairing;
runtime tensor oracles are still required.

## Exact R1 definition before scoring

During the one native-shaped multimodal prefix forward, at all36language attention
layers AFTER q_norm and native RoPE, alter ONLY image-token rows of Q. No edits to
the vision encoder, DeepStack injection, K,V, attention masks or text query rows.
Let d=128, half=64. Use the model's actual rotary inv_freq and temporal axis mask:
initialize64true; set indices `1:3*mrope_section[1]:3` and
`2:3*mrope_section[2]:3` false, exactly mirroring apply_interleaved_mrope.
For native8B section[24,20,20],24frequency pairs remain temporal. Assert the mapping
against actual model rotary outputs, not only this assumed list.

For query head h=0..31 use delta_h=0 for even h and.5 for odd h. For temporal pair
(j,j+64), apply ordinary2D rotation with angle delta_h*inv_freq[j], after the
native rotation. Compute the extra rotation in FP32, cast back to BF16; preserve
all other coordinates and rows by direct assignment rather than identity matmul.
Keys and values at that layer remain untouched directly, but later-layer keys/
values and text states may change through the intervened earlier attention.
This is not a claim that all cached prefix representations stay native.

Delta.5 is in native temporal POSITION units, not frames, bins or seconds for
our multi-image input. All32heads/all36layers; no head selection, layer selection,
phase scan, extra video tokenization, extra frames or new preprocessing. The0/.5
pattern follows source default/group-cycling; it is not centered at zero despite
the source's general description of opposed offsets.

After prefix encoding, disable the intervention and read global question,
append that ARM'S own Yes/No answer, then read ordinary isolated visual/speech
questions on that prefix. Unlike prior candidates that held global reads fixed,
this intervention defines the entire shared M1 encoding and allows global/answer/
both branch values to change coherently. Native baseline independently re-encodes
the identical input and must exactly reproduce base_gridA. This is not a mixture
of best arms: main has one encoding, one global and its own answer on both corpora.
The same three-step cached execution and downstream r6 are otherwise unchanged.

## Costs and tests

Deployed3+B forwards (B=V+available speech), same as native; paired native+phase
collection6+2B. Initial full333 estimate20–25GPUmin paired,10–13min deployment on
5090; replace with real5video smoke. Extra rotations are O(image_tokens*heads*
temporal_dims*layers), but save measured wall/peak overhead. No multiple phase
model passes in the deployed method. Reuse all images/ASR/model assets.

Independent code review: actual36layer multimodal Qwen FP32/BF16,32/8heads,
20noncontiguousimages, native split-half/interleaved temporal coordinate oracle,
norm preservation before cast, literal unchanged spatial/text coordinates and K/V
at interception, no changes during global/window suffix calls, all36visits,
zero-phase native equality, native restore across videos, distinct arm-specific
global/answers, exact calls,4fps/333coverage/canonical analysis. First2/corpus plus
long-prefixHMM114 smoke with no GT; native full comparison and final3metrics.

Same acceptance: final within+.01 BOTH vs current/native; pooled no loss>.005,
within no loss>.01. No main metric gains>=.01 =>archive. Some gain>=.01 permits
rule9's maximum3revisions; no silent selection of different phases per corpus.
If main qualifies, required controls set ALL heads to.25 (same mean offset),
ALL heads to.5 (same nonzero phase, larger aggregate perturbation), and swap
the0/.5head groups. The.25control does not match per-head amplitude or squared
perturbation and cannot alone identify diversity. All.5 also changes the mean;
swap tests reliance on the chosen head identities. These are diagnostic controls,
not options to select the best main arm post hoc. Removing diversity must cost
>=.01 on a common main metric both to support it as a component; interpret any
positive result jointly with amplitude/group controls, with no general robustness
claim if it depends on only the selected odd heads. Also declare a matched temporal-phase-jitter
diagnostic before running it: compare score sensitivity to added uniform offsets
−.25/0/+.25 using exactly the same frames/text. It measures sensitivity to this
intervention, not frame resampling or true timestamp noise. Report raw visual,
speech/max ordering, final metrics, changes in global answers, unchanged-order/
single-window effects, uncertainty and inspected gain/loss cases.
Every jitter case adds its offset to all heads of the named arm, independently
re-encodes the prefix, recomputes its own global and answer, and re-reads both
branches. Fixed sensitivity summaries: per-video mean absolute change in raw
visual and available speech margins against that arm's zero-jitter values;
fraction of changed within-video pairwise orderings (ties counted as a distinct
order state), and global answer-flip fraction. Report both jitter signs separately
and their equal-weight average, then corpus video-macro means and paired video
bootstrap95%CI (2000draws,seed0). Do not choose the sign or subset after results.
Lower sensitivity alone is not higher correctness; score/GT comparisons remain
separate canonical reports. No confidence threshold or routing is fitted.

Independent proposal review2026-10-03 PASS: actual target-task search completed,
no rule4 STOP. `docs/reviews/20261003_m1_stabilizer_proposal.md`.

Source mathematical limits: the scalar-kernel approximation assumes conditions
on content-frequency coefficients; different learned heads followed by softmax
and o_proj are not a literal convex average of the same kernel. Finite-window
DFT magnitudes need not be invariant under arbitrary fractional phase shifts.
Do not inherit a stability/spectrum theorem without its conditions, or treat
norm-preserving rotations as proof of semantic robustness. Any claimed mechanism
needs our empirical component and sensitivity results.

Prior evidence read: prior candidate failures and native Qwen rotary code. No
Highlighter performance or new GT has been read for this candidate. Unlabeled
ASR-timing statistics and exploratory previous-score correlations did not isolate
a common timing-error explanation; no forced-alignment candidate was selected.

## Implementation and execution

Implementation2026-10-03: `stabilizer.py`, `measure.py`, `analyze.py`; no shared
scoring or evaluation source was changed. `selfcheck.py` passed FP32/BF16 scalar
rotation oracles (32heads/128coordinates), actual36layer multimodal Qwen encoding
with20noncontiguousimages, layer0KV identity, laterKV change, zero-phase/native
restoration and suffix crop. `runs/20261003_m1_stabilizer/selfcheck/numerics.json`.
Independent code review requested; no real-video result yet.

Independent code review PASS, including the actual32/8-head,128-dimensional,
36-layer FP32/BF16 reader, native rotary tensor oracle,18/20frames and deliberately
opposite arm-global plumbing: `docs/reviews/20261003_m1_stabilizer_code.md`.
No real-video run/performance yet.

Execution update2026-10-03: lab ordinary SSH sessions now inherit systemd
user.slice DevicePolicy=closed from50-slurm-gpu-only.conf; nvidiactl open returns
EPERM. Driver/module versions match. The lab cluster qian_pilot exposes per-node
local partitions, with sc474399 currently idle. Use the legitimate Slurm allocation
via `sbatch experiments/20261003_m1_stabilizer/launch/lab.sbatch smoke` then `main`.
The32G host-memory request fits the node's56000MB configured memory. This changes
only execution, not scoring/constants. No system policy or GPU reset is performed.

Intended host sc474399 (uoa-lab2), selected by live machine check while sc448960
runs Highlighter. Run after independent code PASS and synced source:
`bash experiments/20261003_m1_stabilizer/launch/run_lab.sh smoke` then `main`,
using detached setsid/nohup with output in each run directory. Return full outputs
to the local repository before updating STATUS, then
`bash experiments/20261003_m1_stabilizer/launch/run_analysis.sh`.
Smoke validation (noGT): `analyze.py --stage prepare --smoke`.
Each full arm calls the canonical evaluator and unchanged r6; report is
`runs/20261003_m1_stabilizer/r1_main_analysis/summary.json`.
