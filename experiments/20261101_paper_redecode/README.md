# Paper re-decode of the cached reads with the current method (2026-11-01)

Host: sc474397 (uoa-lab1), CPU only. Development-selected numbers (rule 10). No new reads, no GPU.

## 1. What and why

The paper's experiment tables (`paper/overleaf/sections/experiments.tex`: `tab:ablation` decoder and integrator rows,
`tab:smoothing`, `tab:modality` rows that need no new reads, §"Two Orders, One Score", §"Cost") still carry the
r6-reader numbers with the calibrated video key (`experiments/20260926_twolevel/README.md` §20.1,
`experiments/20260928_infer/README.md` §12.1 and §15.1). The current method is:

- reader adopted on 2026-10-31 (`experiments/20261031_window_evidence/README.md`): each window branch carries its own
  0.5 fps frames next to the visual question and the forced-aligned words in the speech question;
- time level r6_bma with the raw key (the calibration was removed by the user on 2026-10-07):
  `twolevel_r2.py --noleak --transform nscore --key raw --duration bma --bma-prior length --min-windows 2 --bma-grid 6 --arm m2`.

This directory re-decodes the cached adopted reads with the same scripts and flags as the old runs
(`experiments/20260926_twolevel/launch/run_final.sh`, `experiments/20260928_infer/launch/run_dehate_all.sh`,
`experiments/20260928_infer/launch/run_gate.sh`), switching only the read set and the key (raw). Nothing in the method
changes. The "one joint branch" modality row and the reader ablations need new reads and are done elsewhere
(`experiments/20261101_reader_ablation/`).

## 2. Read sets (identity checked, `identity_check.py`, predictions only)

| role | HateMM / HateClipSeg | DeHate |
|---|---|---|
| adopted ("both") | `runs/20261014_m1_audio_bound/vl_aligned/vl_text_aligned` | `runs/20261031_window_evidence/DeHate/adopted` |
| old reader (r6) | `runs/20260926_glr/base_gridA` (the reads of `runs/20260926_twolevel/final_rawkey`) | `runs/20261031_window_evidence/DeHate/r6_replay` |
| frames only | `runs/20261007_m1_streamingtom/placement_controls_main/adjacent_local` | `runs/20261031_window_evidence/DeHate/frames` |
| words only | `runs/20261014_m1_audio_bound/vl_aligned/vl_r6_aligned` | `runs/20261031_window_evidence/DeHate/words` |
| gate verdicts (frames-only prefix / transcript-only prefix) | `runs/20260928_infer/main/reads_noctx` / `reads_noframes` | `runs/20260928_infer/dehate/reads_noctx` / `reads_noframes` |

Check (`runs/20261101_paper_redecode/identity/{main,dehate}.txt`): on both corpus sets the four read sets have the same
videos and windows and the same verdict z_video (max |d| = 0); frames only has exactly the adopted visual reads and the
r6 speech reads, words only exactly the r6 visual reads and the adopted speech reads (max |d| = 0 everywhere). So the
four sets form an exact 2 x 2 design (visual read r6 / in-window frames, speech read proportional / aligned). The gate
verdict sets cover every video (0 missing of 333 / 1341); they depend only on the prefix, which the adopted reader
does not change.

## 3. How to run

```
setsid nohup bash experiments/20261101_paper_redecode/launch/run_all.sh repro    > runs/20261101_paper_redecode/launch_repro.out 2>&1 &
NP=8 setsid nohup bash experiments/20261101_paper_redecode/launch/run_all.sh decode > runs/20261101_paper_redecode/launch_decode.out 2>&1 &
NP=8 setsid nohup bash experiments/20261101_paper_redecode/launch/run_all.sh extra  > runs/20261101_paper_redecode/launch_extra.out 2>&1 &
setsid nohup bash experiments/20261101_paper_redecode/launch/run_all.sh analysis > runs/20261101_paper_redecode/launch_analysis.out 2>&1 &
python experiments/20261101_paper_redecode/summarize.py     # -> runs/20261101_paper_redecode/tables.md
python experiments/20261101_paper_redecode/cost.py          # -> runs/20261101_paper_redecode/cost/cost.json
```

- Every arm writes `runs/20261101_paper_redecode/{main,dehate}/<arm>/{metrics.json,predictions.jsonl,run.log,config.json}`;
  the exact command of each arm is in `runs/20261101_paper_redecode/jobs_decode.txt` / `jobs_extra.txt`.
- All runs use `OPENBLAS_NUM_THREADS=1`. Metrics only from `src/eval/evaluate_four_datasets.py` (called by the
  decoders). Bootstrap: `experiments/20260927_dvd/analyze_dvd.py` (pooled and within, base first) and
  `within_boot.py` here (a copy of the within bootstrap of `experiments/20260926_twolevel/analyze.py`, whose root and arm
  lists are hard-coded; one change: a fresh `default_rng(0)` per corpus and pair).
- No existing file under `src/` or another experiment directory was edited.

Arms (flags beyond the method's; `M` = the method flags above):

| arm | flags | row |
|---|---|---|
| `m2` | `M` | the method |
| `full` | `M --arm full` | interval output |
| `nocoupling` | `M --nocoupling` | temporal coupling off |
| `k1` | `M` with `--min-windows 0.5` | minimum length k = 1 |
| `fixed80` | `M --bma-fixed 80` | mean length fixed at 80 s |
| `fixed80_k4` | `--noleak --transform nscore --key raw --k 4` | old DeHate flags of the same row (plumbing: must equal `fixed80`) |
| `sharedchain` | `M --sharedchain` | one shared chain |
| `rawscale` | `M` with `--transform none` | normal scores off |
| `r2nl`, `linstd` | `--noleak --k 4` vs `--evidence linear --k 4 --fusion or` (raw reads, fixed 80 s, raw key) | EM emissions vs reads / corpus s.d., the old setting (`r2_noleak` vs `diag_lin_or_k4`; DeHate `abl_r2nl` vs `abl_linstd`) |
| `lin_bma_nscore` | `M --evidence linear` | extra: EM -> reads / s.d. under the current chain, everything else the method |
| `lin_bma_raw` | `M --evidence linear --transform none` (no `--noleak`) | extra: the same on raw reads; pair `rawscale` |
| `nokey`, `norank` | `M` with `--key none`; `M --norank` | video term; within term |
| `speechonly`, `visualonly` | `M --chains z_speech`; `M --chains z_visual` | modality rows |
| `gate` | `M --gate verdicts --gate-visual <noctx> --gate-speech <noframes>` | per-video verdict gate (20260928_infer §15, version 1) |
| `gauss{2,4,8,16}` | `til_infer.py --model gauss --sigma s --fusion max` | smoothing table |
| `rs_old`, `rs_frames`, `rs_words` | `M` on the other read sets | window-evidence inputs |
| `s{100,...,0.125}` | `M` with `--key scaled --key-scale s` | video-term scale |
| `old_s{...}` | the same on the old (r6) reads | extra: the sweep under the old reader |

## 4. Results

Run 2026-11-01 on sc474397 (repro 22:54–23:00, decode 23:01–23:20, extra 23:21–23:30, analysis 23:30–23:44), code
at commits bb287f2 → af73158 of the branch (only other experiments' files changed in between; no decoder or evaluator
change). No job failed, no Traceback. Every number below is copied from the evaluator's
`runs/20261101_paper_redecode/{main,dehate}/<arm>/metrics.json` (main = HateMM + HateClipSeg, dehate = DeHate);
intervals from `<set>/analysis_within/summary.json` (within, `within_boot.py`),
`<set>/analysis_within_evidence_{fixed80,bma_raw}/summary.json`, and `<set>/analysis_{ablation,modality,readsets}/summary.json`
(`analyze_dvd.py`, pooled and within; its within intervals agree with `within_boot.py` to ±.002). All tables
regenerate with `summarize.py` → `runs/20261101_paper_redecode/tables.md`. Old values: the r6 reads with the
calibrated key as quoted in the paper (within does not depend on the key: the key is constant within a video).
Development-selected.

### 4.0 Reproduction and plumbing

- `m2` on the adopted reads equals the stored decodes exactly (difference 0.0 in all three metrics on all three
  corpora; score curves max |d| = 0): HateMM .8989 / .6921 / .7799, HateClipSeg .7327 / .6895 / .6866
  (`runs/20261031_window_evidence/decoded_main/adopted_rawkey/metrics.json`), DeHate .6940 / .1636 / .6603
  (`runs/20261031_window_evidence/decoded/adopted/metrics.json`). Self-test of the chain passed (`main/m2/run.log`).
- `s1` (`--key scaled --key-scale 1`) = `m2` and `old_s1` = `rs_old` = `runs/20260926_twolevel/final_rawkey` exactly;
  `fixed80` (`--bma-fixed 80`) = `fixed80_k4` (the old `--k 4` flags) exactly on all three corpora.

### 4.1 Task 1: decoder and integrator ablations (arm − method; within unless stated; 95 % interval; `*` excludes 0)

| removed or replaced | HateMM old → new | HateClipSeg old → new | DeHate old → new |
|---|---|---|---|
| temporal coupling (`nocoupling`) | −.114 → **−.147** [−.198, −.094]* | −.057 → **−.062** [−.094, −.032]* | −.021 → **−.074** [−.111, −.036]* |
| minimum length k = 1 (`k1`) | −.030 → **−.052** [−.088, −.021]* | −.009 → **−.013** [−.024, −.001]* | −.016 → −.014 [−.037, +.011] |
| mean length fixed at 80 s (`fixed80`) | +.002 (paper prints −.002) → −.002 [−.023, +.017] | +.002 (paper −.002) → −.013 [−.027, −.000]* | +.011 → +.016 [−.008, +.040] |
| one shared chain (`sharedchain`) | −.007 → **−.021** [−.053, +.010] | −.043 → **−.068** [−.100, −.038]* | −.013 → −.006 [−.027, +.016] |
| normal scores off (`rawscale`) | +.011 → −.013 [−.038, +.010] | +.011 → +.009 [−.007, +.023] | −.010 → −.025 [−.048, −.002]* |
| EM emissions → reads / corpus s.d., fixed 80 s chain (`linstd` − `r2nl`, the old setting) | −.010 (paper prints +.010) → **−.020** [−.050, +.007] | −.024 (paper +.024) → **−.031** [−.047, −.015]* | −.014 → −.015 [−.040, +.008] |
| video term, pooled ROC (`nokey`) | −.325 → **−.321** [−.363, −.273]* | −.147 → **−.139** [−.200, −.076]* | −.150 → −.124 [−.173, −.073]* |
| within term (`norank`; within becomes .500) | −.251 → **−.280** [−.328, −.229]* | −.137 → **−.187** [−.222, −.149]* | −.143 → −.160 [−.198, −.122]* |

Bold: |change| ≥ .01 with the same sign on both main corpora (the paper's bold rule). Old sources: within
`runs/20260926_twolevel/analysis_final/table.txt` (§20.1), `r3_m2` / `r2_noleak` / `diag_lin_or_k4` / `final_nokey`
`metrics.json`; DeHate `runs/20260928_infer/dehate/analysis_all/table.txt` and `abl_*/metrics.json` (§12.1).
The calibration row is dropped. With the raw key as base the old video-term row would read −.324 / −.144 / −.149
(`final_nokey` − `final_rawkey`, `abl_nokey` − `abl_rawkey`).

Pooled ROC and PR of every decoder ablation stay within .003 of the method, except `norank` (ROC −.002 / −.003 /
−.002, PR −.006 / −.008 / +.003) and `nokey` (PR −.404 / −.138 / −.070). Table in `runs/20261101_paper_redecode/tables.md`.

EM emissions under the current chain (extra arms, not in the old table; within):

| replaced | HateMM | HateClipSeg | DeHate |
|---|---|---|---|
| EM → normal scores / s.d., everything else the method (`lin_bma_nscore` − `m2`) | +.016 [−.004, +.036] | −.034 [−.051, −.018]* | −.003 [−.020, +.014] |
| EM → raw reads / s.d. (`lin_bma_raw` − `rawscale`) | +.011 [−.015, +.038] | −.049 [−.069, −.032]* | −.000 [−.025, +.025] |

So under the current chain both evidence models were run: EM is clearly better only on HateClipSeg, and the linear
evidence is slightly (not significantly) better on HateMM. The paper's footnote (only the fixed 80 s setting had both)
no longer has to hold.

### 4.2 Task 2: smoothing (within)

| time level | HateMM old → new | HateClipSeg old → new | DeHate old → new |
|---|---|---|---|
| none (independent cells) | .637 → .633 | .580 → .624 | .623 → .587 |
| Gaussian σ = 2 s | .676 → .673 | .622 → .673 | — → .650 |
| Gaussian σ = 4 s | .685 → **.687** | .628 → **.678** | .641 → .661 |
| Gaussian σ = 8 s | .688 → .683 | .631 → .671 | .647 → **.663** |
| Gaussian σ = 16 s | .673 → .667 | .621 → .644 | — → .653 |
| explicit-duration chain (method) | .751 → **.780** | .637 → **.687** | .643 → .660 |
| chain − best σ (best picked on test) | +.063 → **+.093** [+.038, +.149]* (σ 4 s) | +.006 → +.009 [−.021, +.043] (σ 4 s) | −.004 → −.002 [−.040, +.034] (σ 8 s) |

Old DeHate `none` = `abl_nocoupling` (.6226; `final_dehate/final_nocoupling` .6233). The best σ moves from 8 s to 4 s
on the main corpora. Posterior mean hate length per video (median, 10–90 %; `main/m2/run.log`, `dehate/m2/run.log`):
HateMM 67–74 s (18–204 s), HateClipSeg 136–149 s (57–245 s), DeHate 43–47 s (16–154 s).

### 4.3 Task 3: modality rows (within; difference to two branches + OR with 95 % interval)

| window branches | HateMM old → new | HateClipSeg old → new | DeHate old → new |
|---|---|---|---|
| speech branch only (`speechonly`) | .738 → .744 (−.036 [−.073, −.001]*) | .578 → .612 (−.075 [−.120, −.034]*) | .653 → .650 (−.010 [−.043, +.021]) |
| visual branch only (`visualonly`) | .677 → .661 (−.119 [−.178, −.059]*) | .580 → .605 (−.082 [−.117, −.047]*) | .567 → .585 (−.075 [−.106, −.046]*) |
| two isolated branches, OR (method) | .751 → **.780** | .637 → **.687** | .643 → .660 |
| per-video verdict gate (`gate`) | .751 → **.787** (+.007 [−.012, +.027]) | .633 → .682 (−.005 [−.023, +.012]) | .664 → **.665** (+.005 [−.013, +.022]) |
| one joint branch | needs new reads (other agent) | | |

- Pooled ROC / PR of all four arms within .002 / .005 of the method (`analysis_modality/table.txt`).
- Gate (20260928_infer §15 version 1: each chain weighted by the two-Gaussian-calibrated verdict from a prefix with
  only that modality; this calibration belongs to the gate, the video key stays raw): visual gate < .5 on 63 % / 36 % /
  63 % of videos, speech gate on 37 % / 17 % / 41 % (`<set>/gate/run.log`), the same as on the r6 reads.
- The verdict reads cover every video (0 missing); they are prefix-only reads and do not depend on the window evidence.

### 4.4 Task 4: interval output (`--arm full`, raw key)

| corpus | old (`final_full`, calibrated key) | new pooled ROC / PR / within | F1 @ IoU .3 / .5 / .7 | precision | recall |
|---|---|---|---|---|---|
| HateMM | .8904 / .7133 / .7508; F1 .325 / .295 / .235 | .8347 / .5882 / .7799 | .333 / **.304** / .250 | .361 / .329 / .271 | .309 / .282 / .232 |
| HateClipSeg | .7391 / .6779 / .6373; F1 .246 / .145 / .077 | .7472 / .6864 / .6866 | .285 / **.164** / .092 | .507 / .293 / .164 | .198 / .114 / .064 |
| DeHate | .6947 / .1559 / .6431; F1 .178 / .133 / .105 | .6844 / .1387 / .6603 | .175 / **.127** / .102 | .113 / .082 / .066 | .384 / .280 / .224 |

Caveat: with `--key raw` the interval arm's P(V = 1) is not P(V = 1 | K_v) (that needed the removed calibration). In
`twolevel_r2.py` the `full` arm uses `lo = key` only under `--key calib`; otherwise `lo` is the time level's own video
posterior (`video_terms(...)["lo"]`: prior, verdict mixture and the chains' likelihood ratio). So the paper's formula
`P(V=1|K_v)P(t) ≥ .5` should be stated as `P(V=1 | verdict, reads) P(t) ≥ .5`. The frame score of this arm is
log P(V) + log P(t) (not the method's score), hence its lower pooled numbers on HateMM.

### 4.5 Task 5: contribution of the two window-evidence inputs (current decoder, raw key)

| read set | HateMM ROC / PR / within | HateClipSeg | DeHate |
|---|---|---|---|
| old reader (r6) `rs_old` | .8955 / .6883 / .7508 | .7137 / .6666 / .6373 | .6994 / .1568 / .6431 |
| frames only `rs_frames` | .8966 / .6846 / .7690 | .7353 / .6839 / .6637 | .6933 / .1582 / .6510 |
| words only `rs_words` | .8978 / .6905 / .7677 | .7114 / .6773 / .6664 | .6994 / .1604 / .6471 |
| both (adopted) `m2` | .8989 / .6921 / .7799 | .7327 / .6895 / .6866 | .6940 / .1636 / .6603 |

Difference to the old reader (`analysis_readsets/summary.json`, ROC / PR / within):

| | HateMM | HateClipSeg | DeHate |
|---|---|---|---|
| frames only | +.001 / −.004 / +.018 [−.008, +.044] | +.022* / +.017 / +.026 [+.003, +.051]* | −.006* / +.001 / +.008 [−.015, +.029] |
| words only | +.002 / +.002 / +.017 [−.006, +.041] | −.002 / +.011 / +.029 [+.009, +.051]* | .000 / +.004 / +.004 [−.012, +.022] |
| both | +.003 / +.004 / +.029 [−.004, +.062] | +.019* / +.023 / +.049 [+.017, +.083]* | −.005 / +.007 / +.017 [−.008, +.041] |

Read-set identity: §2. All four read sets of each corpus come from Qwen3-VL-8B with the same prefix and verdict.

### 4.6 Task 6: video-term scale (frame score = s · K + centred rank, no calibration; s = 1 is the method)

| s | HateMM ROC / PR | HateClipSeg ROC / PR | DeHate ROC / PR |
|---|---|---|---|
| 100 | .8977 / .6887 | .7303 / .6836 | .6922 / .1623 |
| 8 | .8979 / .6901 | .7304 / .6838 | .6925 / .1625 |
| 4 | .8980 / .6904 | .7309 / .6848 | .6927 / .1626 |
| 2 | .8983 / .6907 | .7317 / .6870 | .6931 / .1630 |
| **1 (method)** | .8989 / .6921 | .7327 / .6895 | .6940 / .1636 |
| 0.5 | .9001 / .6963 | .7341 / .6926 | .6954 / .1648 |
| 0.25 | .9012 / .6988 | .7363 / .6959 | .6977 / .1673 |
| 0.125 | .9030 / .7019 | .7407 / .7014 | .7014 / .1710 |

Within is .7799 / .6866 / .6603 at every s (the key is constant within a video). Pooled ROC and PR rise monotonically
as s falls on all three corpora; the method's s = 1 is not the best point, and the best point would have to be picked
on labels. Same sweep on the old r6 reads under the current decoder (`old_s*`): HateMM .8941 → .8997, HateClipSeg
.7113 → .7212, DeHate .6983 → .7043 (s = 100 → 0.125), also monotone. The paper's old numbers (.894 → .899, .711 → .722)
came from the round-C time level (`runs/20260926_twolevel/c_s*`, `diag_r2v_nl_current_mix_s100`), not from r6_bma.

### 4.7 Task 7: cost (`cost.py` → `runs/20261101_paper_redecode/cost/cost.json`)

| corpus | read seconds per video, mean / median | what the time covers | GPU | median prefix tokens |
|---|---|---|---|---|
| HateMM | 2.80 / 2.35 (r6 from the same records 1.41 / 1.18) | estimate: prefix + verdict 0.40 + in-window-frame visual branches 1.91 + one speech branch per window 0.49 (`placement_controls_main/records`, `checks.times`) | RTX 5090 32 GB (sc474399) | 2743 |
| HateClipSeg | 4.53 / 4.50 (r6 2.09 / 2.00) | same estimate (0.43 + 3.33 + 0.77) | RTX 5090 (sc474399) | 3040 |
| DeHate | 2.89 / 2.09 | measured, one pass that reads all four read sets (r6 visual and proportional speech branches included), so an upper bound for the adopted reader | RTX 5090 (sc474397) | 2606 |

- The adopted reads of HateMM / HateClipSeg were composed from two runs (in-window-frame visual reads from the
  StreamingTOM placement run, aligned speech reads from `measure_vl_aligned.py`), so no single-pass time exists; the
  estimate takes the placement run's per-component times. For reference, `measure_vl_aligned.py` (prefix + aligned
  speech branches, sc474397): 0.90 / 0.78 s and 1.22 / 1.18 s.
- The in-window frames dominate: the visual branches with frames cost about 3.7 times the r6 visual branches.
- Peak GPU memory 17.6 GB (DeHate run). Windows per video median 15 / 30 / 8.
- Decoder (`m2`, one corpus per process, one CPU core, `OPENBLAS_NUM_THREADS=1`, evaluator call included;
  `<set>/time_<corpus>.txt`): HateMM 40.7 s (215 videos), HateClipSeg 38.2 s (118), DeHate 208 s (1341; EM 183 s).
  "Under a minute per corpus" holds for the two main corpora, not for DeHate. Fixed-length arms (`fixed80`, `r2nl`)
  take about 12 s on DeHate.

### 4.8 Paper issues found while mapping the rows

- `tab:ablation`, row "durations integrated out (fixed mean of 80 s)": the main-corpus entries −.002 / −.002 have the
  wrong sign. Fixed 80 s minus the method was +.0017 / +.0024 (`r3_m2` .7525 / .6397 vs `final_m2` .7508 / .6373);
  the DeHate entry +.011 has the right sign.
- Row "EM emissions (reads divided by the corpus s.d.)": the main-corpus entries +.010 / +.024 have the wrong sign.
  Replacing EM lowered within by .0096 / .0243 (`diag_lin_or_k4` .7483 / .6185 vs `r2_noleak` .7579 / .6428); DeHate
  −.014 has the right sign. So EM was better on all three corpora. The text "within noise of a simpler alternative on
  HateMM and HCS and slightly better on DeHate" mixes the two signs.
- §"Two Orders": the sweep numbers are from an older time level (§4.6), and the interval output's P(V) is no longer
  the calibrated key (§4.4).
- §"Cost": 1.5 s per video and "under a minute per corpus" are r6 numbers; see §4.7.

## 5. Rule-10 log (test files read)

- Test GT `data/gt_4fps/{HateMM,HateClipSeg,DeHate}.npz`: read only by the evaluator (inside every decode) and by the
  two bootstrap scripts (`analyze_dvd.py`, `within_boot.py`). No per-video GT was inspected by hand.
- Read predictions (no labels): the read sets of §2 (identity check), and the per-video read records
  `runs/20261031_window_evidence/DeHate/records/`, `runs/20261007_m1_streamingtom/placement_controls_main/records/`,
  `runs/20261014_m1_audio_bound/vl_aligned/records/` (cost only).
- Old numbers: `runs/20260926_twolevel/{final_*,r3_m2,r2_noleak,diag_lin_or_k4}/metrics.json`,
  `runs/20260926_twolevel/final_dehate/*/metrics.json`, `runs/20260928_infer/{dehate,gate}/*/metrics.json`,
  `runs/20260926_twolevel/analysis_final/table.txt`.
- No design was changed; no number here selects anything in the method.
