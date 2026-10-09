# M1 candidate39: change-preserving quantized source memory

Prepared on sc474397, 2026-10-07. Original candidate `streamingtom_causal_dualpath_quantized_media_memory` passed the single independent proposal jury, rank3 (`docs/reviews/20261006_m1_ideation_jury.md`). This is the original dual-path source-memory mechanism, not a reset of ReKV37, TTF or OTT. Current complete R1 prototype, author scientific CPU checks and real333 input/processor preflight PASS; one independent scientific code review PASS. No GPU, GT or performance evidence exists for39. Current method remains r6_bma and the user goal remains active.

Source: [StreamingTOM v1](https://arxiv.org/html/2510.18269v1), method equations4–14 and implementation section4.1; [official CTR](https://github.com/YIGE24/StreamingTOM/blob/main/streamingtom/modules/ctr.py) and [OQM](https://github.com/YIGE24/StreamingTOM/blob/main/streamingtom/modules/oqm.py). Reading records are in `runs/20261007_m1_streamingtom/source_reading/reading.json`. The transferred chain uses adjacent-frame static/dynamic classification, clustered static support and saliency-selected dynamic support, followed by frame-aligned quantized KV and query-conditioned retrieval. Qwen-specific media/layout and LOCAL support are adaptations. No claim of the paper's numerical reproduction or speedup.

Scientific hypothesis: scene changes and salient current objects should survive memory construction while repeated background tokens are consolidated. Correct contextual groups can then help interpret a current-window act. Test raw V/max ordering and matched wrong-source interventions, alongside final metrics; storage reduction or a common margin shift is not localization evidence.

Proposed constants, to be finalized in spec before any run: native frozen Qwen3-VL-8B/native20/fullASR/G/own stance and fresh S unchanged; supplementary actual0.5fps, original independent8s questions/max/fixedr6. Similarity>.9 defines static tokens only when consecutive actual processor grids match; first frame/grid change treats all as dynamic. Retain min(50,N) visual tokens. Static quota floor(G×static/N), remainder dynamic; clip to available counts and give remainder to the other pool. Static clusters use k=min(7,n−1) nearest distances including self, beta=.6 center plus .4 mean of noncenter members; singleton retains center. Dynamic tokens use actual last-vision attention saliency. Ties use actual source/token indices. The official DPC density-comparison direction selects lower-density neighbors despite its variable name; the final spec must explicitly fix its direction and deterministic tie treatment. Apply exactly the same group members and weights to projector and every DeepStack level. No fake image grid or deletion of source text/boundary tokens.

Source history, representative layer/head mapping, full-precision text versus visual quantization groups, zero-scale/odd-size packing, position packing and query layout still require a complete frozen spec. No source prefill or cost claim may silently assume a shorter history. Native full-video context means overall inference remains offline. LOCAL remains uncompressed actual current-window media; context groups retain their real source times and ancestry. Missing LOCAL gives fresh native V, not an absence label. Quantization is an implementation/efficiency detail unless independent contribution evidence supports otherwise.

New-video cost: F=actual deduplicated0.5fps frames incurs F full vision encodings, chunked last-vision attention saliency, dual-path clustering/aggregation and reduced-source LM prefills; per-window fresh V includes actual local media and retrieval/dequantization. No generated captions and no separate scored predictor. If a retrieval probe is needed it must be charged, not hidden. Native inputs may be reused with provenance; all new work is paid on a new video. Planning range from the original candidate15–80GPU seconds per minute of source video plus .2–1s/window is unmeasured, not a capacity guarantee. Source frames may reuse the already validated read-only0.5fps media inputs; derived source features/groups/KV and provenance are39-owned. Full persistent/temporary disk and allocated host/GPU memory must be audited before launch; no deletion of prior evidence for space.

Same fixed215/118,4fps sole evaluator, within84/99, allsix development-selected results. Promotion requires a common primary +.01 both and other pooled/within losses no worse than−.005/−.01. No labels in source, fitting, scoring or thresholds; no ensembles, corpus routes, alternate GT or output mixtures. Full333 only after actual originalfixed5 input/native/state/DeepStack/quantization/source guards and one independent scientific code review. No source-family budget reset;39 starts R1 only when its own complete mechanism is implemented and run.

Required mechanism controls after performance passes: same-budget uniform token selection instead of static/dynamic allocation; no explicit remote with identical LOCAL; samecount chronological context; fullprecision KV as numerical/efficiency comparison; matched wrong donor/current-to-previous static-classification binding. Every retained novelty part must have deletion loss>=.01 on a common primary both, and correct binding must beat matched wrong binding beyond original noise floors. Report raw V/S/max, actual exposure and all negative results. ReKV's failed remote/history/per-layer claims cannot be inherited as evidence for39.

Partial prototype modules only: `ctr.py` explicitly retains the official lower-density comparison direction, replacing random jitter with deterministic source-index ties; `quantization.py` uses FP32 per-channel ranges, zero-scale constant-channel reconstruction and storage-only odd-nibble padding with the true logical token count. Isolated CPU 14 dtype/token-size cases (1/2/49/50/51/196/320 FP32/BF16, each four first/static/dynamic/grid-reset grouping cases) satisfy independent weighted-member and half-quantization-step bounds, constant channels exact. Evidence `runs/20261007_m1_streamingtom/algorithm_cpu_checks/`; reproducible `algorithms_selfcheck.py`. `vision.py` prepared actual Qwen last-attention observer uses64-row chunks and charges its additional attention/transfer cost; not yet tested. No complete source/reader, no independent code review, no GPU/GT/quality evidence. Final full spec still pending before execution.

Actual isolated Qwen vision observer now CPU PASS, four FP32/BF16 ×64/576rawpatch cases; projector and all three DeepStack outputs are exactly unchanged versus original vision. Independent dense FP64 attention saliency oracle maximum error7.05e-9 (within declared2e-6 software tolerance); chunk observation is separately timed and paid. Evidence runs/vision_cpu_checks/{summary.json,run.log}, reproducible vision_selfcheck.py. This is random-weight isolated vision/software evidence, not a complete source/reader/pretrained performance or independent review.

**R1 full specification frozen before implementation/GPU/GT, 2026-10-07:** spec.json fixes four previous reduced/dequantized source blocks plus inherited ancestry under native full context (explicit Qwen adaptation of the source encoding window; no linear total-memory claim). Source visual KV alone is frame-group uint4 with FP32 scales/minima; source text/boundary KV stays original full BF16, native prefix unquantized. Representatives use actual pre-RoPE normalized reduced visual keys, per-layer FP32 mean over visual rows/concatenated8KV heads. Actual question-content queries pool32Q→8KV by four contiguous groups, no probe, top4 nonlocal by FP32 cosine/actual source-index ties. At each layer context blocks pack by source order; whole uncompressed LOCAL/question suffix shifts as one three-axis block after them; nativeprefix unchanged. LOCAL embeddings and every DeepStack row come from fresh single-frame actual vision cached within this video, without source-KV history; added memory is context only. All literal source/reader text is in spec and shared by both corpora. Missing LOCAL is fresh nativeV, fresh nativeG/S/stance/max/r6 remain.

Full source vision features and quantized/fulltext KV are temporary per video, validated before atomic record and released; persistent compressed projector/allDeepStack and representative/group/input/position/cost proofs permit exact rebuilt-source comparison. At50visual tokens and current Qwen4096width/three DeepStack stages, compressed-feature proof is ~1.6MB/frame (~47GB for28895 historical source frames), plus representative/native/input traces; full temporary feature/textKV cost is separate and must be measured/audited. These are conditional capacity estimates, not actual run sizes or a claim that audit reconstruction is free. No complete prototype or independent code review yet.

Before any GPU/GT, actual prior333 source-token capacity audit counted28895frames/2,592,359visual tokens/max45,500pervideo (`source_reading/capacity_from_actual37.json`). At Qwen4096/BF16, fullprojector+threeDeepStack proof is ~84.95GB total/1.49GBlargestvideo; compressed50-token projector+3DS upper bound47.34GB. Native/representative/trace payloads are extra. Fullprojector/allDeepStack are now persisted alongside compressed values so strict CPU replay can recompute every static/dynamic/group/mean mapping, rather than treating compressed-only files as sufficient. Working copies and quantized/textKV remain temporary/released; persistent proof costs are not free preprocessing. The estimated total full-run footprint must fit host and ROOT return disk before submission; unchanged ReKV/VTime evidence will not be removed. This resolves the earlier compressed-only draft before any GPU/scientific run.

Complete collector+LOCAL/context reader actual36-layer CPU four FP32/BF16×native18/20 cases PASS, runs/model_cpu_checks/summary.json. Real vision64tokens/frame reduces to50; firstdynamic and identical-pair static paths both enter LM/threeDeepStack,7sourcevision+7sparseLM, actualuint4 history and remote reads, last4direct/allolderinherited ancestry correct, onefinalquery/no vision/probe, nativeG/S/allKV/rope and full cloned read identical. Explicit fixture tokenizer/renderer/randomweights only; not real8B/semantic/performance. Initial entrypoint ROOT import failure was fixed and retained in run.log; actual completed test run_import_fixed.log. Isolated memory6cases physical36layer8KV128 FP32/BF16 visual5/50/51 quant/textbits/source representative/fullLOCAL DS/release checks PASS; initial BF16 cast-error test omitted half-step magnitude from rounding bound, test corrected with failure preserved, no producer correction. Full measurement/strict validation/realprocessor preflight and one independent code review remain pending.

Before GPU/code review, author identified and fixed a concrete real-processor question-row issue: LOCAL images expand one raw image placeholder to multiple processor IDs. reader now binds raw tokenizer offsets through each actual image-token run/count and asserts the complete expanded sequence equals processor IDs before selecting question rows; role/image/special rows remain excluded. Software fixture had already expanded placeholders and did not expose this real interface issue. Added shared expanded_image_offsets helper and separated read-only encode_local_input from model embedding assembly for CPU strict replay; scientific spec/question/LOCAL/media/constants unchanged. Actual native processor preflight required next; no39GPU or performance run has occurred.

Full333 real-input preflight now running: current RAW/PTS/PNG, nativepixel/IDs/3axes and every source+LOCAL expanded image/question sequence. It uses the prior native stance only as a declared renderer condition; fresh GPU G/S still required. A report-only field originally called max_query_suffix_tokens actually measured native question characters; code renamed max_native_question_chars. The in-flight already-loaded output will receive the same explicit unit correction after completion; no value/gate/media/question/score changes, no rerun selection.

Actual full333 current realprocessor preflight PASS, runs/full_input_preflight/summary.json,603.020937s CPU:28895sourceframes/7359windows/7missingLOCAL/7352 genuine expanded-image branches. Current nativepixels/IDs/3axis equal prior native input proofs; oldstance only a declared rendercondition, fresh GPU judgment still required. Every source and LOCAL image grid/token count/question-offset sequence is bound to current rawPTS/RGB. Based on exact currenttoken counts/4096width/threeDS/BF16, full+compressed feature proof estimate132,287,987,712bytes (~123.21GiB); native/reps/query/JSON/temp/I/O extra. Report unit field corrected explicitly to max_native_question_chars with unchanged values in summary.unit_correction; not a token-cap result.

Full read_video/strictvalidator actual36layer two CPU FP32native18/BF16native20 cases PASS, runs/measurement_cpu_checks/run_syntax_fixed.log andsummary.json: noLOCAL latewindow/nativeS orNone/2clones/nativeG/S/KV/rope, fullprojector+all3DS group weighted means/3axis/GQA/querytoken budgets, permanent proof replay after temporary release and question/KVbytes/S/group corruption refusal. Initial test with-line syntax failure retainedrun.log. Cross-host proof descriptors now ROOTrelative; only owned temporary payload releases. Per-layer textKV uses mmap to read just that layer instead of reloading all36. Measurement/analyzer/launch complete; one independent scientific code review in progress. No39GPU/GT/performance/novelty evidence yet.

One independent Rule6 scientific code review PASS (same-family provisional), docs/reviews/20261007_m1_streamingtom_code.md/fulltraces runs/independent_code_review/. Four actual36 FP32BF16/native18/20 own NumPy uint4/text/history exact and144perlayer denseGQA/threeaxisRoPE/causality vsSDPA reference max3.58e-7FP32/.00390625BF16; nativeS/KV/rope/clone exact. Two completeproduction CPU chains/5corruption categories/failedtemporary-preserve-rebuild/permanentproofafterrelease/earlyASR-config-model audits PASS; actual fixed5 624source/158LOCALcurrentrawPTS/RGB/nativeprocessor and independent raw-run token/question mapping PASS. Own computational LOCAL DS/remote-V zeroing changes fixture scores, only execution evidence. Reviewer did not independently rerun author333 preflight, did not runGPU/GT/evaluator, and found no observation-changing bug. Actual8B originalfixed5 next; allcapacity/costforecasts remain forecasts, source/noise/performance/novelty/control gates unchanged.

Originalfixed5 actual8B source+reader submitted sc474398/lab3 Slurm2412026-10-07 02:36, after author36CPU/full333currentrealprocessor/onceindependentRule6PASS and all4operativecodeagreement/foreignworkpreserved. Lab3 actualidle afterHL0job240DONE,1.26099TBfree; source0.5fps/read-onlycache/fullnative/ASR/model/runtime present andwholefixed5 onthisonehost, newmodelsource+reader fresh/no hostsplice. ROOT238 othernormalactiveGPU/held193unchanged. ROOT return_capacity_plan.json atcheck309.778GBfree reservesfeature132.288GB+extra3950GB+HL0return21GB+VTime20GB leaving86.49GB; estimates notactual39payload, recheckbeforemain/return. Standardwait_smoke.log matchesjob disappearance/Traceback/FAILED. ActualGPU first2video readers complete02:36:48/53; noGT/metrics/effectiveness yet. Whole333 cannot submit beforeactualfixed5/BOTHROOTreturn/strictsource/native/dualpath/realintervention guards.

Actual8B originalfixed5/sc474398/Slurm241 DONE2026-10-07 02:39:55.916,3:38/0:0; immediateBOTH ROOTreturncomplete02:44,new3,213,302,879bytesrunpayload/sourceinputunchanged. FirststrictprepareFAILEDbecause defaultROOTCPU1thread vsgenerationexplicit4; independentactualall624 replay isolatesonly52 DPCdiagnostic末位arrays (scientific IDs/budgets/centers/members/weights all exact), 4threads everyfullplan/projector/3DS exact. Targeted validatorentrytorch.set_num_threads4+reportthreads fixed only replay configuration; all exactguards/spec/scoring retained, noGPUrerun/oldscoreedit/tolerance. Failedrun/costattempt retained. NarrowindependentconfirmationPASS docs/reviews/20261007_m1_streamingtom_group_replay_diagnosis.md, owncompleteproductionstrict5PASS25.413s fromthreads1→4. OfficialROOTsamebackend strictprepare nowPASS r1_full_smoke_analysis/plumbing_summary.json, noGT/metrics.

Real source/input/native allrawexact, static/dynamicpaths both datasets entered actualsourceLM; total624frames/1099actualLM+629vision forwards,158changedV/nonzero perlayerremote5688reads,10clones/nativeG/S/stance exact. Static retained tokens HMM4992/HCS7737, dynamic14008/4463. Actualprocessing173.265232s/source148.751100s/standalone197.870241s, peak18.152959GiB. Actualallocated3:38 andROOTstrict/failure setup/validation/I/O audits separate, original nativeacquisition additional. Forecastwhole3333–5allocatedGPUh fromactual7359/158scaling≈2.24hprocessing/2.56hstandalone, source/shape/proof/IO/validation/longvideo overhead mayvary; source main_forecast_from_actual_smoke.json, not actualwhole333cost orperformance. Complete333samehost ready aftercommit/sync/resources, budget0/3, all6 andnegativecosts preserved.

Complete333 R1 submitted samehost/sc474398/lab3 Slurm2422026-10-07 02:57 andRUNNING02:58 after originalactual8B fixed5/BOTH ROOTreturn/strict exactPASS/threadfix narrowconfirmationPASS, all4operativecodeagree/foreignwork preserved/lab3idle1.2Tfree. ROOTreturncurrentplan281.577GBfree-feature132.288GB-extra39native/reps/query/JSON50GB-VTime20GB leaves79.289GB; HHreturnalreadycomplete. Plan estimates remain subject to currentdisk check, no oldevidence removal. Freshfullsource+pairedreader all215/118 ononehost/frozeninput/model/constants/sourcehistory/quant/query/G/S/stance/max/r6, no pilotsplice/routing/sharding/GT. Source/storageproof/decode/extra native checks/diagnostics/allfailedattempts paid; native acquisition additional. Forecast3–5allocatedGPUh frommeasuredfixed5 remains forecast. Standardwait_main.log matchesjobgone/Traceback/FAILED, parentmustcheckactualDONE; immediateBOTHROOTreturnbeforeSTATUSnumbers/strict+canonicalall6. BudgetR1 0/3/oldfamilies unchanged, no39mainperformance/noveltyyet.

## Completed R1; stopped after this round at user's request

Complete333 on sc474398 / Slurm242 finished2026-10-07 05:26:28 NZDT, allocated2:29:02, exit0:0. Entire run (~144GiB) and current source cache returned to sc474397. Final whole-directory rsync traversed267742entries and transferred0newfiles after disjoint transfer-only directory copies; no producer changes, host splicing, checksum comparison or destination deletion. Return attempt/logs remain under this run; `data/temporal_retrieved_kv/PROVENANCE.md` records actual source-generation/return hosts. Strict current-input/native/source/group/three-DS/uint4/text/history/query/speech replay passed all333 before any main GT evaluation (`r1_full_main_analysis/alignment.json`; prepare1739.126985s). Original native raw values and allsix fixed-r6 final metrics are exact; all frame counts, eligible cohorts and skipped IDs match the fixed baseline.

| Dataset | pooled ROC-AUC | pooled PR-AUC | within macro ROC-AUC | defined videos |
|---|---:|---:|---:|---:|
| HateMM | .8987682566241558 | .6949421109369128 | .7636230856117475 | 84/215 |
| HateClipSeg | .7357724935435498 | .6872929933506378 | .6597430006184163 | 99/118 |

Authority: `runs/20261007_m1_streamingtom/r1_full_main_decoded/optimized/metrics.json`; same evaluator4fps/test/frozenr6, all development-selected. Against `runs/20260926_twolevel/r6_bma/metrics.json`, HMM ROC/PR/within deltas+.0016495952100460/+.0007075075907018/+.0128410416457806; HCS+.0189476296288902/+.0162209089504508/+.0223938868143175. Allsix increase; the predeclared common-within>=.01 and no-loss-beyond-noise performance gate passes (`r1_full_main_analysis/summary.json`). This is descriptive performance evidence, not established component necessity, binding mechanism, confidence/significance or novelty acceptance. Main GT files `data/gt_4fps/{HateMM,HateClipSeg}.npz` were first used after complete inference and strict validation by the sole evaluator; no main-GT error analysis or design change was performed in wrap-up.

Actual source/read processing7534.731744s (~125.58min), including source6691.353936s (~111.52min); standalone8857.576657s (~147.63min) includes source decode accounting. Paid51192LM/29228vision forwards,28895sourceframes,7352changedV,264528recordedremote_reads,7missingLOCAL, peak18.134750GiB; no main clones (originalfixed5 had10). Source costs, proof storage/replay and native20/fullASR original acquisition remain charged separately as specified; allocation, ROOT strict28.99min, transfer/setup/I/O and prior failed smoke-validator attempt are retained, no efficiency claim.

User's latest instruction is to finish the already-running round and stop. Therefore preserve this positive R1 and budget0/3; do not start mechanism controls, scientific revisions or new candidates. Formal method remains r6_bma. Performance passed, mechanism untested and overall research goal not declared complete.

## R1 mechanism controls 1–3 (declared 2026-10-08, before any control run)

User instruction 2026-10-08: run controls 1, 2 and 3 of the list above (no remote, same-count chronological context,
same-budget uniform token selection). The full-precision comparison and the wrong-donor binding control are not part
of this request, so the binding claim stays untested.

Each arm changes one thing relative to R1. Unchanged in every arm: native prefix, G, own stance and S; the LOCAL
frames of each window (actual 0.5 fps frames in the half-open 8 s window, uncompressed); every prompt text, including
`reader_role_text`; uint4 source memory; 4 remote frames per layer where remote frames exist; windows without LOCAL keep
the native V; fusion max(V, S); fixed r6; sole evaluator.

| arm | the one change |
|---|---|
| `no_remote` | no remote blocks at any layer: native prefix, then LOCAL, then the question |
| `nearest` | per-layer question matching replaced by time: the 4 non-LOCAL source frames closest to the window (distance start − t for t < start, t − end for t ≥ end; ties to the earlier frame), the same 4 at every layer |
| `uniform` | memory construction keeps g = min(50, n) tokens per frame at raster positions floor((k + 0.5) n / g), k = 0 … g − 1: no static/dynamic split, no merging, no saliency ranking (saliency is still computed by the shared vision path and is unused). Retrieval is R1's per-layer question matching |
| `replay` | R1 itself, rebuilt in the same job as `no_remote` and `nearest`; plumbing check, not a control. It must reproduce the R1 record exactly |

Gate (declared in this README before R1): a component is supported when R1 − control ≥ .01 on the same primary metric
in both corpora. R1 reference `runs/20261007_m1_streamingtom/r1_full_main_decoded/optimized/metrics.json`. Each
control is also reported against r6 (`runs/20260926_twolevel/r6_bma/metrics.json`); `no_remote` − r6 is the gain from
LOCAL frames alone. All numbers development-selected.

Implementation:
- `controls.py`; hooks with R1 defaults added to `reader.visual_margin(pick=None)`, `source_encoding.reduce(grouping=group)`
  and `collect.collect(grouping=group, persist=True)`. With defaults R1 is unchanged; the CPU fixture check runs
  production R1 `read_video` + strict `validate_bundle` with the hooks in place.
- Controls persist no tensor proofs. Instead each video compares the rebuilt full projector, all DeepStack features and
  saliency with the persisted R1 proofs (exact equality), and in job `dualpath` also the grouping plans, source
  representatives and replay question vectors. `controls_analyze.py prepare` checks native reads = r6 reads, replay =
  R1 predictions, the `no_remote` / `nearest` / `uniform` selection rules, and recomputes uniform retrieval from saved
  vectors.
- CPU fixture check (random-weight 36-layer model, FP32 native18 and BF16 native20): `controls_selfcheck.py`, PASS,
  `runs/20261007_m1_streamingtom/controls_cpu_checks/{run.log,summary.json}`.

Runs (one host each, full 333, after a fixed-5 smoke):
- job `dualpath` (`replay`, `no_remote`, `nearest`) on sc474398, the R1 host: `launch/controls_lab3.sbatch`,
  `SCOPE=smoke|main`, output `runs/20261007_m1_streamingtom/controls_dualpath_{smoke,main}/`.
- job `uniform` on sc474397: `launch/controls_lab1.sbatch`, output `controls_uniform_{smoke,main}/`.
- Analysis on sc474397: `STAGE=prepare|evaluate JOB=… bash launch/run_controls_analysis.sh`, then `STAGE=report`;
  table and gates `runs/20261007_m1_streamingtom/controls_analysis/summary.json`.

Cost estimate from R1 (forecast, not measured): source memory 111.5 min per job; each reading arm about 14 min.
Job `dualpath` about 2.6 h of GPU, job `uniform` about 2.2 h.

### Control results (2026-10-08; development-selected)

Runs: job `dualpath` sc474398 / Slurm 281, 2:59:08, exit 0:0; job `uniform` sc474397 / Slurm 282, 2:17:15, exit 0:0.
Smoke (fixed 5) Slurm 279 / 280 passed first. `dualpath` returned to sc474397 with rsync (no deletion, no checksum).
Binding checks (`controls_{dualpath,uniform}_main_analysis/alignment.json`) PASS on all 333 videos:
- native reads equal the r6 reads;
- every rebuilt projector, DeepStack and saliency tensor equals the persisted R1 proof (both hosts), and in `dualpath`
  also every grouping plan, source representative and replay question vector;
- `replay` reproduces the R1 predictions and all six R1 metrics exactly;
- the `no_remote`, `nearest` and `uniform` selection rules hold in every covered window (7352) and layer.

Pooled ROC / pooled PR / within (84 / 99 videos). Sources: `controls_dualpath_main_decoded/<arm>/metrics.json`,
`controls_uniform_main_decoded/uniform/metrics.json`; table and gates `controls_analysis/summary.json`.

| arm | HateMM | HateClipSeg | R1 − arm, HateMM | R1 − arm, HateClipSeg |
|---|---|---|---|---|
| R1 (= `replay`) | .8988 / .6949 / .7636 | .7358 / .6873 / .6597 | 0 | 0 |
| `no_remote` | .9003 / .6952 / .7655 | .7384 / .6911 / .6572 | −.0015 / −.0003 / −.0019 | −.0026 / −.0038 / +.0025 |
| `nearest` | .8975 / .6925 / .7663 | .7338 / .6860 / .6631 | +.0012 / +.0025 / −.0027 | +.0019 / +.0013 / −.0033 |
| `uniform` | .8978 / .6944 / .7648 | .7341 / .6863 / .6616 | +.0010 / +.0005 / −.0011 | +.0017 / +.0010 / −.0019 |
| r6 | .8971 / .6942 / .7508 | .7168 / .6711 / .6373 | | |

Gate: no control loses ≥ .01 against R1 on any metric in either corpus. None of the three components is supported:
- remote memory (`no_remote` is as good as R1, within noise on all six numbers);
- per-layer question matching (`nearest`);
- static/dynamic compression (`uniform`).

Where the R1 gain comes from: `no_remote` keeps only the LOCAL frames and still has the whole R1 gain over r6:
- HateMM +.0032 / +.0010 / +.0148;
- HateClipSeg +.0216 / +.0201 / +.0199.

So the gain comes from giving every 8 s window its own actual frames.

Raw visual reads over the 7352 covered windows:
- Remote memory changes the read (`no_remote` vs R1: mean |difference| 1.98 against an R1 standard deviation of 7.83;
  Spearman .965).
- The change does not improve the ranking.
- LOCAL frames change the read more (native vs R1: Spearman .852).
- R1 retrieval shares on average .17 of its 4 frames with `nearest` (identical in 0.46 % of layer reads).

Cost (actual, summed over both corpora):
- Reading per arm: `no_remote` 14.3 min; `replay` 30.6 min; `nearest` 29.1 min; `uniform` 30.6 min.
- Source memory per job: about 110 min.
- From the R1 records, the source vision encoding alone was 3.9 min of the 85.9 min of R1 source acquisition.

Conclusion: the StreamingTOM components (dual-path compression, quantized remote memory, per-layer retrieval) fail
rule 14(g) and cannot be claimed. Candidate 39 is not promoted; its mechanism is not established.

The positive finding is LOCAL frames on r6. That is an input change (rule 5: a new input is not novelty by itself),
and adopting it is the user's decision.

DeHate was not run. R1 and these controls cover HateMM and HateClipSeg only.

## Decomposing control #1 against r6 (declared 2026-10-08, before the run)

User instruction 2026-10-08: confirm where the gain of control #1 (`no_remote`) over r6 comes from, on HateMM and
HateClipSeg first. Control #1 differs from r6 in three things: the R1 attention code (`reader.factory`) instead of
the model's own forward; the LOCAL frames with their time labels; and the R1 role text, which still describes remote
memory. Two new arms change these one at a time. All share the fresh native prefix, G, own stance and S of each video:

| arm | attention code | LOCAL frames + time labels | role text |
|---|---|---|---|
| r6 (A) | model's own forward | no | no |
| `custom_native` (B) | R1 code, nothing inserted | no; r6's exact question ids and positions | no |
| `local_clean` (E) | R1 code, no remote | yes | no |
| control #1 | R1 code, no remote | yes | yes |

- B reads every window. E and control #1 keep the native V for the 7 windows without LOCAL, as control #1 does.
- `no_remote_replay` re-reads control #1 in the same job and must reproduce `controls_dualpath_main/no_remote` exactly.
- LOCAL features are read from the persisted R1 proofs, which equal fresh features exactly (controls alignment, all
  333 videos). The single-frame vision cost of these frames is therefore not in this job's times; it is reported from R1.
- Code: `local_controls.py`; default-preserving `role` argument added to `reader.encode_local_input`,
  `reader.encode_local` and `reader.visual_margin` (None omits the role text). CPU fixture check
  `local_selfcheck.py` PASS (B equals the native read exactly in FP32 and BF16 on the fixture; E has no role text and
  the same time labels/images), `runs/20261007_m1_streamingtom/local_cpu_checks/`.

Declared reading (pooled noise floor .005, within .01):
- code path: B − r6. No effect if all six differences are inside the noise floor.
- frames: E − B. Supported if one metric gains ≥ .01 in both corpora and no metric loses beyond the noise floor.
- role text: control #1 − E. No effect if all six differences are inside the noise floor.
- The three steps add up exactly to control #1 − r6.

Run: `launch/local_lab1.sbatch` on sc474397 (`SCOPE=smoke`, then `SCOPE=main`), output
`runs/20261007_m1_streamingtom/local_controls_{smoke,main}/`. Analysis: `STAGE=prepare|evaluate|report bash
launch/run_local_analysis.sh`; table `runs/20261007_m1_streamingtom/local_controls_analysis/summary.json`.
DeHate is not part of this run.

### Error analysis: empty versus covered windows (2026-10-08, sc474397, CPU; development, not a gate)

Question (user): does control #1 gain because windows that had no frame now have one, or elsewhere? Split from r6's
own input layout, no labels: a window is "empty" if none of the 20 shared prefix frames (`data/frames_k20` times)
falls inside [start, end), else "covered". Two diagnostic prediction sets take control #1's visual read in one kind of
window and r6's read elsewhere (speech, G and stance are r6's), then go through the sole evaluator and the fixed r6
decoder. Files read: `controls_dualpath_main/{base,no_remote}/predictions.jsonl`, `runs/20260926_glr/base_gridA`,
`data/frames_k20` file names; test GT only inside `src/eval/evaluate_four_datasets.py` and `twolevel_r2.py`.
Code `coverage_analysis.py`; output `runs/20261007_m1_streamingtom/coverage_analysis/{summary.json,decoded/}`.
No design was changed.

| | HateMM empty windows | HateClipSeg empty windows | HateMM gain vs r6 | HateClipSeg gain vs r6 |
|---|---|---|---|---|
| control #1 in empty windows only | 1030 / 3768 (114 videos) | 1233 / 3591 (all 118) | +.0008 / −.0005 / +.0034 | +.0071 / +.0066 / +.0092 |
| control #1 in covered windows only | | | +.0025 / +.0015 / +.0167 | +.0163 / +.0140 / +.0102 |
| control #1 everywhere | | | +.0032 / +.0010 / +.0148 | +.0216 / +.0201 / +.0199 |

Reading:
- The coverage explanation is not supported on HateMM. The within gain comes from covered windows (+.0167). Empty
  windows add +.0034.
- On HateClipSeg the within gain splits about evenly (+.0092 empty, +.0102 covered). The pooled gain comes more from
  covered windows.
- The visual read changes by a similar amount in both kinds of window (mean |Δ| 3.9 / 3.5 HateMM, 4.2 / 4.4
  HateClipSeg).
- So most of the gain is in windows that already had a shared frame inside. That points to more frames per window
  and/or frames placed next to the window's question, not only to filling empty windows. These two causes are not
  separated yet.
- The swaps are not additive: the decoder is fitted on the whole corpus. On HateMM the covered-only set is above full
  control #1.

### Decomposition results (2026-10-08; development-selected)

Run sc474397 / Slurm 315, 53:13, exit 0:0 (smoke Slurm 304 PASS first). `local_controls_main_analysis/alignment.json`
PASS on all 333 videos:
- native reads equal r6;
- `no_remote_replay` reproduces control #1 exactly in every window and in all six metrics;
- B (`custom_native`) equals the native visual read exactly in all 7359 windows (maximum |difference| 0.0);
- E has no role text and the same time labels and images as control #1 in every covered window.

Sources `local_controls_main_decoded/<arm>/metrics.json`; steps and verdicts `local_controls_analysis/summary.json`.

| | HateMM | HateClipSeg |
|---|---|---|
| r6 | .8971 / .6942 / .7508 | .7168 / .6711 / .6373 |
| B `custom_native` | .8971 / .6942 / .7508 | .7168 / .6711 / .6373 |
| E `local_clean` | .8978 / .6905 / .7644 | .7373 / .6906 / .6669 |
| control #1 | .9003 / .6952 / .7655 | .7384 / .6911 / .6572 |

| step | HateMM | HateClipSeg | verdict (declared) |
|---|---|---|---|
| code path, B − r6 | 0 / 0 / 0 | 0 / 0 / 0 | no effect (bit-identical reads) |
| frames, E − B | +.0006 / −.0037 / +.0136 | +.0205 / +.0195 / +.0296 | supported: within ≥ .01 in both corpora, no loss beyond noise |
| role text, #1 − E | +.0025 / +.0047 / +.0011 | +.0011 / +.0005 / −.0097 | within noise |

Conclusion:
- The gain of control #1 over r6 comes from the window's own LOCAL frames.
- The R1 attention code changes nothing.
- The leftover role text has no effect beyond the noise floor. It is close to the floor on HateMM PR (+.0047) and
  HateClipSeg within (−.0097).
- E (r6 plus LOCAL frames with time labels, no role text) against r6:
  - HateMM: +.0007 / −.0037 / +.0136;
  - HateClipSeg: +.0205 / +.0195 / +.0296;
  - within ≥ .01 in both corpora, every loss inside the noise floor.
- Not yet separated: more frames per window versus frames placed next to the window's question (see the error
  analysis above). DeHate not run. E is an input change (rule 5); adoption is the user's decision.

Cost (actual, both corpora, LOCAL features from R1 proofs):
- E reading: 18.3 min. Native visual read: 7.3 min.
- The 0.5 fps single-frame vision encoding is extra: at most 3.9 min in R1, which also includes saliency.

**2026-10-09 disk cleanup (user-approved):** `runs/20261007_m1_streamingtom/r1_full_main/proof/` (143 GB of per-frame audit tensors) was deleted on sc474397 and sc474398. Records, predictions, metrics and analyses are kept. The R1 strict CPU replay and `local_controls.py` (which read LOCAL features from these proofs) can no longer run as written; a rerun must recompute the single-frame vision features.

**2026-10-09 disk cleanup (user-approved, category A):** runs/20261007_m1_streamingtom/r1_full_smoke/proof/ and the controls vectors/ directories (retrieval vectors used only for the completed binding checks) were deleted on sc474397. controls_analyze.py prepare for the uniform job can no longer be rerun as written.

## Separating "more frames inside the window" from "frames next to the question" (declared 2026-10-09, before the run)

User question 2026-10-09: the decomposition above left two explanations of the LOCAL-frame gain unseparated: the window
now has more frames inside it (information), or those frames sit right before the window's question (placement). Four
arms separate them. All share the fresh native prefix of r6 and r6's G, own stance and S; only the visual read of
windows that have the arm's frames changes, every other window keeps the native V (as E does):

| arm | frames added | where | label text |
|---|---|---|---|
| r6 | none | | |
| E `local_clean` (stored, `local_controls_main`) | the window's LOCAL 0.5 fps frames | right before the question | `Actual LOCAL source at 12.000 seconds.` |
| `local_clean_replay` | same as E | same as E | same as E; re-read through the model's own forward with fresh vision (the R1 proofs are deleted) |
| `adjacent_local` | the same LOCAL frames | right before the question | `[t=12.0s]`, the prefix frames' own label |
| `prefix_local` | the same LOCAL frames | inserted into the prefix frame list in time order (native frame first on a tie); the question is r6's plain question | `[t=12.0s]` |
| `adjacent_native` | none new: the prefix frames whose time falls inside the window are shown again | right before the question | `[t=6.5s]` |

- `prefix_local` cannot share the KV cache: each covered window is one standalone forward of the whole conversation
  (prefix with 20 + k frames and the intro stating that count, transcript, rules, whole-video question, the native
  stance answer, window question). The whole-video question is not re-asked; G and S stay native.
- Windows without LOCAL frames keep the native V in the three LOCAL arms (7 windows, as E). `adjacent_native`
  changes only windows with at least one of the 20 prefix frames inside (HateMM 2738 / 3768, HateClipSeg 2358 /
  3591, from the coverage analysis above).
- Numeric floor of the no-cache path: for one window per video, r6's exact input is run as one standalone forward
  (`standalone_native`) and compared with the cached V; its token ids must equal the cached conversation's.
- Code: `placement_controls.py` (reading), `placement_analyze.py` (binding checks, sole evaluator + fixed r6 decoder,
  report), CPU fixture check `placement_selfcheck.py`. Launch `launch/placement_lab3.sbatch` (`SCOPE=smoke`, then
  `SCOPE=main`) on sc474398; `launch/placement_lab1.sbatch` and `launch/placement_lab2.sbatch` are the same job for sc474397 / sc474399 (queued on all three on 2026-10-09 because every lab GPU was held by baseline training under the two-GPU-per-user quota; the first to start runs, the others are cancelled). Output
  `runs/20261007_m1_streamingtom/placement_controls_{smoke,main}/`; analysis `STAGE=prepare|evaluate|report bash
  launch/run_placement_analysis.sh`; table `runs/20261007_m1_streamingtom/placement_analysis/summary.json`.

Declared readings (pooled noise floor .005, within .01; "supported" = one metric gains ≥ .01 in both corpora and no
metric loses beyond the floor):
- implementation: `local_clean_replay` − E inside the floor on all six; the per-window max |Δz| is reported.
- label wording: `adjacent_local` − E. Inside the floor on all six = no effect.
- position, same frames and labels: `adjacent_local` − `prefix_local`. Inside the floor on all six = position does
  not matter and the gain is from having the window's own frames. Supported for `adjacent_local` = placement next
  to the question contributes. Supported for `prefix_local` = the prefix placement is better. Otherwise mixed.
- more frames without adjacency: `prefix_local` − r6, supported or not.
- adjacency without new frames: `adjacent_native` − r6, supported or not.
- Nothing is promoted by this run; E stays an input change (rule 5). DeHate is not part of this run.

Independent code review (rule 6, 2026-10-09, before the run): one BLOCKER, the job and analysis assumed 20 native
frames while HateClipSeg `bit_AxrVklzh9Cyf` has 18 in `data/frames_k20` (as in r6); fixed to use the video's own
count. Record `docs/reviews/20261009_m1_streamingtom_placement_controls_code.md`. CPU fixture check
`runs/20261007_m1_streamingtom/placement_cpu_checks/summary.json` PASS (standalone native read within 4e-7 of the
cached read in FP32).

Planned cost: `adjacent_local` and `local_clean_replay` about E's 18 min each; `adjacent_native` less (fewer windows,
one to three images); `prefix_local` one full prefill per covered window, about 24 M prefix tokens plus the LOCAL
image tokens over both corpora, estimated 1 to 1.5 h on a 5090. Measured times go in the records.

### Results (2026-10-10; development-selected)

Run sc474399 / Slurm 344 (`SCOPE=both`: the fixed five, the in-job plumbing check PASS, then the full 333), 1:47
wall, exit 0; the lab3 / lab1 copies (343 / 345) were cancelled when 344 started. Binding
`placement_controls_main_analysis/alignment.json` PASS on all 333 videos: native reads equal r6 exactly; the
standalone conversation tokenizes as the cached one in every video; the arms change only the declared windows
(LOCAL arms 7352 / 7359 windows, `adjacent_native` 5096 / 7359). Metrics
`placement_controls_main_decoded/<arm>/metrics.json`; table, verdicts and per-window numbers
`placement_analysis/summary.json`.

| arm | HateMM ROC / PR / within (84) | HateClipSeg ROC / PR / within (99) |
|---|---|---|
| r6 | .8971 / .6942 / .7508 | .7168 / .6711 / .6373 |
| E `local_clean` (stored) | .8978 / .6905 / .7644 | .7373 / .6906 / .6669 |
| `local_clean_replay` | .8976 / .6899 / .7638 | .7374 / .6907 / .6649 |
| `adjacent_local` | .8977 / .6887 / .7690 | .7383 / .6893 / .6637 |
| `prefix_local` | .8985 / .6963 / .7640 | .7267 / .6812 / .6518 |
| `adjacent_native` | .8970 / .6938 / .7732 | .7252 / .6771 / .6501 |

| declared reading | HateMM | HateClipSeg | verdict |
|---|---|---|---|
| implementation, replay − E | −.0001 / −.0006 / −.0006 | +.0001 / +.0001 / −.0020 | inside the floor |
| wording, `adjacent_local` − E | −.0001 / −.0018 / +.0045 | +.0010 / −.0013 / −.0032 | inside the floor |
| position, `adjacent_local` − `prefix_local` | −.0008 / −.0076 / +.0049 | +.0116 / +.0081 / +.0119 | mixed: adjacent better on all three in HCS; HMM PR beyond the floor the other way |
| more frames only, `prefix_local` − r6 | +.0013 / +.0020 / +.0132 | +.0099 / +.0101 / +.0145 | supported (within both; no loss) |
| adjacency only, `adjacent_native` − r6 | −.0001 / −.0004 / +.0224 | +.0084 / +.0061 / +.0128 | supported (within both; no loss) |
| `adjacent_local` − r6 | +.0006 / −.0056 / +.0182 | +.0215 / +.0182 / +.0264 | within both, but HMM PR −.0056 is beyond the floor |

Per window (no labels): the replay differs from the stored E by mean 0.33 / max 3.7 logit (k images in one vision
call against E's single-frame features), Spearman .998, no mean shift; the no-cache standalone read of r6's own
input differs from the cached read by mean 0.16 / max 1.4. The LOCAL label wording raises z by 0.6 (HMM) / 0.9
(HCS) on average against the `[t=..s]` label without changing the ordering.

Reading, why the LOCAL frames help:
- Both causes are real and each is sufficient on its own for the within gain.
- Placement. r6's question tells the model to use only the frames whose timestamps fall inside the window, which
  it has to find among the 20 prefix frames by their labels. Showing the in-window prefix frames again right
  before the question, with no new image, gives within +.022 on HateMM (the largest within gain of any arm
  there) and +.013 on HateClipSeg.
- Information. Putting the 0.5 fps frames into the prefix frame list (about 4 per window instead of 1.1 on
  HateMM / 0.66 on HateClipSeg; 27% / 34% of windows have no prefix frame inside) gives within +.013 / +.015 and,
  on HateClipSeg only, pooled +.010 / +.010. HateClipSeg videos are longer (240 s against 136 s), so 20 frames
  are sparser there and extra frames carry more.
- The two do not add on HateMM: E and `adjacent_local` (within +.014 / +.018) stay below adjacency alone
  (+.022). On HateClipSeg they stack (E: pooled +.02, within +.030).
- No arm moves the pooled HateMM metrics; video-level ranking there stays with G.

Cost (measured, 333 videos, both corpora): native visual read 3.6 min; `adjacent_native` +4.7 min with no new
extraction; `adjacent_local` / replay +13.4 / +14.7 min plus the 0.5 fps extraction; `prefix_local` +57.2 min
(one prefill of 3.8 k tokens on average, 6354 max, per window); peak 18.0 GiB; whole job 107.5 min.

Files read: predictions of every arm, r6 and E, and the run records (no GT outside the sole evaluator). No
design was changed. Nothing is promoted; which input to adopt, if any, is the user's decision. DeHate not run.

### Error analysis: can a label-free rule pick the windows that need the extra read? (2026-10-10; development, not a gate)

User question 2026-10-10: the extra frames probably do not help in every window; can the windows that need them be
found without labels, so that only part of the windows is re-read and most of the gain is kept (about +.009
within on HateMM would be enough)? The same idea was tried as candidate 18 Explorer (2026-10-03/04,
`archive/experiments/20261003_m1_explorer/README.md`): entropy-gated acquisition of 2 to 4 new frames next to the
question, HateMM within +.0091 at R4, and the gate fired in 7344 of 7359 windows, so it did not select.

Simulation with the reads already in hand (`selection_analysis.py`, no GPU): for each arm, the arm's visual read is
taken only in the windows a rule selects and r6's read elsewhere; speech, G and stance stay r6's; sole evaluator
and fixed r6 decoder. Rules use only quantities known before the extra read: native visual margin v, native speech
margin s, number of prefix frames inside the window. `random_q50_*` are size-matched random controls (two
seeds). `oracle_help` selects with the test labels (window positive if any GT frame in it is positive; selected when
the arm moved z toward that label): a ceiling for selection, never a method. Files read: `placement_controls_main`
records, `local_controls_main/local_clean` predictions, `data/gt_4fps` (evaluator and oracle only). Outputs
`runs/20261007_m1_streamingtom/selection_analysis/{summary.json,<arm>__<rule>/,decoded/}`.

| arm | rule | changed windows HMM / HCS | HateMM gain vs r6 ROC / PR / within | HateClipSeg gain vs r6 |
|---|---|---|---|---|
| adjacent_native | full | 0.73 / 0.66 | -0.0001 / -0.0004 / +0.0224 | +0.0084 / +0.0061 / +0.0128 |
|  | unc_abs_1 | 0.06 / 0.07 | +0.0005 / +0.0002 / +0.0095 | +0.0012 / +0.0007 / +0.0012 |
|  | unc_abs_2 | 0.12 / 0.13 | +0.0005 / +0.0010 / +0.0079 | +0.0029 / +0.0015 / +0.0093 |
|  | unc_abs_3 | 0.18 / 0.21 | -0.0001 / -0.0014 / +0.0034 | +0.0042 / +0.0024 / +0.0136 |
|  | unc_q50 | 0.36 / 0.33 | -0.0006 / -0.0011 / +0.0167 | +0.0045 / +0.0046 / +0.0104 |
|  | top_q50 | 0.36 / 0.32 | +0.0003 / -0.0002 / +0.0105 | +0.0054 / +0.0031 / +0.0054 |
|  | disagree | 0.24 / 0.24 | -0.0007 / -0.0022 / +0.0075 | +0.0051 / +0.0029 / +0.0123 |
|  | random_q50_0 | 0.36 / 0.32 | -0.0003 / +0.0002 / +0.0096 | +0.0044 / +0.0047 / +0.0121 |
|  | random_q50_1 | 0.35 / 0.32 | -0.0007 / -0.0025 / +0.0140 | +0.0046 / +0.0025 / +0.0054 |
|  | oracle_help | 0.38 / 0.36 | +0.0064 / +0.0165 / +0.0531 | +0.0216 / +0.0185 / +0.0475 |
| local_clean | full | 1.00 / 1.00 | +0.0006 / -0.0037 / +0.0136 | +0.0205 / +0.0195 / +0.0296 |
|  | unc_abs_1 | 0.08 / 0.10 | +0.0003 / -0.0011 / +0.0082 | +0.0050 / +0.0056 / +0.0054 |
|  | unc_abs_2 | 0.16 / 0.20 | +0.0008 / -0.0040 / +0.0151 | +0.0087 / +0.0088 / +0.0109 |
|  | unc_abs_3 | 0.24 / 0.30 | +0.0008 / -0.0044 / +0.0039 | +0.0126 / +0.0108 / +0.0204 |
|  | unc_q50 | 0.49 / 0.49 | -0.0009 / -0.0053 / +0.0208 | +0.0107 / +0.0123 / +0.0132 |
|  | top_q50 | 0.49 / 0.49 | +0.0004 / -0.0006 / +0.0099 | +0.0085 / +0.0050 / +0.0072 |
|  | disagree | 0.30 / 0.36 | +0.0000 / -0.0005 / +0.0095 | +0.0092 / +0.0088 / +0.0200 |
|  | empty | 0.27 / 0.34 | -0.0003 / -0.0031 / +0.0017 | +0.0067 / +0.0065 / +0.0145 |
|  | covered | 0.73 / 0.66 | +0.0008 / -0.0004 / +0.0233 | +0.0144 / +0.0136 / +0.0156 |
|  | random_q50_0 | 0.49 / 0.49 | +0.0003 / -0.0014 / +0.0215 | +0.0106 / +0.0115 / +0.0167 |
|  | random_q50_1 | 0.49 / 0.49 | +0.0001 / -0.0030 / +0.0056 | +0.0109 / +0.0106 / +0.0189 |
|  | oracle_help | 0.51 / 0.57 | +0.0098 / +0.0234 / +0.0673 | +0.0413 / +0.0382 / +0.0900 |
| prefix_local | full | 1.00 / 1.00 | +0.0013 / +0.0020 / +0.0132 | +0.0099 / +0.0101 / +0.0145 |
|  | unc_abs_1 | 0.08 / 0.10 | +0.0004 / -0.0000 / +0.0057 | +0.0026 / +0.0033 / +0.0025 |
|  | unc_abs_2 | 0.16 / 0.20 | +0.0004 / +0.0000 / +0.0114 | +0.0041 / +0.0048 / +0.0057 |
|  | unc_abs_3 | 0.24 / 0.30 | +0.0001 / +0.0004 / +0.0093 | +0.0068 / +0.0073 / +0.0109 |
|  | unc_q50 | 0.49 / 0.49 | +0.0003 / +0.0004 / +0.0092 | +0.0068 / +0.0068 / +0.0093 |
|  | top_q50 | 0.49 / 0.49 | +0.0010 / +0.0012 / +0.0029 | +0.0039 / +0.0027 / +0.0067 |
|  | disagree | 0.30 / 0.36 | +0.0004 / +0.0015 / +0.0123 | +0.0036 / +0.0035 / +0.0085 |
|  | empty | 0.27 / 0.34 | +0.0001 / +0.0019 / +0.0098 | +0.0037 / +0.0030 / +0.0035 |
|  | covered | 0.73 / 0.66 | +0.0014 / +0.0007 / +0.0166 | +0.0075 / +0.0077 / +0.0078 |
|  | random_q50_0 | 0.49 / 0.49 | +0.0010 / +0.0028 / +0.0117 | +0.0058 / +0.0066 / +0.0060 |
|  | random_q50_1 | 0.49 / 0.49 | +0.0003 / +0.0013 / +0.0077 | +0.0060 / +0.0059 / +0.0130 |
|  | oracle_help | 0.59 / 0.57 | +0.0068 / +0.0142 / +0.0409 | +0.0227 / +0.0216 / +0.0609 |

Reading:
- The user's premise holds, strongly. Selecting only the windows where the change moves toward the label (oracle)
  gives within +.053 / +.048 (`adjacent_native`) and +.067 / +.090 (E), two to four times the full arm, and the
  pooled metrics move too. About half of the changed windows move the wrong way.
- None of the label-free rules finds those windows. On HateMM within, every rule lands inside the band of the two
  size-matched random subsets (random 50 %: +.0096 / +.0140 for `adjacent_native`, +.0056 / +.0215 for E), and the
  two random seeds themselves differ by .005 to .016, so a partial-application number below about .015 cannot be
  read as a rule working. No rule beats random on both corpora.
- Partial application keeps a proportional part of the gain: `adjacent_native` on the least-confident half of each
  video's windows (36 % / 33 % of windows) gives within +.0167 / +.0104, above the user's +.009 target on HateMM,
  but random halves give +.0096 / +.0121 and +.0140 / +.0054, so this is dilution, not selection.
- The only structural rule with a consistent sign is coverage, and it is corpus-specific: on HateMM, LOCAL frames
  in windows that already hold a prefix frame give within +.0233 (above the full +.0136, so the empty windows
  hurt there), on HateClipSeg the empty windows carry half the gain (+.0145 of +.0296).
- Conclusion: the headroom is in telling helpful from harmful changes, and the pre-read signals (confidence, rank,
  speech disagreement, coverage) do not carry that information. A rule would have to use the content of the extra
  read itself. Nothing changed in the method; no selection rule is adopted.
