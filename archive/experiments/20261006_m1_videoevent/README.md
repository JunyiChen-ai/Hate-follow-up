**归档第28项：完整333 R1无任一主指标提升≥.01，HMM PR退化超过噪声；按规则9不跑修订/控制GPU。**

# Candidate36: query-relevant events and representative background sources

Original unchanged nine-candidate pool C9/rank9, once-only independent proposal
PASS: `docs/reviews/20261005_m1_ideation_jury.md`. Run host will be recorded before
submission. Implementation starts 2026-10-06 on sc474397. No GPU, GT or result.
Current formal method remains r6_bma; complete215HateMM+118HateClipSeg, 4fps,
fixed native20/fullASR/own hard stance, independent V/S/max and r6. All subsequent
performance comparisons are development-selected.

Primary actual read: https://aclanthology.org/2026.lrec-1.395.pdf §§3.1–3.5/Fig4
and4.1, retained `runs/20261006_m1_videoevent/source_reading/`.
Original VideoEvent captions clips, computes MCQA candidate-answer relevance,
aggregates answers, smooths relevance, selects above-mean contiguous events,
acquires a representative background frame/description, and reasons from
selected captions/background. The adaptation uses one frozen Qwen3-VL-8B,
one neutral current-context request, no answer options/confidence/hate relevance.
Three-point smoothing, ±4-window candidate range and four selected event windows
are pool-defined bounded adaptations, not constants given by the source paper.
This is a functional adaptation, not a literal original MCQA/model reproduction.

## Complete pre-generation method

1. Decode actual PTS-based two-thirds frames for every original8s window using
   shared actual_video_frames, preserving actual source/ownership/ASR metadata.
   Same Qwen generates one neutral caption from those frames and literal ASR.
   Explicit instruction≤12words; unchanged fixed validation≤16words/64content
   tokens/96whole generation. UNKNOWN/invalid/capped remains unavailable.
2. For each current window form one literal neutral search request from its
   available caption and actual ASR. No inferred target/group/intent or hate
   answer. Consider window IDs within±4, including current. One text-only Qwen
   call gives each available candidate caption relevance1–10 or UNKNOWN to that
   request; unavailable captions or missing actual frames permit only UNKNOWN. Maximum128generated
   tokens. Scores only select sources, never enter final numeric predictions.
3. Three-point centered input-selection smoothing: a point remains unavailable
   if its own relevance is UNKNOWN. Otherwise mean its available immediate
   candidate neighbors and itself. Mean threshold over available smoothed points;
   choose points≥that mean, join consecutive original window IDs. Choose the
   event with highest smoothed member score, ties earliest event/window. Keep
   at most4 of its windows nearest current (distance then ID), restore chronology.
   Representative is highest smoothed score among retained windows, earliest
   for ties. These deterministic limits use no labels/fitted thresholds.
4. Its first actual third-selected frame is the representative background.
   Same Qwen generates a neutral scene/object-layout/action description,
   ≤12word instruction,16word/64content/96whole caps. Reuse only identical
   actual frame ownership and the same task within this video; charge original
   decoding/caption/relevance/background generation for every new video.
5. Final V sees original native conversation, original question, current actual
   LOCAL frames/ASR, selected event actual frames/ASR with owning times, caption
   interpretations clearly marked unverified, and representative frame with its
   unverified background interpretation. No query, relevance or threshold scalar
   enters reader. Remote observations are context, never current occurrence.
   Final S sees only actual remote selected-event ASR with timestamps, independently
   from V; no generated captions/background and no visual-derived hate decision.
   Missing source packets use the unchanged original branch. Native G/ownstance,
   8s/4fps/max and fixedr6 unchanged. At most11 actual images per newV (current2,
   selected4×2, representative1), real duplicate image input honestly counted.

No generated description is certified ground truth. UNKNOWN is legal; no forced
semantic source, salvage, alternate cohort or lower execution guard. Strict actual
current tokens/pixels/source selection/native allraw/clone checks precede GT.
Fixed5 remains first2per corpus+HMM114; require real caption/relevance/background
acquisition, event packet and changed V/S in each corpus before main. Whole333 on
one host, source+reader; remote BOTH inputs/results returned before STATUS results.
Cache `data/temporal_query_events`; outputs `runs/20261006_m1_videoevent`.
No imports from other experiments; stable shared src only.

New-video source calls W captions+at mostW relevance+at mostW backgrounds;
then native setup and at most2W fresh independent readings. Background reuse
requires exact same owned frame ID/task, never global corpus sharing. Native
JPEG/ASR may be reused, but decode/selection/source images and all new processing
are charged. Rough unmeasured extra15–50GPUmin/100windows plus ordinary reader;
actual wall clock/tokens/vision forwards/peak and cached-origin costs reported.
No rented compute/new dataset or new predictor.

Performance gate: against `runs/20260926_twolevel/r6_bma/metrics.json`, same main
metric +.01 both corpora, other pooled losses≤.005/within≤.01, report all6 and
eligible84/99. Canonical src evaluator only. Rule9 no qualifying+.01 archives;
any qualifying signal permits actual recorded GT error analysis and≤3 revisions.
No posthoc calibration/ensemble/dataset route or score smoothing.

Only after mainPASS run full333 controls: same source count independent top-k
instead of contiguous events; remove background acquisition/interpretation;
replace representative background by current-window source at same image count;
wrong event background with true donor ownership and generated donor description;
wrong whole-event packet at matched frame/text budget. Require dual-corpus common
metric≥.01 deletion losses for each claimed novel component, actual rawV/S/max
ordering and wrong-source falsification. Unsupported components are removed or
downgraded, never claimed from finalr6 alone. Complete necessary independent code
and final review, local evidence/STATUS precede goal success.

## Implementation and pre-GPU evidence

Complete prototype `events.py/interface.py/inputs.py/extract.py/measure.py/analyze.py` implements the full source-function chain; canonical evaluator and native Judge unchanged. Whole333 realraw/JPEG/ASR/native-token preflight PASS. 30,948 exhaustive selector cases against independently enumerated contiguous groups/UNKNOWN/means/ties/limits PASS. Real synthetic80s/PTS/pixel collector with actualtokenizer/processor/scripted provider acquire→strictvalidate, owned background reuse/costs and ASR/event/background/input-token corruption rejection PASS. 8actual36layer FP32/BF16×18/20×2/11new-image cached/full-position/uncachedreference/KV/clone checks PASS at unchanged tolerances; FP32 max4.2e-7, BF16 max.006942. 16productionreader CPU cases cover empty/current/max4-window event/no-frame packets, G/nativeallraw unchanged, independentS excludes all generated text, source costs retained and 11-image max exercised. All are software/random-weight checks, no pretrained semantics or performance. Evidence paths in manifest. Unique independent code review required before GPU.

Unique independent Rule6 review closed PASS: `docs/reviews/20261006_m1_videoevent_code.md`. Original independent realcollector/oracle/token-seam/16productionreader including11image fullreference artifacts were inspected by the finishing reviewer after the original handle disappeared; no second proposal/general review or pretended rerun. Largest independent BF16 fullreference difference.008237 within predeclared.025. Still only CPU software, not8B success.

Configured lab-server/sc448960 currently idle with existing identical HateVLM2.11cu128/5.15.1/av18.1/numpy2.5.2, nativeJPEG/ASR/reference/raw benchmark/model cache. All333 originaltest media copied from ROOT existingmanifest to documented laboratory `~/data/<dataset>/{video,videos}`; no annotations/newdatasets/GT used, every generated project output remains repository. Whole333 actualraw/native/ASR CPU preflight onsc448960 PASS and returned ROOT `runs/_setup_lab_server_hatevlm/input_preflight/summary.json`. New launchers `lab_server_native.sbatch` / `lab_server.sbatch`; independent narrow output-path/parser/resource/activation/failure-stop PASS `docs/reviews/20261006_lab_server_runtime_launch.md`. Actual native5 reproduction is next, then same5 scientific source+reader onthat machine. No GPU execution/performance conclusion yet. Rawcopy/process/runtime evidence `runs/_setup_lab_server_hatevlm/`.

Scientific samefixed5/sc448960 submittedSlurm212 after actualnative5/sourceinput and independentcode/launch PASS; source+paired wholecohort samejob/host. Pending/RUNNING after211 uses normal configured Slurm, no dependencies/chaining/script-submission or policy change. Original initialconstants/guard unchanged; real observations/UNKNOWN retained, noGT/performance until strict afterBOTHreturn.

Actualfixed5/sc448960/212 DONE08:55:50, BOTHsource+pairedruns andderivedcache returned ROOT; matching-runtime noGT prepare PASS `runs/20261006_m1_videoevent/r1_full_smoke_analysis/plumbing_summary.json`. Nativeallraw/Gexact; HMM96caption/16relevance/6BGcalls,13remote/16eventpackets/16changedV/2changedS; HCS62/41/8calls,37remote/41packets/41changedV/30changedS; 5actual V+5S clones exact. ActualHMM/HCSnewvideo processing84.0122/83.6160s includes source72.8135/73.7622s,18.78/18.74GiBpeak. F5 demonstrates execution not semantic correctness/performance; UNKNOWN and originalguard remain. Full333 samehost ready, sourceinputcost retained for pilot caches, noGT/mainmetrics.

Full333 scheduling declaration: aftersamefixed5PASS, select now-idle sc474398,1.3Tfree, whole333 realraw/native inputs already available (Spatial31 fullsource justcompleted there). Its existing reviewed `launch/lab3.sbatch` with SCOPE=main runs all333 source+reader as one experiment ononehost. No `data/temporal_query_events` exists there; acquire all333 fresh, do not splice sc448960 pilotinputs/readings. This avoids waiting behind fullR2/sc448960 and prioritizes lower measured167.63s fixed5cost over MERIT334.08s. LabuserQOS mayqueue behind207/209; no scheduler changes. Samecode/constants/method, not a revision; pilot evidence retainsits originalhost/cost separately. No mainGT yet.

Freshcomplete333/sc474398 submitted214 afterall4 committed3016995 match/unchanged foreignwork/STRAY/emptycandidatecache/1.3Tfree, evidence`machines_before_main{,_note}.txt`. NormalQOS waiting thenRUNNING after209; whole source+reader all333 samehost, sameoriginalR1 version/spec/guard. No mainGT/metrics.

Complete333/sc474398/Slurm214 COMPLETED2026-10-06 11:50:47, elapsed2:34:07/0:0; BOTH wholefreshsourcecache andruns returned ROOT before result/status claims. Originalfive/sc448960 source saved at data/temporal_query_events_smoke_sc448960 before overwriting standardcache with freshwholemain; main does not splice pilot inputs/margins. CachePROVENANCE records generating host/date/commands/readable paths. ROOT matchingruntime strictprepare then canonicalall6/fixedr6 running; no mainperformance/GT claim yet.

## Final complete R1 outcome and destination

Full333/sc474398/214 BOTH returned; matchingruntime strict currentraw/PTS/pixels/source-selection/caption/background/input/nativeallraw/G PASS, paired native canonical six metrics exactly equal currentr6. Authority `runs/20261006_m1_videoevent/r1_full_main_decoded/optimized/metrics.json`: HateMM ROC/PR/within .895782100769/.679467477326/.749919032928 (84/215); HateClipSeg .722113988793/.670560196809/.633512239814 (99/118). Deltas HMM −.001336561/−.014767126/−.000863011; HCS +.005289125/−.000511888/−.003836874. All development-selected. No same-primary dualgain and no individual+.01; HMM PR exceeds allowableloss. Rule9 archive28 directly, noR2/no fullmechanism controls or salvage.

Real acquisition/read execution was verified (1471/1762 eventpackets andchangedV, 1146/1361changedS), but neither source selection nor representative background receives a mechanism claim from this negative main result. Actual new-video processing HMM59.225217/HCS62.111983minutes, total121.337200; includes source50.749848/54.012187minutes, all actualcaption/relevance/background/media/reader cost retained. Paired native9.149947minutes, new processing13.261fold; Slurm2:34:07 wallclock includes paired checking/source replay/I/O, not sole deployment estimate. Costauthority `r1_full_main_analysis/alignment.json`; all outputs/source origins remain locally readable. Canonical evaluator, GT/split/4fps/r6/commonconstants unchanged. Only canonical evaluation/report read test GT/metrics; no GT-driven redesign after this noqualifyinggain outcome. Currentmethod and autonomous goal unchanged.

**2026-10-09 disk cleanup (user-approved, category A):** the derived cache data/temporal_query_events (with its PROVENANCE.md) was deleted on sc474397. Run outputs and metrics under runs/ are kept.
