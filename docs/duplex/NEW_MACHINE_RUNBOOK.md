# New-Machine Runbook — Duplex Reading Kill-Test

Audience: the coding agent operating on a machine with direct command-line
GPU access (no Slurm). Mission: run the pre-registered kill-test probe and
report the P1–P5 verdict. This is a diagnostic experiment; the rules that
make it valid are in `docs/duplex/PREREG_duplex_killtest.md` — read that
file first.

## Hard rules

1. **Do not edit** `READER_BLOCKS` in `src/duplex/score_duplex_probe.py`,
   the prompt skeleton, or any threshold in the pre-registration. If a
   prompt seems broken, report it; a changed operationalization requires a
   new dated amendment in the prereg file, never silent wordsmithing.
2. **Do not touch the ImpliHateVid test split.** The probe runs on
   `train_clean` only.
3. EX/IM/NH prefixes are diagnostic ground truth for the analysis script
   only. No scoring component may consume them.
4. Report all pre-registered arms regardless of outcome — no cherry-picking.

## Environment

Reference environment (University of Auckland cluster, verified working):

| Component | Version |
|---|---|
| Python | 3.11.14 |
| vllm | 0.11.0 (pin this; scorer behavior is tied to it) |
| torch | 2.8.0+cu129 |
| transformers | 4.57.1 |
| numpy | 2.2.6 |

Setup: `pip install vllm==0.11.0 transformers==4.57.1` (vllm pulls a
matching torch). GPU needs: Qwen3-VL-8B-Instruct in bf16 wants ≥24 GB VRAM
at `--gpu-mem 0.90`; the 2B fits in ~8 GB. Models download automatically
from Hugging Face on first run (`Qwen/Qwen3-VL-2B-Instruct`,
`Qwen/Qwen3-VL-8B-Instruct`, not gated, ~4.5 GB + ~17 GB).

## Data

Minimal payload (~1.7 GB — do NOT transfer the 51 GB `video/` directory):

```
$HVD_DATA_ROOT/ImpliHateVid/
├── annotation(new).json     # 2009 entries: Video_ID, Label, Title, Transcript
├── splits/                  # train/val/test csv + *_clean.csv
└── frames_16/               # 2009 subdirs of 16 pre-extracted jpgs
```

The payload is pre-packed on the source cluster as a single tarball:

```
/data/jehc223/ihv_killtest_payload.tar   # 1.5 GB
md5: b5a7fc7eda9aa102248c8dac81348f17
```

**Preferred route — Backblaze B2 (no SSH connectivity needed).** The
tarball is uploaded at `b2:junyi-data/hate-followup/ihv_killtest_payload.tar`
(1,590,435,840 bytes). Install rclone, then copy the owner's rclone config
(a 90-byte `[b2]` section with `account` + `key`, from
`~/.config/rclone/rclone.conf` on the cluster — the owner supplies it) to
`~/.config/rclone/rclone.conf` on this machine, and pull:

```
rclone copy b2:junyi-data/hate-followup/ihv_killtest_payload.tar . --progress
```

Fallback — direct scp if this machine can reach the campus network / VPN:

```
scp jehc223@foscsmlprd02.its.auckland.ac.nz:/data/jehc223/ihv_killtest_payload.tar .
```

Either way, then:

```
md5sum ihv_killtest_payload.tar          # must match the hash above
tar -xf ihv_killtest_payload.tar -C $HVD_DATA_ROOT
```

Integrity checks after extraction:

- `ls $HVD_DATA_ROOT/ImpliHateVid/frames_16 | wc -l` → 2009
- `wc -l $HVD_DATA_ROOT/ImpliHateVid/splits/train_clean.csv` → 1283
- prefix census of train_clean: 634 NH / 325 EX / 324 IM

## Run

```
export HVD_DATA_ROOT=/path/that/contains/ImpliHateVid
cd <repo root>
bash scripts/duplex/run_duplex_probe_local.sh
```

Or invoke the stages manually — the runner is just the two probe calls
(2B batch 16, then 8B batch 8, both with `--no-video`) followed by the two
analysis calls. Notes:

- `--no-video` feeds the 16 pre-extracted frames as images. This is
  required here (no mp4s in the minimal payload) and must be held constant
  across all readers of a run. The source-cluster replication (job 18386)
  runs in mp4 video mode; the two runs are parallel replications — never
  mix their output files.
- **Resume**: output is appended per batch with fsync; rerunning the same
  command skips already-scored videos. A killed run loses at most one batch.
- Runtime ballpark on a single modern GPU: 2B ~1–1.5 h, 8B ~2.5–4 h for
  5 readers × 1283 videos each.
- Scores land in `results/duplex_probe/ImpliHateVid/train_<reader>_<slug>.jsonl`
  (gitignored). The analysis JSONs land in `docs/duplex/reports/` (tracked).

## Report back

1. Run the analysis for both model slugs (the runner does this) and read
   the verdict against the prereg table: P1 AUCs ≥ 0.60, P4a ≥ 0.70,
   P4b < 0, P5a placebo gap ≥ 0.05, P5b noise floor ≤ 0.55.
2. Commit `docs/duplex/reports/*.json` and push — that is how results flow
   back to the main line. Do not commit anything under `results/`.
3. In the report, include the failure-mode fingerprint on the IM subset
   (lit-high+prag-high = leakage; lit-low+prag-low = knowledge absence) —
   it decides what happens next, not just whether the test passed.
