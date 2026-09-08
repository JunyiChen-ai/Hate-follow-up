# Follow-up working tree

This tree is a follow-up workspace branched from the EMNLP submission repo. It
was created on 2026-08-04 by cloning `/data/jehc223/EMNLP2` at commit `927d9aa`
(branch `label-free`) and checking out a new branch, `follow-up`.

## Why a separate directory

`/data/jehc223/EMNLP2` holds the frozen submission and its rebuttal artifacts.
New experiments must not overwrite those results, so code and outputs live here
while the 44 GB of inputs stay shared through symlinks.

## Layout

| Path | Kind | Notes |
| --- | --- | --- |
| `datasets` | symlink → `../EMNLP2/datasets` | 7.9 GB, read-only inputs |
| `external_repos` | symlink → `../EMNLP2/external_repos` | upstream baseline checkouts |
| `third_party` | symlink → `../EMNLP2/third_party` | vendored code |
| `results_frozen` | symlink → `../EMNLP2/results` | submission-era outputs, read only |
| `paper_emnlp2_frozen` | symlink → `../EMNLP2/paper` | Overleaf-synced submission paper |
| `.cache` | copy | 191 MB of resized frames, copied so writes stay local |
| `results` | new empty directory | every new experiment writes here |
| `logs` | new empty directory | Slurm output |

Treat everything reached through `results_frozen` and `paper_emnlp2_frozen` as
read-only. Writing through those links mutates the submission record.

## Absolute paths

The 191 tracked `.py` and `.sh` files that hardcoded `/data/jehc223/EMNLP2` now
point at `/data/jehc223/EMNLP3`. Documentation and CSV files under `docs/`,
`rebuttal/INVENTORY_REPORT.md`, and `STATE_ARCHIVE.md` keep the old path on
purpose, because they record where the submission artifacts were produced.

A script that reads a submission-era result file will now fail with a missing
path under `results/` rather than silently reading the frozen copy. That is
intended. Repoint such a script at `results_frozen/` explicitly when the old
numbers are genuinely what you want.

## Git remotes

`emnlp2-frozen` fetches from the local `/data/jehc223/EMNLP2` clone. Its push
URL is set to a nonexistent target so that no push can reach the submission
repo by accident. No GitHub remote is configured yet.

## Not carried over

`release/triage` lives in the submission tree only. It is an independently
published repository at `github.com/JunyiChen-ai/TRIAGE`; clone it separately if
the follow-up work needs it.
