#!/usr/bin/env bash
# All declared arms (README §5) from runs/20260922_til/gridA and gridB; writes runs/20260922_til/table.md.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY="${TIL_PYTHON:-$HOME/miniconda3/envs/HateVideo/bin/python}"
A=runs/20260922_til/gridA; B=runs/20260922_til/gridB
I="$PY experiments/20260922_til/til_infer.py"
{
$I --runs $A --model none --dwell 0 --tag A0_spvl_replicate
$I --runs $A --model average --dwell 80 --tag A1_gridA_prior80
$I --runs $A $B --model average --dwell 0 --tag A2_AB_average
$I --runs $A $B --model average --dwell 80 --tag A3_AB_average_prior80
$I --runs $A $B --model interval --dwell 80 --tag A4_AB_interval_prior80
$I --runs $A $B --model interval --dwell 0 --tag A5_AB_interval_noprior
$I --runs $A $B --model interval --dwell 80 --fusion sum --tag A6_AB_interval_prior80_sum
$I --runs $B --model average --dwell 80 --tag B1_gridB_prior80
for D in 40 160; do
  $I --runs $A --model average --dwell $D --tag A1_gridA_prior$D
  $I --runs $A $B --model average --dwell $D --tag A3_AB_average_prior$D
  $I --runs $A $B --model interval --dwell $D --tag A4_AB_interval_prior$D
done
} | tee runs/20260922_til/table.txt
echo "INFER_DONE $(date -Is)"
