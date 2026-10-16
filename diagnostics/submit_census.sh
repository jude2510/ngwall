#!/bin/bash
# Approach-speed census on Sol. Array index decoding as submit_production.sh:
# seed = idx % 16, omega = (0, 0.3, 0.6)[(idx / 16) % 3], l_max = (4, 8, 12)[idx / 48].
# Environment: NSUB (default 4), TMAX (default 2.0); results go to
# ~/ngwall/census_v2_n$NSUB/. Submit from ~/ngwall:
#   full production grid:  sbatch diagnostics/submit_census.sh
#   resolution check (l_max 4/8/12 x seeds 0-7, omega 0, to t = 1.5):
#     NSUB=4 TMAX=1.5 sbatch --array=0-7,48-55,96-103 diagnostics/submit_census.sh
#     NSUB=5 TMAX=1.5 sbatch --array=0-7,48-55,96-103 --partition=public \
#         --time=2-00:00:00 diagnostics/submit_census.sh
#SBATCH --job-name=ngwall-census
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-143
#SBATCH --output=logs/%x_%A_%a.out

NSUB=${NSUB:-4}
TMAX=${TMAX:-2.0}
cd ~/ngwall/diagnostics
idx=${SLURM_ARRAY_TASK_ID}
seed=$(( idx % 16 ))
omegas=(0 0.3 0.6)
w=${omegas[$(( (idx / 16) % 3 ))]}
lmaxes=(4 8 12)
l_max=${lmaxes[$(( idx / 48 ))]}
~/.conda/envs/ngwall/bin/python approach_census.py "$l_max" "$seed" "$w" "$TMAX" \
    ~/ngwall/census_v2_n"$NSUB" --nsub "$NSUB"
