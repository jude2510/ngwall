#!/bin/bash
# Approach-speed census on Sol: the 144 production initial conditions
# (seeds 0-15 x omega {0, 0.3, 0.6} x l_max {4, 8, 12}, same index decoding
# as submit_production.sh), evolved as pure Nambu-Goto to t = 2.
# Submit from ~/ngwall:  sbatch diagnostics/submit_census.sh
#SBATCH --job-name=ngwall-census
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-143
#SBATCH --output=logs/%x_%a.out

cd ~/ngwall/diagnostics
idx=${SLURM_ARRAY_TASK_ID}
seed=$(( idx % 16 ))
omegas=(0 0.3 0.6)
w=${omegas[$(( (idx / 16) % 3 ))]}
lmaxes=(4 8 12)
l_max=${lmaxes[$(( idx / 48 ))]}
~/.conda/envs/ngwall/bin/python approach_census.py "$l_max" "$seed" "$w" 2.0 ~/ngwall/census_bare
