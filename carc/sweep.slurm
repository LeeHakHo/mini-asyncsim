#!/bin/bash
# Evaluate every grid cell of scripts/sweep_grid.py in parallel as a Slurm job array.
#   BC_CKPT=... DP_CKPT=... sbatch -A <account> --array=0-$(( $(python scripts/sweep_grid.py --count) - 1 )) carc/sweep.slurm
#SBATCH --job-name=asyncsim-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep_%A_%a.out

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asyncsim
cd "$SLURM_SUBMIT_DIR"
export MUJOCO_GL=egl

ARGS=$(python scripts/sweep_grid.py --index "$SLURM_ARRAY_TASK_ID")
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): $ARGS"
python scripts/eval.py $ARGS -n "${N_ROLLOUTS:-50}" --out "results/sweep_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}.csv"
