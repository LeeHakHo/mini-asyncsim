#!/bin/bash
# Train a robomimic policy on Lift (low-dim) on one GPU.
#   sbatch -A <account> carc/train.slurm bc
#   sbatch -A <account> carc/train.slurm diffusion_policy
#SBATCH --job-name=asyncsim-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%x_%j.out

set -e
ALGO=${1:?usage: sbatch carc/train.slurm <bc|diffusion_policy>}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asyncsim
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader

CFG=$(python scripts/make_config.py "$ALGO")
python -m robomimic.scripts.train --config "$CFG"
