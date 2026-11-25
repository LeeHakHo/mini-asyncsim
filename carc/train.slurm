#!/bin/bash
# Train a low-dim robomimic policy. TASK: lift (default) or square.
#   sbatch -A <account> carc/train.slurm bc
#   sbatch -A <account> carc/train.slurm diffusion_policy square [num_epochs]
#SBATCH --job-name=asyncsim-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%x_%j.out

set -e
ALGO=${1:?usage: sbatch carc/train.slurm <bc|diffusion_policy> [lift|square]}
TASK=${2:-lift}
EPOCHS=${3:-}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asyncsim
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader

DS=datasets/$TASK/ph/low_dim_v15.hdf5
if [ ! -f "$DS" ]; then
  python -m robomimic.scripts.download_datasets --download_dir datasets --tasks "$TASK" --dataset_types ph --hdf5_types low_dim
fi
CFG=$(python scripts/make_config.py "$ALGO" "$TASK" $EPOCHS)
python -m robomimic.scripts.train --config "$CFG"
