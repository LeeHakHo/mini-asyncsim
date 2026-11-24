#!/bin/bash
# Re-render the comparison videos only (no evaluation rollouts).
#   BC_CKPT=... DP_CKPT=... [OUT=... HORIZON=...] sbatch -A <account> carc/videos_only.slurm
#SBATCH --job-name=asyncsim-videos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/videos_only_%j.out

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asyncsim
cd "$SLURM_SUBMIT_DIR"
export MUJOCO_GL=egl
OUT=${OUT:-videos_new}
H="--horizon ${HORIZON:-400}"
python scripts/make_video.py --ckpt "$BC_CKPT" $H --extra_latency 0.2 --out "$OUT/bc_lat200"
python scripts/make_video.py --ckpt "$DP_CKPT" $H --extra_latency 0.0 --out "$OUT/dp_lat0"
python scripts/make_video.py --ckpt "$DP_CKPT" $H --extra_latency 0.2 --out "$OUT/dp_lat200"
echo VIDEOS_DONE
