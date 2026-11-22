#!/bin/bash
# Render sync / async_repeat / async_stop videos for BC and Diffusion Policy,
# plus a quick success-rate check of the three modes for BC at +200 ms.
#   BC_CKPT=... DP_CKPT=... sbatch -A <account> carc/videos.slurm
#SBATCH --job-name=asyncsim-videos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/videos_%j.out

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asyncsim
cd "$SLURM_SUBMIT_DIR"
export MUJOCO_GL=egl

# skip videos that already exist
[ -f videos/bc_lat200_side_by_side.mp4 ] || python scripts/make_video.py --ckpt "$BC_CKPT" --extra_latency 0.2 --out videos/bc_lat200
[ -f videos/dp_lat0_side_by_side.mp4 ] || python scripts/make_video.py --ckpt "$DP_CKPT" --extra_latency 0.0 --out videos/dp_lat0
[ -f videos/dp_lat200_side_by_side.mp4 ] || python scripts/make_video.py --ckpt "$DP_CKPT" --extra_latency 0.2 --out videos/dp_lat200

python scripts/eval.py --ckpt "$BC_CKPT" --mode sync  --extra_latency 0.2 -n 20 --out results/bc_lat200_modes.csv
python scripts/eval.py --ckpt "$BC_CKPT" --mode async --hold repeat --extra_latency 0.2 -n 20 --out results/bc_lat200_modes.csv
python scripts/eval.py --ckpt "$BC_CKPT" --mode async --hold stop   --extra_latency 0.2 -n 20 --out results/bc_lat200_modes.csv
echo VIDEOS_DONE
