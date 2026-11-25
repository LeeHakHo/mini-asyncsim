#!/bin/bash
# Render comparison videos for BC and Diffusion Policy and evaluate BC at +200 ms.
#   BC_CKPT=... DP_CKPT=... OUT=videos_new sbatch -A <account> carc/videos.slurm
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
OUT=${OUT:-videos_new}

# Smoke test with a random policy.
python scripts/eval.py --dummy --dataset datasets/lift/ph/low_dim_v15.hdf5 --mode async -n 2 \
    --horizon 60 --extra_latency 0.05 --out results/smoke_mp.csv

# Videos.
python scripts/make_video.py --ckpt "$BC_CKPT" --extra_latency 0.2 --out "$OUT/bc_lat200"
python scripts/make_video.py --ckpt "$DP_CKPT" --extra_latency 0.0 --out "$OUT/dp_lat0"
python scripts/make_video.py --ckpt "$DP_CKPT" --extra_latency 0.2 --out "$OUT/dp_lat200"

# BC at +200 ms, 20 rollouts per mode.
CSV=results/bc_lat200_modes_mp.csv
python scripts/eval.py --ckpt "$BC_CKPT" --mode sync               --extra_latency 0.2 -n 20 --out $CSV
python scripts/eval.py --ckpt "$BC_CKPT" --mode async --hold repeat --extra_latency 0.2 -n 20 --out $CSV
python scripts/eval.py --ckpt "$BC_CKPT" --mode async --hold stop   --extra_latency 0.2 -n 20 --out $CSV
echo VIDEOS_DONE
