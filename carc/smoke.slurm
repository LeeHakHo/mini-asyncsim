#!/bin/bash
# Download the Lift dataset and run sync/async rollouts with a random policy.
#   sbatch -A <account> carc/smoke.slurm
#SBATCH --job-name=asyncsim-smoke
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_%j.out

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate asyncsim
cd "$SLURM_SUBMIT_DIR"

DS=datasets/lift/ph/low_dim_v15.hdf5
if [ ! -f "$DS" ]; then
  python -m robomimic.scripts.download_datasets --download_dir datasets --tasks lift --dataset_types ph --hdf5_types low_dim
fi
ls -lh "$DS"

for MODE in sync async; do
  python scripts/eval.py --dummy --dataset "$DS" --mode $MODE -n 2 --horizon 100 \
      --extra_latency 0.05 --out results/smoke.csv --tag smoke
done
echo SMOKE_DONE
