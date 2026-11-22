#!/bin/bash
# Create the `asyncsim` conda env on CARC (run once on a login node).
set -ex
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n asyncsim python=3.10
conda activate asyncsim
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# robosuite 1.5.1 only sets mujoco>=3.2.3; newer mujoco (e.g. 3.17) breaks env creation
pip install robosuite==1.5.1 mujoco==3.3.0
pip install "git+https://github.com/ARISE-Initiative/robomimic.git@v0.5.0"
python -c "import torch, robosuite, robomimic, mujoco; print('OK', torch.__version__, robosuite.__version__, robomimic.__version__, mujoco.__version__)"
