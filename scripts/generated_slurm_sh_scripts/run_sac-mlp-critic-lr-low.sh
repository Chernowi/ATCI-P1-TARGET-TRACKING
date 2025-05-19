#!/bin/bash
#SBATCH --job-name=dl-sac-mlp-critic-lr-low
#SBATCH --account=nct328
#SBATCH --qos=acc_training
#SBATCH --time=01-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nct/nct01026/ATCI-P1/
#SBATCH --output=/home/nct/nct01026/ATCI-P1/out_logs/sac-mlp-critic-lr-low/job_output.log
#SBATCH --error=/home/nct/nct01026/ATCI-P1/out_logs/sac-mlp-critic-lr-low/job_error.log

module purge
module load impi intel hdf5 mkl python/3.12.1-gcc
# Ensure your Python environment (e.g., virtual environment) with PyTorch, Pydantic, etc.
# is activated here if not part of the loaded Python module.
# Example: source /path/to/your/venv/bin/activate

cd ~/ATCI-P1/  # Redundant if --chdir is effective, but harmless
python src/bsc_main.py -c sac_mlp_critic_lr_low
