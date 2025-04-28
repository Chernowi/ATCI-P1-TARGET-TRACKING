#!/bin/bash
#SBATCH --job-name=dl-target
#SBATCH --account=nct328
#SBATCH --qos=acc
#SBATCH --time=01-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/nct/nct01026/ATCI-P1/
#SBATCH --output=/home/nct/nct01026/ATCI-P1/out_logs/job_output.log
#SBATCH --error=/home/nct/nct01026/ATCI-P1/out_logs/job_error.log

module purge

module load  impi  intel  hdf5  mkl  python/3.12.1-gcc

cd ~/ATCI-P1/
time python src/main.py