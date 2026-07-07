#!/bin/bash
#SBATCH -n 1
#SBATCH -t 1-18:00:00
#SBATCH -J ms
#SBATCH --output=logs/ms_run_%j.log
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G

# Use this to run generic scripts:
# sbatch --export=CMD="python my_python_script --my-arg" launcher_scripts/generic_slurm.sh

set +u
source /etc/profile
source "$HOME/.bashrc"
conda activate ms-gen
set -u

export HDF5_USE_FILE_LOCKING='FALSE'
ulimit -s unlimited
ulimit -u 65535
ulimit -n 65535

# Evaluate the passed in command... in this case, it should be python
eval $CMD
