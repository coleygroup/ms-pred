#!/bin/bash
#SBATCH -n 1           # 1 core
#SBATCH -t 0-12:00:00   # 12 hours
#SBATCH -J ms # sensible name for the job
#SBATCH --output=logs/ms_run_%j.log   # Standard output and error log
#SBATCH --gres=gpu:volta:2  # full node on supercloud
#SBATCH -c 40  # full node on supercloud
#SBATCH --exclusive

## engaging configs
##SBATCH -p sched_mit_ccoley
##SBATCH --mem-per-cpu=20000 # 10 gb
##SBATCH --mem=20000 # 20 gb

##SBATCH -w node1236
##SBATCH --gres=gpu:1 #1 gpu
##SBATCH --mem=20000  # 20 gb 
##SBATCH -p {Partition Name} # Partition with GPUs

# Use this to run generic scripts:
# sbatch --export=CMD="python my_python_script --my-arg" src/scripts/slurm_scripts/generic_slurm.sh

# On Engaging
## Import module
#source /etc/profile
#source /home/samlg/.bashrc
#
## Activate conda
## source {path}/miniconda3/etc/profile.d/conda.sh
#
## Activate right python version
## conda activate {conda_env}
#conda activate ms-gen

# On Supercloud
module load anaconda/2022a
module load cuda/11.3

export HDF5_USE_FILE_LOCKING='FALSE'
ulimit -s unlimited
ulimit -u 65535
ulimit -n 65535

# Evaluate the passed in command... in this case, it should be python
eval $CMD
