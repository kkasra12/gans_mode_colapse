#!/bin/bash -l
#SBATCH -J mnist_train
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 7
#SBATCH -G 1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH -p gpu

print_error_and_exit() {
    echo "***ERROR*** $*"
    exit 1
}
module purge || print_error_and_exit "No 'modulcae' command"
# module load numlib/cuDNN   # Example with cuDNN

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK # Propagate Slurm 'cpus-per-task' to srun
# module load lang/python3
echo "Running on $(hostname)"
conda activate llama_env
echo "Running main.py"
srun --unbuffered python -m fire main.py Main train --num_epochs 40
echo "Done"
# to run this file we can use the following command
# sbatch train_model.sh
