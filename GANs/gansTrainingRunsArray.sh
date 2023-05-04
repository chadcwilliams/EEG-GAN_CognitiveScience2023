#!/bin/bash

# Request a GPU partition node and access to 8 GPUs
#SBATCH -p gpu --gres=gpu:8

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 2 CPU core
#SBATCH -n 2

# Request 23GB of memort
#SBATCH --mem 32G

# Request 24 hours
#SBATCH -t 24:00:00
# Specifiy the output file 

# Determine number of files to process
#SBATCH --array=1-35

# Job handling
#SBATCH -J arrayjob-readline
#SBATCH -o %x-%A_%a.out

#Load Python etc.
module load python/3.9.0
module load gcc/10.2
module load cuda/11.7.1
module load cudnn/8.2.0

#Environment
source ./venv/bin/activate

dataFile="`sed -n ${SLURM_ARRAY_TASK_ID}p gansTrainingFiles.txt`"
python gan_training_main.py ddp filter_generator path_dataset="${dataFile}" n_epochs=8000
