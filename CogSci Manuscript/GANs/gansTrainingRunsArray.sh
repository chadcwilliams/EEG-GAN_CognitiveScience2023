#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:8

#SBATCH --constraint=titanv|titanrtx|quadrortx

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 1
#SBATCH --mem 32G

#SBATCH -t 01:00:00
# Specifiy the output file 

#SBATCH --array=1-20

# Job handling
#SBATCH -J arrayjob-readline
#SBATCH -o %x-%A_%a.out

#Load Python etc.
module load python/3.9.0
module load gcc/5.4
module load cuda/8.0.61

#Environment
source ./venv/bin/activate

dataFile="`sed -n ${SLURM_ARRAY_TASK_ID}p gansTrainingFiles.txt`"
python gan_training_main.py ddp filter_generator path_dataset="${dataFile}" n_epochs=1000
