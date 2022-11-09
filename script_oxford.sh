#!/bin/bash  
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 02:00:00  # Job time limit
#SBATCH -o slurm-oxford-%j.out  # %j = job ID

module load cuda/11.3.1
python src/train_resnet_oxford102.py