#!/bin/bash  
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 02:00:00  # Job time limit
#SBATCH -o slurm-cub-%j.out  # %j = job ID

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs670-project

module load cuda/11.3.1
python src/train_resnet_cub.py