#!/bin/bash  
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o ../slurm/slurm-%j.out  # %j = job ID

module load cuda/11.3.1
python train_resnet.py --config_path configs/pred_models/cub200_res50.yaml