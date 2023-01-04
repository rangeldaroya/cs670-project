#!/bin/bash  
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=32G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o ../slurm/slurm-%j.out  # %j = job ID

module load cuda/11.3.1
source ~/miniconda3/etc/profile.d/conda.sh conda activate cs670-project
python explain_lime.py --config_path configs/lime/cifar10.yaml