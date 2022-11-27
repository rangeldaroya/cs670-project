# CS 670 Project: Analysis of Saliency Frameworks on Fine Grained Image Classification

# Requirements
1. Create an environment with python3.8.13:
```conda create -n cs670-project python=3.8.13```
2. Use cuda 11.3 and pytorch:
    - Local: `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch`
    - Unity:
        - `module load cuda/11.3.1`
        - activate environment: `conda activate cs670-project`
        - install pytorch 1.12.1: `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch`
3. Install additional packages with: `pip install -r requirements.txt`
4. Run scripts
    - To train resnet50 on cifar: 
        - Local: `python src/train_resnet_cifar10.py`
        - Unity: `sbatch script_cifar.sh`
5. To run scripts for explainability frameworks:
    - Lime: Create directories: `output/lime/cifar`, `output/lime/oxford`, `output/lime/cub`
    - Semantic CVE: Create directories:  `output/semantic-cve/cifar`, `output/semantic-cve/oxford`, `output/semantic-cve/cub`