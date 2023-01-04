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
4. Run scripts to train model. Go to `src`.
    - Local: 
        - `python train_resnet.py --config_path configs/pred_models/cifar10_res50.yaml`
        - `python train_resnet.py --config_path configs/pred_models/cub200_res50.yaml`
        - `python train_resnet.py --config_path configs/pred_models/oxford102_res50.yaml`
    - Unity: 
        - `sbatch script_train_resnet.sh`
        - Note: change specified config_path to desired dataset
5. To run scripts for explainability frameworks:
    - Lime: 
        - Local:
            - `python explain_lime.py --config_path configs/lime/cifar10.yaml`
            - `python explain_lime.py --config_path configs/lime/cub200.yaml`
            - `python explain_lime.py --config_path configs/lime/oxford102.yaml`
    - Semantic CVE: 
        - Create directories:  `output/semantic-cve/cifar`, `output/semantic-cve/oxford`, `output/semantic-cve/cub`
        - Install the following:
            - `conda install yaml`
            - `pip install pytorch-lightning`
            - `pip install -U albumentations`
            - `pip install pandas`
        - Run the code with one of the ff: 
            - `python explain_counterfactuals.py --config_path configs/counterfactuals/counterfactuals_ours_cub_res50.yaml`
            - `python explain_counterfactuals_cifar_gpu.py --config_path configs/counterfactuals/counterfactuals_ours_cifar_res50.yaml`
            - `python explain_counterfactuals_oxford_gpu.py --config_path configs/counterfactuals/counterfactuals_ours_oxford_res50.yaml`