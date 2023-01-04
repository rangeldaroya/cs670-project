import argparse
import yaml
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
from loguru import logger


import torch
import torch.nn.functional as F

import pandas as pd
import os

from dataloader.dataset import get_dataset, get_transforms
from utils.model import get_model
from utils.results import save_and_update_results
from utils.transforms import get_lime_result

RANDOM_SEED = 0     # set this to have consistent results over runs
device = "cuda"

parser = argparse.ArgumentParser(description="Train image classification model")
parser.add_argument("--config_path", type=str, required=True)

np.random.seed(RANDOM_SEED)

pill_transf, preprocess_transform = get_transforms()

def batch_predict(images, random=False):
    m = random_model if random else model
    m.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)
    batch = batch.to(device)
    
    logits = m(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def generate_affine_vals(num_imgs, config_dir, im_size=224):
    rot_vals_deg = np.random.randint(low=-80, high=80, size=num_imgs)

    t_1 = np.zeros((num_imgs//2,2))    # half don't have translation
    t_2 = np.random.randint(low=-0.1*im_size, high=0.1*im_size, size=(num_imgs-(num_imgs//2),2))
    trans_vals = np.concatenate((t_1,t_2))

    s_1 = np.ones(num_imgs//2)          # half don't have scaling
    s_2 = np.random.rand(num_imgs-(num_imgs//2))+1  # generates n-dim array in range of [1,2)
    scales = np.concatenate((s_1,s_2))

    np.savetxt(f"{config_dir}/lime_cifar_rot_vals_deg.txt", rot_vals_deg)
    np.savetxt(f"{config_dir}/lime_cifar_trans_vals.txt", trans_vals)
    np.savetxt(f"{config_dir}/lime_cifar_scales.txt", scales)

    return rot_vals_deg, trans_vals, scales

if __name__=="__main__":
    # Load config file
    args = parser.parse_args()
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)
    logger.debug(f"LIME run with {config}")
    if config["dataset"] == "cub200":
        idx2label = open("./data/CUB_200_2011/classes.txt", "r").readlines()
        idx2label = [x[:-1].split(" ")[1].split(".")[1] for x in idx2label]
    else:
        idx2label = config['idx2label']

    # Check for GPU
    if torch.cuda.is_available():  
        dev = "cuda:0"
        logger.debug(f"Found GPU. Using: {dev}")
    else:  
        dev = "cpu"
    device = torch.device(dev)

    # Define data loaders
    trainset, testset = get_dataset(dataset=config["dataset"])
    test_dl = torch.utils.data.DataLoader(
        testset, batch_size=config["test_batch_size"], shuffle=False#, num_workers=2
    )
    num_imgs = len(test_dl)

    # Define model
    model = get_model(config["dataset"], device, model_path=config['model_path'])
    random_model = get_model(config["dataset"], device, is_random=True)
    model.eval()
    random_model.eval()

    rot_vals_deg, trans_vals, scales = generate_affine_vals(num_imgs, config["config_dir"])

    # Define output directories
    if not os.path.exists(config['out_dir']):
        os.makedirs(config['out_dir'])
    if not os.path.exists(f"{config['out_dir']}/{config['dataset']}"):
        os.makedirs(f"{config['out_dir']}/{config['dataset']}")

    # Load previous results to append to
    results = []
    if config['to_append_results']:
        prev_results = pd.read_csv(f"{config['out_dir']}/lime_{config['dataset']}_results.csv")
        results = prev_results.values.tolist()
    pos_ious, neg_ious = [], []
    for i, (inputs, targets) in enumerate(test_dl):
        if i < config['start_idx']:
            continue

        # Define rotation, translation, and scale
        is_rot_only = False
        rot_val_deg = rot_vals_deg[i]
        rot_val = rot_val_deg*np.pi/180.0
        trans_val = trans_vals[i]
        scale_val = scales[i]
        if (all(trans_val == np.array([0,0]))) and (scale_val==1) and (rot_val!=0):
            is_rot_only = True

        target_class = idx2label[targets]

        img = inputs.cpu().detach().numpy()
        img = img[0,:,:,:].swapaxes(0,1).swapaxes(1,2)

        # Get result for original image
        pred_class, orig_mask = get_lime_result(
            i,
            img,
            # lime_img=lime_image,
            t_type="orig",
            batch_predict=batch_predict,
            idx2label=idx2label,
            pill_transf=pill_transf,
            target_class=target_class,
            config=config,
            random_seed=config['random_seed'],
            to_save_imgs=config['to_save_imgs'],
        )

        # Get result for each of the transformed images
        pos_ious = {k:None for k in config['transform_types']}
        neg_ious = {k:None for k in config['transform_types']}
        class_preds = {k:None for k in config['transform_types']}
        for t_type in config['transform_types']:
            if t_type != "affine":
                t_pred_class, pos_iou, neg_iou = get_lime_result(
                    i,
                    img,
                    # lime_img=lime_image,
                    t_type=t_type,
                    batch_predict=batch_predict,
                    idx2label=idx2label,
                    pill_transf=pill_transf,
                    target_class=target_class,
                    config=config,
                    random_seed=config['random_seed'],
                    to_save_imgs=config['to_save_imgs'],
                    orig_mask=orig_mask,
                )
            else:
                t_pred_class, pos_iou, neg_iou = get_lime_result(
                    i,
                    img,
                    # lime_img=lime_image,
                    t_type=t_type,
                    batch_predict=batch_predict,
                    idx2label=idx2label,
                    pill_transf=pill_transf,
                    target_class=target_class,
                    config=config,
                    random_seed=config['random_seed'],
                    to_save_imgs=config['to_save_imgs'],
                    orig_mask=orig_mask,
                    is_rot_only=is_rot_only,
                    trans_val=trans_val,
                    rot_val=rot_val,
                    rot_val_deg=rot_val_deg,
                    scale_val=scale_val,
                )
            pos_ious[t_type] = pos_iou
            neg_ious[t_type] = neg_iou
            class_preds[t_type] = t_pred_class

        # Log results to csv file
        results = save_and_update_results(
            i,
            results,
            target_class,
            pred_class,
            pos_ious,
            neg_ious,
            class_preds,
            config,
            is_rot_only,
            rot_val_deg,
            trans_val,
            scale_val,
        )
        logger.info(f"Done marking img {i+1:02d}/{len(test_dl)} [idx={i}]")
        
        if i >= config['end_idx']:
            break