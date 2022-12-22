# from models.resnet import resnet50
import argparse
import torch
from torchvision.models import resnet50, ResNet50_Weights
from loguru import logger
import numpy as np
import yaml


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt


from utils.model import get_model, get_optimizer
from dataloader.dataset import get_dataset

parser = argparse.ArgumentParser(description="Train image classification model")
parser.add_argument("--config_path", type=str, required=True)

def train_model(train_dl, model, optimizer, num_epochs):
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    # val_losses = []
    for epoch in tqdm(range(num_epochs)):
        # model.train()
        train_losses_epoch = 0
        for i, (inputs, targets) in enumerate(train_dl):
            targets = targets.to(device)
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
            train_losses_epoch += loss.item()

        train_losses.append(train_losses_epoch/len(train_dl))

    return train_losses

def evaluate_model(val_dl, model):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_loss, num_correct, num_total = 0, 0, 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_dl):
            targets = targets.to(device)
            inputs = inputs.to(device)
            yhat = model(inputs)
            
            loss = criterion(yhat, targets)

            test_loss += loss.item()
            _, predicted = yhat.max(1)
            num_total += targets.size(0)
            num_correct += (predicted == targets).sum().item()

            logger.debug(f"test_loss: {test_loss}, num_correct: {num_correct}, num_total: {num_total}")

    accuracy = num_correct/num_total
    return test_loss, accuracy

if __name__=="__main__":
    logger.debug("Training prediction model")
    # Load config file
    args = parser.parse_args()
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)

    # Check for GPU
    if torch.cuda.is_available():  
        dev = "cuda:0"
        logger.debug(f"Found GPU. Using: {dev}")
    else:  
        dev = "cpu"
    device = torch.device(dev)

    # Define data loaders
    trainset, testset = get_dataset(dataset=config["dataset"])
    train_dl = torch.utils.data.DataLoader(
        trainset, batch_size=config["train_batch_size"], shuffle=True#, num_workers=8
        )
    test_dl = torch.utils.data.DataLoader(
        testset, batch_size=config["test_batch_size"], shuffle=False#, num_workers=2
    )

    # Define model
    model = get_model(config["dataset"], device)
    
    # Train and evaluate model
    optimizer = get_optimizer(model, config['pred_module'])
    train_losses = train_model(train_dl, model, optimizer, config['trainer']['max_epochs'])
    test_loss, accuracy = evaluate_model(test_dl, model)
    logger.info(f"test_loss: {test_loss}, accuracy: {accuracy}")

    # Save model
    if not os.path.exists(config['out_dir']):
        logger.warning(f"{config['out_dir']} does not exist. Creating directory.")
        os.makedirs(config["out_dir"])
    torch.save(model.state_dict(), f"{config['out_dir']}/resnet50_{config['dataset']}_acc{accuracy:.02f}.pth")