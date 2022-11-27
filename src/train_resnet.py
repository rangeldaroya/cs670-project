import torch
from torchvision.models import resnet50
from loguru import logger
import numpy as np

# from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid, Tanh
from torch.nn import Module
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import pickle
import os
import matplotlib.pyplot as plt

NUM_EPOCHS = 100
EVAL_EPOCH_EVERY = 10
DATAPATH = "cifar-10-python"
TRAIN_BATCHES = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
TEST_BATCHES = ["test_batch"]
NUM_TO_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]



transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # means and stddevs of RGB channels
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # means and stddevs of RGB channels
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_dl = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_dl = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)



def train_model(train_dl, model):
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0)

    train_losses = []
    # val_losses = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        # print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS}")
        model.train()
        train_losses_epoch = 0
        for i, (inputs, targets) in enumerate(train_dl):
            targets = targets.to(device)
            
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
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(val_dl):
            yhat = model(inputs)
            
            yhat = yhat.detach().cpu().numpy()
            actual = targets.cpu().numpy()
            actual = actual.reshape((len(actual), 1))
            
            loss = criterion(yhat, targets)

            test_loss += loss.item()
            _, predicted = yhat.max(1)
            num_total += targets.size(0)
            num_correct += predicted.eq(targets).sum().item()

        
        # predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        # mse = mean_squared_error(actuals, predictions)
    accuracy = num_correct/num_total
    return test_loss, accuracy

if __name__=="__main__":
    logger.debug("Hello world")

    # Check for GPU
    if torch.cuda.is_available():  
        dev = "cuda:0"
        logger.debug(f"Found GPU. Using: {dev}")
    else:  
        dev = "cpu"
    # dev = "cpu"
    device = torch.device(dev) 

    model = resnet50()
    model = model.to(device)
    
    train_losses, val_losses = train_model(train_dl, model)