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
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
from torch.utils.data.sampler import SubsetRandomSampler

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

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def train_model(train_dl, val_dl, model):
    criterion = MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0)

    train_losses = []
    val_losses = []
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
        if (epoch+1)%EVAL_EPOCH_EVERY==0:
            print(f"Train MSE: {loss:.03f}")
            mse = evaluate_model(val_dl, model)
            val_losses.append(mse)
            print('Val MSE: %.3f, RMSE: %.3f' % (mse, np.sqrt(mse)))

    return train_losses, val_losses

def evaluate_model(val_dl, model):
    model.eval()

    with torch.no_grad():
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(val_dl):
            yhat = model(inputs)
            
            yhat = yhat.detach().cpu().numpy()
            actual = targets.cpu().numpy()
            actual = actual.reshape((len(actual), 1))
            
            predictions.append(yhat)
            actuals.append(actual)
        
        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        mse = mean_squared_error(actuals, predictions)
    return mse

if __name__=="__main__":
    logger.debug("Hello world")

    # Check for GPU
    # if torch.cuda.is_available():  
    #     dev = "cuda:0"
    #     logger.debug(f"Found GPU. Using: {dev}")
    # else:  
    #     dev = "cpu"
    dev = "cpu"
    device = torch.device(dev) 

    train_imgs = []
    all_labels = []
    for fn in TRAIN_BATCHES:
        fp = os.path.join(DATAPATH, fn)
        data = unpickle(fp)
        # logger.debug(f"data: {data.keys()}")

        # logger.debug(f"data[b'labels'].shape: {len(data[b'labels'])}")
        imgs = data[b'data']
        # logger.debug(f"img.shape: {imgs.shape}")
        num_imgs = imgs.shape[0]
        reshaped_imgs = np.reshape(imgs,(num_imgs,3,32,32))  # Each CIFAR image is 32x32
        rgb_imgs = np.swapaxes(reshaped_imgs,2,3)
        rgb_imgs = np.swapaxes(rgb_imgs,1,3)    # dim: n x 32 x 32 x 3 (n=num of imgs; 3=num of channels)
        # logger.debug(f"label: {data[b'labels'][0]}")
        # logger.debug(f"rgb_imgs.shape: {rgb_imgs.shape}")
        plt.imshow(rgb_imgs[0])
        plt.savefig("samp.jpg")
        train_imgs.append(rgb_imgs)
        all_labels += data[b'labels']
        # break
    all_imgs = np.concatenate(train_imgs, axis=0)
    logger.debug(f"all_imgs: {all_imgs.shape}")


    model = resnet50()
    model = model.to(device)
    # train_losses, val_losses = train_model(train_dl, val_dl, model)