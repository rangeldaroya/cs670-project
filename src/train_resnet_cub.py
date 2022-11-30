# from models.resnet import resnet50
import torch
from torchvision.models import resnet50, ResNet50_Weights
from loguru import logger
import numpy as np


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

from cub import Cub

NUM_EPOCHS = 100
EVAL_EPOCH_EVERY = 10
# DATAPATH = "cifar-10-python"
# TRAIN_BATCHES = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
# TEST_BATCHES = ["test_batch"]
# NUM_TO_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
trans = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),normalize])

trainset = Cub(train=True, transform=trans, return_image_only=True)
train_dl = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True#, num_workers=2
    )

testset = Cub(train=False, transform=trans, return_image_only=True)
test_dl = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False#, num_workers=2
    )



def train_model(train_dl, model):
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0)

    train_losses = []
    # val_losses = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        # print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS}")
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
    logger.debug("Hello world")

    # Check for GPU
    if torch.cuda.is_available():  
        dev = "cuda:0"
        logger.debug(f"Found GPU. Using: {dev}")
    else:  
        dev = "cpu"
    # dev = "cpu"
    device = torch.device(dev)

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = model.to(device)

    # Freeze the layers
    for param in model.parameters():
        param.requires_grad = False

    # Change the last layer to cifar10 number of output classes, and then just finetune these layers
    num_feats = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_feats, 1024),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 200),   # 200 CUB categories
    )
    model = model.to(device)

    # model = ResNet50(img_channel=3, num_classes=10).to(device)
    # model = resnet50(num_classes=10)
    # model = model.to(device)
    
    train_losses = train_model(train_dl, model)

    test_loss, accuracy = evaluate_model(test_dl, model)
    logger.info(f"test_loss: {test_loss}, accuracy: {accuracy}")

    torch.save(model.state_dict(), f"resnet50_cub_acc{accuracy:.02f}.pth")