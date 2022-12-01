import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from cub import Cub

RANDOM_SEED = 0     # set this to have consistent results over runs
MODEL_PATH = "./models/resnet50_cub_acc0.83.pth"
idx2label = open("./data/CUB_200_2011/classes.txt", "r").readlines()
idx2label = [x[:-1].split(" ")[1].split(".")[1] for x in idx2label]
device = "cpu"

NUM_SAMPLES = 15    # number of samples to generate

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    trans = transforms.Compose([
    transforms.Resize(224), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    normalize
    ])

    return trans  

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


if __name__=="__main__":

    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_feats, 200) # 200 CUB categories
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)

    testset = Cub(train=False, transform=preprocess_transform, return_image_only=False)
    test_dl = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False#, num_workers=2
    )

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(device=="cuda"))

    for i, (inputs, targets) in enumerate(test_dl):
        # print(inputs, inputs.shape)
        # print(targets)
        model.eval()
        logits = model(inputs)
        # print(f"logits: {logits}")
        probs = F.softmax(logits, dim=1)
        probs5 = probs.topk(5)
        # print(tuple((p,c, idx2label[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy())))
        
        img = inputs.cpu().detach().numpy()
        img = img[0,:,:,:]
        img = img.swapaxes(0,1)
        img = img.swapaxes(1,2)
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        pil_img = Image.fromarray(np.uint8(img * 255)).convert('RGB')

        # predict on a single image
        test_pred = batch_predict([pill_transf(pil_img)])
        pred_idx = test_pred.squeeze().argmax()
        pred_class = idx2label[pred_idx]
        target_class = idx2label[targets]

        grayscale_cam = cam(input_tensor=inputs, targets=[ClassifierOutputTarget(targets[0])])
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        plt.imshow(visualization)      
        plt.savefig(f"./outputs/gradcam/cub/{i:02d}_t{target_class}_p{pred_class}_gradcam.jpg")
        plt.close()
        
        # break
        print(f"Done marking img {i+1:02d}/{len(test_dl)}")
