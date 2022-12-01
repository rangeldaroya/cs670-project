import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np

import torch
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

from lime import lime_image
from skimage.segmentation import mark_boundaries
import skimage

RANDOM_SEED = 0     # set this to have consistent results over runs
MODEL_PATH = "/home/rdaroya/Documents/cs670-project/models/resnet50_cifar10_acc0.82.pth"
idx2label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
device = "cpu"

NUM_SAMPLES = 15    # number of samples to generate

np.random.seed(RANDOM_SEED)

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
    trans = transforms.Compose([transforms.ToTensor(),normalize])


    return trans  

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def get_reverse_affine_masks(rotated_mask, rotated_ones_mask, rot_val, scale_val, trans_val):
    # Reverse affine transform
    r_tform = skimage.transform.AffineTransform(
        translation=-1*trans_val,
        rotation=-1*rot_val,
        scale=1/scale_val,
    )
    r_t_mask = skimage.transform.warp(
        rotated_mask,
        r_tform.inverse,
        mode="constant",
        cval=0,
        preserve_range=True,
    )
    r_t_ones_mask = skimage.transform.warp(
        rotated_ones_mask,
        r_tform.inverse,
        mode="constant",
        cval=0,
        preserve_range=True,
    )
    return r_t_mask, r_t_ones_mask

def get_masked_iou(mask, reversed_mask, untrans_ones):
    ones_mask = (untrans_ones>0).astype(int)[:,:,0]
    
    pos_mask = (mask>0).astype(int)*ones_mask
    pos_reversed_mask = (reversed_mask>0).astype(int)*ones_mask
    union_pos = pos_mask + pos_reversed_mask

    neg_mask = (mask<0).astype(int)*ones_mask
    neg_reversed_mask = (reversed_mask<0).astype(int)*ones_mask
    union_neg = neg_mask + neg_reversed_mask

    pos_iou = np.sum(pos_mask&pos_reversed_mask)/(np.sum(union_pos>0) or 1e10)  # if sum is zero, make iou approach 0
    neg_iou = (np.sum(neg_mask&neg_reversed_mask)/(np.sum(union_neg>0) or 1e10))

    return pos_iou, neg_iou

if __name__=="__main__":

    pill_transf = get_pil_transform()
    preprocess_transform = get_preprocess_transform()

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(2048, 256), 
        nn.ReLU(), 
        nn.Linear(256, 10)  # 10 CIFAR classes
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    trans = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize])


    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=trans)
    test_dl = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False#, num_workers=2
    )

    model.eval()
    pos_ious, neg_ious = [], []
    for i, (inputs, targets) in enumerate(test_dl):
        rot_val_deg = np.random.randint(low = -80, high=80) # generate random rotation values
        # print(f"rot_val: {rot_val}")
        rot_val = rot_val_deg*np.pi/180.0
        trans_val = 0
        scale_val = 1

        img = inputs.cpu().detach().numpy()
        img = img[0,:,:,:]
        img = img.swapaxes(0,1)
        img = img.swapaxes(1,2)
        img_np = img.copy()
        ones_mask = np.ones_like(img)   # to be used for computing IoU later
        tform = skimage.transform.AffineTransform(translation=trans_val, rotation=rot_val, scale=scale_val)
        t_img = skimage.transform.warp(
            img,
            tform.inverse,
            mode="constant",
            cval=0,
            preserve_range=True,
        )
        t_ones_mask = skimage.transform.warp(
            ones_mask,
            tform.inverse,
            mode="constant",
            cval=0,
            preserve_range=True,
        )
        img = Image.fromarray(np.uint8((img-np.min(img))*255/(np.max(img)-np.min(img)))).convert('RGB')
        t_img = Image.fromarray(np.uint8((t_img-np.min(t_img))*255/(np.max(t_img)-np.min(t_img)))).convert('RGB')

        # predict on a single image
        test_pred = batch_predict([pill_transf(img)])
        pred_idx = test_pred.squeeze().argmax()
        pred_class = idx2label[pred_idx]
        target_class = idx2label[targets]

        print("Getting explanation for original image")
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np.array(pill_transf(img)), 
            batch_predict, # classification function
            top_labels=5,
            hide_color=0,
            random_seed=RANDOM_SEED,
            num_samples=1000) # number of images that will be sent to classification function

        temp1, mask1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        img_boundry1 = mark_boundaries(temp1/255.0, mask1)

        # Shade areas that contribute to top prediction
        temp2, mask2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        img_boundry2 = mark_boundaries(temp2/255.0, mask2)
        plt.imshow(img_boundry2)      
        plt.savefig(f"../outputs/lime/cifar/{i:02d}_t{target_class}_p{pred_class}_shade.jpg")
        plt.close()


        # Explain rotated img
        print(f"Getting explanation for rotated image, rot_val: {rot_val}, rot_val_deg: {rot_val_deg}")
        transformed_explainer = lime_image.LimeImageExplainer()
        transformed_explanation = transformed_explainer.explain_instance(
            np.array(pill_transf(t_img)), 
            batch_predict, # classification function
            top_labels=5, 
            hide_color=0, 
            random_seed=RANDOM_SEED,
            num_samples=1000) # number of images that will be sent to classification function

        transformed_temp1, transformed_mask1 = transformed_explanation.get_image_and_mask(transformed_explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        transformed_img_boundry1 = mark_boundaries(transformed_temp1/255.0, transformed_mask1)

        # Shade areas that contribute to top prediction
        transformed_temp2, transformed_mask2 = transformed_explanation.get_image_and_mask(transformed_explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        transformed_img_boundry2 = mark_boundaries(transformed_temp2/255.0, transformed_mask2)
        plt.imshow(transformed_img_boundry2)      
        plt.savefig(f"../outputs/lime/cifar/{i:02d}_t{target_class}_p{pred_class}_t_shade.jpg")
        plt.close()

        # Compute iou between rotated images
        # unrot_mask2 = get_reverse_rot_mask(transformed_mask2, rot_val)
        # pos_iou, neg_iou = compute_iou(mask2, unrot_mask2)
        untrans_mask, untrans_ones = get_reverse_affine_masks(transformed_mask2, t_ones_mask, rot_val, scale_val, trans_val)
        pos_iou, neg_iou = get_masked_iou(mask2, untrans_mask, untrans_ones)
        pos_ious.append(pos_iou)
        neg_ious.append(neg_iou)
        print(f"pos_iou: {pos_iou}, neg_iou: {neg_iou}")

        print(f"Done marking img {i+1:02d}/{len(test_dl)}")
        if (i+1) == NUM_SAMPLES:
            break
    print(f"pos_ious: {pos_ious}")
    print(f"neg_ious: {neg_ious}")