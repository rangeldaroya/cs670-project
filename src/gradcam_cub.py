import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np

import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import skimage
import pandas as pd

from cub import Cub

RANDOM_SEED = 0     # set this to have consistent results over runs
MODEL_PATH = "./models/resnet50_cub_acc0.83.pth"
idx2label = open("./data/CUB_200_2011/classes.txt", "r").readlines()
idx2label = [x[:-1].split(" ")[1].split(".")[1] for x in idx2label]
device = "cuda"
TO_APPEND_RESULTS = False   # set to True when there are previous results
NUM_SAMPLES = 5794    # number of samples to generate

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
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return trans  

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

def get_reverse_rot_mask(rotated_mask, rot_val):

    unrot_mask_pos = (rotated_mask>0)
    unrot_mask_pos = skimage.transform.rotate(unrot_mask_pos, -rot_val, preserve_range=True, resize=False, clip=False, mode='constant', cval=0)
    unrot_mask_neg = (rotated_mask<0)
    unrot_mask_neg = skimage.transform.rotate(unrot_mask_neg, -rot_val, preserve_range=True, resize=False, clip=False, mode='constant', cval=0)

    unrot_mask = unrot_mask_pos.astype(int) + unrot_mask_neg.astype(int)*(-1)

    return unrot_mask

def compute_iou(mask, reversed_mask):
    pos_mask = (mask>0).astype(int)
    pos_reversed_mask = (reversed_mask>0).astype(int)
    union_pos = pos_mask + pos_reversed_mask

    neg_mask = (mask<0).astype(int)
    neg_reversed_mask = (reversed_mask<0).astype(int)
    union_neg = neg_mask + neg_reversed_mask

    pos_iou = np.sum(pos_mask&pos_reversed_mask)/(np.sum(union_pos>0) or 1e10)  # if sum is zero, make iou approach 0
    neg_iou = (np.sum(neg_mask&neg_reversed_mask)/(np.sum(union_neg>0) or 1e10))

    return pos_iou, neg_iou


def generate_affine_vals(num_imgs, im_size=224):
    rot_vals_deg = np.random.randint(low=-80, high=80, size=num_imgs)

    t_1 = np.zeros((num_imgs//2,2))    # half don't have translation
    t_2 = np.random.randint(low=-0.1*im_size, high=0.1*im_size, size=(num_imgs-(num_imgs//2),2))
    trans_vals = np.concatenate((t_1,t_2))

    s_1 = np.ones(num_imgs//2)          # half don't have scaling
    s_2 = np.random.rand(num_imgs-(num_imgs//2))+1  # generates n-dim array in range of [1,2)
    scales = np.concatenate((s_1,s_2))

    return rot_vals_deg, trans_vals, scales

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

    random_model = resnet50()
    random_model.fc = nn.Sequential(
        nn.Linear(num_feats, 200) # 200 CUB categories
    )
    random_model = random_model.to(device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    trans = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize])


    testset = Cub(
        root='./data', train=False, transform=trans)
    test_dl = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False#, num_workers=2
    )
    num_imgs = len(test_dl)
    rot_vals_deg, trans_vals, scales = generate_affine_vals(num_imgs)
    np.savetxt("grad_cub_rot_vals_deg.txt", rot_vals_deg)
    np.savetxt("grad_cub_trans_vals.txt", trans_vals)
    np.savetxt("grad_cub_scales.txt", scales)

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(device=="cuda"))

    random_target_layers = [random_model.layer4[-1]]
    random_cam = GradCAM(model=random_model, target_layers=random_target_layers, use_cuda=(device=="cuda"))

    model.eval()
    random_model.eval()
    results = []
    # Load previous results to append to
    if TO_APPEND_RESULTS:
        prev_results = pd.read_csv("grad_cub_results.csv")
        results = prev_results.values.tolist()
    pos_ious, neg_ious = [], []

    for i, (inputs, targets) in enumerate(test_dl):
        is_rot_only = False
        rot_val_deg = rot_vals_deg[i]
        rot_val = rot_val_deg*np.pi/180.0
        trans_val = trans_vals[i]
        scale_val = scales[i]
        if (all(trans_val == np.array([0,0]))) and (scale_val==1) and (rot_val!=0):
            is_rot_only = True

        img = inputs.cpu().detach().numpy()
        img = img[0,:,:,:]
        img = img.swapaxes(0,1)
        img = img.swapaxes(1,2)
        img_np = img.copy()
        rrr_img = img_np.copy()     # red-tinted image
        rrr_img[:,:,1] = 0
        rrr_img[:,:,2] = 0
        bgr_img = img_np[...,::-1].copy()   # bgr image
        ones_mask = np.ones_like(img)   # to be used for computing IoU later

        if not is_rot_only:
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
        else:
            t_img = skimage.transform.rotate(img, rot_val_deg)
    
        for im in [img, t_img, rrr_img, bgr_img]:
            im -= np.min(im)
            im /= np.max(im)

        pil_img = Image.fromarray(np.uint8(img * 255)).convert('RGB')
        pil_t_img = Image.fromarray(np.uint8(t_img * 255)).convert('RGB')
        pil_rrr_img = Image.fromarray(np.uint8(rrr_img * 255)).convert('RGB')
        pil_bgr_img = Image.fromarray(np.uint8(bgr_img * 255)).convert('RGB')
        
        # predict on a single image
        test_pred = batch_predict([pill_transf(pil_img)])
        pred_idx = test_pred.squeeze().argmax()
        pred_class = idx2label[pred_idx]
        target_class = idx2label[targets]

        # predict on transformed image
        t_test_pred = batch_predict([pill_transf(pil_t_img)])
        t_pred_idx = t_test_pred.squeeze().argmax()
        t_pred_class = idx2label[t_pred_idx]

        # predict with a random model
        random_test_pred = batch_predict([pill_transf(pil_img)], True)
        random_pred_idx = random_test_pred.squeeze().argmax()
        random_pred_class = idx2label[random_pred_idx]

        # predict on red-tinted image
        rrr_test_pred = batch_predict([pill_transf(pil_rrr_img)])
        rrr_pred_idx = rrr_test_pred.squeeze().argmax()
        rrr_pred_class = idx2label[rrr_pred_idx]

        # predict on bgr image
        bgr_test_pred = batch_predict([pill_transf(pil_bgr_img)])
        bgr_pred_idx = bgr_test_pred.squeeze().argmax()
        bgr_pred_class = idx2label[bgr_pred_idx]

        # CAM on original image
        grayscale_cam = cam(input_tensor=preprocess_transform(pill_transf(pil_img)).unsqueeze(0), targets=[ClassifierOutputTarget(targets[0])])
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        plt.imshow(visualization)      
        plt.savefig(f"./outputs/gradcam/cub/{i:02d}_t{target_class}_p{pred_class}_gradcam.jpg")
        plt.close()
        mask2 = grayscale_cam > np.mean(grayscale_cam)
        
        # CAM on translated image
        transformed_grayscale_cam = cam(input_tensor=preprocess_transform(pill_transf(pil_t_img)).unsqueeze(0), targets=[ClassifierOutputTarget(targets[0])])
        transformed_grayscale_cam = transformed_grayscale_cam[0, :]
        transformed_visualization = show_cam_on_image(t_img, transformed_grayscale_cam, use_rgb=True)
        plt.imshow(transformed_visualization)
        plt.savefig(f"./outputs/gradcam/cub/{i:02d}_t{target_class}_p{t_pred_class}_gradcam_t.jpg")
        plt.close()
        transformed_mask2 = transformed_grayscale_cam > np.mean(transformed_grayscale_cam)

        # CAM with random model
        random_grayscale_cam = random_cam(input_tensor=preprocess_transform(pill_transf(pil_img)).unsqueeze(0), targets=[ClassifierOutputTarget(targets[0])])
        random_grayscale_cam = random_grayscale_cam[0, :]
        random_visualization = show_cam_on_image(img, random_grayscale_cam, use_rgb=True)
        plt.imshow(random_visualization)
        plt.savefig(f"./outputs/gradcam/cub/{i:02d}_t{target_class}_p{random_pred_class}_gradcam_random.jpg")
        plt.close()
        random_mask2 = random_grayscale_cam > np.mean(random_grayscale_cam)

        # CAM on red tinted image
        rrr_grayscale_cam = cam(input_tensor=preprocess_transform(pill_transf(pil_rrr_img)).unsqueeze(0), targets=[ClassifierOutputTarget(targets[0])])
        rrr_grayscale_cam = rrr_grayscale_cam[0, :]
        rrr_visualization = show_cam_on_image(rrr_img, rrr_grayscale_cam, use_rgb=True)
        plt.imshow(rrr_visualization)
        plt.savefig(f"./outputs/gradcam/cub/{i:02d}_t{target_class}_p{rrr_pred_class}_gradcam_rrr.jpg")
        plt.close()
        rrr_mask2 = rrr_grayscale_cam > np.mean(rrr_grayscale_cam)

        # CAM on bgr tinted image
        bgr_grayscale_cam = cam(input_tensor=preprocess_transform(pill_transf(pil_bgr_img)).unsqueeze(0), targets=[ClassifierOutputTarget(targets[0])])
        bgr_grayscale_cam = bgr_grayscale_cam[0, :]
        bgr_visualization = show_cam_on_image(bgr_img, bgr_grayscale_cam, use_rgb=True)
        plt.imshow(bgr_visualization)
        plt.savefig(f"./outputs/gradcam/cub/{i:02d}_t{target_class}_p{bgr_pred_class}_gradcam_bgr.jpg")
        plt.close()
        bgr_mask2 = bgr_grayscale_cam > np.mean(bgr_grayscale_cam)

        # Compute iou between rotated images
        if not is_rot_only:
            untrans_mask, untrans_ones = get_reverse_affine_masks(transformed_mask2, t_ones_mask, rot_val, scale_val, trans_val)
            pos_iou, neg_iou = get_masked_iou(mask2, untrans_mask, untrans_ones)
        else:
            unrot_mask2 = get_reverse_rot_mask(transformed_mask2, rot_val_deg)
            pos_iou, neg_iou = compute_iou(mask2, unrot_mask2)
        # pos_ious.append(pos_iou)
        # neg_ious.append(neg_iou)
        print(f"t_pos_iou: {pos_iou}, t_neg_iou: {neg_iou}")

        #Compute iou of random model
        random_pos_iou, random_neg_iou = compute_iou(mask2, random_mask2)

        #Compute iou of tinted and bgr images
        rrr_pos_iou, rrr_neg_iou = compute_iou(mask2, rrr_mask2)
        bgr_pos_iou, bgr_neg_iou = compute_iou(mask2, bgr_mask2)

        # Log results
        results.append([i, target_class, pred_class, t_pred_class, random_pred_class, rrr_pred_class, bgr_pred_class, is_rot_only, rot_val_deg, trans_val, scale_val, pos_iou, neg_iou, random_pos_iou, random_neg_iou, rrr_pos_iou, rrr_neg_iou, bgr_pos_iou, bgr_neg_iou])
        print(f"Done marking img {i+1:02d}/{len(test_dl)}")
        
        if (i+1) == NUM_SAMPLES:
            break
        df = pd.DataFrame(results, columns=[
            "test_idx", "target_class", "pred_class", "t_pred_class", "random_pred_class", "rrr_pred_class", "bgr_pred_class", "is_rot_only", "rot_val_deg", "trans_val", "scale_val", "pos_iou", "neg_iou", "random_pos_iou", "random_neg_iou", "rrr_pos_iou", "rrr_neg_iou", "bgr_pos_iou", "bgr_neg_iou"
        ])
        df.to_csv("grad_cub_results.csv", index=False)
    # print(f"pos_ious: {pos_ious}")
    # print(f"neg_ious: {neg_ious}")
    df = pd.DataFrame(results, columns=[
        "test_idx", "target_class", "pred_class", "t_pred_class", "random_pred_class", "rrr_pred_class", "bgr_pred_class", "is_rot_only", "rot_val_deg", "trans_val", "scale_val", "pos_iou", "neg_iou", "random_pos_iou", "random_neg_iou", "rrr_pos_iou", "rrr_neg_iou", "bgr_pos_iou", "bgr_neg_iou"
    ])
    df.to_csv("grad_cub_results.csv", index=False)
