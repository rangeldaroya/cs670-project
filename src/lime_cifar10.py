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
import pandas as pd

RANDOM_SEED = 0     # set this to have consistent results over runs
MODEL_PATH = "/home/rdaroya_umass_edu/Documents/cs670-project/models/resnet50_cifar10_acc0.82.pth"
idx2label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
device = "cpu"
TO_APPEND_RESULTS = False   # set to True when there are previous results

NUM_SAMPLES = 100    # number of samples to generate

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
    model.fc = nn.Sequential(
        nn.Linear(2048, 256), 
        nn.ReLU(), 
        nn.Linear(256, 10)  # 10 CIFAR classes
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)

    random_model = resnet50()
    model.fc = nn.Sequential(
        nn.Linear(2048, 256), 
        nn.ReLU(), 
        nn.Linear(256, 10)  # 10 CIFAR classes
    )
    random_model = random_model.to(device)

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
    num_imgs = len(test_dl)
    rot_vals_deg, trans_vals, scales = generate_affine_vals(num_imgs)
    np.savetxt("lime_cifar_rot_vals_deg.txt", rot_vals_deg)
    np.savetxt("lime_cifar_trans_vals.txt", trans_vals)
    np.savetxt("lime_cifar_scales.txt", scales)

    model.eval()
    random_model.eval()
    results = []
    # Load previous results to append to
    if TO_APPEND_RESULTS:
        prev_results = pd.read_csv("lime_cifar_results.csv")
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
        img = Image.fromarray(np.uint8((img-np.min(img))*255/(np.max(img)-np.min(img)))).convert('RGB')
        t_img = Image.fromarray(np.uint8((t_img-np.min(t_img))*255/(np.max(t_img)-np.min(t_img)))).convert('RGB')
        rrr_img = Image.fromarray(np.uint8((rrr_img-np.min(rrr_img))*255/(np.max(rrr_img)-np.min(rrr_img)))).convert('RGB')
        bgr_img = Image.fromarray(np.uint8((bgr_img-np.min(bgr_img))*255/(np.max(bgr_img)-np.min(bgr_img)))).convert('RGB')
        
        # predict on a single image
        test_pred = batch_predict([pill_transf(img)])
        pred_idx = test_pred.squeeze().argmax()
        pred_class = idx2label[pred_idx]
        target_class = idx2label[targets]

        # predict on transformed image
        t_test_pred = batch_predict([pill_transf(t_img)])
        t_pred_idx = t_test_pred.squeeze().argmax()
        t_pred_class = idx2label[t_pred_idx]

        # predict with a random model
        random_test_pred = batch_predict([pill_transf(img)])
        random_pred_idx = random_test_pred.squeeze().argmax()
        random_pred_class = idx2label[random_pred_idx]

        # predict on red-tinted image
        rrr_test_pred = batch_predict([pill_transf(rrr_img)])
        rrr_pred_idx = rrr_test_pred.squeeze().argmax()
        rrr_pred_class = idx2label[rrr_pred_idx]

        # predict on bgr image
        bgr_test_pred = batch_predict([pill_transf(bgr_img)])
        bgr_pred_idx = bgr_test_pred.squeeze().argmax()
        bgr_pred_class = idx2label[bgr_pred_idx]

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
        plt.savefig(f"../outputs/lime/cifar/{i:02d}_t{target_class}_p{t_pred_class}_t_shade.jpg")
        plt.close()

        # Explain random model
        print(f"Getting explanation for random model")
        random_explainer = lime_image.LimeImageExplainer()
        random_explanation = random_explainer.explain_instance(
            np.array(pill_transf(img)), 
            lambda x: batch_predict(x, random=True), # classification function
            top_labels=5, 
            hide_color=0, 
            random_seed=RANDOM_SEED,
            num_samples=1000) # number of images that will be sent to classification function

        random_temp1, random_mask1 = random_explanation.get_image_and_mask(random_explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        random_img_boundry1 = mark_boundaries(random_temp1/255.0, random_mask1)

        # Shade areas that contribute to top prediction
        random_temp2, random_mask2 = random_explanation.get_image_and_mask(random_explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        random_img_boundry2 = mark_boundaries(random_temp2/255.0, random_mask2)
        plt.imshow(random_img_boundry2)      
        plt.savefig(f"./outputs/lime/cifar/{i:02d}_t{target_class}_p{random_pred_class}_random_shade.jpg")
        plt.close()

        # Explain red-tinted image
        print(f"Getting explanation for red-tinted image")
        rrr_explainer = lime_image.LimeImageExplainer()
        rrr_explanation = rrr_explainer.explain_instance(
            np.array(pill_transf(rrr_img)), 
            batch_predict, # classification function
            top_labels=5, 
            hide_color=0, 
            random_seed=RANDOM_SEED,
            num_samples=1000) # number of images that will be sent to classification function

        rrr_temp1, rrr_mask1 = rrr_explanation.get_image_and_mask(rrr_explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        rrr_img_boundry1 = mark_boundaries(rrr_temp1/255.0, rrr_mask1)

        # Shade areas that contribute to top prediction
        rrr_temp2, rrr_mask2 = rrr_explanation.get_image_and_mask(rrr_explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        rrr_img_boundry2 = mark_boundaries(rrr_temp2/255.0, rrr_mask2)
        plt.imshow(rrr_img_boundry2)      
        plt.savefig(f"../outputs/lime/cifar/{i:02d}_t{target_class}_p{rrr_pred_class}_rrr_shade.jpg")
        plt.close()


        # Explain bgr image
        print(f"Getting explanation for bgr image")
        bgr_explainer = lime_image.LimeImageExplainer()
        bgr_explanation = bgr_explainer.explain_instance(
            np.array(pill_transf(bgr_img)), 
            batch_predict, # classification function
            top_labels=5, 
            hide_color=0, 
            random_seed=RANDOM_SEED,
            num_samples=1000) # number of images that will be sent to classification function

        bgr_temp1, bgr_mask1 = bgr_explanation.get_image_and_mask(bgr_explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        bgr_img_boundry1 = mark_boundaries(bgr_temp1/255.0, bgr_mask1)

        # Shade areas that contribute to top prediction
        bgr_temp2, bgr_mask2 = bgr_explanation.get_image_and_mask(bgr_explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        bgr_img_boundry2 = mark_boundaries(bgr_temp2/255.0, bgr_mask2)
        plt.imshow(bgr_img_boundry2)      
        plt.savefig(f"../outputs/lime/cifar/{i:02d}_t{target_class}_p{bgr_pred_class}_bgr_shade.jpg")
        plt.close()



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
        df.to_csv("lime_cifar_results.csv", index=False)
    # print(f"pos_ious: {pos_ious}")
    # print(f"neg_ious: {neg_ious}")
    df = pd.DataFrame(results, columns=[
        "test_idx", "target_class", "pred_class", "t_pred_class", "random_pred_class", "rrr_pred_class", "bgr_pred_class", "is_rot_only", "rot_val_deg", "trans_val", "scale_val", "pos_iou", "neg_iou", "random_pos_iou", "random_neg_iou", "rrr_pos_iou", "rrr_neg_iou", "bgr_pos_iou", "bgr_neg_iou"
    ])
    df.to_csv("lime_cifar_results.csv", index=False)