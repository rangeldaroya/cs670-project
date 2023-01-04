
import skimage
from PIL import Image
import numpy as np
import numpy as np
import skimage
from lime import lime_image

from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from loguru import logger


def get_affine_img(img, ones_mask, is_rot_only, trans_val, rot_val, rot_val_deg, scale_val):
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
        t_ones_mask = skimage.transform.rotate(ones_mask, rot_val_deg)
    
    return t_img, t_ones_mask


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


def get_lime_result(
    i,
    img,
    # lime_img,
    t_type,
    batch_predict,
    idx2label,
    pill_transf,
    target_class,
    config,
    random_seed=0,
    to_save_imgs=False,
    **kwargs,
):
    img_np = img.copy()
    ones_mask = np.ones_like(img_np)   # to be used for computing IoU later
    if t_type == "rrr":
        t_img = img_np.copy()     # red-tinted image
        t_img[:,:,1] = 0
        t_img[:,:,2] = 0
    elif t_type == "bgr":
        t_img = img_np[...,::-1].copy()   # bgr image
    elif (t_type == "orig") or (t_type == "random"):
        t_img = img_np.copy()
    elif t_type == "affine":
        t_img, t_ones_mask = get_affine_img(
            img,
            ones_mask,
            kwargs['is_rot_only'],
            kwargs['trans_val'],
            kwargs['rot_val'],
            kwargs['rot_val_deg'],
            kwargs['scale_val'],
        )
    else:
        raise NotImplementedError

    t_img = Image.fromarray(np.uint8((t_img-np.min(t_img))*255/(np.max(t_img)-np.min(t_img)))).convert('RGB')

    # predict on transformed image
    t_test_pred = batch_predict([pill_transf(t_img)])
    t_pred_idx = t_test_pred.squeeze().argmax()
    t_pred_class = idx2label[t_pred_idx]

    # Explain rotated img
    logger.debug(f"Getting explanation for {t_type} image")
    transformed_explainer = lime_image.LimeImageExplainer()
    if t_type == "random":
        transformed_explanation = transformed_explainer.explain_instance(
            np.array(pill_transf(t_img)), 
            lambda x: batch_predict(x, random=True), # classification function
            top_labels=5, 
            hide_color=0, 
            random_seed=random_seed,
            num_samples=1000) # number of images that will be sent to classification function
    else:
        transformed_explanation = transformed_explainer.explain_instance(
            np.array(pill_transf(t_img)), 
            batch_predict, # classification function
            top_labels=5, 
            hide_color=0, 
            random_seed=random_seed,
            num_samples=1000) # number of images that will be sent to classification function

    # transformed_temp1, transformed_mask1 = transformed_explanation.get_image_and_mask(transformed_explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    # transformed_img_boundry1 = mark_boundaries(transformed_temp1/255.0, transformed_mask1)

    # Shade areas that contribute to top prediction
    transformed_temp2, transformed_mask2 = transformed_explanation.get_image_and_mask(transformed_explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    transformed_img_boundry2 = mark_boundaries(transformed_temp2/255.0, transformed_mask2)
    if to_save_imgs:
        plt.imshow(transformed_img_boundry2)      
        plt.savefig(f"{config['out_dir']}/{config['dataset']}/{i:02d}_t{target_class}_p{t_pred_class}_{t_type}_shade.jpg")
        plt.close()

    if t_type == "affine":
        if not kwargs['is_rot_only']:
            untrans_mask, untrans_ones = get_reverse_affine_masks(transformed_mask2, t_ones_mask, kwargs['rot_val'], kwargs['scale_val'], kwargs['trans_val'])
            pos_iou, neg_iou = get_masked_iou(kwargs['orig_mask'], untrans_mask, untrans_ones)
        else:
            unrot_mask2 = get_reverse_rot_mask(transformed_mask2, kwargs['rot_val_deg'])
            pos_iou, neg_iou = compute_iou(kwargs['orig_mask'], unrot_mask2)
    elif t_type in ["rrr", "bgr", "random"]:
        pos_iou, neg_iou = compute_iou(kwargs['orig_mask'], transformed_mask2)
    elif t_type == "orig":
        return t_pred_class, transformed_mask2
    
    return t_pred_class, pos_iou, neg_iou