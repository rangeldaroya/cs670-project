# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torchvision
from torchvision import transforms
from loguru import logger
import skimage
from typing import List
import math
import pandas as pd

from utils.visualize import visualize_counterfactuals
from data.cub import Cub

TRANSFORM_TYPE = "color-rrr"    # "affine" or "color-bgr", "color-rrr"
NUM_IMGS = 10000
TO_GENERATE_IMGS = False    # set to True to generate image visualizations
idx2label = open("./data/CUB_200_2011/classes.txt", "r").readlines()
idx2label = [x[:-1].split(" ")[1].split(".")[1] for x in idx2label]
cp_path = f"/home/aaronsun_umass_edu/cs670-project/counterfactuals/cub_counterfactuals_{TRANSFORM_TYPE}.npy"

def get_inverse_affine_matrix(
    center: List[float], angle: float, translate: List[float], scale: float, shear: List[float], inverted: bool = True
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation
    # Reference: https://pytorch.org/vision/main/_modules/torchvision/transforms/functional.html

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    if inverted:
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d, -b, 0.0, -c, a, 0.0]
        matrix = [x / scale for x in matrix]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    m = np.array([[matrix[0],matrix[1],matrix[2]],[matrix[3],matrix[4],matrix[5]]])
    m = np.append(m,[[0,0,1]],axis=0)
    return m


def compute_iou(w,h, orig_edits_ul, orig_edits_lr, r_t_edits_ul, r_t_edits_lr):
    mask = np.zeros((w, h))
    r_t_mask = mask.copy()
    orig_mask = mask.copy()

    for ul,lr in zip(orig_edits_ul, orig_edits_lr):
        if ul[1]<0 or ul[0]<0 or lr[1]<0 or lr[0]<0:
            continue
        orig_mask[ul[1]: lr[1]+1, ul[0]: lr[0]+1]=1

    for ul,lr in zip(r_t_edits_ul, r_t_edits_lr):
        ul, lr = ul.astype(int), lr.astype(int)
        if ul[1]<0 or ul[0]<0 or lr[1]<0 or lr[0]<0:
            continue
        r_t_mask[ul[1]: lr[1]+1, ul[0]: lr[0]+1]=1

    uni = (orig_mask + r_t_mask)
    inter = (orig_mask.astype(bool) & r_t_mask.astype(bool)).astype(int)
    iou = np.sum(inter)/np.sum(uni)
    return iou

def main():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    trans = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    
    if TRANSFORM_TYPE == "affine":
        rot_vals_deg = np.loadtxt("/home/aaronsun_umass_edu/cs670-project/counterfactuals/scve_cub_rot_vals_deg.txt")
        trans_vals = np.loadtxt("/home/aaronsun_umass_edu/cs670-project/counterfactuals/scve_cub_trans_vals.txt")
        scales = np.loadtxt("/home/aaronsun_umass_edu/cs670-project/counterfactuals/scve_cub_scales.txt")
        dataset = Cub(
            root='./data', train=False, transform=trans,
            rot_vals_deg=rot_vals_deg, trans_vals=trans_vals, scales=scales,
        )
    elif TRANSFORM_TYPE == "color-bgr":
        dataset = Cub(
            root='./data', train=False, transform=trans,
            rot_vals_deg=None, to_bgr=True, to_rrr=False,
        )
    elif TRANSFORM_TYPE == "color-rrr":
        dataset = Cub(
            root='./data', train=False, transform=trans,
            rot_vals_deg=None, to_bgr=False, to_rrr=True,
        )

    counterfactuals = np.load(
        cp_path, allow_pickle=True
    ).item()

    # Note that these only contain correct predictions 
    # (the processing of semantic cve doesn't include incorrect model predictions)
    cf_keys = list(counterfactuals.keys())
    num_imgs_w_pairs = 0
    n_pix = 7
    width,height = 224,224
    width_cell = width // n_pix
    height_cell = height // n_pix

    scve_results = []
    for ctr, idx in enumerate(cf_keys):
        if (idx+NUM_IMGS) in cf_keys:
            logger.debug(f"idx={idx} has a pair")
            num_imgs_w_pairs += 1

            orig_idx = idx
            t_idx = idx + NUM_IMGS  # idx of transformed img
            orig_cf = counterfactuals[orig_idx]
            t_cf = counterfactuals[t_idx]

            # Get coordinates of edits of the original untransformed image
            orig_edits_ul = []     # coordinates for upper left corner of edit box
            orig_edits_lr = []     # coordinates for lower right corner of edit box
            for o in orig_cf["edits"]:
                cell_index_query = o[0]
                row_index_query = cell_index_query // n_pix
                col_index_query = cell_index_query % n_pix

                x = int(col_index_query * width_cell)
                y = int(row_index_query * height_cell)
                orig_edits_ul.append([x,y])
                orig_edits_lr.append([x+width_cell, y+height_cell])
            orig_edits_ul = np.array(orig_edits_ul)
            orig_edits_lr = np.array(orig_edits_lr)

            # Get coordinates of edits of the transformed image
            t_edits_ul = []     # coordinates for upper left corner of edit box
            t_edits_lr = []     # coordinates for lower right corner of edit box
            for o in t_cf["edits"]:
                cell_index_query = o[0]
                row_index_query = cell_index_query // n_pix
                col_index_query = cell_index_query % n_pix

                x = int(col_index_query * width_cell)
                y = int(row_index_query * height_cell)
                t_edits_ul.append([x,y])
                t_edits_lr.append([x+width_cell, y+height_cell])
            t_edits_ul = np.array(t_edits_ul)
            t_edits_ul = np.append(t_edits_ul, np.ones((len(t_edits_ul),1)), axis=1)
            t_edits_lr = np.array(t_edits_lr)
            t_edits_lr = np.append(t_edits_lr, np.ones((len(t_edits_lr),1)), axis=1)

            if TRANSFORM_TYPE=="affine":
                # Reproject transformed edits based on given rot_val, trans_val, and scale
                rot_val, trans_val, scale = t_cf['rot_vals_deg'],t_cf['trans_vals'],t_cf['scales']
                reverse_trans = get_inverse_affine_matrix(center=(width//2, height//2), angle=rot_val, scale=scale, shear=[0,0], translate=trans_val)
                
                # print(f"reverse_trans: {reverse_trans}, t_edits_ul[0]: {t_edits_ul[0]}")
                r_t_edits_ul = np.array([np.matmul(reverse_trans, np.reshape(t, (-1,1))) for t in t_edits_ul])
                r_t_edits_ul = r_t_edits_ul[:,:2,0]
                r_t_edits_lr = np.array([np.matmul(reverse_trans, np.reshape(t, (-1,1))) for t in t_edits_lr])
                r_t_edits_lr = r_t_edits_lr[:,:2,0]
            else:   # if no affine transformation, no need to do any change
                r_t_edits_lr = t_edits_lr
                r_t_edits_ul = t_edits_ul
            # Compute IoU
            iou = compute_iou(width,height, orig_edits_ul, orig_edits_lr, r_t_edits_ul, r_t_edits_lr)
            logger.debug(f"iou: {iou}")

            # Log results
            label = dataset.__getitem__(orig_cf["query_index"])[1]

            if TRANSFORM_TYPE=="affine":
                scve_results.append([idx, label, rot_val, trans_val, scale, iou])
                df = pd.DataFrame(scve_results, columns=[
                    "test_idx", "label", "rot_val", "trans_val", "scale", "iou"
                ])
            else:
                scve_results.append([idx, label, TRANSFORM_TYPE, iou])
                df = pd.DataFrame(scve_results, columns=[
                    "test_idx", "label", "transform_type", "iou"
                ])
            df.to_csv(f"scve_cub_results_{TRANSFORM_TYPE}.csv", index=False)

            # Make visualizations
            if TO_GENERATE_IMGS:
                visualize_counterfactuals(
                    edits=orig_cf["edits"],
                    query_index=orig_cf["query_index"],
                    distractor_index=orig_cf["distractor_index"],
                    dataset=dataset,
                    n_pix=7,
                    fname=f"output/counterfactuals_cub_demo/example_{idx}_orig.png",
                    idx2label=idx2label,
                )
                visualize_counterfactuals(
                    edits=t_cf["edits"],
                    query_index=t_cf["query_index"],
                    distractor_index=t_cf["distractor_index"],
                    dataset=dataset,
                    n_pix=7,
                    fname=f"output/counterfactuals_cub_demo/example_{idx}_{TRANSFORM_TYPE}.png",
                    idx2label=idx2label,
                )
    
    if TRANSFORM_TYPE=="affine":
        df = pd.DataFrame(scve_results, columns=[
            "test_idx", "label", "rot_val", "trans_val", "scale", "iou"
        ])
    else:
        df = pd.DataFrame(scve_results, columns=[
            "test_idx", "label", "transform_type", "iou"
        ])
    df.to_csv(f"scve_cub_results_{TRANSFORM_TYPE}.csv", index=False)
    print(f"Found {num_imgs_w_pairs} images with transformed pairs")

if __name__ == "__main__":
    main()
