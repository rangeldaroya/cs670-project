# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def visualize_counterfactuals(
    edits,
    query_index,
    distractor_index,
    dataset,
    n_pix,
    fname=None,
    idx2label=None,
):
    # load image
    query_img = dataset.__getitem__(query_index)[0]
    query_img_label = dataset.__getitem__(query_index)[1]
    print(f"\nQuery label: {idx2label[query_img_label]}, [{query_img_label}]")

    query_img = np.swapaxes(query_img, 0,1)
    query_img = np.swapaxes(query_img, 1,2)
    query_img = query_img.detach().cpu().numpy()
    query_img = (query_img-np.min(query_img,axis=0))/(np.max(query_img,axis=0)-np.min(query_img,axis=0))
    height, width = query_img.shape[0], query_img.shape[1]

    # geometric properties of cells
    width_cell = width // n_pix
    height_cell = height // n_pix

    # create plot
    n_edits = len(edits)
    print(f"n_edits: {n_edits}")
    _, axes = plt.subplots(n_edits, 2)

    # loop over edits
    for ii, edit in enumerate(edits):
        # show query
        cell_index_query = edit[0]
        row_index_query = cell_index_query // n_pix
        col_index_query = cell_index_query % n_pix

        query_left_box = int(col_index_query * width_cell)
        query_top_box = int(row_index_query * height_cell)

        rect = patches.Rectangle(
            (query_left_box, query_top_box),
            width_cell,
            height_cell,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        if n_edits>1:
            axes[ii,0].imshow(query_img)
            axes[ii,0].add_patch(rect)
            axes[ii,0].get_xaxis().set_ticks([])
            axes[ii,0].get_yaxis().set_ticks([])
            if ii == 0:
                axes[ii,0].set_title(f"Query: {idx2label[query_img_label]}")
        else:
            axes[0].imshow(query_img)
            axes[0].add_patch(rect)
            axes[0].get_xaxis().set_ticks([])
            axes[0].get_yaxis().set_ticks([])
            if ii == 0:
                axes[0].set_title(f"Query: {idx2label[query_img_label]}")

        # show distractor
        cell_index_distractor = edit[1]

        index_distractor = distractor_index[cell_index_distractor // (n_pix**2)]
        img_distractor = dataset.__getitem__(index_distractor)[0]
        img_distractor_label = dataset.__getitem__(index_distractor)[1]

        cell_index_distractor = cell_index_distractor % (n_pix**2)
        row_index_distractor = cell_index_distractor // n_pix
        col_index_distractor = cell_index_distractor % n_pix

        distractor_left_box = int(col_index_distractor * width_cell)
        distractor_top_box = int(row_index_distractor * height_cell)

        rect = patches.Rectangle(
            (distractor_left_box, distractor_top_box),
            width_cell,
            height_cell,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        img_distractor = np.swapaxes(img_distractor, 0,1)
        img_distractor = np.swapaxes(img_distractor, 1,2)
        img_distractor = img_distractor.detach().cpu().numpy()
        img_distractor = (img_distractor-np.min(img_distractor))/(np.max(img_distractor)-np.min(img_distractor))
        if n_edits>1:
            axes[ii,1].imshow(img_distractor)
            axes[ii,1].add_patch(rect)
            axes[ii,1].get_xaxis().set_ticks([])
            axes[ii,1].get_yaxis().set_ticks([])
            if ii == 0:
                axes[ii,1].set_title(f"Distractor: {idx2label[img_distractor_label]}")
        else:
            axes[1].imshow(img_distractor)
            axes[1].add_patch(rect)
            axes[1].get_xaxis().set_ticks([])
            axes[1].get_yaxis().set_ticks([])
            if ii == 0:
                axes[1].set_title(f"Distractor: {idx2label[img_distractor_label]}")
        print(f"Distractor: {idx2label[img_distractor_label]}, [{img_distractor_label}]")

    # save or view
    plt.tight_layout()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
