# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchmetrics

from sklearn.metrics import confusion_matrix


def get_query_distractor_pairs(
    dataset,
    confusion_matrix,
    max_num_distractors,
    get_dataset_targets,
    seed=0,
):
    """
    Compute query-distractor image pairs using a confusion matrix:
        (1) For each query class, we select the most confusing
        class as it's distractor class.
        (2) Next, we randomly sample `max_num_distractors`
        from the distractor class as the distractor images.

    We return the result as a a dictionary.
    """
    np.random.seed(seed)
    result = {}

    # process all images
    for query_index in range(len(dataset)):
        # determine the distractor class for this sample
        # query_class = dataset.__getitem__(query_index)["target"]
        query_class = dataset.__getitem__(query_index)[1]
        row = np.copy(confusion_matrix[query_class])
        row[query_class] = -1
        if np.all(row <= 0):  # no signal from confusion matrix - skip
            continue
        distractor_class = np.argmax(row)

        # gater all images belonging to distractor class
        # distractor_index = dataset.get_target(distractor_class)
        distractor_index = get_dataset_targets(dataset, distractor_class)

        # randomly select `max_num_distractors`
        num_random = min(len(distractor_index), max_num_distractors)
        distractor_index = np.random.choice(distractor_index, num_random, replace=False)
        distractor_index = distractor_index.reshape(-1).tolist()

        # add to dictionary
        result[query_index] = {
            "distractor_index": distractor_index,
            "distractor_class": distractor_class,
            "query_index": query_index,
            "query_class": query_class,
        }

    return result


@torch.no_grad()
def process_dataset(model, random_model, dataloader, device, get_model_feats_logits, num_classes, num_imgs):
    """
    Process a dataset using a pre-trained classification model.
    We return the spatial feature representations, the predictions,
    the targets, top-1 accuracy and a confusion matrix as a dictionary.
    """
    top1 = torchmetrics.Accuracy(top_k=1, task="multiclass", num_classes=num_classes)

    top1.to(device)
    model.to(device)
    model.eval()
    if random_model is not None:
        random_model.to(device)
        random_model.eval()

    features = []
    preds = []
    targets = []

    for data_idx, (images, target) in enumerate(dataloader):
        images, target = images.to(device), target.to(device)
        if random_model is not None: # apply trained model on first half, random model on 2nd half
            if data_idx < num_imgs:
                output = get_model_feats_logits(model, images)
            else:
                output = get_model_feats_logits(random_model, images)
        else:
            output = get_model_feats_logits(model, images)
        top1(output["logits"], target)

        features.append(output["features"].cpu())
        targets.append(target.cpu())
        preds.append(torch.argmax(output["logits"], dim=1).cpu())

    features = torch.cat(features, dim=0)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    confusion_mat = confusion_matrix(targets.numpy(), preds.numpy())

    result = {
        "features": features,
        "preds": preds,
        "targets": targets,
        "top1": top1.compute(),
        "confusion_matrix": confusion_mat,
    }

    return result
