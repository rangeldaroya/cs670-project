# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import os
from PIL import Image

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from torchvision.datasets.folder import default_loader

class Cub(Dataset):
    """
    Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version
    of the CUB-200 dataset, with roughly double the number of images per
    class and new part location annotations. For detailed information
    about the dataset, please see the technical report linked below.
    Number of categories: 200
    Number of images: 11,788
    Annotations per image: 15 Part Locations
    Webpage: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    README: http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/README.txt  # noqa
    Download: http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
    """

    def __init__(
        self,
        root="./data",
        train=True,
        transform=None,
        loader=default_loader,
        return_image_only=False,
    ):

        self._dataset_folder = pathlib.Path(os.path.join(root, "CUB_200_2011"))
        self._transform = transform
        self._loader = loader
        self._train = train
        self._class_name_index = {}
        self._return_image_only = return_image_only

        if not self._check_dataset_folder():
            raise RuntimeError(
                "Dataset not downloaded, download it from "
                "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"  # noqa
            )

    def _load_metadata(self):
        images = pd.read_csv(
            self._dataset_folder.joinpath("images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )

        image_class_labels = pd.read_csv(
            self._dataset_folder.joinpath("image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )

        train_eval_split = pd.read_csv(
            self._dataset_folder.joinpath("train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        with open(self._dataset_folder.joinpath("classes.txt")) as f:
            for line in f:
                class_label, class_name = line.strip().split(" ", 1)
                class_label = int(class_label) - 1
                self._class_name_index[class_label] = class_name

        # merge
        data = images.merge(image_class_labels, on="img_id")
        self._data = data.merge(train_eval_split, on="img_id")

        # select split
        if self._train:
            self._data = self._data[self._data.is_training_img == 1]
        else:
            self._data = self._data[self._data.is_training_img == 0]

    def _check_dataset_folder(self):
        try:
            self._load_metadata()
        except Exception as e:
            print(f"Error: {e}")
            return False

        for _, row in self._data.iterrows():
            filepath = self._dataset_folder.joinpath("images", row.filepath)
            if not pathlib.Path.exists(filepath):
                return False
        return True

    @property
    def class_name_index(self):
        return self._class_name_index

    @property
    def parts_name_index(self):
        return self._parts_name_index

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        sample = self._data.iloc[idx]
        path = self._dataset_folder.joinpath("images", sample.filepath)

        img = self._loader(path)
        width, height = img.size

        return self._transform(img), sample.target - 1

    def get_target(self, target):
        return (
            np.argwhere(np.array(self._data["target"].tolist()) == target + 1)
            .reshape(-1)
            .tolist()
        )