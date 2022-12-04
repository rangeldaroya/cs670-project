# Code below is based on https://pytorch.org/vision/stable/_modules/torchvision/datasets/flowers102.html#Flowers102

from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import PIL.Image
import numpy as np

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets import VisionDataset

import torchvision

class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        rot_vals_deg=None,
        trans_vals=None,
        scales=None,

        to_bgr=False,
        to_rrr=False,

        to_double_data_only=False,   # setting this to True will just double the data length (no transformations)
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"
        
        self.rot_vals_deg = rot_vals_deg
        self.trans_vals = trans_vals
        self.scales = scales

        self.to_bgr = to_bgr
        self.to_rrr = to_rrr

        self.to_double_data_only = to_double_data_only

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

        self.len_orig = len(self._image_files)  # number of orig images

        if (self.rot_vals_deg is not None) or to_bgr or to_rrr or to_double_data_only:
            for image_id in image_ids:
                self._labels.append(image_id_to_label[image_id])
                self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")
            

    def __len__(self) -> int:
        return len(self._image_files)


    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")
        if (idx > (self.len_orig-1)):
            if (self.rot_vals_deg is not None):
                # Add data augmentations with random rotations/scale/trans defined
                image = torchvision.transforms.functional.affine(
                    image,
                    angle=self.rot_vals_deg[idx],
                    translate=list(self.trans_vals[idx]),
                    scale=self.scales[idx],
                    shear=0,
                )
            elif self.to_rrr:
                rrr_img = np.array(image).astype(float)
                rrr_img[:,:,1] = 0
                rrr_img[:,:,2] = 0
                image = PIL.Image.fromarray(rrr_img.astype(np.uint8)) # convert to PIL image
            elif self.to_bgr:
                rgb_img = np.array(image).astype(float)
                bgr_img = rgb_img[...,::-1]
                image = PIL.Image.fromarray(bgr_img.astype(np.uint8)) # convert to PIL image

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)
