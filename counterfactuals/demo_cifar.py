# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torchvision
from torchvision import transforms

from utils.visualize import visualize_counterfactuals


idx2label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# parser = argparse.ArgumentParser(description="Visualize counterfactual explanations")
# parser.add_argument("--config_path", type=str, required=True)


def main():
    # args = parser.parse_args()

    # experiment_name = os.path.basename(args.config_path).split(".")[0]
    # dirpath = os.path.join(Path.output_root_dir(), experiment_name)
    dirpath = "/home/rdaroya_umass_edu/Documents/cs670-project/counterfactuals/output/counterfactuals_ours_cifar_res50"

    # dataset = get_test_dataset(get_vis_transform(), return_image_only=True)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    trans = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=trans)

    counterfactuals = np.load(
        os.path.join(dirpath, "counterfactuals.npy"), allow_pickle=True
    ).item()

    for idx in np.random.choice(list(counterfactuals.keys()), 5):
        cf = counterfactuals[idx]

        visualize_counterfactuals(
            edits=cf["edits"],
            query_index=cf["query_index"],
            distractor_index=cf["distractor_index"],
            dataset=dataset,
            n_pix=7,
            fname=f"output/counterfactuals_cifar_demo/example_{idx}.png",
            idx2label=idx2label,
        )


if __name__ == "__main__":
    main()