import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class ADE20KDataset(Dataset):
    split_to_dir = {"train": "training", "val": "validation"}

    def __init__(
        self,
        root,
        transform,
        target_transform,
        split="train",
        skip_other_class=False,
        file_set=None,
        num_classes=None,
        tag=None,
    ):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.root = root
        self.skip_other_class = skip_other_class
        self.file_set = file_set

        # Collect the data
        self.data = self.collect_data()

    def collect_data(self):
        # Get the image and annotation dirs
        image_dir = os.path.join(self.root, f"images/{self.split_to_dir[self.split]}")
        annotation_dir = os.path.join(self.root, f"annotations/{self.split_to_dir[self.split]}")

        # Collect the filepaths
        if self.file_set is None:
            image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
            annotation_paths = [os.path.join(annotation_dir, f) for f in sorted(os.listdir(annotation_dir))]
        else:
            image_paths = [os.path.join(image_dir, f"{f}.jpg") for f in sorted(self.file_set)]
            annotation_paths = [os.path.join(annotation_dir, f"{f}.png") for f in sorted(self.file_set)]

        data = list(zip(image_paths, annotation_paths))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the  paths
        image_path, annotation_path = self.data[index]

        # Load
        image = Image.open(image_path).convert("RGB")
        target = Image.open(annotation_path)

        # Augment
        image = self.transform(image).squeeze(0)
        target = self.target_transform(target)
        if self.skip_other_class == True:
            target = target * 255.0
            target[target.type(torch.int64) == 0] = 255.0
            target /= 255.0
        target = target.squeeze(0)

        batch = {"image": image, "label": target}
        return batch
