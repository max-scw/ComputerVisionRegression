from pathlib import Path

import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


from typing import Union, List, Dict


class LoadImagesAndLabels(Dataset):
    def __init__(
            self,
            path_to_info_file: Union[str, Path],
            transforms = None,
            shuffle_data: bool = False
    ) -> None:
        # assumes an info file with lines structured as this: [path_to_image, regression_value]
        self.path_to_info = Path(path_to_info_file)
        self.transforms = transforms

        with open(self.path_to_info.as_posix(), "r") as fid:
            lines_ = fid.readlines()
        lines = [ln.strip("\n").split(" ") for ln in lines_ if len(ln) > 5]
        self.data_info = [(Path(p), float(v)) for (p, v) in lines]

        self.indices = list(range(len(self)))
        if shuffle_data:
            random.shuffle(self.indices)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        path_to_image, val = self.data_info[index]
        # Load image
        img = Image.open(path_to_image.as_posix()).convert('RGB')
        # apply transformations for augmentation (returns a tensor)
        img = self.transforms(img)
        # return torch tensors in 32-bit precision
        return img.float(), torch.tensor([val]).float()


def build_dataloaders(info_file: Path, transforms=None, batch_size: int = 32):
    dataset_names = {"training": "Trn", "validation": "Val"}

    data_loaders = dict()
    for ky, vl in dataset_names.items():
        image_dataset = LoadImagesAndLabels(
            info_file,
            transforms=transforms,
            shuffle_data=True if ky == "training" else False
        )
        # TODO: add augmentation for training data
        data_loaders[ky] = DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=True
        )
    return data_loaders