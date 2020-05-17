import numpy as np
import torch

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

from lib.utils import boolean_mask, to_tiles

from lib.config.dataset import (
    DATASET_CHOICE,
    IGNORE_COLOR,
    LABEL_COLORS,
    IMAGE_SIZE,
    AUGMENTATION_TRANSFORMS,
    PREPROCESSING_TRANSFORMS,
)
from lib.config.session import DEVICE


class DroneDeploySegmentationDataset(Dataset):
    """
    A Torch Dataset for the tiled images of the Drone Deploy Segmentation Dataset.
    """

    def __init__(
        self,
        sample_set,
        augmentation=AUGMENTATION_TRANSFORMS,
        preprocessing=PREPROCESSING_TRANSFORMS,
    ):
        """
        Initializes the dataset

        Arguments:
            sample_set {str} -- sample set in split ("train", "valid")
            augmentation {torchvision.transforms.Conpose} -- composed list of augmentation transforms
            preprocessing {torchvision.transforms.Conpose} -- composed list of preprocessing transforms
        """
        self.directory = Path(f"data/{DATASET_CHOICE}/tiles/x{IMAGE_SIZE}")
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # Store list of tile filenames
        with open(
            (self.directory / "split" / f"{sample_set}.txt").as_posix(), "r"
        ) as fd:
            self.filenames = fd.read().strip().split("\n")

    def __getitem__(self, idx):
        """
        Indexes an item in the dataset.

        Arguments:
            idx {int} -- index of item

        Returns:
            [tuple(torch.Tensor, torch.Tensor)] -- a tuple of the image and mask tensors
        """
        filename = self.filenames[idx]

        # Load image
        image = Image.open((self.directory / "images" / filename).as_posix()).convert(
            "RGB"
        )

        # Load label
        label = Image.open((self.directory / "labels" / filename).as_posix()).convert(
            "RGB"
        )

        # Apply augmenation transforms
        if self.augmentation:
            image = self.augmentation(image)
            label = self.augmentation(label)

        # Convert label to mask
        label = torch.from_numpy(np.array(label)).to(DEVICE)
        mask = (
            torch.stack([boolean_mask(label, color) for color in LABEL_COLORS])
            .type(torch.float32)
            .to(DEVICE)
        )

        # Apply preprocessing transforms
        if self.preprocessing:
            image = self.preprocessing(image)

        return image, mask

    def __len__(self):
        return len(self.filenames)


class DroneDeploySegmentationTestCase(Dataset):
    """
    A Torch Dataset for a single test case of the Drone Deploy Segmentation Dataset.
    """

    def __init__(self, filename, preprocessing=PREPROCESSING_TRANSFORMS):
        self.preprocessing = preprocessing

        image = Image.open(filename).convert("RGB")
        image = torch.from_numpy(np.array(image))
        image = to_tiles(image, IMAGE_SIZE)

        self.nrows, self.ncols = image.shape[:2]

        self.tiles = image.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3).numpy()

    def __getitem__(self, idx):
        image = self.tiles[idx]
        image = Image.fromarray(image)

        image = self.preprocessing(image)

        return image

    def __len__(self):
        return len(self.tiles)
