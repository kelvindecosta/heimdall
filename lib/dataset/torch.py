import numpy as np
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from lib.dataset.transforms import (
    parse_albumentation_transforms,
    parse_torch_transforms,
)
from lib.utils import boolean_mask, to_tiles

from lib.config import (
    CLASS_COLORS,
    DATASET_CHOICE,
    DEVICE,
    IMAGE_SIZE,
    TRANSFORMATIONS,
)

__all__ = ["DroneDeploySegmentationDataset"]


class DroneDeploySegmentationDataset(Dataset):
    """
    A Torch Dataset for the tiled images of the Drone Deploy Segmentation Dataset.

    Uses the following configuration settings:
        - CLASS_COLORS: RGB colors for each class; used for splitting label into mask
        - DATASET_CHOICE: dataset to reference
        - DEVICE: device upon which torch operations are run
        - IMAGE_SIZE: size of square tiles
        - TRANSFORMATIONS: dictionary of image transformations
    """

    def __init__(self, split=None, filepath=None, transforms=TRANSFORMATIONS):
        """
        Initializes Torch Dataset.

        Keyword Arguments:
            split {str} -- a choice between 'train' & 'valid' (default: {None})
            filepath {str} -- a filepath for a specific image (default: {None})
            transforms {dict} -- a dictionary of image transforms (default: {TRANSFORMATIONS})
        """

        if (
            split is None and filepath is None
        ):  # Neither split nor filenpath are provided
            raise ValueError("Specify sample set split or specific filepath")
        elif (
            split is not None and filepath is not None
        ):  # Both split and filenpath are provided
            raise ValueError(
                "Cannot specify both sample set split and specific filepath"
            )
        elif split is not None:  # split is provided
            # Store directory of tiles
            self.directory = Path(f"data/{DATASET_CHOICE}/tiles/x{IMAGE_SIZE}")

            # Store list of tile filenames
            with open(str(self.directory / "split" / f"{split}.txt"), "r") as fd:
                self.filenames = fd.read().strip().split("\n")
        else:  # filepath is provided
            # Load image
            image = Image.open(str(filepath)).convert("RGB")
            image = torch.from_numpy(np.array(image))

            # Store image as tiles
            image = to_tiles(image, IMAGE_SIZE)
            _, self.nrow = image.shape[:2]
            self.tiles = image.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3).numpy()

        # Store whether a single specific image or the entire dataset is being loaded
        self.single = filepath is not None

        # Parse transformations
        if transforms is not None:
            if split == "train":
                self.augmentation = parse_albumentation_transforms(
                    transforms["augmentation"]
                )
            self.preprocessing = parse_torch_transforms(transforms["preprocessing"])

    def __getitem__(self, idx):
        """
        Indexes an item in the Torch Dataset.

        Arguments:
            idx {int} -- index of item

        Returns:
            [tuple(torch.Tensor, torch.Tensor)] -- a tuple of the image and mask tensors
        """

        # Declare variables
        image = None
        mask = None

        if not self.single:
            # Index image filename
            filename = self.filenames[idx]

            # Load image and label
            image = Image.open(str(self.directory / "images" / filename)).convert("RGB")
            label = Image.open(str(self.directory / "labels" / filename)).convert("RGB")

            # Apply augmentation transforms
            try:
                image = np.array(image)
                label = np.array(label)

                augmented = self.augmentation(image=image, label=label)

                image = Image.fromarray(augmented["image"])
                label = Image.fromarray(augmented["label"])
            except:
                pass

            # Convert label to mask
            label = torch.from_numpy(np.array(label)).to(DEVICE)
            mask = (
                torch.stack([boolean_mask(label, color) for color in CLASS_COLORS])
                .type(torch.float32)
                .to(DEVICE)
            )
        else:
            # Index image tile
            image = self.tiles[idx]
            image = Image.fromarray(image)

        # Apply preprocessing transforms
        try:
            image = self.preprocessing(image)
        except:
            pass

        if self.single:
            return image

        return image, mask

    def __len__(self):
        """
        Returns:
            [int] -- Length of the Torch Dataset
        """

        if self.single:
            return len(self.tiles)

        return len(self.filenames)
