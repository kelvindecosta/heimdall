import numpy as np
import torch
import os
import json

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from lib.utils import tile, mask

from lib.config.data import DATASET_CHOICE, URLS, IGNORE_COLOR, LABEL_COLORS
from lib.config.device import DEVICE
from lib.config.model import INPUT_SIZE


def download():
    """
    Downloads the Drone Deploy dataset.
    """

    # Check if choice is valid
    if DATASET_CHOICE not in URLS:
        print(f"Invalid choice '{DATASET_CHOICE}'.")
        exit(1)

    # Download the archive file, if it isn't downloaded already
    filename = Path(f"downloads/{DATASET_CHOICE}.tar.gz")
    filename.parent.mkdir(parents=True, exist_ok=True)

    if filename.exists():
        print(
            f"Archive file of dataset '{DATASET_CHOICE}' already exists ('{filename.as_posix()}')."
        )
    else:
        print(f"Downloading archive file of dataset '{DATASET_CHOICE}'")
        os.system(f"curl '{URLS[DATASET_CHOICE]}' -o {filename.as_posix()}")

    # Extract the archive file, if it isn't extracted already
    dataset_directory = Path(f"data/{DATASET_CHOICE}")

    if dataset_directory.exists():
        print(
            f"Extracted dataset '{DATASET_CHOICE}' already exists ('{dataset_directory.as_posix()}')."
        )
    else:
        print(f"Extracting dataset '{DATASET_CHOICE}'...")
        dataset_directory.mkdir(parents=True, exist_ok=True)
        os.system(
            f"tar -xf {filename.as_posix()} -C {dataset_directory.as_posix()} --strip-components 1"
        )


def preprocess():
    """
    Preprocesses the dataset by tiling images and labels.
    """

    dataset_directory = Path(f"data/{DATASET_CHOICE}")
    tiles_directory = dataset_directory / "tiles" / f"x{INPUT_SIZE}"

    # Break image & labels into smaller tiles for training
    if tiles_directory.exists():
        print(
            f"Tiles for dataset '{DATASET_CHOICE}' for size '{INPUT_SIZE}' already exist."
        )
        return

    print(f"Creating tiles for dataset '{DATASET_CHOICE}'")

    # Creating required directories
    (tiles_directory / "split").mkdir(parents=True, exist_ok=True)
    for subdir in ["images", "labels"]:
        (tiles_directory / subdir).mkdir(parents=True, exist_ok=True)

    # Create tiles for each scene image & mask
    index = []
    with open((dataset_directory / "index.json").as_posix(), "r") as fd:
        index = json.load(fd)

    for sample_set in ["train", "valid"]:
        print(f"  {sample_set}")
        for scene_id in index[sample_set]:
            # Load PIL image & label
            image = Image.open(
                (dataset_directory / "images" / f"{scene_id}-ortho.tif").as_posix()
            ).convert("RGB")
            label = Image.open(
                (dataset_directory / "labels" / f"{scene_id}-label.png").as_posix()
            ).convert("RGB")

            # Convert to Tensors
            image = torch.from_numpy(np.array(image)).to(DEVICE)
            label = torch.from_numpy(np.array(label)).to(DEVICE)

            # Tile label
            tiled_label = tile(label, INPUT_SIZE)

            # Mask out tiles with incomplete labels
            keep = ~torch.any(torch.any(mask(tiled_label, IGNORE_COLOR), dim=3), dim=2)
            image_tiles = tile(image, INPUT_SIZE)[keep]
            label_tiles = tiled_label[keep]

            # Iterate over each tile
            for count in tqdm(
                range(len(image_tiles)),
                desc=f"    {scene_id:35}",
                bar_format="{l_bar}{bar}|{n:4d}/{total:4d} [{elapsed} <{remaining}]",
            ):
                # Generate filename
                filename = f"{scene_id}-{count+1:06d}.png"

                # Store filename (to be used while loading data)
                with open(
                    (tiles_directory / "split" / f"{sample_set}.txt").as_posix(), "a+"
                ) as fd:
                    fd.write(f"{filename}\n")

                # Save image and label tile
                Image.fromarray(image_tiles[count].cpu().numpy()).save(
                    (tiles_directory / "images" / filename).as_posix()
                )
                Image.fromarray(label_tiles[count].cpu().numpy()).save(
                    (tiles_directory / "labels" / filename).as_posix()
                )


class DroneDeploySegmentationDataset(Dataset):
    """
    A Torch Dataset for the tiled images of the Drone Deploy dataset.
    """

    def __init__(self, sample_set, transforms):
        """
        Initializes the dataset

        Arguments:
            sample_set {str} -- sample set in split ("train", "valid")
            transforms {torchvision.transforms.Conpose} -- composed list of transforms
        """
        self.directory = Path(f"data/{DATASET_CHOICE}/tiles/x{INPUT_SIZE}")
        self.transforms = transforms

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

        # Load image and apply transforms
        image = Image.open((self.directory / "images" / filename).as_posix()).convert(
            "RGB"
        )

        image = self.transforms(image).to(DEVICE)

        # Load label and convert to masks
        label = Image.open((self.directory / "labels" / filename).as_posix()).convert(
            "RGB"
        )

        label = torch.from_numpy(np.array(label)).to(DEVICE)
        masks = (
            torch.stack([mask(label, color) for color in LABEL_COLORS])
            .type(torch.float32)
            .to(DEVICE)
        )

        return image, masks

    def __len__(self):
        return len(self.filenames)
