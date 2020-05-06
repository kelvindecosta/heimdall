import os
import numpy as np
import torch

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms as T

from lib.config import DATASET
from lib.preprocess import create_tiles


def get_sample_set(scene_id):
    if scene_id in DATASET["split"]["train"]:
        return "train"
    if scene_id in DATASET["split"]["valid"]:
        return "valid"
    if scene_id in DATASET["split"]["test"]:
        return "test"


def download():
    """
    Downloads the Drone Deploy dataset.
    See `lib/config.py` to choose between `sample` and `medium` datasets.
    """

    choice = DATASET["choice"]
    urls = DATASET["urls"]

    # Check if choice is valid
    if choice not in urls:
        print(f"Invalid choice '{choice}'.")
        exit(1)

    # Download the archive file, if it isn't downloaded already
    filename = Path(f"downloads/{choice}.tar.gz")
    filename.parent.mkdir(parents=True, exist_ok=True)

    if filename.exists():
        print(
            f"Archive file of dataset '{choice}' already exists ('{filename.as_posix()}')."
        )
    else:
        print(f"Downloading archive file of dataset '{choice}'...")
        os.system(f"curl '{urls[choice]}' -o {filename.as_posix()}")

    # Extract the archive file, if it isn't extracted already
    dataset_directory = Path(f"data/{choice}")

    if dataset_directory.exists():
        print(
            f"Extracted dataset '{choice}' already exists ('{dataset_directory.as_posix()}')."
        )
    else:
        print(f"Extracting dataset '{choice}'...")
        dataset_directory.mkdir(parents=True, exist_ok=True)
        os.system(
            f"tar -xf {filename.as_posix()} -C {dataset_directory.as_posix()} --strip-components 1"
        )


def preprocess():
    choice = DATASET["choice"]
    dataset_directory = Path(f"data/{choice}")

    # Break image & labels into smaller tiles for training
    if (dataset_directory / "tiles").exists():
        print(f"Tiles for dataset '{choice}' already exist.")
    else:
        print(f"Creating tiles for dataset '{choice}'...")

        # Creating required directories
        (dataset_directory / "split").mkdir(parents=True, exist_ok=True)
        for subdir in ["images", "labels"]:
            (dataset_directory / "tiles" / subdir).mkdir(parents=True, exist_ok=True)

        # Create tiles for each scene image & mask
        with open(f"{dataset_directory.as_posix()}/index.csv", "r") as fd:
            lines = fd.read().strip().split("\n")

        for line in lines:
            scene_id = line.split(" ")[1]
            sample_set = get_sample_set(scene_id)

            if sample_set in ["test", None]:
                continue

            create_tiles(dataset_directory, scene_id, sample_set)


class DroneDeployDataset(Dataset):
    def __init__(self, sample_set):
        self.directory = Path(f"data/{DATASET['choice']}")
        self.labels = np.array(DATASET["labels"]["map"][1:], dtype=int)
        self.transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )

        with open(
            (self.directory / "split" / f"{sample_set}.txt").as_posix(), "r"
        ) as fd:
            self.filenames = fd.read().strip().split("\n")

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(
            (self.directory / "tiles" / "images" / filename).as_posix()
        ).convert("RGB")
        label = Image.open(
            (self.directory / "tiles" / "labels" / filename).as_posix()
        ).convert("RGB")

        # Convert RGB label to multiple (number of classes) boolean masks
        masks = torch.as_tensor(
            np.apply_along_axis(
                np.all, -1, np.array(label) == self.labels[:, None, None]
            ),
            dtype=torch.uint8,
        )

        image = self.transforms(image)

        return image, masks

    def __len__(self):
        return len(self.filenames)
