import json
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from lib.utils import to_tiles, boolean_mask

from lib.config import DATASET_CHOICE, DEVICE, IGNORE_COLOR, IMAGE_SIZE

__all__ = ["run"]


def run(**kwargs):
    """
    Creates square tiles of the large images and labels.

    Uses the following configuration settings:
        - DATASET_CHOICE: dataset from which tiles are generated
        - DEVICE: device upon which torch operations are run
        - IGNORE_COLOR: a torch tensor for the RGB color of unlabelled pixels
        - IMAGE_SIZE: size of square tile in pixels
    """

    dataset_directory = Path(f"data/{DATASET_CHOICE}")
    tiles_directory = dataset_directory / "tiles" / f"x{IMAGE_SIZE}"

    # Create tiles, if they do not already exist
    if tiles_directory.exists():
        print(
            f"Tiles for dataset '{DATASET_CHOICE}' for size '{IMAGE_SIZE}' already exist."
        )
        return

    print(f"Creating tiles for dataset '{DATASET_CHOICE}'")

    # Creating required directories
    (tiles_directory / "split").mkdir(parents=True, exist_ok=True)
    for subdir in ["images", "labels"]:
        (tiles_directory / subdir).mkdir(parents=True, exist_ok=True)

    # Create tiles for each scene image & label
    index = []
    with open(str(dataset_directory / "index.json"), "r") as fd:
        index = json.load(fd)

    for sample_set in ["train", "valid"]:
        print(f"  {sample_set}")
        for scene_id in index[sample_set]:
            # Load PIL image & label
            image = Image.open(
                str(dataset_directory / "images" / f"{scene_id}-ortho.tif")
            ).convert("RGB")
            label = Image.open(
                str(dataset_directory / "labels" / f"{scene_id}-label.png")
            ).convert("RGB")

            # Convert to Tensors
            image = torch.from_numpy(np.array(image)).to(DEVICE)
            label = torch.from_numpy(np.array(label)).to(DEVICE)

            # Tile label
            tiled_label = to_tiles(label, IMAGE_SIZE)

            # Mask out tiles with incomplete labels
            keep = ~torch.any(
                torch.any(boolean_mask(tiled_label, IGNORE_COLOR), dim=3), dim=2
            )
            image_tiles = to_tiles(image, IMAGE_SIZE)[keep]
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
                    str(tiles_directory / "split" / f"{sample_set}.txt"), "a+"
                ) as fd:
                    fd.write(f"{filename}\n")

                # Save image and label tile
                Image.fromarray(image_tiles[count].cpu().numpy()).save(
                    str(tiles_directory / "images" / filename)
                )
                Image.fromarray(label_tiles[count].cpu().numpy()).save(
                    str(tiles_directory / "labels" / filename)
                )
