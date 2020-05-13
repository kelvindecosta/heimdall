import torch

from lib.config.device import DEVICE

# Choice of datasets
CHOICE = "sample"

# URLs for dataset downloads
URLS = {
    "sample": "https://dl.dropboxusercontent.com/s/h8a8kev0rktf4kq/dataset-sample.tar.gz?dl=0",
    "medium": "https://dl.dropboxusercontent.com/s/r0dj9mhyv4bgbme/dataset-medium.tar.gz?dl=0",
}

# Names of the classes
LABEL_NAMES = ["BUILDING", "CLUTTER", "VEGETATION", "WATER", "GROUND", "CAR"]

# Color of pixels to ignore
IGNORE_COLOR = torch.tensor([255, 0, 255]).to(DEVICE)

# Color of pixels for the classes
LABEL_COLORS = torch.tensor(
    [
        [230, 25, 75],
        [145, 30, 180],
        [60, 180, 75],
        [245, 130, 48],
        [255, 255, 255],
        [0, 130, 200],
    ]
).to(DEVICE)
