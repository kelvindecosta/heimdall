import torch

from torchvision import transforms as T

from lib.config.session import DEVICE

# Choice of datasets
DATASET_CHOICE = "sample"

# URLs for dataset downloads
URLS = {
    "sample": "https://dl.dropboxusercontent.com/s/h8a8kev0rktf4kq/dataset-sample.tar.gz?dl=0",
    "medium": "https://dl.dropboxusercontent.com/s/r0dj9mhyv4bgbme/dataset-medium.tar.gz?dl=0",
}

# Names of the classes
LABEL_NAMES = ["BUILDING", "CLUTTER", "VEGETATION", "WATER", "GROUND", "CAR"]

# Color of pixels to ignore
IGNORE_COLOR = torch.tensor([255, 0, 255], dtype=torch.uint8).to(DEVICE)

# Color of pixels for the classes
LABEL_COLORS = torch.tensor(
    [
        [230, 25, 75],
        [145, 30, 180],
        [60, 180, 75],
        [245, 130, 48],
        [255, 255, 255],
        [0, 130, 200],
    ],
    dtype=torch.uint8,
).to(DEVICE)

# Image Size
IMAGE_SIZE = 224

# Transformations
AUGMENTATION_TRANSFORMS = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomAffine(
            30, translate=None, scale=None, shear=None, resample=False, fillcolor=0
        ),
    ]
)
PREPROCESSING_TRANSFORMS = T.Compose(
    [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)
