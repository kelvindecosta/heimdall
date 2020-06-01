import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from lib.dataset import DroneDeploySegmentationDataset as Dataset
from lib.utils import mask_to_label

from lib.config import BATCH_SIZES, CLASS_COLORS, DEVICE

__all__ = ["run"]


def run(input_path, model_path, output_path):
    """
    Predicts the segmentation masks for an input image.

    Arguments:
        input_path {str} -- path to input image
        model_path {str} -- path to model file
        output_path {str} -- path to output image

    Uses the following configuration settings:
        - BATCH_SIZES: number of data points fed in a single optimization step
        - CLASS_COLORS: RGB colors for each class; used for creating label from mask
        - DEVICE: device upon which torch operations are run
    """

    # Create path variables
    image_path = Path(input_path)
    model_path = Path(model_path)
    output_path = Path(output_path)

    # Load model
    model = torch.load(str(model_path))

    # Set data loader
    test_ds = Dataset(filepath=image_path)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZES["test"], shuffle=False)

    # Predict
    prediction = None
    with torch.no_grad():
        # Load in batches of multiple tiles
        for tiles in tqdm(
            test_dl,
            desc="test",
            bar_format="{l_bar}{bar}|{n:4d}/{total:4d} [{elapsed} <{remaining}]",
            unit="batch",
        ):
            tiles = tiles.to(DEVICE)
            pred_label = mask_to_label(model(tiles), CLASS_COLORS)

            # Concatenate prediction with previous bath, to form contiguous tiles
            if prediction is None:
                prediction = pred_label.cpu()
            else:
                prediction = torch.cat((prediction, pred_label.cpu()))

    # Create grid out of contiguous tiles
    prediction = make_grid(prediction, nrow=test_ds.nrow, padding=0)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.permute(1, 2, 0).numpy()).save(str(output_path))
