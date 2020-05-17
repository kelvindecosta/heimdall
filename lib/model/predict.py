import torch

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from lib.dataset import DroneDeploySegmentationTestCase as TestCase
from lib.utils import mask_to_label

from lib.config.dataset import LABEL_COLORS
from lib.config.model import BATCH_SIZE
from lib.config.session import DEVICE


def run(image, model):
    image_path = image
    model_path = Path(model)

    # Load model
    model = torch.load(model_path.as_posix())

    # Set image loader
    image_test_case = TestCase(image_path)
    image_loader = DataLoader(image_test_case, batch_size=BATCH_SIZE * 4, shuffle=False)

    # Predict
    prediction = None
    with torch.no_grad():
        for tiles in tqdm(
            image_loader,
            desc="test",
            bar_format="{l_bar}{bar}|{n:4d}/{total:4d} [{elapsed} <{remaining}]",
            unit="batch",
        ):
            tiles = tiles.to(DEVICE)
            pred_label = mask_to_label(model(tiles), LABEL_COLORS)

            if prediction is None:
                prediction = pred_label.cpu()
            else:
                prediction = torch.cat((prediction, pred_label.cpu()))

    prediction = make_grid(prediction, nrow=image_test_case.ncols, padding=0)

    # Write to file
    run_id = model_path.stem[: -len("-model")]
    output_path = Path("predictions") / f"{run_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.permute(1, 2, 0).numpy()).save(output_path.as_posix())
