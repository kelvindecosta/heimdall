import argparse

from lib import dataset
from lib import model

from lib.config.model import BACKBONES, LOSSES, MODELS, METRICS


def parse():
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation for Orthographic Drone Imagery"
    )

    commands = parser.add_subparsers(dest="command")

    # Download
    commands.add_parser("download", help="Download and extract dataset")

    # Tile
    commands.add_parser("tile", help="Create tiles of images and labels in the dataset")

    # Train
    p_train = commands.add_parser(
        "train", help="Train a segmentation model on the tiles of the dataset"
    )

    models = sorted(list(MODELS.keys()))
    p_train.add_argument(
        "-a",
        "--architecture",
        choices=models,
        help="choice of model architecture",
        default="unet",
    )

    backbones = sorted(list(BACKBONES))
    p_train.add_argument(
        "-b",
        "--backbone",
        choices=backbones,
        help="choice of pretrained backbone",
        default="resnet101",
    )

    losses = sorted(list(LOSSES.keys()))
    p_train.add_argument(
        "-c",
        "--criterion",
        choices=losses,
        help="choice of loss function",
        default="dice",
    )

    metrics = sorted(list(x.__name__ for x in METRICS))
    p_train.add_argument(
        "-m",
        "--save-metric",
        choices=metrics,
        help="choice of validation metric for saving model",
        default="iou_score",
    )

    p_train.add_argument(
        "-p",
        "--model-path",
        help="path of saved model weights (for retraining)",
        type=str,
    )

    # Predict
    p_predict = commands.add_parser(
        "predict", help="Predict the segmentation mask for an input image"
    )

    p_predict.add_argument("image", help="path to image file")
    p_predict.add_argument("model", help="path to model weight")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        exit()

    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse()
    commands = {
        "download": dataset.download,
        "tile": dataset.tile,
        "train": model.train,
        "predict": model.predict,
    }

    kwargs = vars(args)
    cmd = kwargs.pop("command")

    # Run command
    commands[cmd](**kwargs)
