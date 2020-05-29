import argparse

from lib import dataset, model

__all__ = ["Interface"]


class Interface:
    """
    A wrapper for the command line interface
    """

    def __init__(self):
        """
        Initializes the parser
        """

        # Create parser
        self.parser = argparse.ArgumentParser(
            description="A pipeline for semantic segmentation on orthographic drone imagery."
        )

        self.functions = {
            "download": dataset.download,
            "tile": dataset.tile,
            "train": model.train,
            "predict": model.predict,
        }

        # Assign commands
        commands = self.parser.add_subparsers(dest="command")

        # download
        commands.add_parser("download", help="Download dataset")

        # tile
        commands.add_parser(
            "tile", help="Create tiles of images and labels in the dataset"
        )

        # train
        commands.add_parser(
            "train", help="Train a segmentation model on the tiles of the dataset"
        )

        # predict
        predict_subparser = commands.add_parser(
            "predict", help="Predict the segmentation mask for an input image"
        )
        predict_subparser.add_argument("input_path", help="path to input image file")
        predict_subparser.add_argument("model_path", help="path to model")
        predict_subparser.add_argument("output_path", help="path to output image file")

    def __call__(self):
        """
        Executes command
        """

        # Parse args
        args = self.parser.parse_args()

        # Show help and exit if no command is specified
        if args.command is None:
            self.parser.print_help()
            exit()

        # Execute command with keyword arguments
        kwargs = vars(args)
        command = kwargs.pop("command")
        self.functions[command](**kwargs)
