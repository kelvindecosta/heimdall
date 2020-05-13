import argparse

from lib import dataset


def parse():
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation for Orthographic Drone Imagery"
    )

    commands = parser.add_subparsers(dest="command")

    # Download
    commands.add_parser("download", help="Download and extract dataset")

    # Preprocess
    commands.add_parser("preprocess", help="Preprocess and tile images of dataset")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        exit()

    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse()
    commands = {"download": dataset.download, "preprocess": dataset.preprocess}

    # Run command
    commands[args.command]()
