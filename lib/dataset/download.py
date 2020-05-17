import os

from pathlib import Path

from lib.config.dataset import DATASET_CHOICE, URLS


def run(**kwargs):
    """
    Downloads & extracts the Drone Deploy Segmentation Dataset.
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
