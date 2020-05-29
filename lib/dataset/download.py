import gdown
import tarfile

from pathlib import Path

from lib.config import DATASET_CHOICE, DATASET_URL

__all__ = ["run"]


def run(**kwargs):
    """
    Downloads the Drone Deploy Segmentation Dataset.

    Uses the following configuration settings:
        - DATASET_CHOICE: dataset to be downloaded
        - DATASET_URL: download url
    """

    # Download the archive file, if it isn't downloaded already
    filename = Path(f"downloads/{DATASET_CHOICE}.tar.gz")
    filename.parent.mkdir(parents=True, exist_ok=True)

    if filename.exists():
        print(
            f"Archive file of dataset '{DATASET_CHOICE}' already exists ('{str(filename)}')."
        )
    else:
        print(f"Downloading archive file of dataset '{DATASET_CHOICE}'")
        gdown.download(DATASET_URL, str(filename), quiet=False)

    # Extract the archive file, if it isn't extracted already
    dataset_directory = Path(f"data/{DATASET_CHOICE}")

    if dataset_directory.exists():
        print(
            f"Extracted dataset '{DATASET_CHOICE}' already exists ('{str(dataset_directory)}')."
        )
    else:
        print(f"Extracting dataset '{DATASET_CHOICE}'...")
        dataset_directory.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(filename), "r:gz") as tar:
            tar.extractall(str(dataset_directory.parent))
