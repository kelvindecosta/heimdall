# Heimdall

Semantic Segmentation on Orthographic Drone Imagery for Disaster Management.

## Instructions

### Setup

- Clone this repository.

  ```bash
  git clone https://github.com/kelvindecosta/heimdall.git && cd heimdall
  ```

- Install dependencies.

  ```bash
  pip install -r requirements.txt
  ```

### Data

- Download and extract the [Drone Deploy Segmentation Dataset](https://github.com/dronedeploy/dd-ml-segmentation-benchmark).

  ```bash
  python main.py download
  ```

  > This command downloads one of two datasets.
  > See [lib.config.data](lib/config/data.py) to change dataset `DATASET_CHOICE`.

- Preprocess the extracted dataset.

  ```bash
  python main.py preprocess
  ```

  > This command preprocesses the dataset specified by `DATASET_CHOICE`.
  > Tiles of size `INPUT_SIZE` are created from the larger images and labels.
