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
  > See [lib.config.dataset](lib/config/dataset.py) to change dataset `DATASET_CHOICE`.

- Index the dataset.

  ```bash
  python scripts/dataset.py
  ```

- Create small tiles from the large images and labels in the dataset.

  ```bash
  python main.py tile
  ```

  > This command preprocesses the dataset specified by `DATASET_CHOICE`.
  > Tiles of size `IMAGE_SIZE` are created from the larger images and labels.
  >
  > `INORE_COLOR` represents the `RGB` values for the unlabelled pixels in the dataset.
  > Label tiles that have this color present are ignore (so are their corresponding images).

### Train

Train a model.

```bash
python main.py train
```

> Run `python main.py train -h` for more options

### Predict

Predict the segmentation mask for an image.

```bash
python main.py predict <path_to_image> <path_to_model>
```

> Run `python main.py predict -h` for more options
