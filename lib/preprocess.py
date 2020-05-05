import cv2
import os
import numpy as np
import pandas

from lib.config import DATASET


def rolling_window_tiles(
    image, width=DATASET["chip-size"], height=DATASET["chip-size"]
):
    _nrows, _ncols, depth = image.shape

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)

    _temp = np.copy(image[: nrows * height, : ncols * width, :])
    _strides = _temp.strides

    return np.lib.stride_tricks.as_strided(
        np.ravel(_temp),
        shape=(nrows, ncols, height, width, depth),
        strides=(height * _strides[0], width * _strides[1], *_strides),
        writeable=False,
    )


def is_ignorable(label_tile):
    """
    Returns whether the tile can be ignored
    """

    return np.squeeze(
        np.apply_over_axes(
            np.any, np.all(label_tile == DATASET["labels"]["map"][0], axis=-1), [-1, -2]
        )
    )


def create_tiles(
    dataset_directory,
    scene_id,
    sample_set,
    windowx=DATASET["chip-size"],
    windowy=DATASET["chip-size"],
    stridex=DATASET["chip-size"],
    stridey=DATASET["chip-size"],
):
    image = cv2.imread(
        (dataset_directory / "images" / f"{scene_id}-ortho.tif").as_posix()
    )
    label = cv2.imread(
        (dataset_directory / "labels" / f"{scene_id}-label.png").as_posix()
    )

    image_tiles = rolling_window_tiles(image)
    label_tiles = rolling_window_tiles(label)

    index_filter = np.invert(is_ignorable(label_tiles))

    image_tiles = image_tiles[index_filter]
    label_tiles = label_tiles[index_filter]

    for count in range(len(image_tiles)):
        filename = f"{scene_id}-{count+1:06d}.png"
        with open(
            (dataset_directory / "split" / f"{sample_set}.txt").as_posix(), "a+"
        ) as fd:
            fd.write(f"{filename}\n")

        cv2.imwrite(
            (dataset_directory / "tiles" / "images" / filename).as_posix(),
            image_tiles[count],
        )
        cv2.imwrite(
            (dataset_directory / "tiles" / "labels" / filename).as_posix(),
            label_tiles[count],
        )
