import torch


def tile(img, size):
    """
    Tiles an image.

    Arguments:
        img {torch.Tensor} -- image tensor [shape = (H, W, 3)]
        size {int} -- tile size in pixels

    Returns:
        torch.Tensor -- tiled images tensor [shape = (H //  size, W // size, size, size, 3)]
    """
    return img.unfold(0, size, size).unfold(1, size, size).unfold(2, 3, 3).squeeze()


def mask(img, color):
    """
    Returns a Boolean mask on a image, based on the presence of a color.

    Arguments:
        img {torch.Tensor} -- image tensor
        color {torch.Tensor} -- RGB color tensor

    Returns:
        torch.BoolTensor -- boolean mask of image
    """
    dim = len(img.shape) - 1

    return torch.all(img == color.view(*([1] * dim), 3), dim=dim)
