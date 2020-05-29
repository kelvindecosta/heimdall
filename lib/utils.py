import torch

__all__ = ["boolean_mask", "mask_to_label", "to_tiles"]


def to_tiles(img, size):
    """
    Tiles an image.

    Arguments:
        img {torch.Tensor} -- image tensor [shape = (H, W, 3)]
        size {int} -- tile size in pixels

    Returns:
        torch.Tensor -- tiled images tensor [shape = (H //  size, W // size, size, size, 3)]
    """

    return img.unfold(0, size, size).unfold(1, size, size).unfold(2, 3, 3).squeeze()


def boolean_mask(img, color):
    """
    Returns a Boolean mask on a image, based on the presence of a color.

    Arguments:
        img {torch.Tensor} -- image tensor [shape = (..., 3)]
        color {torch.Tensor} -- RGB color tensor [shape = (3, )]

    Returns:
        torch.BoolTensor -- boolean mask of image [shape = (..., )]
    """

    dim = len(img.shape) - 1

    return torch.all(img == color.view(*([1] * dim), 3), dim=dim)


def mask_to_label(mask, classes):
    """
    Returns the combined masks, i.e. label for a mask based on the color classes.

    Arguments:
        mask {torch.Tensor} -- mask tensor [shape = (B, N, H, W) ; B = Batch Size, N = Number of Classes]
        classes {torch.Tensor} -- color classes tensor [shape = (N, 3)]

    Returns:
        torch.Tensor -- combined masks / label [shape = (B, 3, H, W)]
    """

    mask_indices = torch.argmax(mask, dim=1)

    return (
        classes[mask_indices.view(-1)].view(*mask_indices.shape, 3).permute(0, 3, 1, 2)
    )
