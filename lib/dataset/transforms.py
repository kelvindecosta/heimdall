import albumentations
import torchvision

__all__ = ["parse_albumentation_transforms", "parse_torch_transforms"]


def parse_albumentation_transforms(dictionary):
    """
    Parse a dictionary of image transformations defined in albumentations.

    Arguments:
        dictionary {dict} -- dictionary for a transform, with keys `name` and `args`

    Returns:
        a callable albumentation transform pipeline
    """

    name = eval(f"""albumentations.{dictionary["name"]}""")
    args = dict(dictionary["args"])

    # Recursively parse transforms if the current transform is a composition
    if name.__name__ in albumentations.core.composition.__all__:
        args["transforms"] = list(
            map(parse_albumentation_transforms, args["transforms"])
        )

    return name(**args)


def parse_torch_transforms(dictionary):
    """
    Parse a dictionary of image transformations defined in torchvision.transforms.

    Arguments:
        dictionary {dict} -- dictionary for a transform, with keys `name` and `args`

    Returns:
        a callable torchvision transform pipeline
    """

    name = eval(f"""torchvision.transforms.{dictionary["name"]}""")
    args = dict(dictionary["args"])

    # Recursively parse transforms if the current transform is a composition
    if any(
        issubclass(name, eval(f"torchvision.transforms.transforms.{x}"))
        for x in ["Compose", "RandomTransforms"]
    ):
        args["transforms"] = list(map(parse_torch_transforms, args["transforms"]))

    return name(**args)
