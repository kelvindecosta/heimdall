import torch
import segmentation_models_pytorch as smp

# Dictionary of model choices
MODELS = {
    "unet": smp.Unet,
    "linknet": smp.Linknet,
}

# Choices of pretrained backbones
BACKBONES = set(["resnet101", "se_resnet101", "se_resnext101_32x4d"])

# Activation function (for single channel use `sigmoid`)
ACTIVATION = "softmax"

# Dataset on which weights were obtained
WEIGHTS = "imagenet"

# Choice of loss function
LOSSES = {
    "dice": smp.utils.losses.DiceLoss,
    "cross-entropy": smp.utils.losses.CrossEntropyLoss,
}

# Metrics to be tracked
METRICS = [
    smp.utils.metrics.IoU(),
    # smp.utils.metrics.Fscore(),
    # smp.utils.metrics.Accuracy(),
    # smp.utils.metrics.Precision(),
    # smp.utils.metrics.Recall(),
]

# Hyperparameters
BATCH_SIZE = 4
OPTIMIZER = torch.optim.Adam
LEARNING_RATE = 0.0001
EPOCHS = 30
