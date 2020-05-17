import torch
import segmentation_models_pytorch as smp
import json

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.dataset import DroneDeploySegmentationDataset as Dataset

from lib.config.model import (
    MODELS,
    ACTIVATION,
    WEIGHTS,
    METRICS,
    BATCH_SIZE,
    OPTIMIZER,
    LEARNING_RATE,
    EPOCHS,
)
from lib.config.dataset import LABEL_COLORS
from lib.config.session import DEVICE, TIMESTAMP


def run(architecture, backbone, save_metric, model_path):

    # Create data loaders
    data_loaders = {
        "train": DataLoader(Dataset("train"), batch_size=BATCH_SIZE, shuffle=True),
        "valid": DataLoader(Dataset("valid"), batch_size=BATCH_SIZE, shuffle=False),
    }

    # Set model
    model = None
    run_id = None

    if model_path is None:
        model = MODELS.get(architecture)(
            encoder_name=backbone,
            encoder_weights=WEIGHTS,
            classes=len(LABEL_COLORS),
            activation=ACTIVATION,
        )
        run_id = f"{TIMESTAMP}-{architecture}-{backbone}"
    else:
        run_id = model_path.stem[: -len("-model")]
        model = torch.load(model_path)

    # Setup custom logging
    log_data = {}
    log_file = Path("logs") / f"{run_id}.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if log_file.exists():
        with open(log_file.as_posix(), "r") as fd:
            log_data = json.load(fd)
            start_epoch = log_data["epoch"]
    else:
        log_data = {
            "architecture": architecture,
            "backbone": backbone,
            "save_metric": save_metric,
            "batch_size": BATCH_SIZE,
        }

    # Set loss and optimizer
    loss = smp.utils.losses.DiceLoss()
    optimizer = OPTIMIZER(params=model.parameters(), lr=LEARNING_RATE)

    # Set train and valid epoch execution
    execution = {
        "train": smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=METRICS,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        ),
        "valid": smp.utils.train.ValidEpoch(
            model, loss=loss, metrics=METRICS, device=DEVICE, verbose=True,
        ),
    }

    # Set up logging with TensorBoard
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir.as_posix())

    # Iterate over epochs
    best_score = 0
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch: {epoch+1}")

        for phase in ["train", "valid"]:
            logs = execution[phase].run(data_loaders[phase])
            for scalar in logs:
                writer.add_scalar(f"{phase} {scalar}", logs[scalar], epoch + 1)

            # Save the model if it is the best one so far, based on the validation score
            score = logs[save_metric]
            if phase == "valid" and best_score < score:
                best_score = score
                model_path = Path("weights") / f"{run_id}-model.pth"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model, model_path)

                log_data["epoch"] = epoch
                with open(log_file.as_posix(), "w") as fd:
                    json.dump(log_data, fd, indent=2)
