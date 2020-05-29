import segmentation_models_pytorch as smp
import shutil
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.dataset import DroneDeploySegmentationDataset as Dataset

from lib.config import (
    BATCH_SIZES,
    CONFIG_PATH,
    CRITERION,
    CRITERION_ARGS,
    DEVICE,
    EPOCHS,
    METRIC,
    METRIC_ARGS,
    MODEL,
    MODEL_ARGS,
    OPTIMIZER,
    OPTIMIZER_ARGS,
    SCHEDULER,
    SCHEDULER_ARGS,
    TIMESTAMP,
)

__all__ = ["run"]


def run(**kwargs):
    """
    Trains a model on the dataset.

    Uses the following configuration settings:
        - BATCH_SIZES: number of data points fed in a single optimization step
        - CONFIG_PATH: path to configuration file
        - CRITERION: loss function
        - CRITERION_ARGS: arguments for criterion
        - DEVICE: device upon which torch operations are run
        - EPOCHS: number of iterations on the dataset
        - METRIC: accuracy score
        - METRIC_ARGS: arguments for metric
        - MODEL: model architecture
        - MODEL_ARGS: arguments for model
        - OPTIMIZER: gradient descent and backpropagation optimizer
        - OPTIMIZER_ARGS: arguments for optimizer
        - SCHEDULER: learning rate scheduler
        - SCHEDULER_ARGS: arguments for scheduler
        - TIMESTAMP: time at run (unique identifier)
    """

    # Create data loaders
    data_loaders = {
        "train": DataLoader(
            Dataset(split="train"), batch_size=BATCH_SIZES["train"], shuffle=True,
        ),
        "valid": DataLoader(
            Dataset(split="valid"), batch_size=BATCH_SIZES["valid"], shuffle=False,
        ),
    }

    # Assign model, criterion, optimizer, scheduler and metrics
    model = MODEL(**MODEL_ARGS)
    criterion = CRITERION(**CRITERION_ARGS)
    optimizer = OPTIMIZER(params=model.parameters(), **OPTIMIZER_ARGS)
    scheduler = SCHEDULER(optimizer=optimizer, **SCHEDULER_ARGS)
    metric = METRIC(**METRIC_ARGS)

    # Create train and valid epoch executions
    execution = {
        "train": smp.utils.train.TrainEpoch(
            model,
            loss=criterion,
            metrics=[metric],
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        ),
        "valid": smp.utils.train.ValidEpoch(
            model, loss=criterion, metrics=[metric], device=DEVICE, verbose=True,
        ),
    }

    # Create run directory
    run_dir = Path("runs") / TIMESTAMP
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy current configuration settings
    shutil.copy(str(CONFIG_PATH), str(run_dir / "config.yml"))

    # Setup TensorBoard
    writer = SummaryWriter(str(run_dir))

    # Iterate over epochs
    best_score = 0
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1}")

        # Iterate over phases
        for phase in ["train", "valid"]:
            # Evaluate dataset
            logs = execution[phase].run(data_loaders[phase])

            # Write to TensorBoard
            for scalar in logs:
                writer.add_scalar(f"{phase} {scalar}", logs[scalar], epoch + 1)

            # Save the model if it is the best one so far, based on the validation score
            score = logs[metric.__name__]
            if phase == "valid" and best_score < score:
                best_score = score
                torch.save(model, str(run_dir / "model.pth"))

        # Notify scheduler every epoch
        scheduler.step()
