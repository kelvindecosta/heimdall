import segmentation_models_pytorch as smp
import shutil
import torch
import yaml

from datetime import datetime
from pathlib import Path
from yaml import Loader

__all__ = [
    "BATCH_SIZES",
    "CLASS_COLORS",
    "CLASS_NAMES",
    "CONFIG_PATH",
    "CRITERION",
    "CRITERION_ARGS",
    "DATASET_CHOICE",
    "DATASET_URL",
    "DEVICE",
    "EPOCHS",
    "IGNORE_COLOR",
    "IMAGE_SIZE",
    "METRIC",
    "METRIC_ARGS",
    "MODEL",
    "MODEL_ARGS",
    "OPTIMIZER",
    "OPTIMIZER_ARGS",
    "SCHEDULER",
    "SCHEDULER_ARGS",
    "TIMESTAMP",
    "TRANSFORMATIONS",
]

# Default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Timestamp
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M")

# Default config path
CONFIG_PATH = Path("config.yml")

# Create default config if necessary
if not CONFIG_PATH.exists():
    shutil.copy("templates/conf.yml", str(CONFIG_PATH))

# Load settings
with open(str(CONFIG_PATH), "r") as fs:
    settings = yaml.load(fs, Loader=Loader)

data_conf = settings["dataset"]
model_conf = settings["model"]

classes = data_conf["classes"][:-1]

# Class names
CLASS_NAMES = list(map(str.upper, (c["name"] for c in classes)))

# Class colors
CLASS_COLORS = torch.tensor(list(c["color"] for c in classes), dtype=torch.uint8).to(
    DEVICE
)

# Color for unlabelled pixels
IGNORE_COLOR = torch.tensor(data_conf["classes"][-1]["color"], dtype=torch.uint8).to(
    DEVICE
)

# Choice of dataset
DATASET_CHOICE = data_conf["choice"]

# Dataset download url
try:
    DATASET_URL = data_conf["urls"][DATASET_CHOICE]
except:
    raise KeyError(f"Invalid choice of dataset '{DATASET_CHOICE}'")

# Image Size
IMAGE_SIZE = data_conf["size"]

# Transformations
TRANSFORMATIONS = data_conf["transformations"]

# Model
try:
    MODEL = eval(f"""smp.{model_conf["architecture"]["name"]}""")
    MODEL_ARGS = model_conf["architecture"]["args"]
except:
    raise NameError(
        f"""Architecture '{model_conf["architecture"]["name"]}' not defined"""
    )

# Metric
try:
    METRIC = eval(f"""smp.utils.metrics.{model_conf["metric"]["name"]}""")
    METRIC_ARGS = model_conf["metric"]["args"]
except:
    raise NameError(f"""Metric '{model_conf["metric"]["name"]}' not defined""")

# Criterion
try:
    CRITERION = eval(f"""smp.utils.losses.{model_conf["criterion"]["name"]}""")
    CRITERION_ARGS = model_conf["criterion"]["args"]
except:
    raise NameError(f"""Criterion '{model_conf["criterion"]["name"]}' not defined""")

# Optimizer
try:
    OPTIMIZER = eval(f"""torch.optim.{model_conf["optimizer"]["name"]}""")
    OPTIMIZER_ARGS = model_conf["optimizer"]["args"]
except:
    raise NameError(f"""Optimizer '{model_conf["optimizer"]["name"]}' not defined""")

# Scheduler
try:
    SCHEDULER = eval(f"""torch.optim.lr_scheduler.{model_conf["scheduler"]["name"]}""")
    SCHEDULER_ARGS = model_conf["scheduler"]["args"]
except:
    raise NameError(f"""Scheduler '{model_conf["scheduler"]["name"]}' not defined""")

# Batch Sizes
BATCH_SIZES = model_conf["batch-sizes"]

# Epochs
EPOCHS = model_conf["epochs"]
