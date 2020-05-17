import torch

from datetime import datetime

# Default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Timestamp
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H-%M")
