import torch

# Default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
