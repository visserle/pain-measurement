import logging

import torch

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def get_device() -> torch.device:
    """Return the device to be used by the model."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    return device
