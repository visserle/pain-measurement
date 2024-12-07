import logging

import torch


def get_device() -> torch.device:
    """Return the device to be used by the model."""
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    logging.info(f"Using device: {device}")
    return device
