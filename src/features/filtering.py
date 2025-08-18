import numpy as np
import scipy.signal
from numpy.lib.stride_tricks import sliding_window_view


def median_filter(signal, window_size):
    """
    Vectorized implementation of a causal median filter.
    Can be used for real-time applications.
    """
    # Pad signal with first value repeated
    padded = np.pad(signal, (window_size - 1, 0), mode="edge")

    # Create sliding window view
    windowed = sliding_window_view(padded, window_size)

    # Take median along window axis
    return np.median(windowed, axis=1)


def ema_smooth(
    signal,
    alpha: float,
):
    """
    Standard Exponential Moving Average (EMA) smoothing.

    Args:
        signal: Input signal array
        alpha: Smoothing factor (0 < alpha <= 1)
               Higher alpha = more responsive to recent changes
               Lower alpha = more smoothing
    """
    smoothed = np.zeros_like(signal)
    smoothed[0] = signal[0]

    for i in range(1, len(signal)):
        smoothed[i] = alpha * signal[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed
