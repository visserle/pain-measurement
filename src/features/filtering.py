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


def butterworth_filter_non_causal(
    signal: np.ndarray,
    sample_rate: int,
    lowcut=None,
    highcut=None,
    order: int = 5,
):
    """Filter a signal using IIR Butterworth SOS (Second-Order Sections) method.

    Applies the butterworth filter in both directions to avoid phase distortion.
    This is a non-causal filter, only for exploratory purposes.
    """
    freqs, filter_type = _sanitize_filter(
        lowcut=lowcut,
        highcut=highcut,
        sample_rate=sample_rate,
    )
    sos = scipy.signal.butter(
        order,
        freqs,
        btype=filter_type,
        output="sos",
        fs=sample_rate,
    )
    return scipy.signal.sosfiltfilt(sos, signal)


def _sanitize_filter(
    lowcut=None,
    highcut=None,
    sample_rate=1000,
    normalize=False,
):
    """Sanitize the input for filtering.

    Normalize is False by default as there is no need to normalize if `fs` argument is
    provided to the scipy filter.

    Modified from neurokit2 (0.2.10):
    https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/signal/signal_filter.py
    """
    # Sanity checks
    if lowcut is None and highcut is None:
        raise ValueError("At least one of lowcut or highcut must be specified.")

    nyquist_rate = sample_rate / 2
    max_freq = max(filter(None, [lowcut, highcut]))
    if lowcut is not None or highcut is not None:
        if nyquist_rate <= max_freq:
            raise ValueError(
                "The sampling rate is too low. Sampling rate must exceed the Nyquist "
                "rate to avoid aliasing problem. In this analysis, the sampling rate "
                f"has to be higher than {2 * max_freq} Hz"
            )

    # Replace 0 by None
    lowcut = None if lowcut == 0 else lowcut
    highcut = None if highcut == 0 else highcut

    # Determine filter type and frequencies
    if lowcut is not None and highcut is not None:
        filter_type = "bandstop" if lowcut > highcut else "bandpass"
        # pass frequencies in order of lowest to highest to the scipy filter
        freqs = sorted([lowcut, highcut])
    elif lowcut is not None:
        filter_type = "highpass"
        freqs = lowcut
    elif highcut is not None:
        filter_type = "lowpass"
        freqs = highcut
    else:
        return None, None

    # Normalize frequency to Nyquist Frequency (Fs/2) if required
    # However, no need to normalize if `fs` argument is provided to the scipy filter
    if normalize:
        freqs = np.array(freqs) / nyquist_rate

    return freqs, filter_type
