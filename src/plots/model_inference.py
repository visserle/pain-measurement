import logging
from pathlib import Path

import holoviews as hv
import hvplot.polars  # noqa
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d

from src.experiments.measurement.stimulus_generator import StimulusGenerator

logger = logging.Logger(__name__.rsplit(".", 1)[-1])


def create_samples_full_stimulus(df, features, sample_duration=5000, step_size=1000):
    """
    Create samples from dataframe with specified duration and step size.

    Args:
        df: Polars DataFrame with a 'normalized_timestamp' column
        sample_duration: Duration of each sample in milliseconds (default: 5000)
        step_size: Spacing between sample starts in milliseconds (default: 1000)

    Returns:
        List of DataFrames, each representing a sample
    """
    # Get min and max timestamps
    min_time = df["normalized_timestamp"].min()
    max_time = df["normalized_timestamp"].max()

    # Generate sample start times
    sample_starts = np.arange(min_time, max_time - sample_duration + 1, step_size)

    # Create samples
    samples = []
    for start in sample_starts:
        end = start + sample_duration
        sample = df.filter(
            (df["normalized_timestamp"] >= start) & (df["normalized_timestamp"] < end)
        ).select(features)

        # Only include non-empty samples
        if sample.height > 0:
            samples.append(sample)

    if logging:
        logger.info(
            f"Created {len(samples)} samples of {sample_duration}ms duration with {step_size}ms spacing"
        )
    return samples


def analyze_test_dataset_for_one_stimulus(
    model, db, features, participant_ids, stimulus_seed, logging=True
):
    """
    Analyze the entire test dataset with the specified stimulus seed.
    Args:
        model: Trained model to use for predictions
        db: DatabaseManager instance to access the database
        features: List of features to use for analysis
        participant_ids: List of participant IDs to analyze
        stimulus_seed: Stimulus seed to filter for
    Returns:
        all_predictions: Dictionary mapping participant_id to their predictions
        all_confidences: Dictionary mapping participant_id to their confidences
        participant_trials: Dictionary mapping participant_id to number of trials
    """

    device = next(model.parameters()).device.type

    all_predictions = {}
    all_confidences = {}
    participant_trials = {}

    # Process each participant
    for participant_id in participant_ids:
        if logging:
            logger.info(f"Processing participant {participant_id}...")
        # Get data for this participant and stimulus seed
        with db:
            participant_df = db.get_table("merged_and_labeled_data").filter(
                (pl.col("participant_id") == participant_id)
                & (pl.col("stimulus_seed") == stimulus_seed)
            )

        if participant_df.is_empty():
            if logging:
                logger.info(
                    f"No data found for participant {participant_id} with stimulus seed {stimulus_seed}"
                )
            continue

        #
        participant_predictions = []
        participant_confidences = []

        # Create samples for this trial
        samples = create_samples_full_stimulus(participant_df, features)

        if not samples:
            if logging:
                logger.info(
                    f" No valid samples for trial {participant_df.get_column('trial_id').unique().item()}"
                )
            continue

        # Get predictions and confidences for this trial
        trial_predictions = []
        trial_confidences = []
        for sample in samples:
            tensor = (
                torch.tensor(sample.to_numpy(), dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            pred_class = probabilities.argmax(dim=1).item()
            pred_confidence = probabilities[0, pred_class].item()
            trial_predictions.append(pred_class)
            trial_confidences.append(pred_confidence)

        participant_predictions.append(trial_predictions)
        participant_confidences.append(trial_confidences)

        if participant_predictions:
            all_predictions[participant_id] = participant_predictions
            all_confidences[participant_id] = participant_confidences
            participant_trials[participant_id] = len(participant_predictions)

    return all_predictions, all_confidences, participant_trials


def create_aggregate_visualization(
    aligned_predictions, aligned_confidences, stimulus_seed
):
    """
    Create a heatmap visualization of model predictions across all trials.
    Args:
        aligned_predictions: Dictionary of aligned predictions for each participant
        aligned_confidences: Dictionary of aligned confidences for each participant
        stimulus_seed: Seed used for the stimulus generation
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))

    # Combine all trials into a single matrix with signed confidences
    all_confidences_matrix = []

    for participant_id in aligned_confidences:
        for i, trial_confidences in enumerate(aligned_confidences[participant_id]):
            trial_predictions = aligned_predictions[participant_id][i]

            # Transform classifier confidences to centered, scaled values [-1, 1]
            signed_confidences = [
                -(conf - 0.5) * 2 if pred == 0 else (conf - 0.5) * 2
                for conf, pred in zip(trial_confidences, trial_predictions)
            ]
            all_confidences_matrix.append(signed_confidences)

    # Convert to numpy array and sort by average confidence
    confidence_array = np.array(all_confidences_matrix)
    avg_confidence = np.mean(np.abs(confidence_array), axis=1)
    sorted_confidence_array = confidence_array[np.argsort(-avg_confidence)]

    # Create custom colormap: blue for negative, white for 0, orange for positive
    colors = [(0, 0, 1), (1, 1, 1), (1, 0.42, 0.21)]  # Blue, White, Orange
    cmap = LinearSegmentedColormap.from_list("BlueWhiteOrange", colors, N=100)

    # Create time axis
    time_points = np.linspace(0, 180, confidence_array.shape[1])

    # Get and process stimulus signal
    stim = StimulusGenerator(seed=stimulus_seed)
    stimulus = stim.y[::10]  # Downsample to match typical alignment length

    # Resample stimulus to match confidence array length
    f = interp1d(np.linspace(0, 1, len(stimulus)), stimulus)
    stimulus_resampled = f(np.linspace(0, 1, confidence_array.shape[1]))

    # Normalize stimulus to [-1, 1]
    stimulus_line = (
        2
        * (
            (stimulus_resampled - stimulus_resampled.min())
            / (stimulus_resampled.max() - stimulus_resampled.min())
        )
        - 1
    )

    # Plot heatmap
    im = ax.imshow(
        sorted_confidence_array,
        aspect="auto",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        extent=(0, 180, 0, len(sorted_confidence_array)),
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "Signed Confidence (- = Decrease, + = Increase)", rotation=270, labelpad=20
    )

    # Add stimulus line overlay
    scaled_stimulus_y = len(sorted_confidence_array) / 2
    ax.plot(
        time_points,
        stimulus_line * (scaled_stimulus_y / 3) + scaled_stimulus_y,
        "k-",
        linewidth=2,
        label="Stimulus",
    )

    # Add labels and statistics
    num_participants = len(aligned_confidences)
    total_trials = len(sorted_confidence_array)
    ax.set_title(
        f"Stimulus {stimulus_seed}: {num_participants} participants, {total_trials} trials",
        fontsize=16,
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Participants (sorted by confidence)")

    plt.tight_layout()
    return fig
