import logging

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
from scipy.stats import rankdata

from src.data.database_manager import DatabaseManager
from src.experiments.measurement.stimulus_generator import StimulusGenerator

logger = logging.Logger(__name__.rsplit(".", 1)[-1])


def analyze_test_dataset_for_one_stimulus(
    model: torch.nn.Module,
    db: DatabaseManager,
    features: list,
    participant_ids: list,
    stimulus_seed: int,
    sample_duration: int = 3000,
    log: bool = True,
) -> tuple[dict, dict]:
    """
    Analyze the entire test dataset with the specified stimulus seed.
    Args:
        model: Trained model to use for predictions
        db: DatabaseManager instance to access the database
        features: List of features to use for analysis
        participant_ids: List of participant IDs to analyze
        stimulus_seed: Stimulus seed to filter for
    Returns:
        probabilities: Dictionary mapping participant_id to their class probabilities
        participant_trials: Dictionary mapping participant_id to number of trials
    """

    device = next(model.parameters()).device.type

    # Renamed to result_probabilities to avoid name collision
    result_probabilities = {}
    participant_trials = {}

    # Process each participant
    for participant_id in participant_ids:
        if log:
            logger.info(f"Processing participant {participant_id}...")
        # Get data for this participant and stimulus seed
        with db:
            participant_df = db.get_table("merged_and_labeled_data").filter(
                (pl.col("participant_id") == participant_id)
                & (pl.col("stimulus_seed") == stimulus_seed)
            )

        if participant_df.is_empty():
            if log:
                logger.info(
                    f"No data found for participant {participant_id} with stimulus seed {stimulus_seed}"
                )
            continue

        # Create a proper key for the participant (use the actual ID, not the index)
        participant_key = str(participant_id)

        participant_probabilities = []

        # Create samples for this trial
        samples = _create_samples_full_stimulus(
            participant_df, features, sample_duration
        )

        if not samples:
            if log:
                logger.info(
                    f" No valid samples for trial {participant_df.get_column('trial_id').unique().item()}"
                )
            continue

        # Get probabilities for this trial
        batch_samples = []
        with torch.inference_mode():
            for sample in samples:
                batch_samples.append(sample.to_numpy())

            # Process all samples in a single batch
            if batch_samples:
                # Convert all samples to a single tensor batch
                batch_tensor = torch.tensor(
                    np.stack(batch_samples), dtype=torch.float32
                ).to(device)
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1)
                trial_probabilities = probs.cpu().detach().numpy().tolist()
                participant_probabilities.append(trial_probabilities)

        if participant_probabilities:
            result_probabilities[participant_key] = participant_probabilities
            participant_trials[participant_key] = len(participant_probabilities)
    return result_probabilities, participant_trials


def _create_samples_full_stimulus(
    df: pl.DataFrame,
    features: list,
    sample_duration: int = 3000,
    step_size: int = 1000,
) -> list[pl.DataFrame]:
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
            f"Created {len(samples)} samples of {sample_duration}ms duration with {step_size} ms spacing"
        )
    return samples


def plot_prediction_confidence_heatmap(
    probabilities: dict,
    stimulus_seed: int,
    sample_duration: int = 3000,
    step_size: int = 1000,
    classification_threshold: float = 0.5,
    pseudonymize: bool = True,
) -> None:
    """
    Create a heatmap visualization of model predictions across all participants.
    Args:
        probabilities: Dictionary of all probabilities for each participant
        stimulus_seed: Seed used for the stimulus generation
        classification_threshold: Threshold used for binary classification (default: 0.5)
    """
    if sample_duration % 1000:
        raise ValueError("Sample duration must be a multiple of 1000 milliseconds.")

    # Prepare data structures to track participant IDs with their confidence data
    confidence_data = []  # List of (participant_id, confidence_array) tuples

    for participant_id, trial_probabilities_list in probabilities.items():
        for i, trial_probabilities in enumerate(trial_probabilities_list):
            # Extract increase (class 1) probabilities
            increase_probs = np.array([probs[1] for probs in trial_probabilities])

            # Scale values to account for the classification threshold
            # Values below threshold will be negative (decrease)
            # Values above threshold will be positive (increase)
            # The exact threshold will map to 0
            signed_confidences = np.zeros_like(increase_probs)

            # For probabilities below threshold (decrease predictions)
            below_threshold = increase_probs < classification_threshold
            if np.any(below_threshold):
                # Scale from [0, threshold] to [-1, 0]
                signed_confidences[below_threshold] = (
                    increase_probs[below_threshold] - classification_threshold
                ) / classification_threshold

            # For probabilities above threshold (increase predictions)
            above_threshold = increase_probs >= classification_threshold
            if np.any(above_threshold):
                # Scale from [threshold, 1] to [0, 1]
                signed_confidences[above_threshold] = (
                    increase_probs[above_threshold] - classification_threshold
                ) / (1 - classification_threshold)

            confidence_data.append((participant_id, signed_confidences))

    # Convert confidence values to array and calculate average confidence for sorting
    confidence_arrays = [data[1] for data in confidence_data]
    confidence_array = np.array(confidence_arrays)
    # pad with zeros to reflect that the first 5 seconds are needed for the first sample
    padded_array = np.zeros((confidence_array.shape[0], 180))
    padded_array[:, int(sample_duration / 1000 - 1) :] = confidence_array
    confidence_array = padded_array

    avg_confidence = np.mean(np.abs(confidence_array), axis=1)

    # Sort by average confidence
    sort_indices = np.argsort(-avg_confidence)
    sorted_confidence_array = confidence_array[sort_indices]
    sorted_participant_ids = [confidence_data[i][0] for i in sort_indices]

    # Create custom colormap: orange for negative, white for 0, blue for positive
    colors = [(1, 0.42, 0.21), (1, 1, 1), (0, 0, 1)]  # Orange, White, Blue
    cmap = LinearSegmentedColormap.from_list("OrangeWhiteBlue", colors, N=256)

    # Create time axis
    time_points = np.linspace(0, 180, confidence_array.shape[1])

    # Get and process stimulus signal
    stimulus = StimulusGenerator(seed=stimulus_seed, config={"sample_rate": 1}).y
    # Normalize stimulus to [-1, 1]
    stimulus_line = (
        2 * ((stimulus - stimulus.min()) / (stimulus.max() - stimulus.min())) - 1
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot heatmap
    im = ax.imshow(
        sorted_confidence_array,
        aspect="auto",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        extent=(0, 183, 0, len(sorted_confidence_array)),
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
        linewidth=3,
        color="#404040",
        label="Stimulus",
    )

    # Add labels and statistics
    total_trials = len(sorted_confidence_array)
    ax.set_title(
        f"Stimulus {stimulus_seed}: {total_trials} trials, classification threshold {classification_threshold}",
        fontsize=16,
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Participants (sorted by confidence)")

    # Add participant IDs as y-ticks
    y_positions = np.linspace(
        0,
        len(sorted_confidence_array) - 1,
        min(10, len(sorted_confidence_array)),
    )
    y_positions = np.round(y_positions).astype(int)

    if not pseudonymize:
        # Use actual participant IDs
        ax.set_yticks(np.arange(len(sorted_confidence_array)))
        ax.set_yticklabels(sorted_participant_ids[::-1])
        # we have to reverse the order of the y-ticks else they are upside down
    else:
        # Use pseudonymized participant IDs
        sorted_participant_ids = np.array(
            sorted_participant_ids, dtype=int
        )  # ids are strings
        pseudonymized_ids = rankdata(sorted_participant_ids, method="dense")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(pseudonymized_ids[::-1])
        # we have to reverse the order of the y-ticks else they are upside down

    plt.tight_layout()
    plt.show()
