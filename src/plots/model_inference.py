import logging

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from matplotlib.colors import LinearSegmentedColormap

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
        device = next(model.parameters()).device
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
    leaderboard: list | None = None,
    figure_size: tuple = (15, 8),
    stimulus_linewidth: int = 4,
    stimulus_color: str = "black",
    stimulus_scale: float = 0.4,
) -> plt.Figure:
    """
    Create a heatmap visualization of model predictions across all participants.
    Optimized for presentation in PowerPoint.

    Args:
        probabilities: Dictionary of all probabilities for each participant
        stimulus_seed: Seed used for the stimulus generation
        sample_duration: Duration of each sample in milliseconds
        step_size: Step size between samples in milliseconds
        classification_threshold: Threshold used for binary classification (default: 0.5)
        leaderboard: Optional list of participant IDs ordered by performance (best first)
        figure_size: Size of the figure (width, height)
        stimulus_linewidth: Width of the stimulus line
        stimulus_color: Color of the stimulus line
        stimulus_scale: Scale factor for the stimulus amplitude
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

    # Sort by leaderboard if provided, otherwise by average confidence
    if leaderboard is not None:
        # Create a mapping from participant ID to leaderboard position
        leaderboard_positions = {pid: i for i, pid in enumerate(leaderboard)}
        participant_ids = [data[0] for data in confidence_data]

        # Sort by leaderboard position (best first = position 0)
        sort_indices = sorted(
            range(len(participant_ids)),
            key=lambda i: leaderboard_positions.get(participant_ids[i], float("inf")),
        )
    else:
        # Fall back to sorting by average confidence
        avg_confidence = np.mean(np.abs(confidence_array), axis=1)
        sort_indices = np.argsort(-avg_confidence)

    sorted_confidence_array = confidence_array[sort_indices]
    sorted_participant_ids = [confidence_data[i][0] for i in sort_indices]

    # Create custom colormap: orange for negative, white for 0, blue for positive
    colors = [
        (1.0, 0.35, 0.1),
        (0.98, 0.98, 0.98),
        (0.0, 0.2, 0.8),
    ]  # Orange, White, Blue
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
    fig, ax = plt.subplots(figsize=figure_size, dpi=120)

    # Plot heatmap with slightly transparent colors to make stimulus more visible
    im = ax.imshow(
        sorted_confidence_array,
        aspect="auto",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        extent=(0, 183, 0, len(sorted_confidence_array)),
        alpha=0.9,
    )

    # Add colorbar with larger font size
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "Prediction Confidence\n(+ = Decrease, - = Increase)",
        rotation=270,
        labelpad=30,
        fontsize=14,
        fontweight="bold",
    )
    cbar.ax.tick_params(labelsize=10)

    # Add stimulus line overlay with improved visibility
    scaled_stimulus_y = len(sorted_confidence_array) / 2
    ax.plot(
        time_points,
        stimulus_line * (scaled_stimulus_y * stimulus_scale) + scaled_stimulus_y,
        linewidth=stimulus_linewidth,
        color=stimulus_color,
        label="Normalized Temperature Curve",
        zorder=10,  # Ensure stimulus is on top
        path_effects=[
            path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
            path_effects.Normal(),
        ],  # Add shadow for better visibility
    )

    # Add legend for the stimulus line
    ax.legend(loc="lower right", fontsize=14, framealpha=0.8)

    # Add labels and statistics with larger font sizes
    total_trials = len(sorted_confidence_array)
    ax.set_title(
        f"Stimulus {stimulus_seed}: {total_trials} trials, threshold {classification_threshold}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Time (seconds)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Participant", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Add participant IDs as y-ticks (with improved formatting)
    num_ticks = min(10, len(sorted_confidence_array))
    y_positions = np.linspace(0, len(sorted_confidence_array) - 1, num_ticks)
    y_positions = np.round(y_positions).astype(int)

    # Use actual participant IDs
    tick_indices = np.linspace(0, len(sorted_participant_ids) - 1, num_ticks).astype(
        int
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([sorted_participant_ids[i] for i in tick_indices[::-1]])

    # Add gridlines for better readability
    ax.grid(which="major", axis="x", linestyle="--", alpha=0.3)

    # Add x-axis ticks every 30 seconds for better readability
    ax.set_xticks(np.arange(0, 181, 30))

    plt.tight_layout()
    return fig


def main():
    import json
    import logging
    import os
    from pathlib import Path

    import holoviews as hv
    import hvplot.polars  # noqa
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import tomllib
    from dotenv import load_dotenv

    from src.data.database_manager import DatabaseManager
    from src.features.labels import add_labels
    from src.features.resampling import add_normalized_timestamp
    from src.log_config import configure_logging
    from src.models.data_loader import create_dataloaders
    from src.models.data_preparation import prepare_data
    from src.models.main_config import RANDOM_SEED
    from src.models.utils import load_model
    from src.plots.model_inference import (
        analyze_test_dataset_for_one_stimulus,
        plot_prediction_confidence_heatmap,
    )
    from src.plots.model_performance_per_participant import analyze_per_participant

    load_dotenv()
    FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))

    configure_logging(
        stream_level=logging.DEBUG,
        ignore_libs=["matplotlib", "Comm", "bokeh", "tornado"],
    )

    pl.Config.set_tbl_rows(12)  # for the 12 trials

    config_path = Path("src/experiments/measurement/measurement_config.toml")
    with open(config_path, "rb") as file:
        config = tomllib.load(file)
    stimulus_seeds = config["stimulus"]["seeds"]
    print(f"Using seeds for stimulus generation: {stimulus_seeds}")

    feature_combination = "eda_raw_heart_rate_pupil"
    # "heart_rate",
    # "pupil",
    # "eda_raw",
    # "eda_raw_pupil",
    # "eda_raw_heart_rate",
    # "eda_raw_heart_rate_pupil",
    # "brow_furrow_cheek_raise_mouth_open_nose_wrinkle_upper_lip_raise",
    # "brow_furrow_cheek_raise_eda_raw_heart_rate_mouth_open_nose_wrinkle_pupil_upper_lip_raise",
    # "c3_c4_cz_f3_f4_oz_p3_p4",

    results = {}

    # Load data from database
    db = DatabaseManager()
    with db:
        if feature_combination == "c3_c4_cz_f3_f4_oz_p3_p4":
            eeg = db.get_table(
                "Preprocess_EEG",
                exclude_trials_with_measurement_problems=True,
            )
            trials = db.get_table(
                "Trials",
                exclude_trials_with_measurement_problems=True,
            )
            eeg = add_normalized_timestamp(eeg)
            df = add_labels(eeg, trials)
        else:
            df = db.get_table(
                "Merged_and_Labeled_Data",
                exclude_trials_with_measurement_problems=True,
            )

    # Load model
    json_path = Path(f"results/experiment_{feature_combination}/results.json")
    dictionary = json.loads(json_path.read_text())
    model_path = Path(dictionary["overall_best"]["model_path"].replace("\\", "/"))

    model, feature_list, sample_duration_ms, intervals, label_mapping, offsets_ms = (
        load_model(model_path, device="cpu")
    )

    # Get participant leaderboard
    # = a list of participant IDs in the test set, sorted by their accuracy on the test set, from highest to lowest.

    # Prepare data
    X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test = (
        prepare_data(
            df=df,
            feature_list=feature_list,
            sample_duration_ms=sample_duration_ms,
            intervals=intervals,
            label_mapping=label_mapping,
            offsets_ms=offsets_ms,
            random_seed=RANDOM_SEED,
        )
    )
    test_groups = prepare_data(
        df=df,
        feature_list=feature_list,
        sample_duration_ms=sample_duration_ms,
        intervals=intervals,
        label_mapping=label_mapping,
        offsets_ms=offsets_ms,
        random_seed=RANDOM_SEED,
        only_return_test_groups=True,
    )
    test_ids = np.unique(test_groups)

    _, test_loader = create_dataloaders(
        X_train_val, y_train_val, X_test, y_test, batch_size=64
    )

    results_df = analyze_per_participant(
        model,
        test_loader,
        test_groups,
        threshold=0.50,
    )

    leaderboard = (
        results_df.remove(participant="overall")
        .sort("accuracy", descending=True)
        .get_column("participant")
        .to_list()
    )

    # Analyze the entire test dataset
    all_probabilities = {}
    all_participant_trials = {}

    for stimulus_seed in stimulus_seeds:
        probabilities, participant_trials = analyze_test_dataset_for_one_stimulus(
            model,
            db,
            feature_list,
            test_ids,
            stimulus_seed,
            sample_duration_ms,
            log=True,
        )

        all_probabilities[stimulus_seed] = probabilities
        all_participant_trials[stimulus_seed] = participant_trials

    stimulus_seed = stimulus_seeds[0]
    threshold = 0.7

    a = plot_prediction_confidence_heatmap(
        probabilities=all_probabilities[stimulus_seed],
        stimulus_seed=stimulus_seed,
        classification_threshold=threshold,
        sample_duration=sample_duration_ms,
        leaderboard=leaderboard,
    )
    plt.show()


if __name__ == "__main__":
    main()
