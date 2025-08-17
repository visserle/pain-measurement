import logging

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from matplotlib.colors import LinearSegmentedColormap

from src.experiments.measurement.stimulus_generator import StimulusGenerator
from src.models.data_loader import transform_sample_df_to_arrays
from src.models.data_preparation import (
    EEG_FEATURES,
    load_data_from_database,
    prepare_data,
)
from src.models.main_config import INTERVALS, LABEL_MAPPING, OFFSETS_MS, RANDOM_SEED
from src.models.scalers import StandardScaler3D

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])

plt.style.use("./src/plots/style.mplstyle")


def analyze_test_dataset_for_one_stimulus(
    df: pl.DataFrame,
    model: torch.nn.Module,
    features: list,
    participant_ids: list,
    stimulus_seed: int,
    sample_duration: int,
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
    # Get scaler from train+validation data
    scaler: StandardScaler3D = prepare_data(
        df=df,
        feature_list=features,
        sample_duration_ms=sample_duration,
        intervals=INTERVALS,
        label_mapping=LABEL_MAPPING,
        offsets_ms=OFFSETS_MS,
        random_seed=RANDOM_SEED,
        only_return_scaler=True,
    )  # ignore

    result_probabilities = {}
    participant_trials = {}

    # for debugging purposes
    participant_ids = [31]

    # Process each participant
    for participant_id in participant_ids:
        logger.info(f"Processing participant {participant_id}...")
        # Get data for this participant and stimulus seed
        participant_df = df.filter(participant_id=participant_id).filter(
            stimulus_seed=stimulus_seed
        )

        if participant_df.is_empty():
            logger.info(
                f"No data found for participant {participant_id} with stimulus seed {stimulus_seed}"
            )
            continue

        # Create a proper key for the participant (use the actual ID, not the index)
        participant_key = str(participant_id)

        participant_probabilities = []

        # Create samples for this trial
        samples_df = _create_samples_full_stimulus(
            participant_df, features, sample_duration
        )

        if samples_df is None or samples_df.is_empty():
            logger.info(
                f" No valid samples for trial {participant_df.get_column('trial_id').unique().item()}"
            )
            continue

        # Transform samples and scale them
        X, _, _ = transform_sample_df_to_arrays(
            samples_df,  # Combine all sample DataFrames
            feature_columns=features,
            label_column=None,  # No labels needed for inference
            group_column=None,  # No grouping needed
        )
        X = scaler.transform(X)

        # Get predictions
        device = next(model.parameters()).device
        with torch.inference_mode():
            batch_tensor = torch.tensor(X, dtype=torch.float32).to(device)
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
    sample_duration: int,
    step_size: int = 1000,
) -> pl.DataFrame:
    """
    Create samples from dataframe with specified duration and step size.
    Returns a single DataFrame with sample_id column for grouping.

    Args:
        df: Polars DataFrame with a 'normalized_timestamp' column
        features: List of features to include in the samples
        sample_duration: Duration of each sample in milliseconds
        step_size: Spacing between sample starts in milliseconds

    Returns:
        DataFrame with sample_id column and all samples concatenated
    """
    # Get min and max timestamps
    min_time, max_time = 0.0, 180000.0
    # Generate sample start times
    sample_starts = np.arange(min_time, max_time - sample_duration + 1, step_size)

    # Initialize list to store sample DataFrames
    sample_dfs = []

    # Check if we're dealing with EEG data
    eeg_features_present = [f for f in features if f in EEG_FEATURES]
    target_length = 250 * (sample_duration // 1000) if eeg_features_present else None

    for sample_idx, start in enumerate(sample_starts):
        end = start + sample_duration
        sample = df.filter(
            (pl.col("normalized_timestamp") >= start)
            & (pl.col("normalized_timestamp") < end)
        )

        if sample.height == 0:
            continue

        if eeg_features_present:
            # Handle EEG data with specific length requirements
            if sample.height >= target_length or (target_length - sample.height < 20):
                if sample.height > target_length:
                    sample = sample.head(target_length)
                elif sample.height < target_length:
                    # Pad with repeated last row
                    last_row = sample.tail(1)
                    padding = pl.concat([last_row] * (target_length - sample.height))
                    sample = pl.concat([sample, padding])
            else:
                continue

        # Add sample_id column
        sample = sample.with_columns(sample_id=pl.lit(sample_idx))
        sample_dfs.append(sample)

    if not sample_dfs:
        logger.info("No valid samples created")
        return None

    # Combine all samples into one DataFrame
    result_df = pl.concat(sample_dfs)

    logger.info(
        f"Created {len(sample_dfs)} samples of {sample_duration}ms duration "
        f"with {step_size}ms spacing"
    )

    return result_df.select(["sample_id"] + features)


def plot_single_prediction_confidence_heatmap(
    probabilities: dict,
    stimulus_seed: int,
    sample_duration: int,
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
            # Extract decrease (class 1) probabilities
            increase_probs = np.array([probs[1] for probs in trial_probabilities])

            # Scale values to account for the classification threshold
            # Values below threshold will be negative (increase)
            # Values above threshold will be positive (decrease)
            # The exact threshold will map to 0
            signed_confidences = np.zeros_like(increase_probs)

            # For probabilities below threshold (increase predictions)
            below_threshold = increase_probs < classification_threshold
            if np.any(below_threshold):
                # Scale from [0, threshold] to [-1, 0]
                signed_confidences[below_threshold] = (
                    increase_probs[below_threshold] - classification_threshold
                ) / classification_threshold

            # For probabilities above threshold (decrease predictions)
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
    cmap = LinearSegmentedColormap.from_list("CustomColors", colors, N=256)

    # Create time axis
    time_points = np.linspace(0, 180, confidence_array.shape[1])

    # Get and process stimulus signal
    stimulus = StimulusGenerator(seed=stimulus_seed, config={"sample_rate": 1}).y
    # Normalize stimulus to [-1, 1]
    stimulus_line = (
        2 * ((stimulus - stimulus.min()) / (stimulus.max() - stimulus.min())) - 1
    )
    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)  # Removed dpi=120

    # Plot heatmap with slightly transparent colors to make stimulus more visible
    im = ax.imshow(
        sorted_confidence_array,
        aspect="auto",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        extent=(0, 180, len(sorted_confidence_array)),
        alpha=0.9,
    )

    # Add colorbar - removed font size specifications
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        "Prediction Confidence\n(+ = Decrease, - = Increase)",
        rotation=270,
        labelpad=30,
    )

    # Add stimulus line overlay with improved visibility
    scaled_stimulus_y = len(sorted_confidence_array) / 2
    ax.plot(
        time_points,
        stimulus_line * (scaled_stimulus_y * stimulus_scale) + scaled_stimulus_y,
        linewidth=stimulus_linewidth,
        color=stimulus_color,
        label="Normalized Temperature Curve",
        zorder=10,
        path_effects=[
            path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
            path_effects.Normal(),
        ],
    )

    # Add legend for the stimulus line
    ax.legend(loc="lower right", framealpha=0.8)

    # Add labels and statistics - removed font sizes and weights
    total_trials = len(sorted_confidence_array)
    ax.set_title(
        f"Stimulus {stimulus_seed}: {total_trials} trials, threshold {classification_threshold}"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Participant")

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


def plot_prediction_confidence_heatmap(
    all_probabilities: dict,
    sample_duration: int = 3000,
    classification_threshold: float = 0.5,
    figure_size: tuple = (2.8, 1.8),
    stimulus_linewidth: float = 1.5,
    stimulus_scale: float = 0.25,
    seeds_to_plot: list | None = None,
    ncols: int = 2,
    only_decreases: bool = True,
) -> plt.Figure:
    """
    Create compact heatmap visualizations of model predictions across all participants for multiple stimulus seeds.
    Publication-ready version optimized for two-column layout with minimal spacing.

    Args:
        all_probabilities: Dictionary structured as {seed: {participant: probabilities}}
        sample_duration: Duration of each sample in milliseconds
        classification_threshold: Threshold used for binary classification (default: 0.5)
        figure_size: Size of each subplot (width, height)
        stimulus_linewidth: Line width for stimulus overlay
        stimulus_scale: Scale factor for stimulus amplitude
        seeds_to_plot: Optional list of specific seeds to plot. If None, plots all seeds.
        ncols: Number of columns in the subplot grid
    """
    # Validate inputs
    _validate_inputs(sample_duration, all_probabilities, seeds_to_plot)

    # Setup plotting parameters
    seeds_to_plot = _get_seeds_to_plot(all_probabilities, seeds_to_plot)
    fig, axes = _create_figure_and_axes(seeds_to_plot, ncols, figure_size)
    cmap = _create_colormap(only_decreases)

    # Plot each seed
    for idx, stimulus_seed in enumerate(seeds_to_plot):
        _plot_single_heatmap(
            axes.flatten()[idx],
            all_probabilities[stimulus_seed],
            stimulus_seed,
            sample_duration,
            classification_threshold,
            cmap,
            stimulus_linewidth,
            stimulus_scale,
            idx,
            ncols,
            len(seeds_to_plot) // ncols,
            only_decreases,
        )

    # Hide unused subplots
    _hide_empty_subplots(axes.flatten(), len(seeds_to_plot))

    # Add colorbar and finalize layout
    _finalize_figure_layout(fig, axes.flatten()[0])

    return fig


def _validate_inputs(
    sample_duration: int, all_probabilities: dict, seeds_to_plot: list | None
):
    """Validate input parameters."""
    if sample_duration % 1000:
        raise ValueError("Sample duration must be a multiple of 1000 milliseconds.")


def _get_seeds_to_plot(all_probabilities: dict, seeds_to_plot: list | None) -> list:
    """Get list of seeds to plot, validating availability."""
    available_seeds = sorted(all_probabilities.keys())

    if seeds_to_plot is None:
        return available_seeds

    valid_seeds = [s for s in seeds_to_plot if s in available_seeds]
    if not valid_seeds:
        raise ValueError("None of the specified seeds are available in the data.")

    return valid_seeds


def _create_figure_and_axes(seeds_to_plot: list, ncols: int, figure_size: tuple):
    """Create figure with optimized subplot layout."""
    n_seeds = len(seeds_to_plot)
    nrows = (n_seeds + ncols - 1) // ncols

    total_width = figure_size[0] * min(ncols, n_seeds)
    total_height = figure_size[1] * nrows

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(total_width, total_height),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )

    # Ensure consistent 2D array structure
    if nrows == 1:
        axes = axes.reshape(1, -1) if ncols > 1 else np.array([[axes]])
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    return fig, axes


def _create_colormap(only_decreases) -> LinearSegmentedColormap:
    """Create custom colormap."""
    if only_decreases:
        # For decreases only: white to blue
        colors = [
            (1, 1, 1),
            (0.0, 0.2, 0.8),
        ]
    else:
        # For both increases and decreases: orange to white to blue
        colors = [
            (1.0, 0.35, 0.1),
            (1, 1, 1),
            (0.0, 0.2, 0.8),
        ]

    return LinearSegmentedColormap.from_list("CustomColors", colors, N=256)


def _process_confidence_data(
    probabilities: dict, classification_threshold: float, sample_duration: int
):
    """Process raw probabilities into signed confidence values."""
    confidence_data = []
    for participant_id, trial_probabilities_list in probabilities.items():
        for trial_probabilities in trial_probabilities_list:
            increase_probs = np.array([probs[1] for probs in trial_probabilities])
            signed_confidences = _calculate_signed_confidence(
                increase_probs, classification_threshold
            )
            confidence_data.append((participant_id, signed_confidences))

    # Convert to array and sort by participant ID
    confidence_arrays = [data[1] for data in confidence_data]
    confidence_array = np.array(confidence_arrays)

    # Add padding for initial samples
    padded_array = np.zeros((confidence_array.shape[0], 180))
    padded_array[:, int(sample_duration / 1000 - 1) :] = confidence_array

    # Sort by participant ID as integers
    participant_ids = [data[0] for data in confidence_data]
    sort_indices = sorted(
        range(len(participant_ids)), key=lambda i: int(participant_ids[i])
    )
    sorted_confidence_array = padded_array[sort_indices]
    sorted_participant_ids = [participant_ids[i] for i in sort_indices][::-1]

    return sorted_confidence_array, sorted_participant_ids


def _calculate_signed_confidence(
    increase_probs: np.ndarray, threshold: float
) -> np.ndarray:
    """Calculate signed confidence values based on classification threshold."""
    signed_confidences = np.zeros_like(increase_probs)

    # Below threshold (increase predictions)
    below_mask = increase_probs < threshold
    signed_confidences[below_mask] = (
        increase_probs[below_mask] - threshold
    ) / threshold

    # Above threshold (decrease predictions)
    above_mask = increase_probs >= threshold
    signed_confidences[above_mask] = (increase_probs[above_mask] - threshold) / (
        1 - threshold
    )

    return signed_confidences


def _plot_single_heatmap(
    ax,
    probabilities,
    stimulus_seed,
    sample_duration,
    classification_threshold,
    cmap,
    stimulus_linewidth,
    stimulus_scale,
    subplot_idx,
    ncols,
    nrows,
    only_decreases,
):
    """Plot heatmap for a single stimulus seed."""
    # Process confidence data
    confidence_array, participant_ids = _process_confidence_data(
        probabilities, classification_threshold, sample_duration
    )

    # Set vmin and vmax based on only_decreases parameter
    vmin = 0 if only_decreases else -1
    vmax = 1

    # Plot heatmap
    im = ax.imshow(
        confidence_array,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(0, 180, 0, len(confidence_array)),
        alpha=0.9,
        interpolation="nearest",
    )

    # Add stimulus overlay
    _add_stimulus_overlay(
        ax, stimulus_seed, confidence_array, stimulus_linewidth, stimulus_scale
    )

    # Format axes
    _format_subplot_axes(ax, subplot_idx, ncols, nrows, participant_ids)

    return im


def _add_stimulus_overlay(ax, stimulus_seed, confidence_array, linewidth, scale):
    """Add stimulus signal overlay to heatmap."""
    stimulus = StimulusGenerator(seed=stimulus_seed, config={"sample_rate": 1}).y
    stimulus_normalized = (
        2 * ((stimulus - stimulus.min()) / (stimulus.max() - stimulus.min())) - 1
    )

    time_points = np.linspace(0, 180, confidence_array.shape[1])
    y_center = len(confidence_array) / 2
    y_amplitude = y_center * scale

    ax.plot(
        time_points,
        stimulus_normalized * y_amplitude + y_center,
        linewidth=linewidth,
        color="black",
        zorder=10,
        alpha=0.8,
    )


def _format_subplot_axes(ax, subplot_idx, ncols, nrows, participant_ids):
    """Format individual subplot axes with minimal styling."""

    # X-axis formatting (bottom row only)
    if subplot_idx >= (nrows - 1) * ncols:
        ax.set_xticks([0, 90, 180])
        ax.tick_params(axis="x", which="major", pad=2)
        # No individual subplot labels - will add figure-level label
    else:
        ax.set_xlabel("")
        ax.set_xticks([])

    # Y-axis formatting (left column only)
    if subplot_idx % ncols == 0:
        # Set y-ticks to show participant IDs
        n_participants = len(participant_ids)
        y_positions = np.arange(0.5, n_participants, 1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(participant_ids)
        ax.tick_params(axis="y", which="major", pad=2)
        # No individual subplot labels - will add figure-level label
    else:
        ax.set_ylabel("")
        ax.set_yticks([])

    # Clean styling
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def _hide_empty_subplots(axes_flat, n_used_subplots):
    """Hide unused subplot axes."""
    for idx in range(n_used_subplots, len(axes_flat)):
        axes_flat[idx].set_visible(False)


def _finalize_figure_layout(fig, sample_ax):
    """Add colorbar and adjust figure layout."""
    # Adjust subplot spacing
    fig.subplots_adjust(
        bottom=0.15,
        right=0.85,
        top=0.97,
        wspace=0.12,
        hspace=0.15,
    )

    # Add figure-level axis labels
    fig.text(0.488, 0.12, "Time (s)", ha="center", va="center")
    fig.text(0.091, 0.55, "Participant ID", ha="center", va="center", rotation=90)

    # Add colorbar
    cbar_ax = fig.add_axes([0.87, 0.2, 0.015, 0.7])
    cbar = fig.colorbar(sample_ax.images[0], cax=cbar_ax)
    cbar.set_label(
        "Prediction Confidence for Decreases",
        labelpad=8,
    )
    cbar.outline.set_linewidth(0.0)


def main():
    import json
    import logging
    import os
    from pathlib import Path

    import numpy as np
    import polars as pl
    import tomllib
    from dotenv import load_dotenv

    from src.log_config import configure_logging
    from src.models.data_loader import create_dataloaders
    from src.models.data_preparation import (
        expand_feature_list,
        prepare_data,
    )
    from src.models.utils import load_model
    from src.plots.model_inference import (
        analyze_test_dataset_for_one_stimulus,
        plot_prediction_confidence_heatmap,
    )

    configure_logging(
        stream_level=logging.DEBUG,
        ignore_libs=["matplotlib", "Comm", "bokeh", "tornado"],
    )

    config_path = Path("src/experiments/measurement/measurement_config.toml")
    with open(config_path, "rb") as file:
        config = tomllib.load(file)
    stimulus_seeds = config["stimulus"]["seeds"]
    logging.info(f"Using seeds for stimulus generation: {stimulus_seeds}")

    feature_lists = [
        # ["eda_raw", "pupil"],
        # ["eda_raw", "heart_rate"],
        # ["eda_raw", "heart_rate", "pupil"],
        # ["face"],
        ["eeg"],
    ]
    feature_lists = expand_feature_list(feature_lists)

    for feature_list in feature_lists:
        feature_list_str = "_".join(feature_list)
        # Load data from database
        df = load_data_from_database(feature_list=feature_list)

        # Load model
        json_path = Path(f"results/experiment_{feature_list_str}/results.json")
        dictionary = json.loads(json_path.read_text())
        model_path = Path(dictionary["overall_best"]["model_path"].replace("\\", "/"))

        (
            model,
            feature_list,
            sample_duration_ms,
            intervals,
            label_mapping,
            offsets_ms,
        ) = load_model(model_path, device="cpu")

        # Prepare data
        _, _, _, _, X_train_val, y_train_val, X_test, y_test = prepare_data(
            df=df,
            feature_list=feature_list,
            sample_duration_ms=sample_duration_ms,
            intervals=intervals,
            label_mapping=label_mapping,
            offsets_ms=offsets_ms,
            random_seed=RANDOM_SEED,
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
        # train data is not used in this analysis, but we need to create the dataloaders
        _, test_loader = create_dataloaders(
            X_train_val, y_train_val, X_test, y_test, batch_size=64
        )

        # Analyze the entire test dataset
        all_probabilities = {}
        all_participant_trials = {}

        for stimulus_seed in stimulus_seeds:
            probabilities, participant_trials = analyze_test_dataset_for_one_stimulus(
                df,
                model,
                feature_list,
                test_ids,
                stimulus_seed,
                sample_duration_ms,
            )

            all_probabilities[stimulus_seed] = probabilities
            all_participant_trials[stimulus_seed] = participant_trials

        # Plot all available stimuli
        fig = plot_prediction_confidence_heatmap(
            all_probabilities,
            sample_duration_ms,
            classification_threshold=0.8,
            ncols=2,
            figure_size=(7, 2),
            stimulus_scale=0.5,
            stimulus_linewidth=1.5,
            only_decreases=True,
        )

        # Save the figure
        load_dotenv()
        FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))

        fig_path = FIGURE_DIR / f"model_inference_{feature_list_str}.png"
        # fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        logging.info(f"Saved figure to {fig_path}")
        plt.show()


if __name__ == "__main__":
    main()
