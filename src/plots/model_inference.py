import logging
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from joblib import Memory
from matplotlib.colors import LinearSegmentedColormap

from src.experiments.measurement.stimulus_generator import StimulusGenerator
from src.models.data_loader import transform_sample_df_to_arrays
from src.models.data_preparation import EEG_FEATURES, prepare_data
from src.models.main_config import INTERVALS, LABEL_MAPPING, OFFSETS_MS, RANDOM_SEED
from src.models.scalers import StandardScaler3D

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])

memory = Memory(".cache/stimulus", verbose=0)

plt.style.use("./src/plots/style.mplstyle")

TOTAL_DURATION_MS = 180000
TOTAL_DURATION_S = TOTAL_DURATION_MS // 1000
DEFAULT_RATING_CI_LEVEL = 0.95


def analyze_test_dataset_for_one_stimulus(
    df: pl.DataFrame,
    model: torch.nn.Module,
    features: list,
    participant_ids: list,
    stimulus_seed: int,
    sample_duration: int,
    step_size: int = 1000,
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

    # Process each participant
    for participant_id in participant_ids:
        # Get data for this participant and stimulus seed
        participant_df = df.filter(participant_id=participant_id).filter(
            stimulus_seed=stimulus_seed
        )

        if participant_df.is_empty():
            continue

        # Create a proper key for the participant (use the actual ID, not the index)
        participant_key = str(participant_id)

        participant_probabilities = []

        # Create samples for this trial
        samples_df = _create_samples_full_stimulus(
            participant_df, features, sample_duration, step_size
        )

        if samples_df is None or samples_df.is_empty():
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
    min_time, max_time = 0.0, float(TOTAL_DURATION_MS)
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
        return None

    # Combine all samples into one DataFrame
    result_df = pl.concat(sample_dfs)

    return result_df.select(["sample_id"] + features)


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
    step_size: int = 1000,
    ratings_df: pl.DataFrame | None = None,
    show_actual_rating_ci: bool = False,
    rating_confidence_level: float = DEFAULT_RATING_CI_LEVEL,
    rating_linewidth: float = 1.2,
    rating_scale: float = 0.25,
    rating_color: str = "#2ca25f",
    rating_ci_alpha: float = 0.12,
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
        ratings_df: Optional dataframe with measured ratings (e.g., Feature_Data/Feature_Stimulus).
        show_actual_rating_ci: Whether to overlay the mean rating trajectory and CI.
        rating_confidence_level: Confidence level used for CI (e.g., 0.95).
        rating_linewidth: Line width for the rating mean overlay.
        rating_scale: Vertical scale factor for the rating overlay.
        rating_color: Line/fill color for the rating overlay.
        rating_ci_alpha: Alpha for rating CI fill.
    """
    # Validate inputs
    _validate_inputs(sample_duration, all_probabilities, seeds_to_plot, step_size)

    # Setup plotting parameters
    seeds_to_plot = _get_seeds_to_plot(all_probabilities, seeds_to_plot)

    # Get all unique participant IDs across all seeds
    all_participant_ids = _get_all_participant_ids(all_probabilities, seeds_to_plot)
    nrows = (len(seeds_to_plot) + ncols - 1) // ncols

    rating_ci_by_seed = {}
    if show_actual_rating_ci:
        if ratings_df is None:
            logger.warning(
                "show_actual_rating_ci=True but ratings_df is None. Skipping rating CI overlay."
            )
        else:
            rating_ci_by_seed = compute_actual_rating_ci_by_seed(
                ratings_df=ratings_df,
                all_probabilities=all_probabilities,
                seeds_to_plot=seeds_to_plot,
                step_size=step_size,
                confidence_level=rating_confidence_level,
            )

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
            nrows,
            only_decreases,
            all_participant_ids,
            step_size,
            rating_ci_by_seed.get(stimulus_seed),
            rating_linewidth,
            rating_scale,
            rating_color,
            rating_ci_alpha,
        )

    # Hide unused subplots
    _hide_empty_subplots(axes.flatten(), len(seeds_to_plot))

    # Add colorbar and finalize layout
    _finalize_figure_layout(fig, axes.flatten()[0])

    return fig


def _validate_inputs(
    sample_duration: int,
    all_probabilities: dict,
    seeds_to_plot: list | None,
    step_size: int = 1000,  # Add step_size parameter
):
    """Validate input parameters."""
    if sample_duration % step_size:
        raise ValueError(
            f"Sample duration must be a multiple of {step_size} milliseconds."
        )

    if TOTAL_DURATION_MS % step_size:
        raise ValueError(
            f"Step size {step_size} must evenly divide {TOTAL_DURATION_MS} ms (180 seconds)."
        )


def _get_all_participant_ids(all_probabilities: dict, seeds_to_plot: list) -> list:
    """Get sorted list of all unique participant IDs across all seeds."""
    all_participants = set()
    for seed in seeds_to_plot:
        if seed in all_probabilities:
            all_participants.update(all_probabilities[seed].keys())

    # Sort participant IDs as integers
    return sorted(all_participants, key=lambda x: int(x))


def _get_seeds_to_plot(
    all_probabilities: dict,
    seeds_to_plot: list | None,
) -> list:
    """Get list of seeds to plot, validating availability."""
    available_seeds = sorted(all_probabilities.keys())

    if seeds_to_plot is None:
        return available_seeds

    valid_seeds = [s for s in seeds_to_plot if s in available_seeds]
    if not valid_seeds:
        raise ValueError("None of the specified seeds are available in the data.")

    return valid_seeds


def _create_figure_and_axes(
    seeds_to_plot: list,
    ncols: int,
    figure_size: tuple,
):
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


def compute_actual_rating_ci_by_seed(
    ratings_df: pl.DataFrame,
    all_probabilities: dict,
    seeds_to_plot: list | None = None,
    step_size: int = 1000,
    confidence_level: float = DEFAULT_RATING_CI_LEVEL,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Compute participant-level rating mean and confidence intervals for each seed.

    The participant subset is inferred from ``all_probabilities`` to match the test set.
    """
    if ratings_df is None or ratings_df.is_empty():
        return {}

    if not 0 < confidence_level < 1:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}."
        )
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}.")
    if TOTAL_DURATION_MS % step_size:
        raise ValueError(
            f"step_size {step_size} must evenly divide {TOTAL_DURATION_MS} ms."
        )

    seeds = (
        _get_seeds_to_plot(all_probabilities, seeds_to_plot)
        if seeds_to_plot is not None
        else sorted(all_probabilities.keys())
    )

    prepared_df, rating_column = _prepare_ratings_df(ratings_df)
    time_points_ms = np.arange(0, TOTAL_DURATION_MS, step_size)
    rating_ci_by_seed = {}

    for seed in seeds:
        participant_ids = [int(pid) for pid in all_probabilities.get(seed, {}).keys()]
        if not participant_ids:
            continue

        seed_df = prepared_df.filter(
            (pl.col("stimulus_seed") == seed)
            & (pl.col("participant_id").is_in(participant_ids))
        )
        if seed_df.is_empty():
            logger.warning(
                "No rating data found for seed %s and test participants. Skipping overlay for this seed.",
                seed,
            )
            continue

        participant_matrix = _build_participant_rating_matrix(
            seed_df=seed_df,
            rating_column=rating_column,
            time_points_ms=time_points_ms,
        )
        if participant_matrix.size == 0:
            continue

        mean, ci_lower, ci_upper, sample_size = _compute_mean_ci(
            participant_matrix,
            confidence_level=confidence_level,
        )
        rating_ci_by_seed[seed] = {
            "mean": mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "sample_size": sample_size,
            "time_points_s": time_points_ms / 1000,
        }

    return rating_ci_by_seed


def _prepare_ratings_df(ratings_df: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    """
    Normalize rating dataframe schema for CI computation.

    Accepts both Feature_Data and Feature_Stimulus-like schemas.
    """
    df = ratings_df
    rating_column = _get_rating_column(df)

    if "stimulus_seed" not in df.columns:
        if "trial_id" not in df.columns:
            raise ValueError(
                "ratings_df must include either 'stimulus_seed' or 'trial_id'."
            )
        # Import lazily to avoid unnecessary DB initialization in standard plotting path.
        from src.data.database_manager import DatabaseManager

        db = DatabaseManager()
        with db:
            trial_seed_map = db.get_trials(
                "Trials_Info",
                exclude_problematic=True,
            ).select(["trial_id", "stimulus_seed"])
        df = df.join(trial_seed_map, on="trial_id", how="left")
        df = df.drop_nulls("stimulus_seed")

    if "normalized_timestamp" not in df.columns:
        if "timestamp" not in df.columns:
            raise ValueError(
                "ratings_df must include either 'normalized_timestamp' or 'timestamp'."
            )

        if "trial_id" in df.columns:
            group_cols = "trial_id"
        else:
            group_cols = ["participant_id", "stimulus_seed"]

        df = df.with_columns(
            (
                pl.col("timestamp") - pl.col("timestamp").min().over(group_cols)
            ).alias("normalized_timestamp")
        )

    # Ensure ratings are in [0, 1] for consistent overlay scaling.
    max_rating = df.select(pl.col(rating_column).max()).item()
    if max_rating is not None and max_rating > 1.5:
        df = df.with_columns((pl.col(rating_column) / 100).alias(rating_column))

    return (
        df.select(
            [
                "participant_id",
                "stimulus_seed",
                "normalized_timestamp",
                rating_column,
            ]
            + (["trial_id"] if "trial_id" in df.columns else [])
        ),
        rating_column,
    )


def _get_rating_column(ratings_df: pl.DataFrame) -> str:
    """Pick rating column from known variants."""
    if "pain_rating" in ratings_df.columns:
        return "pain_rating"
    if "rating" in ratings_df.columns:
        return "rating"
    raise ValueError("ratings_df must contain either 'pain_rating' or 'rating'.")


def _build_participant_rating_matrix(
    seed_df: pl.DataFrame,
    rating_column: str,
    time_points_ms: np.ndarray,
) -> np.ndarray:
    """Interpolate each participant's rating signal to a common time grid."""
    participant_to_trials = {}
    group_cols = ["participant_id"]
    if "trial_id" in seed_df.columns:
        group_cols.append("trial_id")

    for trial_df in seed_df.partition_by(group_cols, maintain_order=True):
        participant_id = int(trial_df["participant_id"][0])
        interpolated = _interpolate_to_time_grid(
            time_ms=trial_df["normalized_timestamp"].to_numpy(),
            values=trial_df[rating_column].to_numpy(),
            target_time_points_ms=time_points_ms,
        )
        if interpolated is None:
            continue
        participant_to_trials.setdefault(participant_id, []).append(interpolated)

    if not participant_to_trials:
        return np.array([])

    participant_signals = []
    for participant_id in sorted(participant_to_trials.keys()):
        trial_signals = np.vstack(participant_to_trials[participant_id])
        participant_signals.append(np.nanmean(trial_signals, axis=0))

    return np.vstack(participant_signals)


def _interpolate_to_time_grid(
    time_ms: np.ndarray,
    values: np.ndarray,
    target_time_points_ms: np.ndarray,
) -> np.ndarray | None:
    """Interpolate signal to a fixed time grid."""
    valid_mask = np.isfinite(time_ms) & np.isfinite(values)
    if np.count_nonzero(valid_mask) < 2:
        return None

    time_ms = np.asarray(time_ms[valid_mask], dtype=float)
    values = np.asarray(values[valid_mask], dtype=float)

    sort_idx = np.argsort(time_ms)
    time_ms = time_ms[sort_idx]
    values = values[sort_idx]

    # Drop duplicate timestamps to keep interpolation stable.
    unique_time, unique_idx = np.unique(time_ms, return_index=True)
    unique_values = values[unique_idx]
    if unique_time.size < 2:
        return None

    return np.interp(
        target_time_points_ms,
        unique_time,
        unique_values,
        left=np.nan,
        right=np.nan,
    )


def _compute_mean_ci(
    participant_matrix: np.ndarray,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and two-sided CI from participant-level signals."""
    sample_size = np.sum(np.isfinite(participant_matrix), axis=0)
    mean = np.nanmean(participant_matrix, axis=0)
    std = np.nanstd(participant_matrix, axis=0, ddof=0)

    sem = np.full_like(mean, np.nan, dtype=float)
    valid_sem = sample_size > 1
    # Apply Bessel's correction for the sample standard deviation.
    std[valid_sem] *= np.sqrt(sample_size[valid_sem] / (sample_size[valid_sem] - 1))
    sem[valid_sem] = std[valid_sem] / np.sqrt(sample_size[valid_sem])

    z_score = NormalDist().inv_cdf((1 + confidence_level) / 2)
    ci_lower = mean - z_score * sem
    ci_upper = mean + z_score * sem

    single_sample = sample_size == 1
    ci_lower[single_sample] = mean[single_sample]
    ci_upper[single_sample] = mean[single_sample]

    return mean, ci_lower, ci_upper, sample_size


def _process_confidence_data(
    probabilities: dict,
    classification_threshold: float,
    sample_duration: int,
    all_participant_ids: list | None = None,
    step_size: int = 1000,
):
    """Process raw probabilities into signed confidence values."""
    # If no complete participant list provided, use only available participants
    if all_participant_ids is None:
        all_participant_ids = sorted(probabilities.keys(), key=lambda x: int(x))

    # Calculate total number of time points based on step_size
    num_time_points = TOTAL_DURATION_MS // step_size

    # Calculate how many steps to skip at the beginning (padding)
    padding_steps = sample_duration // step_size

    # Create a mapping of participant_id to confidence array
    confidence_map = {}

    for participant_id, trial_probabilities_list in probabilities.items():
        for trial_probabilities in trial_probabilities_list:
            decrease_probs = np.array([probs[1] for probs in trial_probabilities])
            signed_confidences = _calculate_signed_confidence(
                decrease_probs, classification_threshold
            )

            padded_array = np.zeros(num_time_points)
            num_predictions = len(signed_confidences)
            start_idx = padding_steps
            end_idx = start_idx + num_predictions

            if end_idx > num_time_points:
                signed_confidences = signed_confidences[: num_time_points - start_idx]
                end_idx = num_time_points

            padded_array[start_idx:end_idx] = signed_confidences
            confidence_map[participant_id] = padded_array

    # Build the final array with all participants, using NaN for missing data
    confidence_array = []
    sorted_participant_ids = []

    for participant_id in all_participant_ids:
        sorted_participant_ids.append(participant_id)
        if participant_id in confidence_map:
            confidence_array.append(confidence_map[participant_id])
        else:
            # Use NaN for missing participants (will be displayed as grey)
            confidence_array.append(np.full(num_time_points, np.nan))

    sorted_confidence_array = np.array(confidence_array)

    return sorted_confidence_array, sorted_participant_ids


def _calculate_signed_confidence(
    decrease_probs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Calculate signed confidence values based on classification threshold."""
    signed_confidences = np.zeros_like(decrease_probs)

    # Below threshold (increase predictions)
    below_mask = decrease_probs < threshold
    signed_confidences[below_mask] = (
        decrease_probs[below_mask] - threshold
    ) / threshold

    # Above threshold (decrease predictions)
    above_mask = decrease_probs >= threshold
    signed_confidences[above_mask] = (decrease_probs[above_mask] - threshold) / (
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
    all_participant_ids=None,
    step_size=1000,
    rating_ci=None,
    rating_linewidth=1.2,
    rating_scale=0.25,
    rating_color="#2ca25f",
    rating_ci_alpha=0.12,
):
    """Plot heatmap for a single stimulus seed."""
    # Process confidence data with complete participant list and step_size
    confidence_array, participant_ids = _process_confidence_data(
        probabilities,
        classification_threshold,
        sample_duration,
        all_participant_ids,
        step_size,
    )

    # Create a masked array to handle NaN values (missing participants)
    masked_array = np.ma.masked_invalid(confidence_array)

    # Set vmin and vmax based on only_decreases parameter
    vmin = 0 if only_decreases else -1
    vmax = 1

    # Plot heatmap with grey color for masked (missing) values
    cmap.set_bad(color="#f0f0f0")  # Very light grey, almost white

    im = ax.imshow(
        masked_array,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(0, TOTAL_DURATION_S, 0, len(confidence_array)),
        alpha=0.9,
        interpolation="nearest",
        origin="lower",
    )

    # Add stimulus overlay
    _add_stimulus_overlay(
        ax,
        stimulus_seed,
        confidence_array,
        stimulus_linewidth,
        stimulus_scale,
        step_size,
    )
    _add_rating_ci_overlay(
        ax,
        confidence_array=confidence_array,
        rating_ci=rating_ci,
        linewidth=rating_linewidth,
        scale=rating_scale,
        color=rating_color,
        ci_alpha=rating_ci_alpha,
    )

    # Format axes
    _format_subplot_axes(ax, subplot_idx, ncols, nrows, participant_ids)

    return im


@memory.cache
def _get_cached_stimulus(stimulus_seed: int, sample_rate: int = 1) -> np.ndarray:
    """Get cached normalized stimulus signal."""
    stimulus = StimulusGenerator(
        seed=stimulus_seed, config={"sample_rate": sample_rate}
    ).y
    return 2 * ((stimulus - stimulus.min()) / (stimulus.max() - stimulus.min())) - 1


def _add_stimulus_overlay(
    ax,
    stimulus_seed,
    confidence_array,
    linewidth,
    scale,
    step_size=1000,  # Add step_size parameter
):
    """Add stimulus signal overlay to heatmap."""
    # Get stimulus at appropriate sample rate based on step_size
    sample_rate = 1000 // step_size  # Convert step_size to sample_rate
    stimulus_normalized = _get_cached_stimulus(stimulus_seed, sample_rate)

    # Ensure stimulus matches the confidence array length
    if len(stimulus_normalized) != confidence_array.shape[1]:
        # Resample stimulus to match confidence array length
        from scipy import signal

        stimulus_normalized = signal.resample(
            stimulus_normalized, confidence_array.shape[1]
        )

    time_points = np.linspace(0, TOTAL_DURATION_S, confidence_array.shape[1])
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


def _add_rating_ci_overlay(
    ax,
    confidence_array: np.ndarray,
    rating_ci: dict[str, np.ndarray] | None,
    linewidth: float,
    scale: float,
    color: str,
    ci_alpha: float,
):
    """Add actual rating trajectory with confidence interval as overlay."""
    if not rating_ci:
        return

    mean = rating_ci["mean"]
    ci_lower = rating_ci["ci_lower"]
    ci_upper = rating_ci["ci_upper"]
    time_points = rating_ci.get("time_points_s")
    if time_points is None:
        time_points = np.linspace(0, TOTAL_DURATION_S, len(mean))

    if mean.size != confidence_array.shape[1]:
        logger.warning(
            "Rating CI length (%s) does not match heatmap time points (%s). Skipping rating overlay.",
            mean.size,
            confidence_array.shape[1],
        )
        return

    y_center = len(confidence_array) / 2
    y_amplitude = y_center * scale

    # Convert [0, 1] ratings to [-1, 1] for centered overlay.
    mean_scaled = (2 * mean - 1) * y_amplitude + y_center
    lower_scaled = (2 * ci_lower - 1) * y_amplitude + y_center
    upper_scaled = (2 * ci_upper - 1) * y_amplitude + y_center

    y_min = 0
    y_max = len(confidence_array)
    mean_scaled = np.clip(mean_scaled, y_min, y_max)
    lower_scaled = np.clip(lower_scaled, y_min, y_max)
    upper_scaled = np.clip(upper_scaled, y_min, y_max)

    valid_fill = (
        np.isfinite(time_points) & np.isfinite(lower_scaled) & np.isfinite(upper_scaled)
    )
    if np.any(valid_fill):
        ax.fill_between(
            time_points[valid_fill],
            lower_scaled[valid_fill],
            upper_scaled[valid_fill],
            color=color,
            alpha=ci_alpha,
            linewidth=0,
            zorder=8,
        )

    valid_line = np.isfinite(time_points) & np.isfinite(mean_scaled)
    if np.any(valid_line):
        ax.plot(
            time_points[valid_line],
            mean_scaled[valid_line],
            linewidth=linewidth,
            color=color,
            alpha=0.95,
            zorder=9,
        )


def _format_subplot_axes(
    ax,
    subplot_idx,
    ncols,
    nrows,
    participant_ids,
):
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
        ax.set_yticklabels(participant_ids)  # Keep original order
        ax.tick_params(axis="y", which="major", pad=2)
        # No individual subplot labels - will add figure-level label
    else:
        ax.set_ylabel("")
        ax.set_yticks([])

    # Clean styling
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def _hide_empty_subplots(
    axes_flat,
    n_used_subplots,
):
    """Hide unused subplot axes."""
    for idx in range(n_used_subplots, len(axes_flat)):
        axes_flat[idx].set_visible(False)


def _finalize_figure_layout(
    fig,
    sample_ax,
):
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
