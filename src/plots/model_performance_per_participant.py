import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader


def analyze_per_participant(
    model: nn.Module,
    test_loader: DataLoader,
    test_groups: np.ndarray,
    threshold: float = 0.5,
) -> pl.DataFrame:
    """
    Analyze model performance for each participant separately.

    Args:
        model: The trained model
        test_loader: DataLoader containing test data
        test_groups: Array of participant IDs for each sample
        threshold: Classification threshold for binary predictions

    Returns:
        Polars DataFrame with performance metrics per participant
    """
    device = next(model.parameters()).device
    model.eval()

    # Get unique participants, note that unique() sorts the array
    unique_participants = np.unique(test_groups)

    # Initialize containers for batch results
    all_true = []
    all_scores = []
    batch_index = 0
    sample_indices = []

    # Process all batches
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            batch_size = X_batch.shape[0]

            # Track indices for mapping predictions back to participants
            sample_indices.extend(range(batch_index, batch_index + batch_size))
            batch_index += batch_size

            # Forward pass
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, 1].cpu().numpy()

            # Process labels
            if len(y_batch.shape) > 1:  # one-hot encoded
                y_true = y_batch[:, 1].cpu().numpy()
            else:
                y_true = y_batch.cpu().numpy()

            all_true.append(y_true)
            all_scores.append(scores)

    # Concatenate results
    overall_y_true = np.concatenate(all_true)
    overall_scores = np.concatenate(all_scores)
    overall_y_pred = (overall_scores >= threshold).astype(int)

    # Calculate overall accuracy
    overall_acc = accuracy_score(overall_y_true, overall_y_pred)

    # Group results by participant
    participant_metrics = {}

    for participant in unique_participants:
        participant_id = str(participant)
        # Find samples belonging to this participant
        mask = test_groups == participant

        # Get true labels and predictions for this participant
        p_true = overall_y_true[mask]
        p_scores = overall_scores[mask]
        p_pred = overall_y_pred[mask]

        # Calculate metrics
        p_acc = accuracy_score(p_true, p_pred)
        class_0_count = np.sum(p_true == 0)
        class_1_count = np.sum(p_true == 1)

        # Store metrics
        participant_metrics[participant_id] = {
            "accuracy": p_acc,
            "samples": len(p_true),
            "class_distribution": {"0": int(class_0_count), "1": int(class_1_count)},
        }

    # Add overall metrics
    participant_metrics["overall"] = {
        "accuracy": overall_acc,
        "samples": len(overall_y_true),
        "class_distribution": {
            "0": int(np.sum(overall_y_true == 0)),
            "1": int(np.sum(overall_y_true == 1)),
        },
    }

    return _participant_metrics_to_df(participant_metrics)


def _participant_metrics_to_df(participant_metrics: dict) -> pl.DataFrame:
    """
    Convert participant metrics dictionary to a polars DataFrame.

    Args:
        participant_metrics: Dictionary with participant metrics from analyze_per_participant

    Returns:
        Polars DataFrame with participant metrics
    """
    # Extract data from the dictionary
    data = []

    # Process all participants (including 'overall')
    for participant_id, metrics in participant_metrics.items():
        data.append(
            {
                "participant": participant_id,
                "accuracy": metrics["accuracy"],
                "samples": metrics["samples"],
                "class_0_count": metrics["class_distribution"]["0"],
                "class_1_count": metrics["class_distribution"]["1"],
            }
        )

    # Create polars DataFrame
    df = pl.DataFrame(data)

    # Move 'overall' to the last row
    overall_row = df.filter(pl.col("participant") == "overall")
    regular_rows = df.filter(pl.col("participant") != "overall")

    # Sort participants numerically
    regular_rows = (
        regular_rows.with_columns(
            pl.col("participant").cast(pl.Int32).alias("participant_int")
        )
        .sort("participant_int")
        .drop("participant_int")
    )

    # Combine regular rows with overall at the end
    df = pl.concat([regular_rows, overall_row])

    return df


def get_summary_statistics_single_model(
    results_df: pl.DataFrame,
) -> pl.DataFrame:
    # Extract overall accuracy and create a DataFrame without the overall row
    overall_row = results_df.filter(pl.col("participant") == "overall")
    overall_accuracy = overall_row["accuracy"][0]
    participants_df = results_df.filter(pl.col("participant") != "overall")

    # Generate the table with cleaner code
    return pl.DataFrame(
        {
            "Measure": [
                "Overall accuracy",
                "Participants above chance level",
                "Participants above overall accuracy",
                "Highest accuracy (Participant ID)",
                "Lowest accuracy (Participant ID)",
            ],
            "Value": [
                f"{overall_accuracy:.1%}",
                f"{participants_df.filter(pl.col('accuracy') > 0.5).height} out of {participants_df.height}",
                f"{participants_df.filter(pl.col('accuracy') > overall_accuracy).height} out of {participants_df.height}",
                f"{participants_df['accuracy'].max():.1%} (ID: {participants_df.filter(pl.col('accuracy') == participants_df['accuracy'].max())['participant'][0]})",
                f"{participants_df['accuracy'].min():.1%} (ID: {participants_df.filter(pl.col('accuracy') == participants_df['accuracy'].min())['participant'][0]})",
            ],
        }
    )


def plot_feature_accuracy_comparison(results_dict, labels, figsize=(10, 6)):
    """
    Plot bars for each feature combination across all participants using academic styling.
    Args:
        results_dict: Dictionary with feature combination names as keys and polars DataFrames as values
        figsize: Tuple for figure size (width, height)
    """
    # Prepare data for seaborn (long format)
    plot_data = []
    for feature_combo, df in results_dict.items():
        # Filter out 'overall' rows
        participant_data = df.filter(pl.col("participant") != "overall")
        # Use the labels dictionary for better naming
        display_name = labels.get(
            feature_combo, feature_combo.replace("_", " ").title()
        )

        for row in participant_data.iter_rows(named=True):
            plot_data.append(
                {
                    "participant": f"P{row['participant']}",
                    "feature_combination": display_name,
                    "accuracy": row["accuracy"],
                }
            )

    # Convert to pandas DataFrame for seaborn
    plot_df = pd.DataFrame(plot_data)

    # Check how many participants we have
    n_participants = len(plot_df["participant"].unique())

    # Academic journal color palette - grayscale and muted colors
    # These colors work well in both color and grayscale printing
    journal_colors = [
        "#2F2F2F",  # Dark gray
        "#7F7F7F",  # Medium gray
        "#BFBFBF",  # Light gray
        "#4A90E2",  # Muted blue
        "#7ED321",  # Muted green
        "#F5A623",  # Muted orange
        "#BD10E0",  # Muted purple
        "#B8E986",  # Light green
        "#50E3C2",  # Teal
        "#F8E71C",  # Yellow
    ]

    # Use only as many colors as needed
    palette = journal_colors[:n_participants]

    # Set academic style
    plt.style.use("default")

    # Create figure with proper size for academic papers
    fig, ax = plt.subplots(figsize=figsize)

    # Create the grouped bar plot with academic styling
    sns.barplot(
        data=plot_df,
        x="feature_combination",
        y="accuracy",
        hue="participant",
        palette=palette,
        alpha=0.8,
        ax=ax,
        width=0.7,
    )

    # Academic styling
    ax.set_xlabel("", fontsize=0)
    ax.set_ylabel("Classification Accuracy", fontsize=12)

    # Rotate x-axis labels for better readability
    ax.tick_params(axis="x", rotation=45, labelsize=10)

    # Customize legend for academic style - place outside the plot
    legend = ax.legend(
        title="Participant",
        frameon=True,
        fancybox=False,
        shadow=False,
        fontsize=10,
        title_fontsize=11,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor("black")

    # Add chance level reference line
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, linewidth=1)

    # Clean grid styling
    ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis limits and ticks
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Clean up spines (academic standard)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    # Tight layout for proper spacing
    plt.tight_layout()

    return fig, ax


def plot_participant_accuracy_comparison(results_dict, labels, figsize=(10, 6)):
    """
    Plot bars for each participant across all feature combinations using academic styling.
    Args:
        results_dict: Dictionary with feature combination names as keys and polars DataFrames as values
        labels: Dictionary mapping feature combination keys to display names
        figsize: Tuple for figure size (width, height)
    """
    # Prepare data for seaborn (long format)
    plot_data = []
    for feature_combo, df in results_dict.items():
        # Filter out 'overall' rows
        participant_data = df.filter(pl.col("participant") != "overall")
        # Use the labels dictionary for better naming
        display_name = labels.get(
            feature_combo, feature_combo.replace("_", " ").title()
        )

        for row in participant_data.iter_rows(named=True):
            plot_data.append(
                {
                    "participant": f"P{row['participant']}",
                    "feature_combination": display_name,
                    "accuracy": row["accuracy"],
                }
            )

    # Convert to pandas DataFrame for seaborn
    plot_df = pd.DataFrame(plot_data)

    # Check how many feature combinations we have
    n_combinations = len(plot_df["feature_combination"].unique())

    # Academic journal color palette - grayscale and muted colors
    # These colors work well in both color and grayscale printing
    journal_colors = [
        "#2F2F2F",  # Dark gray
        "#7F7F7F",  # Medium gray
        "#BFBFBF",  # Light gray
        "#4A90E2",  # Muted blue
        "#7ED321",  # Muted green
        "#F5A623",  # Muted orange
        "#BD10E0",  # Muted purple
        "#B8E986",  # Light green
        "#50E3C2",  # Teal
    ]

    # Use only as many colors as needed
    palette = journal_colors[:n_combinations]

    # Set academic style
    plt.style.use("default")

    # Create figure with proper size for academic papers
    fig, ax = plt.subplots(figsize=figsize)

    # Create the grouped bar plot with academic styling
    sns.barplot(
        data=plot_df,
        x="participant",
        y="accuracy",
        hue="feature_combination",
        palette=palette,
        alpha=0.8,
        ax=ax,
        width=0.7,
    )

    # Academic styling
    ax.set_xlabel("Participant", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)

    # Customize legend for academic style - place outside the plot
    legend = ax.legend(
        title="Feature Combination",
        frameon=True,
        fancybox=False,
        shadow=False,
        fontsize=10,
        title_fontsize=11,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor("black")

    # Add chance level reference line
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, linewidth=1)

    # Clean grid styling
    ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis limits and ticks
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Clean up spines (academic standard)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    # Tight layout for proper spacing
    plt.tight_layout()

    return fig, ax


def main():
    import json
    import logging
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import polars as pl
    from dotenv import load_dotenv

    from src.data.database_manager import DatabaseManager
    from src.features.labels import add_labels
    from src.features.resampling import add_normalized_timestamp
    from src.log_config import configure_logging
    from src.models.data_loader import create_dataloaders
    from src.models.data_preparation import prepare_data
    from src.models.main_config import RANDOM_SEED
    from src.models.utils import load_model

    configure_logging(
        stream_level=logging.DEBUG,
        ignore_libs=["matplotlib", "Comm", "bokeh", "tornado", "filelock"],
    )

    pl.Config.set_tbl_rows(12)  # for the 12 trials

    feature_combinations = [
        "heart_rate",
        "pupil",
        "eda_raw",
        "eda_raw_pupil",
        "eda_raw_heart_rate",
        "eda_raw_heart_rate_pupil",
        "brow_furrow_cheek_raise_mouth_open_nose_wrinkle_upper_lip_raise",
        "brow_furrow_cheek_raise_eda_raw_heart_rate_mouth_open_nose_wrinkle_pupil_upper_lip_raise",
        "c3_c4_cz_f3_f4_oz_p3_p4",
    ]

    labels = {
        "eda_raw": "EDA",
        "pupil": "Pupil",
        "heart_rate": "HR",
        "eda_raw_pupil": "EDA + Pupil",
        "eda_raw_heart_rate": "EDA + HR",
        "eda_raw_heart_rate_pupil": "EDA + HR + Pupil",
        "brow_furrow_cheek_raise_mouth_open_nose_wrinkle_upper_lip_raise": "Facial Expressions",
        "brow_furrow_cheek_raise_eda_raw_heart_rate_mouth_open_nose_wrinkle_pupil_upper_lip_raise": (
            "All combined (w/o EEG)"
        ),
        "c3_c4_cz_f3_f4_oz_p3_p4": "EEG",
    }

    # Load data from database
    db = DatabaseManager()
    with db:
        df = db.get_table(
            "Merged_and_Labeled_Data",
            exclude_trials_with_measurement_problems=True,
        )

    results = {}

    for feature_combination in feature_combinations:
        if feature_combination == "c3_c4_cz_f3_f4_oz_p3_p4":
            with db:
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

        # Load model
        json_path = Path(f"results/experiment_{feature_combination}/results.json")
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
        _, test_loader = create_dataloaders(
            X_train_val, y_train_val, X_test, y_test, batch_size=64
        )

        result_df = analyze_per_participant(
            model,
            test_loader,
            test_groups,
            threshold=0.50,
        )
        results[feature_combination] = result_df

    feature_set_acc, _ = plot_feature_accuracy_comparison(
        results, labels, figsize=(10, 6)
    )
    plt.show()

    feature_set_acc_by_participant, _ = plot_participant_accuracy_comparison(
        results, labels, figsize=(13, 6)
    )
    plt.show()

    # load_dotenv()
    # FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))

    # feature_set_acc.savefig(
    #     FIGURE_DIR / "feature_set_acc.png", dpi=300, bbox_inches="tight"
    # )

    # feature_set_acc_by_participant.savefig(
    #     FIGURE_DIR / "feature_set_acc_by_participant.png", dpi=300, bbox_inches="tight"
    # )


if __name__ == "__main__":
    main()
