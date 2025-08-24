import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src.plots.utils import FEATURE_LABELS

plt.style.use("./src/plots/style.mplstyle")


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


def plot_participant_performance_single_model(
    results_df: pl.DataFrame,
) -> None:
    # Filter and prepare data
    participant_df = results_df.filter(pl.col("participant") != "overall")
    overall_accuracy = results_df.filter(pl.col("participant") == "overall")[
        "accuracy"
    ][0]

    # Sort participants by accuracy for better visualization
    participant_df = participant_df.sort("accuracy", descending=True)

    # Create figure with appropriate dimensions for a journal column
    fig = plt.figure(figsize=(8, 5))

    # Plot with subtle, professional colors
    participants = participant_df["participant"].to_list()
    accuracies = participant_df["accuracy"].to_list()

    # Create color scheme: darker for above overall, lighter for below overall
    colors = ["#2c7bb6" for acc in accuracies]

    # Plot bars
    bars = plt.bar(participants, accuracies, color=colors, width=0.7)  # noqa

    # Add reference lines
    plt.axhline(
        y=overall_accuracy,
        color="#d73027",
        linestyle="-",
        linewidth=1.5,
        label="Overall accuracy",
        zorder=5,
    )
    plt.axhline(
        y=0.5,
        color="#969696",
        linestyle="--",
        linewidth=1,
        label="Chance level (50%)",
        zorder=5,
    )

    # Clean up aesthetics
    plt.xlabel("Participant ID")
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy by Participant")

    # Format y-axis to show percentages
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Set y-axis limits from 40% to 100% to better show differences but remain truthful
    plt.ylim([0.0, 1.0])

    # Add subtle grid lines for readability
    plt.grid(
        axis="y", linestyle="-", linewidth=0.5, color="#E0E0E0", alpha=0.7, zorder=0
    )

    # Position legend in a non-intrusive location
    plt.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1))

    # Ensure clean spacing
    plt.tight_layout()

    return fig


def plot_feature_accuracy_comparison(results_dict, figsize=(10, 6)):
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
        # Use the feature labels dictionary for better naming
        display_name = FEATURE_LABELS.get(
            feature_combo, feature_combo.replace("_", " ").title()
        )

        for row in participant_data.iter_rows(named=True):
            plot_data.append(
                {
                    "participant": f"{row['participant']}",
                    "feature_list_str": display_name,
                    "accuracy": row["accuracy"],
                }
            )

    # Convert to pandas DataFrame for seaborn
    plot_df = pd.DataFrame(plot_data)

    # Check how many participants we have
    n_participants = len(plot_df["participant"].unique())

    # PARTICIPANT-specific color palette - blues/teals for people
    participant_colors = [
        "#1f4e79",  # Navy blue
        "#2e6da4",  # Medium blue
        "#428bca",  # Light blue
        "#5bc0de",  # Cyan
        "#17a2b8",  # Teal
        "#20c997",  # Light teal
        "#6f42c1",  # Indigo
        "#007bff",  # Primary blue
        "#17a2b8",  # Info blue
        "#6c757d",  # Secondary gray
    ]

    # Use only as many colors as needed
    palette = participant_colors[:n_participants]

    # Create figure with proper size for academic papers
    fig, ax = plt.subplots(figsize=figsize)

    # Create the grouped bar plot with academic styling
    sns.barplot(
        data=plot_df,
        x="feature_list_str",
        y="accuracy",
        hue="participant",
        palette=palette,
        alpha=0.8,
        ax=ax,
        width=0.7,
    )

    # Academic styling
    ax.set_xlabel("")
    ax.set_ylabel("Classification Accuracy")

    # Rotate x-axis labels for better readability
    ax.tick_params(axis="x", rotation=45)

    # Customize legend - place outside the plot
    legend = ax.legend(
        title="Participant ID",
        frameon=True,
        fancybox=False,
        shadow=False,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    # Add chance level reference line
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, linewidth=1)

    # Clean grid styling
    ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis limits and ticks
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Tight layout for proper spacing
    plt.tight_layout()

    return fig, ax


def plot_participant_accuracy_comparison(results_dict, figsize=(10, 6)):
    """
    Plot bars for each participant across all feature combinations using academic styling.
    Args:
        results_dict: Dictionary with feature combination names as keys and polars DataFrames as values
        figsize: Tuple for figure size (width, height)
    """
    # Prepare data for seaborn (long format)
    plot_data = []
    for feature_combo, df in results_dict.items():
        # Filter out 'overall' rows
        participant_data = df.filter(pl.col("participant") != "overall")
        # Use the feature labels dictionary for better naming
        display_name = FEATURE_LABELS.get(
            feature_combo, feature_combo.replace("_", " ").title()
        )

        for row in participant_data.iter_rows(named=True):
            plot_data.append(
                {
                    "participant": f"{row['participant']}",
                    "feature_list_str": display_name,
                    "accuracy": row["accuracy"],
                }
            )

    # Convert to pandas DataFrame for seaborn
    plot_df = pd.DataFrame(plot_data)

    # Check how many feature combinations we have
    n_combinations = len(plot_df["feature_list_str"].unique())

    # FEATURE-specific color palette - warm colors for features/methods
    feature_colors = [
        "#d73027",  # Red
        "#f46d43",  # Orange-red
        "#fdae61",  # Orange
        "#fee08b",  # Yellow-orange
        "#abd9e9",  # Light blue (contrast)
        "#74add1",  # Medium blue
        "#4575b4",  # Dark blue
        "#313695",  # Navy
        "#5e3c99",  # Purple
    ]

    # Use only as many colors as needed
    palette = feature_colors[:n_combinations]

    # Create figure with proper size for academic papers
    fig, ax = plt.subplots(figsize=figsize)

    # Create the grouped bar plot with academic styling
    sns.barplot(
        data=plot_df,
        x="participant",
        y="accuracy",
        hue="feature_list_str",
        palette=palette,
        alpha=0.8,
        ax=ax,
        width=0.7,
    )

    # Academic styling
    ax.set_xlabel("Participant ID")
    ax.set_ylabel("Classification Accuracy")

    # Customize legend
    legend = ax.legend(
        title="Feature Combination",
        frameon=True,
        fancybox=False,
        shadow=False,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    # Add chance level reference line
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, linewidth=1)

    # Clean grid styling
    ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis limits and ticks
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Tight layout for proper spacing
    plt.tight_layout()

    return fig, ax
