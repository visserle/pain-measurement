import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from matplotlib import rcParams
from sklearn.metrics import (
    accuracy_score,
)


def analyze_per_participant(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_groups: np.ndarray,
    batch_size: int = 64,
    threshold: float = 0.5,
    pseudonymize: bool = True,
) -> pl.DataFrame:
    """
    Analyze model performance for each participant separately.

    Args:
        model: The trained model
        X_test: Test features
        y_test: Test labels
        test_groups: Array of participant IDs for each sample
        batch_size: Batch size for processing
        threshold: Classification threshold

    Returns:
        Dictionary with participant IDs as keys and dictionaries of metrics as values
    """
    device = next(model.parameters()).device.type
    model.eval()

    # Get unique participants
    unique_participants = np.unique(test_groups)  # note that unique is sorted
    participant_metrics = {}

    # First, evaluate on the entire test set to get the overall performance
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    all_true = []
    all_pred = []
    all_scores = []

    # Create DataLoader for all test data
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, 1].cpu().numpy()

            # Get predicted labels
            y_pred_batch = (scores >= threshold).astype(int)

            # Convert labels to numpy array
            if len(y_batch.shape) > 1:  # one-hot encoded
                y_batch = y_batch[:, 1].cpu().numpy()
            else:
                y_batch = y_batch.cpu().numpy()

            all_true.append(y_batch)
            all_pred.append(y_pred_batch)
            all_scores.append(scores)

    # Concatenate all results
    overall_y_true = np.concatenate(all_true)
    overall_y_pred = np.concatenate(all_pred)
    overall_scores = np.concatenate(all_scores)

    # Calculate overall metrics
    overall_acc = accuracy_score(overall_y_true, overall_y_pred)

    # Store the scores and true labels for each participant
    participant_true = {}
    participant_scores = {}

    for i, (true_label, score, group) in enumerate(
        zip(overall_y_true, overall_scores, test_groups)
    ):
        if str(group) not in participant_true:
            participant_true[str(group)] = []
            participant_scores[str(group)] = []

        participant_true[str(group)].append(true_label)
        participant_scores[str(group)].append(score)

    # Now calculate metrics for each participant
    for idx, participant in enumerate(unique_participants):
        participant_id = str(participant)
        participant_id_pseud = str(idx + 1)

        # Get true labels and predicted scores for this participant
        p_true = np.array(participant_true[participant_id])
        p_scores = np.array(participant_scores[participant_id])
        p_pred = (p_scores >= threshold).astype(int)

        # Count samples for each class
        class_0_count = np.sum(p_true == 0)
        class_1_count = np.sum(p_true == 1)

        # Calculate accuracy
        p_acc = accuracy_score(p_true, p_pred)

        # Store metrics
        participant_metrics[
            participant_id_pseud if pseudonymize else participant_id
        ] = {
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


def get_summary_statistics(
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


def plot_participant_performance(
    results_df: pl.DataFrame,
) -> None:
    # Set publication-quality parameters
    rcParams["font.family"] = "Arial"
    rcParams["font.size"] = 10
    rcParams["axes.linewidth"] = 1
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False

    # Filter and prepare data
    participant_df = results_df.filter(pl.col("participant") != "overall")
    overall_accuracy = results_df.filter(pl.col("participant") == "overall")[
        "accuracy"
    ][0]

    # Sort participants by accuracy for better visualization
    participant_df = participant_df.sort("accuracy", descending=True)

    # Create figure with appropriate dimensions for a journal column
    plt.figure(figsize=(8, 5))

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
    plt.title("Classification Accuracy by Participant", fontsize=12)

    # Format y-axis to show percentages
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Set y-axis limits from 40% to 100% to better show differences but remain truthful
    plt.ylim([0.4, 1.0])

    # Add subtle grid lines for readability
    plt.grid(
        axis="y", linestyle="-", linewidth=0.5, color="#E0E0E0", alpha=0.7, zorder=0
    )

    # Add a small gap between axis and first tick to improve appearance
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)

    # Position legend in a non-intrusive location
    plt.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1))

    # Ensure clean spacing
    plt.tight_layout()

    # Save in high resolution for publication (300 dpi is journal standard)
    plt.savefig("participant_accuracy.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("participant_accuracy.png", dpi=300, bbox_inches="tight")

    plt.show()
