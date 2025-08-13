import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader


def get_model_predictions(
    model: nn.Module,
    test_loader: DataLoader,
) -> tuple:
    """
    Run model inference on test data and return raw probabilities and true labels.

    Returns:
        tuple: (probabilities, true_labels)
    """
    device = next(model.parameters()).device

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)

            # Convert logits to probabilities using softmax
            probs = torch.softmax(logits, dim=1)

            # Convert labels to numpy array
            if len(y_batch.shape) > 1:  # one-hot encoded
                y_batch = y_batch[:, 1].cpu().numpy()
            else:
                y_batch = y_batch.cpu().numpy()

            all_probs.append(
                probs[:, 1].cpu().numpy()  # store probability for class 1 ("decreases")
            )
            all_labels.append(y_batch)

    # Concatenate batches
    y_true = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    return probs, y_true


def get_confusion_matrix(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    threshold: float = 0.5,
    plot: bool = True,
) -> tuple:
    """
    Calculate confusion matrix from probabilities and true labels.

    Args:
        probabilities: Array of probabilities for the positive class
        true_labels: Array of true binary labels
        threshold: Classification threshold
    """
    # Apply threshold to get predictions
    y_pred = (probabilities >= threshold).astype(int)
    accuracy = accuracy_score(true_labels, y_pred)

    conf_matrix = confusion_matrix(true_labels, y_pred)

    if plot:
        conf_matrix_plot = _plot_confusion_matrix(conf_matrix, threshold, accuracy)
        conf_matrix_plot.show()
        return conf_matrix, conf_matrix_plot

    return conf_matrix


def _plot_confusion_matrix(
    conf_matrix: np.ndarray,
    threshold: float,
    accuracy_score: float,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3.5))

    # Create heatmap with professional styling
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Decrease", "Decrease"],
        yticklabels=["Non-Decrease", "Decrease"],
        cbar_kws={"shrink": 0.7},
        square=True,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 10},
        ax=ax,
    )

    # Professional styling for academic papers
    ax.set_title(
        f"Threshold: {threshold:.2f}, Accuracy: {accuracy_score:.2f}",
        fontsize=10,
        pad=10,
    )
    ax.set_ylabel(
        "True label",
        fontsize=10,
    )
    ax.set_xlabel(
        "Predicted label",
        fontsize=10,
    )

    # Smaller tick labels
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")

    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor("black")

    plt.tight_layout()
    return fig


def plot_single_roc_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> plt.Figure:
    """
    Plot ROC curve and calculate AUC score for binary classification.
    Returns:
    plt.Figure: The matplotlib figure object
    """
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    auc_score = auc(fpr, tpr)

    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot ROC curve with professional styling
    ax.plot(
        fpr,
        tpr,
        color="#1f77b4",
        linewidth=2,
        label=f"ROC curve (AUC = {auc_score:.3f})",
    )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1, label="Random classifier")

    # Add marker for optimal threshold
    ax.plot(
        fpr[optimal_idx],
        tpr[optimal_idx],
        "ro",
        markersize=6,
        label=f"Optimal threshold = {optimal_threshold:.2f}",
    )

    # Professional styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)

    # Legend styling
    ax.legend(loc="lower right", fontsize=8, frameon=True, fancybox=False, shadow=False)

    # Tick styling
    ax.tick_params(axis="both", which="major", labelsize=8)

    # Equal aspect ratio for square plot
    ax.set_aspect("equal")

    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor("black")

    plt.tight_layout()
    return fig


def plot_single_pr_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> tuple[plt.Figure, float, float]:
    """
    Plot Precision-Recall curve and calculate Average Precision score for binary classification.
    Returns:
    tuple: (figure, avg_precision, optimal_threshold)
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    avg_precision = average_precision_score(y_true, probs)

    # Calculate F1 score at each threshold to find optimal threshold
    f1_scores = (
        2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    )
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Calculate no-skill line (ratio of positive samples)
    no_skill = sum(y_true) / len(y_true)

    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot PR curve with professional styling
    ax.plot(
        recall,
        precision,
        color="#1f77b4",
        linewidth=2,
        label=f"PR curve (AP = {avg_precision:.3f})",
    )

    # No-skill baseline
    ax.axhline(
        y=no_skill,
        color="k",
        linestyle="--",
        alpha=0.6,
        linewidth=1,
        label=f"No skill baseline = {no_skill:.2f}",
    )

    # Add marker for optimal threshold
    ax.plot(
        recall[optimal_idx],
        precision[optimal_idx],
        "ro",
        markersize=6,
        label=f"Optimal (F1 = {f1_scores[optimal_idx]:.2f})",
    )

    # Styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)

    # Legend styling
    ax.legend(loc="lower left", fontsize=8, frameon=True, fancybox=False, shadow=False)

    # Tick styling
    ax.tick_params(axis="both", which="major", labelsize=8)

    # Equal aspect ratio for square plot
    ax.set_aspect("equal")

    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor("black")

    plt.tight_layout()
    return (fig, avg_precision, optimal_threshold)


def plot_multiple_roc_curves(
    results: dict,
    labels: dict = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiple models and calculate AUC scores for binary classification.
    Args:
        results: Dictionary where keys are model names and values are tuples of (probs, y_true)
        labels: Optional dictionary for clean model names
    Returns:
        plt.Figure: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(6, 4))  # Wider to accommodate external legend

    # Simple color palette
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#000080",
    ]

    for i, (feature_set, (probs, y_true)) in enumerate(results.items()):
        fpr, tpr, thresholds = roc_curve(y_true, probs)

        # Use labels dictionary for clean names
        clean_name = (
            labels.get(feature_set, feature_set.replace("_", " ").title())
            if labels
            else feature_set.replace("_", " ").title()
        )

        color = colors[i % len(colors)]

        # Plot ROC curve with professional styling
        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"{clean_name}",
        )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1, label="Random classifier")

    # Professional styling (matching single plot)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)

    # Legend with smaller font and tighter spacing
    ax.legend(
        loc="lower right",
        fontsize=7,
        frameon=True,
        fancybox=False,
        shadow=False,
        handlelength=1.5,
        handletextpad=0.3,
        labelspacing=0.3,
    )
    # Tick styling
    ax.tick_params(axis="both", which="major", labelsize=8)

    # Equal aspect ratio for square plot
    ax.set_aspect("equal")

    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor("black")

    plt.tight_layout()
    return fig


def plot_multiple_pr_curves(
    results: dict,
    labels: dict = None,
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple models and calculate Average Precision scores for binary classification.
    Args:
        results: Dictionary where keys are model names and values are tuples of (probs, y_true)
        labels: Optional dictionary for clean model names
    Returns:
        plt.Figure: The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(6, 4))  # Wider to accommodate external legend

    # Simple color palette (same as ROC curves for consistency)
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#000080",
    ]

    # Calculate no-skill baseline from first dataset
    first_key = next(iter(results.keys()))
    _, first_y_true = results[first_key]
    no_skill = sum(first_y_true) / len(first_y_true)

    for i, (feature_set, (probs, y_true)) in enumerate(results.items()):
        precision, recall, thresholds = precision_recall_curve(y_true, probs)

        # Use labels dictionary for clean names
        clean_name = (
            labels.get(feature_set, feature_set.replace("_", " ").title())
            if labels
            else feature_set.replace("_", " ").title()
        )

        color = colors[i % len(colors)]

        # Plot PR curve with professional styling
        ax.plot(
            recall,
            precision,
            color=color,
            linewidth=2,
            label=f"{clean_name}",
        )

    # No-skill baseline
    ax.axhline(
        y=no_skill,
        color="k",
        linestyle="--",
        alpha=0.6,
        linewidth=1,
        label="No skill baseline",
    )

    # Professional styling (matching single plot and ROC curves)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)

    # Legend with smaller font and tighter spacing
    ax.legend(
        loc="lower right",
        fontsize=7,
        frameon=True,
        fancybox=False,
        shadow=False,
        handlelength=1.5,
        handletextpad=0.3,
        labelspacing=0.3,
    )

    # Tick styling
    ax.tick_params(axis="both", which="major", labelsize=8)

    # Equal aspect ratio for square plot
    ax.set_aspect("equal")

    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_edgecolor("black")

    plt.tight_layout()
    return fig


def calculate_performance_metrics(
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Calculate performance metrics from probabilities and true labels.
    """
    # Apply threshold to get predictions
    y_pred = (probabilities >= threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, y_pred).ravel()

    # Calculate metrics
    accuracy = accuracy_score(true_labels, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(true_labels, y_pred)
    mcc = matthews_corrcoef(true_labels, y_pred)

    # Calculate AUC and Average Precision
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    auc_score = auc(fpr, tpr)
    avg_precision = average_precision_score(true_labels, probabilities)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "mcc": mcc,
        "auroc": auc_score,
        "auprc": avg_precision,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def create_performance_table(
    results: dict,
    winning_models: dict = None,
    labels: dict = None,
    threshold: float = 0.5,
    sort_by: str = "AUC",
) -> pl.DataFrame:
    """
    Create a publication-ready performance table for multiple models using polars.
    Args:
        results: Dictionary where keys are model names and values are tuples of (probs, y_true)
        winning_models: Dictionary mapping feature combinations to winning model names
        labels: Optional dictionary for clean model names
        threshold: Classification threshold
        sort_by: Metric to sort by (default: "AUC")
    Returns:
        pl.DataFrame: Performance metrics table formatted for academic publishing
    """
    performance_data = []

    for feature_set, (probs, y_true) in results.items():
        metrics = calculate_performance_metrics(probs, y_true, threshold)
        winning_model = winning_models.get(feature_set, {}).get(feature_set, "")

        # Use labels dictionary for clean names
        feature_set = (
            labels.get(feature_set, feature_set.replace("_", " ").title())
            if labels
            else feature_set.replace("_", " ").title()
        )

        performance_data.append(
            {
                "Feature Set": feature_set,
                "Winning Model": winning_model,
                "Accuracy": metrics["accuracy"],
                "Sensitivity (Recall)": metrics["sensitivity"],
                "Specificity": metrics["specificity"],
                "Precision (PPV)": metrics["precision"],
                "F₁-Score": metrics["f1_score"],
                "MCC": metrics["mcc"],
                "AUROC": metrics["auroc"],
                "AUPRC": metrics["auprc"],
            }
        )

    df = pl.DataFrame(performance_data)

    # Sort by specified metric (descending)
    if sort_by in df.columns:
        df = df.sort(sort_by, descending=True)

    # Round all numeric columns to 3 decimal places
    df = df.with_columns(pl.col(pl.Float64, pl.Float32).round(3))

    return df


def main():
    import json
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
    from src.plots.model_performance import (
        get_model_predictions,
        plot_multiple_roc_curves,
    )

    configure_logging(stream=True, ignore_libs=["matplotlib"])

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

    results = {}
    winning_models = {}

    for feature_combination in feature_combinations:
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

        (
            model,
            feature_list,
            sample_duration_ms,
            intervals,
            label_mapping,
            offsets_ms,
        ) = load_model(model_path, device="cpu")
        winning_models[feature_combination] = {
            feature_combination: model.__class__.__name__
        }

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
        _, test_loader = create_dataloaders(
            X_train_val, y_train_val, X_test, y_test, batch_size=64
        )
        probs, y_true = get_model_predictions(
            model,
            test_loader,
        )
        results[feature_combination] = (probs, y_true)

    roc_curves = plot_multiple_roc_curves(results, labels=labels)
    plt.show()
    pr_curves = plot_multiple_pr_curves(results, labels=labels)
    plt.show()

    # Create performance table
    performance_df = create_performance_table(
        results, winning_models=winning_models, labels=labels, threshold=0.5
    )

    # Save the figure
    load_dotenv()
    FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))

    roc_curves.savefig(FIGURE_DIR / "roc_curves.png", dpi=300, bbox_inches="tight")
    pr_curves.savefig(FIGURE_DIR / "pr_curves.png", dpi=300, bbox_inches="tight")
    performance_df.write_json(FIGURE_DIR / "performance_table.json")


if __name__ == "__main__":
    main()
