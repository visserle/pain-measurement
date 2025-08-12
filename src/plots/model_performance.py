import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
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
    model_results: dict,
) -> plt.Figure:
    """
    Plot ROC curves for multiple models and calculate AUC scores for binary classification.
    Args:
        model_results: Dictionary where keys are model names and values are tuples of (probs, y_true)
    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Set publication-ready style
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "axes.linewidth": 1.2,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    # Academic color palette (colorblind-friendly)
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    linestyles = ["-", "-", "-", "-", "-", "-", "--", "--"]

    # Sort by AUC for consistent ordering
    sorted_results = sorted(
        model_results.items(),
        key=lambda x: auc(*roc_curve(x[1][1], x[1][0])[:2]),
        reverse=True,
    )

    for i, (model_name, (probs, y_true)) in enumerate(sorted_results):
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        auc_score = auc(fpr, tpr)

        # Use labels dictionary for clean names, fallback to original method
        clean_name = labels.get(model_name, model_name.replace("_", " ").title())
        if len(clean_name) > 20:
            clean_name = clean_name[:17] + "..."

        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]

        # Plot ROC curve
        ax.plot(
            fpr,
            tpr,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f"{clean_name} (AUC = {auc_score:.3f})",
            alpha=0.8,
        )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="Chance")

    # Professional styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")

    # Grid for better readability
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend positioning and styling
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
        edgecolor="black",
        facecolor="white",
    )
    legend.get_frame().set_linewidth(0.8)

    # Clean spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("black")

    # Set equal aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Tight layout
    plt.tight_layout()

    return fig
