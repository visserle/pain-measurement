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
    device = next(model.parameters()).device.type
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
                probs[:, 1].cpu().numpy()
            )  # Store positive class probability
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
) -> np.ndarray:
    """
    Calculate confusion matrix from probabilities and true labels.

    Args:
        probabilities: Array of probabilities for the positive class
        true_labels: Array of true binary labels
        threshold: Classification threshold
    """
    # Apply threshold to get predictions
    y_pred = (probabilities >= threshold).astype(int)

    print(f"Accuracy: {accuracy_score(true_labels, y_pred):.3f}")

    conf_matrix = confusion_matrix(true_labels, y_pred)

    if plot:
        plot_confusion_matrix(conf_matrix)

    return conf_matrix


def plot_confusion_matrix(conf_matrix: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Increases", "Decreases"],
        yticklabels=["Increases", "Decreases"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def plot_roc_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> None:
    """
    Plot ROC curve and calculate AUC score for binary classification.
    """

    fpr, tpr, thresholds = roc_curve(y_true, probs)
    auc_score = auc(fpr, tpr)

    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--")  # diagonal line

    # Add marker for optimal threshold
    plt.plot(
        fpr[optimal_idx],
        tpr[optimal_idx],
        "ro",
        label=f"Optimal (threshold={optimal_threshold:.2f})",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_pr_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> tuple:
    """
    Plot Precision-Recall curve and calculate Average Precision score for binary classification.
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

    # Plot precision-recall curve
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"PR curve (AP = {avg_precision:.3f})")

    # Add marker for optimal threshold
    plt.plot(
        recall[optimal_idx],
        precision[optimal_idx],
        "ro",
        label=f"Optimal (threshold={optimal_threshold:.2f}, F1={f1_scores[optimal_idx]:.2f})",
    )

    # Calculate no-skill line (ratio of positive samples)
    no_skill = sum(y_true) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], "k--", label=f"No Skill (y={no_skill:.2f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()

    return (avg_precision, optimal_threshold)
