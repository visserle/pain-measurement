import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("./src/plots/style.mplstyle")


def load_results(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def extract_final_test_accuracies(data: dict) -> np.ndarray:
    return np.array([run["test_accuracy"] for run in data["detailed_results"]])


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
    }


def plot_hist(
    values: list[float],
    bins: int = 20,
    vline: float | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot histogram with refined aesthetics
    ax.hist(
        values, bins=bins, color="#2171b5", edgecolor="white", alpha=0.9, linewidth=0.5
    )

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Labels
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Count")

    # Add optional vertical line
    if vline is not None:
        ax.axvline(
            vline,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
