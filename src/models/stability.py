import json
import logging
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.models.data_loader import create_dataloaders
from src.models.data_preparation import prepare_data
from src.models.evaluation import evaluate_model
from src.models.training_loop import train_model
from src.models.utils import get_input_shape, initialize_model, set_seed

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


class StabilityResults:
    """Container for stability evaluation results (variance across random seeds)."""

    def __init__(self, results: list[dict]):
        self.results = results
        self.accuracies = [r["test_accuracy"] for r in results]
        self.losses = [r["test_loss"] for r in results]

    @property
    def mean_accuracy(self) -> float:
        return np.mean(self.accuracies)

    @property
    def std_accuracy(self) -> float:
        return np.std(self.accuracies)

    @property
    def mean_loss(self) -> float:
        return np.mean(self.losses)

    @property
    def std_loss(self) -> float:
        return np.std(self.losses)

    @property
    def accuracy_range(self) -> tuple[float, float]:
        return (np.min(self.accuracies), np.max(self.accuracies))

    def get_confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """t-based confidence interval for accuracy."""
        from scipy import stats

        n = len(self.accuracies)
        mean = self.mean_accuracy
        if n < 2:
            return (mean, mean)
        std_err = self.std_accuracy / np.sqrt(n)
        h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
        return (mean - h, mean + h)

    def summary_stats(self) -> dict:
        """Summary statistics describing seed-based variability."""
        return {
            "n_runs": len(self.results),
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "mean_loss": self.mean_loss,
            "std_loss": self.std_loss,
            "accuracy_range": self.accuracy_range,
            "coefficient_of_variation": self.std_accuracy / self.mean_accuracy
            if self.mean_accuracy > 0
            else 0,
        }


def load_best_model_config(features: list[str]) -> tuple[str, dict]:
    """Load best model configuration chosen previously."""
    feature_list_str = "_".join(features)
    json_path = Path(f"results/experiment_{feature_list_str}/results.json")

    if not json_path.exists():
        raise FileNotFoundError(
            f"No previous results found at {json_path}. "
            "Run model selection before stability evaluation."
        )

    results = json.loads(json_path.read_text())
    if "overall_best" not in results:
        raise ValueError("No best model found in results file.")

    best_model = results["overall_best"]
    return best_model["model_name"], best_model["params"]


def prepare_stability_data(
    df,
    features: list[str],
    seed: int,
    sample_duration_ms: int,
    intervals: dict,
    label_mapping: dict,
    offsets_ms: dict,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Prepare train_val and test loaders for a specific seed."""
    X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test = (
        prepare_data(
            df=df,
            feature_list=features,
            sample_duration_ms=sample_duration_ms,
            intervals=intervals,
            label_mapping=label_mapping,
            offsets_ms=offsets_ms,
            random_seed=seed,
        )
    )

    train_val_loader, test_loader = create_dataloaders(
        X_train_val, y_train_val, X_test, y_test, batch_size=batch_size
    )
    return train_val_loader, test_loader


def run_single_stability_run(
    df,
    features: list[str],
    seed: int,
    sample_duration_ms: int,
    intervals: dict,
    label_mapping: dict,
    offsets_ms: dict,
    batch_size: int,
    n_epochs: int,
    device,
) -> dict:
    """Run one stability replication for a seed."""
    logger.info(f"Stability run start (seed={seed})")
    set_seed(seed)

    train_val_loader, test_loader = prepare_stability_data(
        df,
        features,
        seed,
        sample_duration_ms,
        intervals,
        label_mapping,
        offsets_ms,
        batch_size,
    )

    model_name, params = load_best_model_config(features)

    model, criterion, optimizer, scheduler = initialize_model(
        model_name,
        get_input_shape(model_name, test_loader),
        device,
        **params,
    )

    history = train_model(
        model,
        train_val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        n_epochs,
        device,
        is_test=True,
    )

    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

    logger.info(f"Seed {seed}: acc={test_accuracy:.4f}, loss={test_loss:.4f}")

    return {
        "seed": seed,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "history": history,
        "model_name": model_name,
        "params": params,
    }


def run_stability_evaluation(
    df,
    features: list[str],
    stability_seeds: list[int],
    sample_duration_ms: int,
    intervals: dict,
    label_mapping: dict,
    offsets_ms: dict,
    batch_size: int,
    n_epochs: int,
    device,
    save_results: bool = True,
) -> StabilityResults:
    """
    Evaluate performance stability across random stability_seeds.

    Args:
        df: Source dataframe.
        features: Selected feature names.
        stability_seeds: Random stability_seeds to evaluate.
        sample_duration_ms: Window length (ms).
        intervals: Interval configuration.
        label_mapping: Label remapping dict.
        offsets_ms: Offset configuration.
        batch_size: Batch size.
        n_epochs: Training epochs.
        device: Torch device.
        save_results: Persist results (JSON + summary).

    Returns:
        StabilityResults with per-seed metrics and aggregate stats.
    """
    logger.info(f"Stability evaluation over {len(stability_seeds)} stability_seeds")

    results = []
    failed = []

    for seed in stability_seeds:
        try:
            results.append(
                run_single_stability_run(
                    df,
                    features,
                    seed,
                    sample_duration_ms,
                    intervals,
                    label_mapping,
                    offsets_ms,
                    batch_size,
                    n_epochs,
                    device,
                )
            )
        except Exception as e:
            logger.error(f"Seed {seed} failed: {e}")
            failed.append(seed)

    if not results:
        raise RuntimeError("All stability runs failed.")

    stability_results = StabilityResults(results)
    stats = stability_results.summary_stats()

    logger.info(
        "Stability: mean acc {:.4f} ± {:.4f} (range {:.4f}-{:.4f}) over {} runs".format(
            stats["mean_accuracy"],
            stats["std_accuracy"],
            stats["accuracy_range"][0],
            stats["accuracy_range"][1],
            stats["n_runs"],
        )
    )

    if failed:
        logger.warning(f"Failed seeds: {failed}")

    if save_results:
        save_stability_results(features, stability_results, failed)

    return stability_results


def save_stability_results(
    features: list[str],
    stability_results: StabilityResults,
    failed_seeds: list[int] = None,
) -> None:
    """Persist stability evaluation results."""
    feature_string = "_".join(features)
    results_dir = Path(f"results/experiment_{feature_string}")
    results_dir.mkdir(exist_ok=True)

    stability_file = results_dir / "stability_results.json"
    stats = stability_results.summary_stats()
    data = {
        "features": features,
        "summary_stats": stats,
        "detailed_results": stability_results.results,
        "failed_seeds": failed_seeds or [],
    }
    with open(stability_file, "w") as f:
        json.dump(data, f, indent=2)

    summary_file = results_dir / "stability_summary.txt"
    with open(summary_file, "w") as f:
        f.write("STABILITY EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Features: {', '.join(features)}\n")
        f.write(f"Successful runs: {stats['n_runs']}\n")
        f.write(f"Failed seeds: {failed_seeds or 'None'}\n\n")

        f.write("ACCURACY:\n")
        f.write(f"Mean: {stats['mean_accuracy']:.4f}\n")
        f.write(f"Std: {stats['std_accuracy']:.4f}\n")
        f.write(
            f"Range: {stats['accuracy_range'][0]:.4f} - {stats['accuracy_range'][1]:.4f}\n"
        )
        f.write(
            f"Coefficient of Variation: {stats['coefficient_of_variation']:.4f}\n\n"
        )

        f.write("LOSS:\n")
        f.write(f"Mean: {stats['mean_loss']:.4f}\n")
        f.write(f"Std: {stats['std_loss']:.4f}\n\n")

        f.write("RUNS:\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(stability_results.results, 1):
            f.write(
                f"Run {i} (seed {result['seed']}): "
                f"Accuracy={result['test_accuracy']:.4f}, "
                f"Loss={result['test_loss']:.4f}\n"
            )

    logger.info(f"Stability results saved: {stability_file}")
    logger.info(f"Stability summary saved: {summary_file}")
