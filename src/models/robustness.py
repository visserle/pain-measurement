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


class RobustnessResults:
    """Container for robustness testing results with analysis methods."""

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
        """Calculate confidence interval for accuracy."""
        from scipy import stats

        n = len(self.accuracies)
        mean = self.mean_accuracy
        std_err = self.std_accuracy / np.sqrt(n)
        h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
        return (mean - h, mean + h)

    def summary_stats(self) -> dict:
        """Return comprehensive summary statistics."""
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
    """Load the best model configuration from previous experiment results."""
    feature_list_str = "_".join(features)
    json_path = Path(f"results/experiment_{feature_list_str}/results.json")

    if not json_path.exists():
        raise FileNotFoundError(
            f"No previous results found at {json_path}. "
            "Run model selection first before robustness testing."
        )

    results = json.loads(json_path.read_text())
    if "overall_best" not in results:
        raise ValueError("No best model found in results file.")

    best_model = results["overall_best"]
    return best_model["model_name"], best_model["params"]


def prepare_robustness_data(
    df,
    features: list[str],
    seed: int,
    sample_duration_ms: int,
    intervals: dict,
    label_mapping: dict,
    offsets_ms: dict,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Prepare data for robustness testing with a specific seed."""
    # Prepare data
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

    # Create dataloaders (we only need train_val and test for robustness)
    train_val_loader, test_loader = create_dataloaders(
        X_train_val, y_train_val, X_test, y_test, batch_size=batch_size
    )

    return train_val_loader, test_loader


def run_single_robustness_test(
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
    """Run robustness test for a single seed."""
    logger.info(f"Starting robustness test with seed {seed}")
    set_seed(seed)

    try:
        # Prepare data
        train_val_loader, test_loader = prepare_robustness_data(
            df,
            features,
            seed,
            sample_duration_ms,
            intervals,
            label_mapping,
            offsets_ms,
            batch_size,
        )

        # Load best model configuration
        model_name, params = load_best_model_config(features)

        # Initialize and train model
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

        # Evaluate on test set
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

        logger.info(
            f"Robustness test with seed {seed} - "
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

        return {
            "seed": seed,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "history": history,
            "model_name": model_name,
            "params": params,
        }

    except Exception as e:
        logger.error(f"Robustness test failed for seed {seed}: {e}")
        raise


def run_robustness_testing(
    df,
    features: list[str],
    robustness_seeds: list[int],
    sample_duration_ms: int,
    intervals: dict,
    label_mapping: dict,
    offsets_ms: dict,
    batch_size: int,
    n_epochs: int,
    device,
    save_results: bool = True,
) -> RobustnessResults:
    """
    Run comprehensive robustness testing with multiple seeds.

    Args:
        df: Input dataframe
        features: list of features to use
        robustness_seeds: list of random seeds to test
        sample_duration_ms: Duration of samples in milliseconds
        intervals: Interval configuration
        label_mapping: Label mapping configuration
        offsets_ms: Offset configuration
        batch_size: Batch size for training
        n_epochs: Number of epochs to train
        device: Device to train on
        save_results: Whether to save results to disk

    Returns:
        RobustnessResults object containing all results and analysis
    """
    logger.info(f"Starting robustness testing with {len(robustness_seeds)} seeds")

    results = []
    failed_seeds = []

    for seed in robustness_seeds:
        try:
            result = run_single_robustness_test(
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
            results.append(result)
        except Exception as e:
            logger.error(f"Robustness test failed for seed {seed}: {e}")
            failed_seeds.append(seed)
            continue

    if not results:
        raise RuntimeError("All robustness tests failed!")

    # Create results object
    robustness_results = RobustnessResults(results)

    # Log summary
    stats = robustness_results.summary_stats()
    logger.info(
        f"Robustness testing completed. "
        f"Mean accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f} "
        f"(range: {stats['accuracy_range'][0]:.4f}-{stats['accuracy_range'][1]:.4f}) "
        f"over {stats['n_runs']} successful runs"
    )

    if failed_seeds:
        logger.warning(f"Failed seeds: {failed_seeds}")

    # Save results if requested
    if save_results:
        save_robustness_results(features, robustness_results, failed_seeds)

    return robustness_results


def save_robustness_results(
    features: list[str],
    robustness_results: RobustnessResults,
    failed_seeds: list[int] = None,
) -> None:
    """Save robustness testing results to disk."""
    feature_string = "_".join(features)
    results_dir = Path(f"results/experiment_{feature_string}")
    results_dir.mkdir(exist_ok=True)

    # Save detailed results
    robustness_file = results_dir / "robustness_results.json"
    robustness_data = {
        "features": features,
        "summary_stats": robustness_results.summary_stats(),
        "detailed_results": robustness_results.results,
        "failed_seeds": failed_seeds or [],
    }

    with open(robustness_file, "w") as f:
        json.dump(robustness_data, f, indent=2)

    # Save summary report
    summary_file = results_dir / "robustness_summary.txt"
    stats = robustness_results.summary_stats()

    with open(summary_file, "w") as f:
        f.write("ROBUSTNESS TESTING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Features: {', '.join(features)}\n")
        f.write(f"Number of successful runs: {stats['n_runs']}\n")
        f.write(f"Failed seeds: {failed_seeds or 'None'}\n\n")

        f.write("ACCURACY STATISTICS:\n")
        f.write(f"Mean: {stats['mean_accuracy']:.4f}\n")
        f.write(f"Standard Deviation: {stats['std_accuracy']:.4f}\n")
        f.write(
            f"Range: {stats['accuracy_range'][0]:.4f} - {stats['accuracy_range'][1]:.4f}\n"
        )
        f.write(
            f"Coefficient of Variation: {stats['coefficient_of_variation']:.4f}\n\n"
        )

        f.write("LOSS STATISTICS:\n")
        f.write(f"Mean: {stats['mean_loss']:.4f}\n")
        f.write(f"Standard Deviation: {stats['std_loss']:.4f}\n\n")

        f.write("INDIVIDUAL RUNS:\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(robustness_results.results, 1):
            f.write(f"Run {i} (seed {result['seed']}): ")
            f.write(f"Accuracy={result['test_accuracy']:.4f}, ")
            f.write(f"Loss={result['test_loss']:.4f}\n")

    logger.info(f"Robustness results saved to {robustness_file}")
    logger.info(f"Robustness summary saved to {summary_file}")
