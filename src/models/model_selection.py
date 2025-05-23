import json
import logging
from datetime import datetime
from pathlib import Path

import optuna
import torch
from torch.utils.data import DataLoader

from src.models.evaluation import evaluate_model
from src.models.hyperparameter_tuning import create_objective_function
from src.models.main_config import SAMPLE_DURATION_MS
from src.models.training_loop import train_model
from src.models.utils import get_input_shape, initialize_model, save_model

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def run_model_selection(
    train_loader: DataLoader,
    val_loader: DataLoader,
    feature_list: list[str],
    model_names: list[str],
    models_config: dict,
    n_trials: int,
    n_epochs: int,
    device: str | torch.device,
    experiment_tracker: "ExperimentTracker",
) -> "ExperimentTracker":
    """
    Run model selection and hyperparameter tuning for the specified models.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        feature_list: List of features used
        model_names: List of model names to evaluate
        models_config: Configuration for models
        n_trials: Number of hyperparameter optimization trials
        n_epochs: Number of training epochs
        device: Device to train on (cuda/cpu)
        experiment_tracker: ExperimentTracker instance for recording results

    Returns:
        ExperimentTracker with updated results
    """
    logger.info(
        f"Starting model selection with features: {experiment_tracker.feature_string}"
    )

    # Evaluate each model
    for model_name in model_names:
        if model_name not in models_config:
            logger.warning(f"Model {model_name} not found in config, skipping")
            continue

        model_info = models_config[model_name]
        logger.info(
            f"Training {model_name} with features: {experiment_tracker.feature_string}"
        )

        # Create and run optimization study
        study_name = f"{model_name}_{experiment_tracker.feature_string}"
        study_name += "_" + experiment_tracker.timestamp
        objective_function = create_objective_function(
            train_loader, val_loader, model_name, model_info, device, n_epochs
        )

        study = optuna.create_study(
            direction="maximize",
            storage="sqlite:///db.sqlite3",
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        study.optimize(objective_function, n_trials=n_trials)

        # Get the best parameters
        best_model_params = study.best_params.copy()

        # Transform exponential parameters for actual use
        for param_name in model_info["hyperparameters"]:
            if model_info["hyperparameters"][param_name]["type"] == "exp":
                exp_value = best_model_params[param_name]
                best_model_params[param_name] = 2**exp_value

        # Save results for this model
        experiment_tracker.add_model_result(
            model_name=model_name,
            validation_accuracy=study.best_value,
            best_params=best_model_params,
            study_name=study_name,
        )

        logger.info(
            f"Best value for {model_name}: {study.best_value:.4f} (params: {best_model_params})"
        )

    # Save intermediate results
    experiment_tracker.save_results()

    return experiment_tracker


def train_evaluate_and_save_best_model(
    train_val_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int,
    device: str | torch.device,
    experiment_tracker: "ExperimentTracker",
) -> "ExperimentTracker":
    """
    Train the final model on combined training+validation data and evaluate on test set.
    """
    best_model = experiment_tracker.results["overall_best"]
    model_name = best_model["model_name"]
    params = best_model["params"]

    logger.info(
        f"Overall Best Model: {best_model['model_name']} with validation accuracy: {best_model['validation_accuracy']:.4f}"
    )
    logger.info(f"Training final {model_name} model on combined train+val data...")

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
        f"Final Model | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}"
    )

    # Generate a model path directly instead of using get_model_path
    model_filename = f"{model_name}_{experiment_tracker.timestamp}.pt"
    model_path = experiment_tracker.models_dir / model_filename

    # Save best model
    save_model(
        model=model,
        accuracy=test_accuracy,
        best_params=params,
        best_model_name=model_name,
        data_sample=test_loader,
        feature_list=experiment_tracker.features,
        sample_duration_ms=experiment_tracker.sample_duration_ms,
        model_path=model_path,
    )

    # Record best model test performance
    experiment_tracker.record_best_model_test_performance(
        test_accuracy=test_accuracy,
        test_loss=test_loss,
        history=history,
        model_path=model_path,
    )

    # Save updated results
    experiment_tracker.save_results()

    return experiment_tracker


class ExperimentTracker:
    """
    Handles experiment tracking, result storage, and reporting.
    """

    def __init__(
        self,
        features: list[str],
        sample_duration_ms: int = SAMPLE_DURATION_MS,
        result_dir: str | Path = RESULT_DIR,
    ):
        """
        Initialize a new experiment tracker.

        Args:
            features: List of features used in the experiment
            result_dir: Base directory for storing results
        """
        self.features = features
        self.feature_string = "_".join(sorted(features))  # Sort for consistency
        self.sample_duration_ms = sample_duration_ms
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Initialize result storage
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)

        # Create a directory for this feature combination
        self.experiment_dir = self.result_dir / f"experiment_{self.feature_string}"
        self.experiment_dir.mkdir(exist_ok=True)

        # Create subdirectories for model
        self.models_dir = self.experiment_dir / "model"
        self.models_dir.mkdir(exist_ok=True)

        # Initialize results dictionary
        self.results = {
            "features": self.features,
            "feature_string": self.feature_string,
            "sample_duration_ms": self.sample_duration_ms,
            "timestamp": self.timestamp,
            "models": {},
        }

        logger.info(f"Created experiment with features: {self.feature_string}")

    def add_model_result(
        self,
        model_name: str,
        validation_accuracy: float,
        best_params: dict,
        study_name: str,
    ) -> None:
        """
        Add model training result to the experiment.

        Args:
            model_name: Name of the model
            validation_accuracy: Validation accuracy achieved
            best_params: Best hyperparameters found
            study_name: Name of the Optuna study
        """
        self.results["models"][model_name] = {
            "validation_accuracy": validation_accuracy,
            "best_model_params": best_params,
            "study_name": study_name,
        }

        # Update best model if this is the best so far
        if (
            "overall_best" not in self.results
            or validation_accuracy > self.results["overall_best"]["validation_accuracy"]
        ):
            self.results["overall_best"] = {
                "model_name": model_name,
                "validation_accuracy": validation_accuracy,
                "params": best_params,
            }

            # Log when we find a new best model
            logger.info(
                f"New best model: {model_name} with validation accuracy: {validation_accuracy:.4f}"
            )

        logger.info(
            f"Added result for {model_name}: validation accuracy = {validation_accuracy:.4f}"
        )

    def record_best_model_test_performance(
        self,
        test_accuracy: float,
        test_loss: float,
        history: dict | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        """
        Record the test performance of the best model.

        Args:
            test_accuracy: Accuracy on test set
            test_loss: Loss on test set
            history: Training history dictionary
            model_path: Path where model was saved
        """
        if "overall_best" not in self.results:
            raise ValueError("No best model has been selected yet")

        self.results["overall_best"].update(
            {
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
            }
        )

        if history is not None:
            self.results["overall_best"]["history"] = history

        if model_path is not None:
            self.results["overall_best"]["model_path"] = str(model_path)

        # Log the test results for the best model
        best_model = self.results["overall_best"]["model_name"]
        logger.info(
            f"Best model {best_model} test results: accuracy = {test_accuracy:.4f}, loss = {test_loss:.4f}"
        )

    def save_results(self) -> None:
        """
        Save experiment results to disk.

        Returns:
            Tuple of (results_file_path, summary_file_path)
        """
        # Create paths for result files
        results_file = self.experiment_dir / "results.json"
        summary_file = self.experiment_dir / "summary.txt"

        # Save results to JSON file
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Create summary file
        with open(summary_file, "w") as f:
            f.write(f"Features: {', '.join(self.features)}\n")
            f.write(f"Sample Duration (ms): {self.sample_duration_ms}\n\n")
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            f.write("=" * 60 + "\n")

            # Sort models by validation accuracy (descending)
            sorted_models = sorted(
                self.results["models"].items(),
                key=lambda x: x[1]["validation_accuracy"],
                reverse=True,
            )

            for model_name, model_data in sorted_models:
                f.write(f"Model: {model_name}\n")
                f.write(
                    f"Validation Accuracy: {model_data['validation_accuracy']:.4f}\n"
                )
                f.write(f"Best Parameters: {model_data['best_model_params']}\n")
                f.write("-" * 60 + "\n")

            if "overall_best" in self.results:
                f.write("\nOVERALL BEST MODEL:\n")
                best_model = self.results["overall_best"]
                f.write(f"Model: {best_model['model_name']}\n")
                f.write(
                    f"Validation Accuracy: {best_model['validation_accuracy']:.4f}\n"
                )
                f.write(f"Parameters: {best_model['params']}\n")

                # Include test results if available
                if "test_accuracy" in best_model:
                    f.write("\nTEST PERFORMANCE:\n")
                    f.write(f"Test Accuracy: {best_model['test_accuracy']:.4f}\n")
                    f.write(f"Test Loss: {best_model['test_loss']:.4f}\n")

        logger.info(f"Results saved to {results_file}")
        logger.info(f"Summary saved to {summary_file}")
