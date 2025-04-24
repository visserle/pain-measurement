import json
import logging
from datetime import datetime
from pathlib import Path

import optuna

from src.models.evaluation import evaluate_model
from src.models.hyperparameter_tuning import create_objective_function
from src.models.training_loop import train_model
from src.models.utils import get_input_shape, initialize_model, save_model

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def run_model_selection(
    train_loader,
    val_loader,
    X_train_val,
    feature_list,
    model_names,
    models_config,
    n_trials,
    n_epochs,
    device,
    experiment_tracker,
):
    """
    Run model selection and hyperparameter tuning for the specified models.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        X_train_val: Combined training and validation features
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
        study_name = f"{model_name}_{experiment_tracker.feature_string}_{experiment_tracker.experiment_id}"
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
    train_val_loader,
    test_loader,
    X_train_val,
    n_epochs,
    device,
    experiment_tracker,
):
    """
    Train the final model on combined training+validation data and evaluate on test set.

    Args:
        train_val_loader: DataLoader for combined training and validation data
        test_loader: DataLoader for test data
        X_train_val: Combined training and validation features
        n_epochs: Number of training epochs
        device: Device to train on
        experiment_tracker: ExperimentTracker with best model information

    Returns:
        ExperimentTracker with updated test results
    """
    best_model_info = experiment_tracker.get_best_model_info()
    model_name = best_model_info["model_name"]
    params = best_model_info["params"]

    logger.info(f"Training final {model_name} model on combined train+val data...")

    # Initialize and train model
    model, criterion, optimizer, scheduler = initialize_model(
        model_name,
        get_input_shape(model_name, X_train_val),
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

    # Save model
    model_path = experiment_tracker.get_model_path(model_name)
    save_model(
        model=model,
        accuracy=test_accuracy,
        best_params=params,
        best_model_name=model_name,
        X_train_val=X_train_val,
        feature_list=experiment_tracker.features,
        model_path=model_path,
    )

    # Update results with test performance and history
    experiment_tracker.update_best_model_test_results(
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

    def __init__(self, features, base_dir=None):
        """
        Initialize a new experiment tracker.

        Args:
            features: List of features used in the experiment
            base_dir: Base directory for storing results (default: 'results')
        """
        self.timestamp = datetime.now()
        self.experiment_id = self.timestamp.strftime("%Y%m%d-%H%M%S")
        self.features = features
        self.feature_string = "_".join(features)

        # Initialize result storage
        self.base_dir = Path(base_dir or "results")
        self.base_dir.mkdir(exist_ok=True)

        self.experiment_dir = self.base_dir / f"experiment_{self.experiment_id}"
        self.experiment_dir.mkdir(exist_ok=True)

        # Initialize results dictionary
        self.results = {
            "experiment_id": self.experiment_id,
            "features": self.features,
            "feature_string": self.feature_string,
            "timestamp": self.timestamp.isoformat(),
            "models": {},
        }

        logger.info(
            f"Created experiment {self.experiment_id} with features: {self.feature_string}"
        )

    def add_model_result(
        self, model_name, validation_accuracy, best_params, study_name=None
    ):
        """
        Add model training result to the experiment.

        Args:
            model_name: Name of the model
            validation_accuracy: Validation accuracy achieved
            best_params: Best hyperparameters found
            study_name: Name of the Optuna study (if applicable)
        """
        self.results["models"][model_name] = {
            "validation_accuracy": validation_accuracy,
            "best_model_params": best_params,
            "study_name": study_name,
        }

        # Update best model if this is the best so far
        if (
            not hasattr(self, "best_accuracy")
            or validation_accuracy > self.best_accuracy
        ):
            self.best_accuracy = validation_accuracy
            self.best_model_name = model_name
            self.best_params = best_params
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

    def update_best_model_test_results(
        self, test_accuracy, test_loss, history=None, model_path=None
    ):
        """
        Update the best model with test results.

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

    def get_best_model_info(self):
        """
        Get information about the best performing model.

        Returns:
            Dictionary with best model information
        """
        if "overall_best" not in self.results:
            raise ValueError("No best model has been selected yet")

        best_model = self.results["overall_best"]
        logger.info(
            f"Overall Best Model: {best_model['model_name']} with validation accuracy: {best_model['validation_accuracy']:.4f}"
        )

        return best_model

    def save_results(self):
        """
        Save experiment results to disk.

        Returns:
            Tuple of (results_file_path, summary_file_path)
        """
        # Create paths for result files
        results_file = self.experiment_dir / f"results_{self.feature_string}.json"
        summary_file = self.experiment_dir / f"summary_{self.feature_string}.txt"

        # Save results to JSON file
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Create summary file
        with open(summary_file, "w") as f:
            f.write(f"Experiment ID: {self.experiment_id}\n")
            f.write(f"Features: {', '.join(self.features)}\n\n")
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

        return results_file, summary_file

    def get_model_path(self, model_name):
        """
        Get the file path for saving a model.

        Args:
            model_name: Name of the model

        Returns:
            Path object for the model file
        """
        return self.experiment_dir / f"model_{model_name}_{self.feature_string}.pt"
