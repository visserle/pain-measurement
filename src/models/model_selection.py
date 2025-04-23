import json
import logging
from datetime import datetime
from pathlib import Path

import optuna

from src.models.evaluation import evaluate_model
from src.models.hyperparameter_tuning import create_objective_function
from src.models.training_loop import train_model
from src.models.utils import get_input_shape, initialize_model, save_model


def save_results(
    results,
    experiment_dir,
    feature_str,
    include_test_results=False,
):
    """Save experiment results to JSON and generate a summary text file."""
    # Save results to JSON file
    results_file = experiment_dir / f"results_{feature_str}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Create summary file
    summary_file = experiment_dir / f"summary_{feature_str}.txt"
    with open(summary_file, "w") as f:
        f.write(f"Experiment ID: {results['experiment_id']}\n")
        f.write(f"Features: {', '.join(results['features'])}\n\n")
        f.write("MODEL PERFORMANCE SUMMARY:\n")
        f.write("=" * 60 + "\n")

        # Sort models by validation accuracy (descending)
        sorted_models = sorted(
            results["models"].items(),
            key=lambda x: x[1]["validation_accuracy"],
            reverse=True,
        )

        for model_name, model_data in sorted_models:
            f.write(f"Model: {model_name}\n")
            f.write(f"Validation Accuracy: {model_data['validation_accuracy']:.4f}\n")
            f.write(f"Best Parameters: {model_data['best_model_params']}\n")
            f.write("-" * 60 + "\n")

        f.write("\nOVERALL BEST MODEL:\n")
        best_model = results["overall_best"]
        f.write(f"Model: {best_model['model_name']}\n")
        f.write(f"Validation Accuracy: {best_model['validation_accuracy']:.4f}\n")
        f.write(f"Parameters: {best_model['params']}\n")

        if include_test_results:
            f.write("\nTEST PERFORMANCE:\n")
            f.write(f"Test Accuracy: {best_model['test_accuracy']:.4f}\n")
            f.write(f"Test Loss: {best_model['test_loss']:.4f}\n")

    return results_file, summary_file


def run_model_selection(
    train_loader,
    val_loader,
    X_train_val,
    feature_list,
    model_names,
    models_config,
    n_trials,
    n_epochs,
    output_dir,
    device,
):
    """
    Run model selection and hyperparameter tuning for the specified models.
    """
    # Create experiment directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = output_dir / f"experiment_{experiment_id}"
    experiment_dir.mkdir(exist_ok=True)

    # Setup results tracking
    feature_str = "_".join(feature_list)
    results = {
        "experiment_id": experiment_id,
        "features": feature_list,
        "models": {},
        "feature_string": feature_str,
    }

    logging.info(f"Starting experiment with features: {feature_str}")

    # Track best model overall
    best_value = float("-inf")
    best_params = None
    best_model_name = None

    # Evaluate each model
    for model_name in model_names:
        if model_name not in models_config:
            logging.warning(f"Model {model_name} not found in config, skipping")
            continue

        model_info = models_config[model_name]
        logging.info(f"Training {model_name} with features: {feature_str}")

        # Create and run optimization study
        study_name = f"{model_name}_{feature_str}_{experiment_id}"
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
        model_best_value = study.best_value
        results["models"][model_name] = {
            "validation_accuracy": model_best_value,
            "best_model_params": best_model_params,
            "study_name": study_name,
        }

        logging.info(
            f"Best value for {model_name}: {model_best_value:.4f} (params: {best_model_params})"
        )

        # Update overall best model
        if model_best_value > best_value:
            best_value = model_best_value
            best_params = best_model_params
            best_model_name = model_name

    # Record overall best model
    results["overall_best"] = {
        "model_name": best_model_name,
        "validation_accuracy": best_value,
        "params": best_params,
    }

    logging.info(f"Overall Best Model: {best_model_name} with value: {best_value:.4f}")

    # Save results
    save_results(results, experiment_dir, feature_str)

    return results, experiment_dir


def train_evaluate_and_save_best_model(
    model_name,
    params,
    X_train_val,
    train_val_loader,
    test_loader,
    feature_list,
    n_epochs,
    device,
    results,
    experiment_dir,
):
    """
    Train the final model on combined training+validation data and evaluate on test set.
    """
    logging.info(f"Training final {model_name} model on combined train+val data...")

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
    logging.info(
        f"Final Model | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}"
    )

    # Save model
    save_model(
        model=model,
        accuracy=test_accuracy,
        best_params=params,
        best_model_name=model_name,
        X_train_val=X_train_val,
        feature_list=feature_list,
    )

    # Update results with test performance
    results["overall_best"].update(
        {"test_accuracy": test_accuracy, "test_loss": test_loss, "history": history}
    )

    # Save updated results
    feature_str = "_".join(feature_list)
    save_results(results, experiment_dir, feature_str, include_test_results=True)

    return results
