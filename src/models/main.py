import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import optuna.logging

from src.log_config import configure_logging
from src.models.data_loader import create_dataloaders
from src.models.data_preparation import (
    expand_feature_list,
    load_data_from_database,
    prepare_data,
)
from src.models.main_config import (
    BATCH_SIZE,
    INTERVALS,
    LABEL_MAPPING,
    N_EPOCHS,
    N_TRIALS,
    OFFSETS_MS,
    RANDOM_SEED,
    SAMPLE_DURATION_MS,
    STABILITY_SEEDS,
)
from src.models.model_selection import (
    ExperimentTracker,
    run_model_selection,
    train_evaluate_and_save_best_model,
)
from src.models.models_config import MODELS
from src.models.stability import run_stability_evaluation
from src.models.utils import (
    get_device,
    set_seed,
)

configure_logging(
    stream_level=logging.DEBUG,
    file_path=Path(f"runs/models/logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
)
optuna.logging.disable_default_handler()
optuna.logging.enable_propagation()

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train models with different feature combinations"
    )
    parser.add_argument(
        # for sanity checks, we can also train on temperature/rating
        "--features",
        nargs="+",
        help="List of features to include in the model.",
    )
    available_models = list(MODELS.keys())
    parser.add_argument(
        "--models",
        nargs="+",
        choices=available_models,
        type=lambda x: next(
            m for m in available_models if m.lower() == x.lower()
        ),  # case-insensitive matching
        help="List of model architectures to train.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of optimization trials to run. Default: {N_TRIALS}",
    )
    parser.add_argument(
        "--stability",
        action="store_true",
        help="If set, perform stability testing with multiple random seeds.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    args.features = expand_feature_list(args.features)
    df = load_data_from_database(args.features)

    if not args.stability:
        # Normal run: single seed for model comparison
        set_seed(RANDOM_SEED)

        # Create experiment tracker
        experiment_tracker = ExperimentTracker(
            features=args.features,
            sample_duration_ms=SAMPLE_DURATION_MS,
            intervals=INTERVALS,
            label_mapping=LABEL_MAPPING,
            offsets_ms=OFFSETS_MS,
        )

        # Prepare data
        X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test = (
            prepare_data(
                df=df,
                feature_list=args.features,
                sample_duration_ms=SAMPLE_DURATION_MS,
                intervals=INTERVALS,
                label_mapping=LABEL_MAPPING,
                offsets_ms=OFFSETS_MS,
                random_seed=RANDOM_SEED,
            )
        )

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, is_test=False
        )

        train_val_loader, test_loader = create_dataloaders(
            X_train_val, y_train_val, X_test, y_test, batch_size=BATCH_SIZE
        )

        # Run model selection and hyperparameter tuning
        experiment_tracker = run_model_selection(
            train_loader=train_loader,
            val_loader=val_loader,
            model_names=args.models,
            models_config=MODELS,
            n_trials=args.trials,
            n_epochs=N_EPOCHS,
            device=device,
            experiment_tracker=experiment_tracker,
            random_seed=RANDOM_SEED,
        )

        # Train final model on combined training+validation data
        experiment_tracker = train_evaluate_and_save_best_model(
            train_val_loader=train_val_loader,
            test_loader=test_loader,
            n_epochs=N_EPOCHS,
            device=device,
            experiment_tracker=experiment_tracker,
        )

    else:
        # stability testing: multiple seeds for the best model
        logging.info("Starting stability testing")

        stability_results = run_stability_evaluation(
            df=df,
            features=args.features,
            stability_seeds=STABILITY_SEEDS,
            sample_duration_ms=SAMPLE_DURATION_MS,
            intervals=INTERVALS,
            label_mapping=LABEL_MAPPING,
            offsets_ms=OFFSETS_MS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            device=device,
            save_results=True,
        )

        # Log summary statistics
        stats = stability_results.summary_stats()
        logging.info(
            f"Stability testing completed with {stats['n_runs']} runs. "
            f"Mean accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}"
        )

    logging.info(f"Experiment with features {args.features} completed successfully")


if __name__ == "__main__":
    main()
