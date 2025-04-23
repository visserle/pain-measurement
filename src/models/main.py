# src/models/training.py (updated)
import argparse
import logging
from datetime import datetime
from pathlib import Path

from src.data.database_manager import DatabaseManager
from src.log_config import configure_logging
from src.models.data_loader import create_dataloaders
from src.models.data_preparation import prepare_data
from src.models.model_selection import (
    run_model_selection,
    train_evaluate_and_save_best_model,
)
from src.models.models_config import MODELS
from src.models.utils import get_device, set_seed

RANDOM_SEED = 42
BATCH_SIZE = 64
N_EPOCHS = 10
N_TRIALS = 2

configure_logging(
    stream_level=logging.DEBUG,
    file_path=Path(f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
)
device = get_device()
set_seed(RANDOM_SEED)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train models with different feature combinations"
    )

    # Default feature list
    default_features = [
        "eda_tonic",
        "eda_phasic",
        "pupil_mean",
        "pupil_mean_tonic",
        "heartrate",
        # "brow_furrow",
        # "cheek_raise",
        # "mouth_open",
        # "upper_lip_raise",
        # "nose_wrinkle",
        #
        # for sanity checks only:
        # "temperature",
        # "rating",
    ]

    parser.add_argument(
        "--features",
        nargs="+",
        default=default_features,
        help="List of features to include in the model. Default: all available features",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        help="List of model architectures to train. Default: all available models",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of optimization trials to run. Default: {N_TRIALS}",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results. Default: 'results'",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load data from database
    db = DatabaseManager()
    with db:
        df = db.get_table(
            "Merged_and_Labeled_Data",
            exclude_trials_with_measurement_problems=True,
        )

    # Prepare data
    X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test = (
        prepare_data(df, args.features, 5000, RANDOM_SEED)
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, is_test=False
    )

    train_val_loader, test_loader = create_dataloaders(
        X_train_val, y_train_val, X_test, y_test, batch_size=BATCH_SIZE
    )

    # Run model selection and hyperparameter tuning
    results, experiment_dir = run_model_selection(
        train_loader=train_loader,
        val_loader=val_loader,
        X_train_val=X_train_val,
        feature_list=args.features,
        model_names=args.models,
        models_config=MODELS,
        n_trials=args.trials,
        n_epochs=N_EPOCHS,
        output_dir=args.output,
        device=device,
    )

    # Train final model on combined training+validation data
    best_model_info = results["overall_best"]
    train_evaluate_and_save_best_model(
        model_name=best_model_info["model_name"],
        params=best_model_info["params"],
        X_train_val=X_train_val,
        train_val_loader=train_val_loader,
        test_loader=test_loader,
        feature_list=args.features,
        n_epochs=N_EPOCHS,
        device=device,
        results=results,
        experiment_dir=experiment_dir,
    )


if __name__ == "__main__":
    main()
