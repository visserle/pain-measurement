import argparse
import logging
from datetime import datetime
from pathlib import Path

from src.data.database_manager import DatabaseManager
from src.log_config import configure_logging
from src.models.data_loader import create_dataloaders
from src.models.data_preparation import prepare_eeg_data
from src.models.model_selection import (
    ExperimentTracker,
    run_model_selection,
    train_evaluate_and_save_best_model,
)
from src.models.models_config import MODELS
from src.models.utils import (
    get_device,
    set_seed,
)

RANDOM_SEED = 42
BATCH_SIZE = 64
N_EPOCHS = 100
N_TRIALS = 30

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

configure_logging(
    stream_level=logging.DEBUG,
    file_path=Path(f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
)
device = get_device()
set_seed(RANDOM_SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Train models for EEG data")

    # Default feature list
    default_features = ["f3", "f4", "c3", "c4", "cz", "p3", "p4", "oz"]

    parser.add_argument(
        "--features",
        nargs="+",
        default=default_features,
        help="List of features to use for training the model.",
    )

    available_models = list(MODELS.keys())
    parser.add_argument(
        "--models",
        nargs="+",
        default=available_models,
        choices=available_models,
        type=lambda x: next(
            m for m in available_models if m.lower() == x.lower()
        ),  # case-insensitive matching
        help="List of model architectures to train. Default: all available models",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of optimization trials to run. Default: {N_TRIALS}",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create experiment tracker
    experiment_tracker = ExperimentTracker(
        features=args.features,
        base_dir=RESULT_DIR,
    )

    # Load data from database
    db = DatabaseManager()
    with db:
        eeg = db.get_table("Preprocess_EEG")
        trials = db.get_table("Trials")

    # Prepare EEG data
    X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test = (
        prepare_eeg_data(eeg, trials, args.features, RANDOM_SEED)
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
        X_train_val=X_train_val,
        feature_list=args.features,
        model_names=args.models,
        models_config=MODELS,
        n_trials=args.trials,
        n_epochs=N_EPOCHS,
        device=device,
        experiment_tracker=experiment_tracker,
    )

    # Train final model on combined training+validation data
    train_evaluate_and_save_best_model(
        train_val_loader=train_val_loader,
        test_loader=test_loader,
        X_train_val=X_train_val,
        n_epochs=N_EPOCHS,
        device=device,
        experiment_tracker=experiment_tracker,
    )

    logging.info(f"Experiment with features {args.features} completed successfully")


if __name__ == "__main__":
    main()
