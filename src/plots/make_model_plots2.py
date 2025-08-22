import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tomllib
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.log_config import configure_logging
from src.models.data_loader import create_dataloaders
from src.models.data_preparation import (
    expand_feature_list,
    load_data_from_database,
    prepare_data,
)
from src.models.main_config import RANDOM_SEED
from src.models.utils import load_model
from src.plots.model_inference import (
    analyze_test_dataset_for_one_stimulus,
    plot_prediction_confidence_heatmap,
)
from src.plots.model_performance import (
    create_performance_table,
    get_model_predictions,
    plot_multiple_roc_curves,
)
from src.plots.model_performance_per_participant import (
    analyze_per_participant,
    plot_feature_accuracy_comparison,
    plot_participant_accuracy_comparison,
)

load_dotenv()
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))

plt.style.use("./src/plots/style.mplstyle")

feature_lists = [
    ["eda_raw"],
    ["heart_rate"],
    ["pupil"],
    ["eda_raw", "pupil"],
    ["eda_raw", "heart_rate"],
    ["eda_raw", "heart_rate", "pupil"],
    ["face"],
    ["face", "eda_raw", "heart_rate", "pupil"],
    ["eeg"],
    ["eeg", "eda_raw"],
    ["eeg", "face", "eda_raw", "heart_rate", "pupil"],
]
feature_lists = expand_feature_list(feature_lists)


class ModelDataCache:
    """Cache for models, data, and dataloaders to avoid redundant loading."""

    def __init__(self):
        self.models: Dict[str, nn.Module] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.datasets: Dict[str, Any] = {}
        self.dataloaders: Dict[str, Tuple[DataLoader, DataLoader]] = {}
        self.test_groups: Dict[str, np.ndarray] = {}
        self.raw_dfs: Dict[str, Any] = {}

    def get_model_and_config(
        self, feature_list_str: str
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load model and its configuration if not cached."""
        if feature_list_str not in self.models:
            json_path = Path(f"results/experiment_{feature_list_str}/results.json")
            dictionary = json.loads(json_path.read_text())
            model_path = Path(
                dictionary["overall_best"]["model_path"].replace("\\", "/")
            )
            (
                model,
                feature_list,
                sample_duration_ms,
                intervals,
                label_mapping,
                offsets_ms,
            ) = load_model(model_path, device="cpu")

            self.models[feature_list_str] = model
            self.model_configs[feature_list_str] = {
                "feature_list": feature_list,
                "sample_duration_ms": sample_duration_ms,
                "intervals": intervals,
                "label_mapping": label_mapping,
                "offsets_ms": offsets_ms,
                "model_name": model.__class__.__name__,
            }

        return self.models[feature_list_str], self.model_configs[feature_list_str]

    def get_data_and_loaders(self, feature_list: list, feature_list_str: str) -> Tuple:
        """Load data and create dataloaders if not cached."""
        if feature_list_str not in self.datasets:
            # Load raw dataframe if not cached
            if feature_list_str not in self.raw_dfs:
                self.raw_dfs[feature_list_str] = load_data_from_database(feature_list)

            df = self.raw_dfs[feature_list_str]

            # Get model config to ensure we have the right parameters
            _, config = self.get_model_and_config(feature_list_str)

            # Prepare data
            _, _, _, _, X_train_val, y_train_val, X_test, y_test = prepare_data(
                df=df,
                feature_list=config["feature_list"],
                sample_duration_ms=config["sample_duration_ms"],
                intervals=config["intervals"],
                label_mapping=config["label_mapping"],
                offsets_ms=config["offsets_ms"],
                random_seed=RANDOM_SEED,
            )

            # Get test groups
            test_groups = prepare_data(
                df=df,
                feature_list=config["feature_list"],
                sample_duration_ms=config["sample_duration_ms"],
                intervals=config["intervals"],
                label_mapping=config["label_mapping"],
                offsets_ms=config["offsets_ms"],
                random_seed=RANDOM_SEED,
                only_return_test_groups=True,
            )

            # Create dataloaders
            train_loader, test_loader = create_dataloaders(
                X_train_val, y_train_val, X_test, y_test, batch_size=64
            )

            self.datasets[feature_list_str] = {
                "X_train_val": X_train_val,
                "y_train_val": y_train_val,
                "X_test": X_test,
                "y_test": y_test,
            }
            self.dataloaders[feature_list_str] = (train_loader, test_loader)
            self.test_groups[feature_list_str] = test_groups

        return (
            self.datasets[feature_list_str],
            self.dataloaders[feature_list_str],
            self.test_groups[feature_list_str],
            self.raw_dfs[feature_list_str],
        )


def model_inference(cache: ModelDataCache):
    """Run model inference analysis using cached data."""
    config_path = Path("src/experiments/measurement/measurement_config.toml")
    with open(config_path, "rb") as file:
        config = tomllib.load(file)
    stimulus_seeds = config["stimulus"]["seeds"]
    logging.info(f"Using seeds for stimulus generation: {stimulus_seeds}")

    for feature_list in feature_lists:
        feature_list_str = "_".join(feature_list)

        # Get cached model and data
        model, model_config = cache.get_model_and_config(feature_list_str)
        datasets, loaders, test_groups, df = cache.get_data_and_loaders(
            feature_list, feature_list_str
        )

        test_ids = np.unique(test_groups)

        # Analyze the entire test dataset
        all_probabilities = {}
        all_participant_trials = {}

        for stimulus_seed in stimulus_seeds:
            probabilities, participant_trials = analyze_test_dataset_for_one_stimulus(
                df,
                model,
                model_config["feature_list"],
                test_ids,
                stimulus_seed,
                model_config["sample_duration_ms"],
            )

            all_probabilities[stimulus_seed] = probabilities
            all_participant_trials[stimulus_seed] = participant_trials

        # Plot all available stimuli
        fig = plot_prediction_confidence_heatmap(
            all_probabilities,
            model_config["sample_duration_ms"],
            classification_threshold=0.88,
            ncols=2,
            figure_size=(7, 2),
            stimulus_scale=0.5,
            stimulus_linewidth=1.5,
            only_decreases=True,
        )

        # Save the figure
        fig_path = FIGURE_DIR / f"model_inference_{feature_list_str}.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)  # Free memory
        logging.info(f"Saved figure to {fig_path}")


def model_performance_per_participant(cache: ModelDataCache):
    """Analyze model performance per participant using cached data."""
    results = {}

    for feature_list in feature_lists:
        feature_list_str = "_".join(feature_list)

        # Get cached model and data
        model, _ = cache.get_model_and_config(feature_list_str)
        datasets, loaders, test_groups, _ = cache.get_data_and_loaders(
            feature_list, feature_list_str
        )
        _, test_loader = loaders

        result_df = analyze_per_participant(
            model,
            test_loader,
            test_groups,
            threshold=0.50,
        )
        results[feature_list_str] = result_df

    feature_set_acc, _ = plot_feature_accuracy_comparison(results, figsize=(10, 6))
    feature_set_acc_by_participant, _ = plot_participant_accuracy_comparison(
        results, figsize=(13, 6)
    )

    # Save results
    feature_set_acc.savefig(
        FIGURE_DIR / "feature_set_acc.png", dpi=300, bbox_inches="tight"
    )
    plt.close(feature_set_acc)

    feature_set_acc_by_participant.savefig(
        FIGURE_DIR / "feature_set_acc_by_participant.png", dpi=300, bbox_inches="tight"
    )
    plt.close(feature_set_acc_by_participant)

    # Save samples size per test set participant
    results["_".join(feature_lists[0])].drop("accuracy").write_json(
        FIGURE_DIR / "samples_per_test_participant.json"
    )


def model_performance(cache: ModelDataCache):
    """Analyze overall model performance using cached data."""
    results = {}
    winning_models = {}

    for feature_list in feature_lists:
        feature_list_str = "_".join(feature_list)

        # Get cached model and data
        model, model_config = cache.get_model_and_config(feature_list_str)
        datasets, loaders, _, _ = cache.get_data_and_loaders(
            feature_list, feature_list_str
        )
        _, test_loader = loaders

        winning_models[feature_list_str] = {
            feature_list_str: model_config["model_name"]
        }

        probs, y_true = get_model_predictions(model, test_loader)
        results[feature_list_str] = (probs, y_true)

    roc_curves = plot_multiple_roc_curves(results)

    # Create performance table
    performance_df = create_performance_table(
        results, winning_models=winning_models, threshold=0.5
    )

    # Save the figures
    roc_curves.savefig(FIGURE_DIR / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(roc_curves)

    performance_df.write_json(FIGURE_DIR / "performance_table.json")


if __name__ == "__main__":
    configure_logging(
        stream_level=logging.DEBUG,
        ignore_libs=["matplotlib", "Comm", "bokeh", "tornado", "filelock"],
    )

    # Create cache instance to share across functions
    cache = ModelDataCache()

    # Run all analyses with shared cache
    # model_performance_per_participant(cache)
    model_performance(cache)
    # model_inference(cache)

    # Log cache statistics
    logging.info(f"Loaded {len(cache.models)} unique models")
    logging.info(f"Cached {len(cache.raw_dfs)} dataframes")
