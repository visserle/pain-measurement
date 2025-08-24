import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tomllib
from dotenv import load_dotenv

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
    aggregate_accuracy_stats,
    analyze_per_participant,
    plot_feature_accuracy_boxplot,
)

load_dotenv()
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))

plt.style.use("./src/plots/style.mplstyle")

feature_lists = [
    ["eda_raw"],
    ["heart_rate"],
    ["pupil"],
    ["eda_raw", "heart_rate"],
    ["eda_raw", "pupil"],
    ["eda_raw", "heart_rate", "pupil"],
    ["face"],
    ["face", "eda_raw", "heart_rate", "pupil"],
    ["eeg"],
    ["eeg", "eda_raw"],
    ["eeg", "face", "eda_raw", "heart_rate", "pupil"],
]
feature_lists = expand_feature_list(feature_lists)


class InferenceCache:
    """Lightweight cache for model inference results only."""

    def __init__(self, cache_dir: Path = Path(".cache/model_inference")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_timestamp(self, model_path: Path) -> str:
        """Extract timestamp from model filename."""
        filename = model_path.name
        try:
            parts = filename.split("_")
            if len(parts) >= 2:
                timestamp_part = parts[-1].replace(".pt", "")
                return timestamp_part
        except:
            pass
        return str(datetime.fromtimestamp(model_path.stat().st_mtime))

    def _get_cache_key(self, feature_list_str: str, cache_type: str, **kwargs) -> str:
        """Generate unique cache key including any additional parameters."""
        key_parts = [feature_list_str, cache_type]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (list, tuple)):
                v = "_".join(map(str, v))
            key_parts.append(f"{k}_{v}")
        return "_".join(key_parts)

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given cache key."""
        return (
            self.cache_dir / f"{cache_key}.joblib"
        )  # Changed extension from .pkl to .joblib

    def _is_cache_valid(self, feature_list_str: str, model_path: Path) -> bool:
        """Check if cached data is still valid based on model timestamp."""
        timestamp_file = self.cache_dir / f"{feature_list_str}_timestamp.txt"

        if not timestamp_file.exists():
            return False

        cached_timestamp = timestamp_file.read_text().strip()
        current_timestamp = self._get_model_timestamp(model_path)

        return cached_timestamp == current_timestamp

    def _update_timestamp(self, feature_list_str: str, model_path: Path):
        """Update the model timestamp for cache validation."""
        timestamp_file = self.cache_dir / f"{feature_list_str}_timestamp.txt"
        timestamp = self._get_model_timestamp(model_path)
        timestamp_file.write_text(timestamp)

    def get(self, feature_list_str: str, cache_type: str, **kwargs) -> Any:
        """Get cached data if available and valid."""
        cache_key = self._get_cache_key(feature_list_str, cache_type, **kwargs)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                data = joblib.load(cache_path)
                logging.debug(f"Cache hit: {cache_key}")
                return data
            except Exception as e:
                logging.warning(f"Failed to load cache {cache_key}: {e}")

        return None

    def set(self, feature_list_str: str, cache_type: str, data: Any, **kwargs):
        """Save data to cache."""
        cache_key = self._get_cache_key(feature_list_str, cache_type, **kwargs)
        cache_path = self._get_cache_path(cache_key)

        try:
            # Use joblib instead of pickle
            joblib.dump(data, cache_path, compress=3)  # Use compression level 3
            logging.debug(f"Cached: {cache_key}")
        except Exception as e:
            logging.warning(f"Failed to cache {cache_key}: {e}")

    def clear(self, feature_list_str: str = None):
        """Clear cache for specific feature list or all."""
        if feature_list_str:
            pattern = f"{feature_list_str}_*"
            for file in self.cache_dir.glob(pattern):
                file.unlink()
                logging.info(f"Removed cache: {file.name}")
        else:
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink()
            for file in self.cache_dir.glob("*.txt"):
                file.unlink()
            logging.info("Cleared all cache")


def load_model_and_data(feature_list: list, feature_list_str: str) -> tuple:
    """Load model and prepare data - no caching, direct loading."""
    # Load model
    json_path = Path(f"results/experiment_{feature_list_str}/results.json")
    dictionary = json.loads(json_path.read_text())
    model_path = Path(dictionary["overall_best"]["model_path"].replace("\\", "/"))

    logging.info(f"Loading model for {feature_list_str}")
    (
        model,
        feature_list_loaded,
        sample_duration_ms,
        intervals,
        label_mapping,
        offsets_ms,
    ) = load_model(model_path, device="cpu")

    # Load data
    logging.info(f"Loading data for {feature_list_str}")
    df = load_data_from_database(feature_list)

    # Prepare data
    _, _, _, _, X_train_val, y_train_val, X_test, y_test = prepare_data(
        df=df,
        feature_list=feature_list_loaded,
        sample_duration_ms=sample_duration_ms,
        intervals=intervals,
        label_mapping=label_mapping,
        offsets_ms=offsets_ms,
        random_seed=RANDOM_SEED,
    )

    # Get test groups
    test_groups = prepare_data(
        df=df,
        feature_list=feature_list_loaded,
        sample_duration_ms=sample_duration_ms,
        intervals=intervals,
        label_mapping=label_mapping,
        offsets_ms=offsets_ms,
        random_seed=RANDOM_SEED,
        only_return_test_groups=True,
    )

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        X_train_val, y_train_val, X_test, y_test, batch_size=64
    )

    model_config = {
        "feature_list": feature_list_loaded,
        "sample_duration_ms": sample_duration_ms,
        "intervals": intervals,
        "label_mapping": label_mapping,
        "offsets_ms": offsets_ms,
        "model_name": model.__class__.__name__,
        "model_path": model_path,
    }

    return model, model_config, df, test_loader, test_groups


def model_inference(cache: InferenceCache, classification_threshold: float = 0.9):
    """Run model inference analysis with caching of results only."""
    config_path = Path("src/experiments/measurement/measurement_config.toml")
    with open(config_path, "rb") as file:
        config = tomllib.load(file)
    stimulus_seeds = config["stimulus"]["seeds"]
    logging.info(f"Using seeds for stimulus generation: {stimulus_seeds}")

    for feature_list in feature_lists:
        feature_list_str = "_".join(feature_list)

        # Check cache validity first
        json_path = Path(f"results/experiment_{feature_list_str}/results.json")
        dictionary = json.loads(json_path.read_text())
        model_path = Path(dictionary["overall_best"]["model_path"].replace("\\", "/"))

        # Try to get cached inference results
        cache_valid = cache._is_cache_valid(feature_list_str, model_path)

        if cache_valid:
            cached_results = cache.get(
                feature_list_str, "inference_probabilities", seeds=tuple(stimulus_seeds)
            )

            if cached_results:
                logging.info(f"Using cached inference results for {feature_list_str}")
                all_probabilities = cached_results["probabilities"]
                sample_duration_ms = cached_results["sample_duration_ms"]
            else:
                cached_results = None
        else:
            cached_results = None

        if cached_results is None:
            # Load model and data (not cached)
            model, model_config, df, _, test_groups = load_model_and_data(
                feature_list, feature_list_str
            )

            test_ids = np.unique(test_groups)

            # Analyze the entire test dataset
            all_probabilities = {}

            for stimulus_seed in stimulus_seeds:
                probabilities, _ = analyze_test_dataset_for_one_stimulus(
                    df,
                    model,
                    model_config["feature_list"],
                    test_ids,
                    stimulus_seed,
                    model_config["sample_duration_ms"],
                )
                all_probabilities[stimulus_seed] = probabilities

            # Cache the results
            cache._update_timestamp(feature_list_str, model_config["model_path"])
            cache.set(
                feature_list_str,
                "inference_probabilities",
                {
                    "probabilities": all_probabilities,
                    "sample_duration_ms": model_config["sample_duration_ms"],
                },
                seeds=tuple(stimulus_seeds),
            )

            sample_duration_ms = model_config["sample_duration_ms"]

        # Plot all available stimuli
        fig = plot_prediction_confidence_heatmap(
            all_probabilities,
            sample_duration_ms,
            classification_threshold=classification_threshold,
            ncols=2,
            figure_size=(7, 2),
            stimulus_scale=0.5,
            stimulus_linewidth=1.5,
            only_decreases=True,
        )

        # Save the figure
        fig_path = FIGURE_DIR / f"model_inference_{feature_list_str}.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        logging.info(f"Saved figure to {fig_path}")


def model_performance_per_participant(cache: InferenceCache):
    """Analyze model performance per participant with result caching."""
    results = {}

    for feature_list in feature_lists:
        feature_list_str = "_".join(feature_list)

        # Check cache validity
        json_path = Path(f"results/experiment_{feature_list_str}/results.json")
        dictionary = json.loads(json_path.read_text())
        model_path = Path(dictionary["overall_best"]["model_path"].replace("\\", "/"))

        # Try to get cached results
        if cache._is_cache_valid(feature_list_str, model_path):
            cached_result = cache.get(feature_list_str, "per_participant_results")
            if not cached_result.is_empty():
                logging.info(
                    f"Using cached per-participant results for {feature_list_str}"
                )
                results[feature_list_str] = cached_result
                continue

        # Load and analyze if not cached
        model, model_config, _, test_loader, test_groups = load_model_and_data(
            feature_list, feature_list_str
        )

        result_df = analyze_per_participant(
            model,
            test_loader,
            test_groups,
            threshold=0.50,
        )
        results[feature_list_str] = result_df

        # Cache the result
        cache._update_timestamp(feature_list_str, model_path)
        cache.set(feature_list_str, "per_participant_results", result_df)

    # Create plots
    feature_set_acc, _ = plot_feature_accuracy_boxplot(results, figsize=(10, 6))

    # Create accuracy table
    accuracy_stats = aggregate_accuracy_stats(
        results, include_ci=False, sort_by_avg=False
    )

    # Save results
    feature_set_acc_path = FIGURE_DIR / "feature_set_acc.png"
    feature_set_acc.savefig(feature_set_acc_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved figure to {feature_set_acc_path}")
    plt.close(feature_set_acc)

    # Save samples size per test set participant
    samples_path = FIGURE_DIR / "samples_per_test_participant.json"
    results["_".join(feature_lists[0])].drop("accuracy").write_json(samples_path)
    logging.info(f"Saved data to {samples_path}")

    # Save accuracy stats
    accuracy_stats_path = FIGURE_DIR / "accuracy_stats_per_participant.json"
    accuracy_stats.write_json(accuracy_stats_path)
    logging.info(f"Saved data to {accuracy_stats_path}")


def model_performance(cache: InferenceCache):
    """Analyze overall model performance with prediction caching."""
    results = {}
    winning_models = {}

    for feature_list in feature_lists:
        feature_list_str = "_".join(feature_list)

        # Check cache validity
        json_path = Path(f"results/experiment_{feature_list_str}/results.json")
        dictionary = json.loads(json_path.read_text())
        model_path = Path(dictionary["overall_best"]["model_path"].replace("\\", "/"))

        # Try to get cached predictions
        if cache._is_cache_valid(feature_list_str, model_path):
            cached_predictions = cache.get(feature_list_str, "test_predictions")
            if cached_predictions:
                logging.info(f"Using cached predictions for {feature_list_str}")
                results[feature_list_str] = (
                    cached_predictions["probs"],
                    cached_predictions["y_true"],
                )
                winning_models[feature_list_str] = {
                    feature_list_str: cached_predictions["model_name"]
                }
                continue

        # Load and predict if not cached
        model, model_config, _, test_loader, _ = load_model_and_data(
            feature_list, feature_list_str
        )

        winning_models[feature_list_str] = {
            feature_list_str: model_config["model_name"]
        }

        probs, y_true = get_model_predictions(model, test_loader)
        results[feature_list_str] = (probs, y_true)

        # Cache the predictions
        cache._update_timestamp(feature_list_str, model_path)
        cache.set(
            feature_list_str,
            "test_predictions",
            {
                "probs": probs,
                "y_true": y_true,
                "model_name": model_config["model_name"],
            },
        )

    # Create plots and tables
    roc_curves = plot_multiple_roc_curves(results)
    performance_df = create_performance_table(
        results, winning_models=winning_models, threshold=0.5
    )

    # Save the figures
    roc_curves_path = FIGURE_DIR / "roc_curves.png"
    roc_curves.savefig(roc_curves_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved figure to {roc_curves_path}")
    plt.close(roc_curves)

    performance_table_path = FIGURE_DIR / "performance_table.json"
    performance_df.write_json(performance_table_path)
    logging.info(f"Saved data to {performance_table_path}")


if __name__ == "__main__":
    configure_logging(
        stream_level=logging.DEBUG,
        ignore_libs=["matplotlib", "Comm", "bokeh", "tornado", "filelock"],
    )

    # Create lightweight cache instance
    cache = InferenceCache()

    # Run all analyses with lightweight caching
    model_performance_per_participant(cache)
    # model_performance(cache)
    # model_inference(cache, classification_threshold=0.5)

    logging.info("Completed all model plots")
