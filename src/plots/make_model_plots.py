import json
import logging
import os
from pathlib import Path

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
    analyze_per_participant,
    plot_feature_accuracy_comparison,
    plot_participant_accuracy_comparison,
)

load_dotenv()
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))

plt.style.use("./src/plots/style.mplstyle")

feature_lists = [
    ["eda_raw"],
    # ["heart_rate"],
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


def model_inferece():
    config_path = Path("src/experiments/measurement/measurement_config.toml")
    with open(config_path, "rb") as file:
        config = tomllib.load(file)
    stimulus_seeds = config["stimulus"]["seeds"]
    logging.info(f"Using seeds for stimulus generation: {stimulus_seeds}")

    for feature_list in feature_lists:
        feature_list_str = "_".join(feature_list)
        # Load data from database
        df = load_data_from_database(feature_list=feature_list)

        # Load model
        json_path = Path(f"results/experiment_{feature_list_str}/results.json")
        dictionary = json.loads(json_path.read_text())
        model_path = Path(dictionary["overall_best"]["model_path"].replace("\\", "/"))

        (
            model,
            feature_list,
            sample_duration_ms,
            intervals,
            label_mapping,
            offsets_ms,
        ) = load_model(model_path, device="cpu")

        # Prepare data
        _, _, _, _, X_train_val, y_train_val, X_test, y_test = prepare_data(
            df=df,
            feature_list=feature_list,
            sample_duration_ms=sample_duration_ms,
            intervals=intervals,
            label_mapping=label_mapping,
            offsets_ms=offsets_ms,
            random_seed=RANDOM_SEED,
        )
        test_groups = prepare_data(
            df=df,
            feature_list=feature_list,
            sample_duration_ms=sample_duration_ms,
            intervals=intervals,
            label_mapping=label_mapping,
            offsets_ms=offsets_ms,
            random_seed=RANDOM_SEED,
            only_return_test_groups=True,
        )
        test_ids = np.unique(test_groups)
        # train data is not used in this analysis, but we need to create the dataloaders
        _, test_loader = create_dataloaders(
            X_train_val, y_train_val, X_test, y_test, batch_size=64
        )

        # Analyze the entire test dataset
        all_probabilities = {}
        all_participant_trials = {}

        for stimulus_seed in stimulus_seeds:
            probabilities, participant_trials = analyze_test_dataset_for_one_stimulus(
                df,
                model,
                feature_list,
                test_ids,
                stimulus_seed,
                sample_duration_ms,
            )

            all_probabilities[stimulus_seed] = probabilities
            all_participant_trials[stimulus_seed] = participant_trials

        # Plot all available stimuli
        fig = plot_prediction_confidence_heatmap(
            all_probabilities,
            sample_duration_ms,
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
        logging.info(f"Saved figure to {fig_path}")


def model_performance_per_participant():
    results = {}

    for feature_list in feature_lists:
        feature_list_str = "_".join(feature_list)
        # Load data from database
        df = load_data_from_database(feature_list=feature_list)
        # Load model
        json_path = Path(f"results/experiment_{feature_list_str}/results.json")
        dictionary = json.loads(json_path.read_text())
        model_path = Path(dictionary["overall_best"]["model_path"].replace("\\", "/"))

        (
            model,
            feature_list,
            sample_duration_ms,
            intervals,
            label_mapping,
            offsets_ms,
        ) = load_model(model_path, device="cpu")
        assert "_".join(feature_list) == feature_list_str

        # Prepare data
        _, _, _, _, X_train_val, y_train_val, X_test, y_test = prepare_data(
            df=df,
            feature_list=feature_list,
            sample_duration_ms=sample_duration_ms,
            intervals=intervals,
            label_mapping=label_mapping,
            offsets_ms=offsets_ms,
            random_seed=RANDOM_SEED,
        )
        test_groups = prepare_data(
            df=df,
            feature_list=feature_list,
            sample_duration_ms=sample_duration_ms,
            intervals=intervals,
            label_mapping=label_mapping,
            offsets_ms=offsets_ms,
            random_seed=RANDOM_SEED,
            only_return_test_groups=True,
        )
        _, test_loader = create_dataloaders(
            X_train_val, y_train_val, X_test, y_test, batch_size=64
        )

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
    feature_set_acc_by_participant.savefig(
        FIGURE_DIR / "feature_set_acc_by_participant.png", dpi=300, bbox_inches="tight"
    )
    # Save samples size per test set participant
    results["_".join(feature_lists[0])].drop("accuracy").write_json(
        FIGURE_DIR / "samples_per_test_participant.json"
    )


def model_performance():
    results = {}
    winning_models = {}

    for feature_list in feature_lists:
        feature_list_str = "_".join(feature_list)

        # Load data from database
        df = load_data_from_database(feature_list)

        # Load model
        json_path = Path(f"results/experiment_{feature_list_str}/results.json")
        dictionary = json.loads(json_path.read_text())
        model_path = Path(dictionary["overall_best"]["model_path"].replace("\\", "/"))

        (
            model,
            feature_list,
            sample_duration_ms,
            intervals,
            label_mapping,
            offsets_ms,
        ) = load_model(model_path, device="cpu")
        assert "_".join(feature_list) == feature_list_str
        winning_models[feature_list_str] = {feature_list_str: model.__class__.__name__}

        # Prepare data
        _, _, _, _, X_train_val, y_train_val, X_test, y_test = prepare_data(
            df=df,
            feature_list=feature_list,
            sample_duration_ms=sample_duration_ms,
            intervals=intervals,
            label_mapping=label_mapping,
            offsets_ms=offsets_ms,
            random_seed=RANDOM_SEED,
        )
        _, test_loader = create_dataloaders(
            X_train_val, y_train_val, X_test, y_test, batch_size=64
        )
        probs, y_true = get_model_predictions(
            model,
            test_loader,
        )
        results[feature_list_str] = (probs, y_true)

    roc_curves = plot_multiple_roc_curves(results)

    # Create performance table
    performance_df = create_performance_table(
        results, winning_models=winning_models, threshold=0.5
    )

    # Save the figure
    roc_curves.savefig(FIGURE_DIR / "roc_curves.png", dpi=300, bbox_inches="tight")
    pr_curves.savefig(FIGURE_DIR / "pr_curves.png", dpi=300, bbox_inches="tight")
    performance_df.write_json(FIGURE_DIR / "performance_table.json")


if __name__ == "__main__":
    configure_logging(
        stream_level=logging.DEBUG,
        ignore_libs=["matplotlib", "Comm", "bokeh", "tornado", "filelock"],
    )

    model_performance_per_participant()
    model_performance()
    model_inferece()
