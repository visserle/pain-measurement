# Models Pipeline

This directory contains the machine learning pipeline for training and evaluating pain classification models.

## Overview

The pipeline supports training various deep learning architectures on physiological time-series data. It includes data loading, preprocessing (windowing, balancing, scaling), hyperparameter optimization via Optuna, and seed stability analysis.

## Usage

The main entry point is `main.py`.

### Train and Optimize Models
To train models with specific features and architectures:

```bash
python -m src.models.main --features eda_raw heart_rate --models PatchTST MLP
```

Arguments:
- `--features`: List of features to use (e.g., `eda_raw`, `heart_rate`, `pupil_diameter`, `face`, `eeg`).
- `--models`: List of model architectures to evaluate.
- `--trials`: Number of Optuna trials for hyperparameter optimization (default: 30).

## Key Components

### `main.py`
Orchestrates the training process. It parses arguments, loads data, and triggers either the model selection loop or the stability evaluation.

### `data_preparation.py`
Handles data loading and transformation:
- **Sample Creation**: Cuts continuous time-series into defined windows (`sample_creation.py`).
- **Balancing**: Undersamples the majority class to ensure a balanced dataset.
- **Splitting**: Uses `GroupShuffleSplit` to ensure subjects are not shared between train/val/test sets.
- **Scaling**: Applies 3D scaling (standardization) to the features.

### `model_selection.py`
Manages the hyperparameter optimization loop using Optuna. It trains multiple candidates and selects the best model based on validation accuracy, then retrains it on the combined train+val set.

### `architectures/`
Contains the PyTorch implementations of various models, including:
- **Time-Series Models**: `PatchTST`, `TimesNet`, `LightTS`, `iTransformer`, `FEDformer`.
- **General Deep Learning**: `MLP`, `ResNet`, `LSTM`.
- **Domain-Specific**: `EEGNet` (for EEG data).
- **Ensembles**: `EEGPhysioEnsemble`, `EEGFacePhysioEnsemble`.

## Configuration

Global experiment settings are defined in `main_config.py`, for instance:
- `RANDOM_SEED`: Base seed for reproducibility.
- `SAMPLE_DURATION_MS`: Length of the input window (default: 7000 ms).
- `INTERVALS`: Definitions of class intervals (increasing vs. decreasing temperature).
